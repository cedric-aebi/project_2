import xgboost as xgb
from flwr.client import Client
from flwr.common import FitIns, FitRes, Status, Code, Parameters, EvaluateIns, EvaluateRes, Context
from sklearn.metrics import f1_score

from enums.Dataset import Dataset
from enums.Participant import NurseParticipant, StressParticipant
from enums.ResamplingMethod import ResamplingMethod
from enums.ScalingMethod import ScalingMethod
from learning_methods.federated.xgboost_.task import evaluate, load_data_nurse, load_data_stress
from service.exportservice.ExportService import ExportService


class FlowerClient(Client):
    def __init__(
        self,
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        params,
        participant,
        mongo_id: str,
        run_index: int,
        database: str,
    ):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params
        self.participant = participant
        self.mongo_id = mongo_id
        self.run_index = run_index
        self.export_service = ExportService(collection="federated", database=database)

    def _local_boost(self, bst_input):
        # Update trees based on local training data.
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees for sever aggregation
        bst = bst_input[bst_input.num_boosted_rounds() - self.num_local_round : bst_input.num_boosted_rounds()]

        return bst

    def custom_f1_score(self, preds, dmatrix):
        labels = dmatrix.get_label()
        preds_class = (preds > 0.5).astype(int)
        f1 = f1_score(labels, preds_class)
        return "f1", f1

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        if global_round == 1:
            # First round local training
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                custom_metric=self.custom_f1_score,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )
        else:
            bst = xgb.Booster(params=self.params)
            global_model = bytearray(ins.parameters.tensors[0])

            # Load global model into booster
            bst.load_model(global_model)

            # Local training
            bst = self._local_boost(bst)

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model
        global_round = int(ins.config["global_round"])
        bst = xgb.Booster(params=self.params)
        para_b = bytearray(ins.parameters.tensors[0])
        bst.load_model(para_b)

        preds_train = bst.predict(self.train_dmatrix)
        y_true_train = self.train_dmatrix.get_label()
        y_pred_train = (preds_train > 0.5).astype(int)
        scores_train, _ = evaluate(pred=y_pred_train, y_true=y_true_train)

        preds_test = bst.predict(self.valid_dmatrix)
        y_true_test = self.valid_dmatrix.get_label()
        y_pred_test = (preds_test > 0.5).astype(int)
        scores_test, _ = evaluate(pred=y_pred_test, y_true=y_true_test)

        scores = {"training_set": scores_train, "testing_set": scores_test}

        self.export_service.update_run(
            run_id=self.mongo_id,
            set_dict={
                "$set": {
                    f"training_runs.{self.run_index}.clients.{self.participant}.round.{str(global_round)}.scores": scores
                }
            },
        )

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_val,
            metrics={"f1": round(scores_test["f1"], 4), "server_round": global_round},
        )


def get_client_fn(
    cfg: dict,
    mongo_id: str,
    run_index: int,
    scaling_method: ScalingMethod | None,
    resampling_method: ResamplingMethod | None,
    database: str,
    participant_leave_out: NurseParticipant | StressParticipant,
    dataset: Dataset,
    with_features: bool,
):
    def client_fn(context: Context):
        # Load model and data
        partition_id = context.node_config["partition-id"]
        if dataset == Dataset.NURSE:
            train_dmatrix, valid_dmatrix, num_train, num_val, participant = load_data_nurse(
                which=partition_id,
                scaling_method=scaling_method,
                resampling_method=resampling_method,
                participant_leave_out=participant_leave_out,
                with_features=with_features,
            )
        else:
            train_dmatrix, valid_dmatrix, num_train, num_val, participant = load_data_stress(
                which=partition_id,
                scaling_method=scaling_method,
                resampling_method=resampling_method,
                participant_leave_out=participant_leave_out,
                with_features=with_features,
            )

        num_local_round = cfg["local_epochs"]

        # Return Client instance
        return FlowerClient(
            train_dmatrix=train_dmatrix,
            valid_dmatrix=valid_dmatrix,
            num_train=num_train,
            num_val=num_val,
            num_local_round=num_local_round,
            params=cfg["params"],
            participant=participant,
            mongo_id=mongo_id,
            run_index=run_index,
            database=database,
        )

    return client_fn
