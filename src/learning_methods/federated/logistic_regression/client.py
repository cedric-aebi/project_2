import warnings
from typing import Callable

from flwr.client import NumPyClient
from flwr.common import Context
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from enums.Participant import NurseParticipant
from enums.ResamplingMethod import ResamplingMethod
from enums.ScalingMethod import ScalingMethod
from learning_methods.federated.logistic_regression.task import (
    create_log_reg_and_instantiate_parameters,
    set_model_params,
    get_model_parameters,
    load_data,
    evaluate,
)
from service.exportservice.ExportService import ExportService


# Define Flower client
class FlowerClient(NumPyClient):
    def __init__(
        self, model: LogisticRegression, data: tuple, participant, mongo_id: str, run_index: int, database: str
    ):
        self.model = model
        self.mongo_id = mongo_id
        self.run_index = run_index
        self.participant = participant
        self.x_train, self.x_test, self.y_train, self.y_test = data
        self.export_service = ExportService(collection="federated", database=database)

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.x_train, self.y_train)
        return get_model_parameters(self.model), len(self.x_train), {}

    def evaluate(self, parameters, config):
        global_round = int(config["global_round"])

        set_model_params(self.model, parameters)

        training_loss = log_loss(self.y_train, self.model.predict_proba(self.x_train))
        test_loss = log_loss(self.y_test, self.model.predict_proba(self.x_test))

        y_pred_train = self.model.predict(self.x_train)
        y_pred_train = (y_pred_train > 0.5).astype(int)
        scores_train, _ = evaluate(pred=y_pred_train, y_true=self.y_train)
        scores_train["loss"] = training_loss

        y_pred_test = self.model.predict(self.x_test)
        y_pred_test = (y_pred_test > 0.5).astype(int)
        scores_test, _ = evaluate(pred=y_pred_test, y_true=self.y_test)
        scores_test["loss"] = test_loss

        scores = {"training_set": scores_train, "testing_set": scores_test}

        self.export_service.update_run(
            run_id=self.mongo_id,
            set_dict={
                "$set": {
                    f"training_runs.{self.run_index}.clients.{self.participant}.round.{str(global_round)}.scores": scores
                }
            },
        )

        return test_loss, len(self.x_test), {"f1": scores_test["f1"], "server_round": global_round}


def get_client_fn(
    cfg: dict,
    mongo_id: str,
    run_index: int,
    scaling_method: ScalingMethod | None,
    resampling_method: ResamplingMethod | None,
    database: str,
    participant_leave_out: NurseParticipant,
) -> Callable:
    def client_fn(context: Context):
        """Construct a Client that will be run in a ClientApp."""

        # Read the node_config to fetch data partition associated to this node
        partition_id = context.node_config["partition-id"]
        x_train, x_test, y_train, y_test, participant = load_data(
            which=partition_id,
            scaling_method=scaling_method,
            resampling_method=resampling_method,
            participant_leave_out=participant_leave_out,
        )

        # Read run_config to fetch hyperparameters relevant to this run
        params = cfg["params"]
        penalty = params["penalty"]
        max_iter = params["max_iter"]

        model = create_log_reg_and_instantiate_parameters(
            penalty=penalty, max_iter=max_iter, num_features=x_train.shape[1]
        )

        # Return Client instance
        return FlowerClient(
            model=model,
            data=(x_train, x_test, y_train, y_test),
            participant=participant,
            mongo_id=mongo_id,
            run_index=run_index,
            database=database,
        ).to_client()

    return client_fn
