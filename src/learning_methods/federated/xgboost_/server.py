import pandas as pd
import xgboost as xgb

from flwr.common import Context, Parameters, Scalar, Metrics
from flwr.server import ServerConfig, ServerAppComponents
from flwr.server.strategy import FedXgbBagging

from enums.Participant import NurseParticipant
from enums.ScalingMethod import ScalingMethod
from learning_methods.federated.xgboost_.task import transform_dataset_to_dmatrix, evaluate
from service.exportservice.ExportService import ExportService
from utils import utils


def get_evaluate_fn(test_data, params, num_rounds, mongo_id, run_index, export_service):
    """Return a function for centralized evaluation."""

    def evaluate_fn(server_round: int, parameters: Parameters, config: dict[str, Scalar]):
        if server_round == 0:
            return 0, {}
        else:
            bst = xgb.Booster(params=params)
            for para in parameters.tensors:
                para_b = bytearray(para)

            # Load model
            bst.load_model(para_b)

            # Predict
            preds = bst.predict(test_data)
            y_true = test_data.get_label()
            y_pred = (preds > 0.5).astype(int)

            scores_test, cm = evaluate(pred=y_pred, y_true=y_true)
            scores = {"testing_set": scores_test}
            export_service.update_run(
                run_id=mongo_id,
                set_dict={
                    "$set": {
                        f"training_runs.{run_index}.clients.server.centralized.round.{str(server_round)}.scores": scores
                    }
                },
            )

            if server_round == num_rounds:
                pass
                # bst.save_model(f"./model_{mongo_id}.json")
                # joblib.dump(params, f"./params_{mongo_id}.pkl")

            return 0, {"f1": scores_test["f1"]}

    return evaluate_fn


def get_evaluate_metrics_aggregation_fn(mongo_id, run_index, export_service):
    def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
        # Multiply f1 of each client by number of examples used
        server_round = metrics[0][1]["server_round"]
        f1_scores = [num_examples * m["f1"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        final_score = sum(f1_scores) / sum(examples)

        export_service.update_run(
            run_id=mongo_id,
            set_dict={
                "$set": {
                    f"training_runs.{run_index}.clients.server.distributed.round.{str(server_round)}.scores": final_score
                }
            },
        )

        # Aggregate and return custom metric (weighted average)
        return {"f1": final_score}

    return weighted_average


def config_func(rnd: int) -> dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def get_server_fn(
    cfg: dict,
    mongo_id: str,
    run_index: int,
    participant_leave_out: NurseParticipant,
    export_service: ExportService,
    scaling_method: ScalingMethod | None,
):
    def server_fn(context: Context) -> ServerAppComponents:
        num_rounds = cfg["num_server_rounds"]
        fraction_fit = cfg["fraction_fit"]
        fraction_evaluate = cfg["fraction_evaluate"]
        params = cfg["params"]

        df = pd.read_pickle(f"../../../datasets/nurse/paper/{participant_leave_out}.pkl")
        x = df.drop(columns=["Label", "Participant"])
        y = df["Label"]

        scaler = utils.get_scaler(method=scaling_method)
        if scaler is not None:
            x = scaler.fit_transform(x)

        test_dmatrix = transform_dataset_to_dmatrix(x, y)

        # Init an empty Parameter
        parameters = Parameters(tensor_type="", tensors=[])

        # Define strategy
        strategy = FedXgbBagging(
            evaluate_function=get_evaluate_fn(
                test_data=test_dmatrix,
                params=params,
                num_rounds=num_rounds,
                mongo_id=mongo_id,
                run_index=run_index,
                export_service=export_service,
            ),
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            on_evaluate_config_fn=config_func,
            on_fit_config_fn=config_func,
            evaluate_metrics_aggregation_fn=get_evaluate_metrics_aggregation_fn(
                mongo_id=mongo_id, run_index=run_index, export_service=export_service
            ),
            initial_parameters=parameters,
        )
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)

    return server_fn
