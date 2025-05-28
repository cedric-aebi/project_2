from pathlib import Path

import joblib
import pandas as pd
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from enums.Participant import NurseParticipant
from enums.ScalingMethod import ScalingMethod
from learning_methods.federated.logistic_regression.task import (
    create_log_reg_and_instantiate_parameters,
    get_model_parameters,
    set_initial_params,
    set_model_params,
    evaluate,
)
from service.exportservice.ExportService import ExportService
from utils import utils


def gen_evaluate_fn(
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    num_features: int,
    num_rounds: int,
    mongo_id: str,
    run_index: int,
    penalty: str,
    export_service: ExportService,
):
    """Generate the function for centralized evaluation."""

    model = LogisticRegression(penalty=penalty)
    set_initial_params(model=model, num_features=num_features)

    def evaluate_fn(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        set_model_params(model=model, params=parameters_ndarrays)

        loss = log_loss(y_test, model.predict_proba(x_test))
        y_pred = model.predict(x_test)
        y_pred = (y_pred > 0.5).astype(int)

        scores_test, cm = evaluate(pred=y_pred, y_true=y_test)
        scores_test["loss"] = loss
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
            joblib.dump(
                model,
                Path(__file__).parent.parent.parent.parent.parent
                / "results"
                / "federated"
                / "models"
                / f"{mongo_id}_run_index_{run_index}_model.pkl",
            )
        return loss, {"centralized_f1": scores_test["f1"]}

    return evaluate_fn


def get_evaluate_metrics_aggregation_fn(mongo_id: str, run_index: int, export_service: ExportService):
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
    def server_fn(context: Context):
        """Construct components that set the ServerApp behaviour."""
        params = cfg["params"]

        df = pd.read_pickle(
            Path(__file__).parent.parent.parent.parent.parent
            / "datasets"
            / "nurse"
            / "paper"
            / f"{participant_leave_out}.pkl"
        )
        x = df.drop(columns=["Label", "Participant"])
        y = df["Label"]

        scaler = utils.get_scaler(method=scaling_method)
        if scaler is not None:
            x = scaler.fit_transform(x)

        penalty = params["penalty"]
        max_iter = params["max_iter"]

        model = create_log_reg_and_instantiate_parameters(penalty=penalty, max_iter=max_iter, num_features=x.shape[1])
        ndarrays = get_model_parameters(model)
        global_model_init = ndarrays_to_parameters(ndarrays)

        # Define the strategy
        strategy = FedAvg(
            fraction_fit=cfg["fraction_fit"],
            fraction_evaluate=cfg["fraction_evaluate"],
            min_available_clients=cfg["min_available_clients"],
            initial_parameters=global_model_init,
            on_evaluate_config_fn=config_func,
            on_fit_config_fn=config_func,
            evaluate_fn=gen_evaluate_fn(
                x_test=x,
                y_test=y,
                num_features=x.shape[1],
                num_rounds=cfg["num_server_rounds"],
                mongo_id=mongo_id,
                run_index=run_index,
                export_service=export_service,
                penalty=penalty,
            ),
            evaluate_metrics_aggregation_fn=get_evaluate_metrics_aggregation_fn(
                mongo_id=mongo_id, run_index=run_index, export_service=export_service
            ),
        )

        config = ServerConfig(num_rounds=cfg["num_server_rounds"])

        return ServerAppComponents(strategy=strategy, config=config)

    return server_fn
