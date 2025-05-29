from pathlib import Path

import joblib
import pandas as pd
from flwr.common import Context, Metrics
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg

from enums.Dataset import Dataset
from utils import utils
from enums.Participant import NurseParticipant, StressParticipant
from enums.ScalingMethod import ScalingMethod
from learning_methods.federated.shallow_nn.task import load_model, evaluate
from service.exportservice.ExportService import ExportService


def gen_evaluate_fn(
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    input_shape: int,
    learning_rate: float,
    dropout: float | None,
    batch_normalization: bool,
    regularization: bool,
    optimizer: str,
    num_rounds: int,
    mongo_id: str,
    run_index: int,
    export_service: ExportService,
):
    """Generate the function for centralized evaluation."""

    def evaluate_fn(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        model = load_model(
            dropout=dropout,
            batch_normalization=batch_normalization,
            regularization=regularization,
            learning_rate=learning_rate,
            optimizer=optimizer,
            input_shape=input_shape,
        )
        model.set_weights(parameters_ndarrays)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
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
    participant_leave_out: NurseParticipant | StressParticipant,
    export_service: ExportService,
    scaling_method: ScalingMethod | None,
    dataset: Dataset,
    with_features: bool,
):
    def server_fn(context: Context):
        """Construct components that set the ServerApp behaviour."""
        params = cfg["params"]
        regularization = params["regularization"]
        learning_rate = params["learning_rate"]
        optimizer = params["optimizer"]
        batch_normalization = params["batch_normalization"]
        dropout = None if params["dropout"] == False else params["dropout"]

        # Load data
        if dataset == Dataset.NURSE:
            if with_features:
                df = pd.read_pickle(
                    Path(__file__).parent.parent.parent.parent.parent
                    / "datasets"
                    / "nurse"
                    / "processed"
                    / "with_features"
                    / f"{participant_leave_out}.pkl"
                )
                x = df.drop(columns=["Label", "Participant", "Split"])
                y = df["Label"]
            else:
                df = pd.read_pickle(
                    Path(__file__).parent.parent.parent.parent.parent
                    / "datasets"
                    / "nurse"
                    / "processed"
                    / "no_features"
                    / f"{participant_leave_out}.pkl"
                )
                x = df.drop(columns=["Label", "Participant"])
                y = df["Label"]
        elif dataset == Dataset.STRESS:
            if with_features:
                df = pd.read_pickle(
                    Path(__file__).parent.parent.parent.parent.parent
                    / "datasets"
                    / "stress"
                    / "processed"
                    / "with_features"
                    / f"{participant_leave_out}.pkl"
                )
                x = df.drop(columns=["Label", "Participant", "Split"])
                y = df["Label"]
            else:
                df = pd.read_pickle(
                    Path(__file__).parent.parent.parent.parent.parent
                    / "datasets"
                    / "stress"
                    / "processed"
                    / "no_features"
                    / f"{participant_leave_out}.pkl"
                )
                x = df.drop(columns=["Label", "Participant"])
                y = df["Label"]
        else:
            raise ValueError(f"Dataset {dataset} is not supported.")

        scaler = utils.get_scaler(method=scaling_method)
        if scaler is not None:
            x = scaler.fit_transform(x)

        # Initialize model parameters
        ndarrays = load_model(
            dropout=dropout,
            batch_normalization=batch_normalization,
            regularization=regularization,
            learning_rate=learning_rate,
            optimizer=optimizer,
            input_shape=x.shape[1],
        ).get_weights()
        parameters = ndarrays_to_parameters(ndarrays)

        # Define the strategy
        strategy = FedAvg(
            fraction_fit=cfg["fraction_fit"],
            fraction_evaluate=cfg["fraction_evaluate"],
            min_available_clients=cfg["min_available_clients"],
            initial_parameters=parameters,
            on_evaluate_config_fn=config_func,
            on_fit_config_fn=config_func,
            evaluate_fn=gen_evaluate_fn(
                x_test=x,
                y_test=y,
                dropout=dropout,
                batch_normalization=batch_normalization,
                regularization=regularization,
                learning_rate=learning_rate,
                optimizer=optimizer,
                input_shape=x.shape[1],
                num_rounds=cfg["num_server_rounds"],
                mongo_id=mongo_id,
                run_index=run_index,
                export_service=export_service,
            ),
            evaluate_metrics_aggregation_fn=get_evaluate_metrics_aggregation_fn(
                mongo_id=mongo_id, run_index=run_index, export_service=export_service
            ),
        )

        config = ServerConfig(num_rounds=cfg["num_server_rounds"])

        return ServerAppComponents(strategy=strategy, config=config)

    return server_fn
