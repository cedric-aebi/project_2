import joblib
import pandas as pd
from flwr.common import Context, Metrics
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig, ServerApp, ServerAppComponents
from flwr.server.strategy import FedAvg, FedProx
from sklearn.metrics import f1_score

from task import load_model, load_data


def gen_evaluate_fn(
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    input_shape: int,
    learning_rate: float,
    dropout: float | None,
    batch_normalization: bool,
    regularization: bool,
    optimizer: str,
):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
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
        f1 = f1_score(y_true=y_test, y_pred=y_pred > 0.5)

        joblib.dump(model, "model.pkl")
        return loss, {"centralized_f1": f1}

    return evaluate


def average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    # Extract f1 scores from metrics
    f1_scores = [m["f1"] for _, m in metrics]

    # Calculate and return the average f1 score
    return {"f1": sum(f1_scores) / len(f1_scores)}


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    # Multiply f1 of each client by number of examples used
    f1_scores = [num_examples * m["f1"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"f1": sum(f1_scores) / sum(examples)}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    regularization = context.run_config["regularization"]
    learning_rate = context.run_config["learning-rate"]
    optimizer = context.run_config["optimizer"]
    batch_normalization = context.run_config["batch-normalization"]
    dropout = None if context.run_config["dropout"] == False else context.run_config["dropout"]

    # Initialize model parameters
    ndarrays = load_model(
        dropout=dropout,
        batch_normalization=batch_normalization,
        regularization=regularization,
        learning_rate=learning_rate,
        optimizer=optimizer,
        input_shape=144,
    ).get_weights()
    parameters = ndarrays_to_parameters(ndarrays)

    x_train, x_test, y_train, y_test = load_data(subject="all")

    # Define the strategy
    strategy = FedAvg(
        fraction_fit=context.run_config["fraction-fit"],
        fraction_evaluate=1.0,
        min_available_clients=34,
        initial_parameters=parameters,
        evaluate_fn=gen_evaluate_fn(
            x_test=x_test,
            y_test=y_test,
            dropout=dropout,
            batch_normalization=batch_normalization,
            regularization=regularization,
            learning_rate=learning_rate,
            optimizer=optimizer,
            input_shape=144,
        ),
        evaluate_metrics_aggregation_fn=average,
    )
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
