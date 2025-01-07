from flwr.common import Context, Metrics
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig, ServerApp, ServerAppComponents
from flwr.server.strategy import FedAvg
from task import load_model


# Define metric aggregation function
def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    # Multiply f1 of each client by number of examples used
    f1_scores = [num_examples * m["f1"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"f1": sum(f1_scores) / sum(examples)}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    # Let's define the global model and pass it to the strategy
    parameters = ndarrays_to_parameters(
        load_model(
            dropout=None,
            batch_normalization=False,
            regularization=False,
            learning_rate=0.001,
            optimizer="adam",
            input_shape=144,
        ).get_weights()
    )

    # Define the strategy
    strategy = FedAvg(
        fraction_fit=context.run_config["fraction-fit"],
        fraction_evaluate=1.0,
        min_available_clients=34,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
