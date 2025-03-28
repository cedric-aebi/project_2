import joblib
import xgboost as xgb

from flwr.common import Context, Parameters, Scalar
from flwr.common.config import unflatten_dict
from flwr.server import ServerConfig, ServerAppComponents, ServerApp
from flwr.server.strategy import FedXgbBagging

from learning_methods.federated.xgboost_.task import load_data, replace_keys


def get_evaluate_fn(test_data, params):
    """Return a function for centralised evaluation."""

    def evaluate_fn(server_round: int, parameters: Parameters, config: dict[str, Scalar]):
        # If at the first round, skip the evaluation
        if server_round == 0:
            return 0, {}
        else:
            bst = xgb.Booster(params=params)
            for para in parameters.tensors:
                para_b = bytearray(para)

            # Load global model
            bst.load_model(para_b)
            # Run evaluation
            eval_results = bst.eval_set(
                evals=[(test_data, "valid")],
                iteration=bst.num_boosted_rounds() - 1,
            )
            auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

            # Save results to disk.
            # Note we add new entry to the same file with each call to this function.
            with open(f"./centralised_eval.txt", "a", encoding="utf-8") as fp:
                fp.write(f"Round:{server_round},AUC:{auc}\n")

            joblib.dump(params, "./params.pkl")
            joblib.dump(bst, "./model.pkl")

            return 0, {"AUC": auc}

    return evaluate_fn


def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    metrics_aggregated = {"AUC": auc_aggregated}
    return metrics_aggregated


def config_func(rnd: int) -> dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def server_fn(context: Context):
    # Read from config
    cfg = replace_keys(unflatten_dict(context.run_config))

    num_rounds = cfg["num_server_rounds"]
    fraction_fit = cfg["fraction_fit"]
    fraction_evaluate = cfg["fraction_evaluate"]
    params = cfg["params"]

    _, test_dmatrix, _, _ = load_data(which="all")

    # Init an empty Parameter
    parameters = Parameters(tensor_type="", tensors=[])

    # Define strategy
    strategy = FedXgbBagging(
        evaluate_function=get_evaluate_fn(test_dmatrix, params),
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        on_evaluate_config_fn=config_func,
        on_fit_config_fn=config_func,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(
    server_fn=server_fn,
)
