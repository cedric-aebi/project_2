"""xgboost_quickstart: A Flower / XGBoost app."""

from logging import INFO
from typing import Dict

import joblib
import xgboost as xgb
from flwr.common import Context, Parameters, Scalar, log
from flwr.common.config import unflatten_dict
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic

from learning_methods.federated.xgboost_.task import replace_keys, load_data


class CyclicClientManager(SimpleClientManager):
    """Provides a cyclic client selection rule."""

    def sample(
        self,
        num_clients: int,
        min_num_clients: int | None = None,
        criterion: Criterion | None = None,
    ) -> list[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""

        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)

        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [cid for cid in available_cids if criterion.select(self.clients[cid])]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients" " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        # Return all available clients
        return [self.clients[cid] for cid in available_cids]


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
            f1 = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

            # Save results to disk.
            # Note we add new entry to the same file with each call to this function.
            with open(f"./centralised_eval.txt", "a", encoding="utf-8") as fp:
                fp.write(f"Round:{server_round},F1:{f1}\n")

            # Save results to disk.
            joblib.dump(bst, "model.pkl")

            return 0, {"F1": f1}

    return evaluate_fn


def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (F1) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    f1_aggregated = sum([metrics["F1"] * num for num, metrics in eval_metrics]) / total_num
    metrics_aggregated = {"F1": f1_aggregated}
    return metrics_aggregated


def config_func(rnd: int) -> Dict[str, str]:
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
    train_method = cfg["train_method"]
    params = cfg["params"]
    centralised_eval = cfg["centralised_eval"]

    _, test_dmatrix, _, _ = load_data("all", centralised_eval_client=False)

    # Init an empty Parameter
    parameters = Parameters(tensor_type="", tensors=[])

    # Define strategy
    if train_method == "bagging":
        # Bagging training
        strategy = FedXgbBagging(
            evaluate_function=(get_evaluate_fn(test_dmatrix, params) if centralised_eval else None),
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate if not centralised_eval else 0.0,
            on_evaluate_config_fn=config_func,
            on_fit_config_fn=config_func,
            evaluate_metrics_aggregation_fn=(evaluate_metrics_aggregation if not centralised_eval else None),
            initial_parameters=parameters,
        )
    else:
        # Cyclic training
        strategy = FedXgbCyclic(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
            on_evaluate_config_fn=config_func,
            on_fit_config_fn=config_func,
            initial_parameters=parameters,
        )

    config = ServerConfig(num_rounds=num_rounds)
    client_manager = CyclicClientManager() if train_method == "cyclic" else None

    return ServerAppComponents(strategy=strategy, config=config, client_manager=client_manager)


# Create ServerApp
app = ServerApp(
    server_fn=server_fn,
)
