import flwr as fl
from flwr.common import Scalar, NDArrays
from flwr.server import ServerConfig
from pymongo import MongoClient
from pymongo.collection import Collection

import utils
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from typing import Dict, Optional, Tuple
import pickle
import pandas as pd


def fit_round(rnd: int) -> Dict:
    """Send round number to client."""
    return {"rnd": rnd}


def get_eval_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    collection: Collection = MongoClient().wesad.federated

    mongo_dict = {
        "_id": "centralized",
        "client_nr": "server",
        "epochs": list()
    }

    collection.insert_one(mongo_dict)

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    
    # Load testing
    file_to_read = open("features/all_testing_features.pickle", "rb")
    features1 = pickle.load(file_to_read)
    file_to_read.close()

    file_to_read = open("features/all_testing_labels.pickle", "rb")
    labels1 = pickle.load(file_to_read)
    file_to_read.close()

    X_test = pd.concat(features1)
    y_test = pd.concat(labels1)

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)


    # The `evaluate` function will be called after every round
    def evaluate(
            server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        y_pred = model.predict(X_test)
        print(y_pred)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=2)
        rec = recall_score(y_test, y_pred, pos_label=2)
        f1 = f1_score(y_test, y_pred, pos_label=2)

        mongo_dict["epochs"].append({
            "acc": acc,
            "prec": prec,
            "rec": rec,
            "f1": f1,
        })

        collection.replace_one({"_id": "centralized"}, mongo_dict)

        return loss, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=15,
        min_fit_clients=15,
        evaluate_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
        fraction_evaluate=1
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=ServerConfig(num_rounds=30),
    )
