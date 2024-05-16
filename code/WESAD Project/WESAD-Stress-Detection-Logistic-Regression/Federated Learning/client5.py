import warnings
import flwr as fl
import pickle

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

import utils

if __name__ == "__main__":
    client_nr = 5

    collection: Collection = MongoClient().wesad.federated

    mongo_dict = {
        "client_nr": client_nr,
        "epochs": list()
    }

    collection.insert_one(mongo_dict)

    # Load training
    file_to_read = open("features/training_features" + str(client_nr) + ".pickle", "rb")
    features1 = pickle.load(file_to_read)
    file_to_read.close()

    file_to_read = open("features/training_labels" + str(client_nr) + ".pickle", "rb")
    labels1 = pickle.load(file_to_read)
    file_to_read.close()

    X_train = features1.reset_index(drop=True)
    y_train = labels1.reset_index(drop=True)

    # Load testing
    file_to_read = open("features/testing_features" + str(client_nr) + ".pickle", "rb")
    features1 = pickle.load(file_to_read)
    file_to_read.close()

    file_to_read = open("features/testing_labels" + str(client_nr) + ".pickle", "rb")
    labels1 = pickle.load(file_to_read)
    file_to_read.close()

    X_test = features1.reset_index(drop=True)
    y_test = labels1.reset_index(drop=True)

    # Create LogisticRegression Model
    # warm_start = prevent refreshing weights when fitting
    model = LogisticRegression(warm_start=True)

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)


    # Define Flower client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['rnd']}")

            return list(utils.get_model_parameters(model)), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            print("Evaluate")
            y_pred = model.predict(X_test)
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

            result = collection.replace_one({"client_nr": client_nr}, mongo_dict)
            print("MongoDB result: ", result.modified_count)

            return loss, len(X_test), {"accuracy": accuracy}


    # Start Flower client
    fl.client.start_client(server_address="0.0.0.0:8080", client=MnistClient().to_client())
