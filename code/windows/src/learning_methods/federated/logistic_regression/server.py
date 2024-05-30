from pathlib import Path

import pandas as pd
import flwr as fl
from flwr.common import NDArrays, Scalar
from flwr.server import ServerConfig
from imblearn.combine import SMOTEENN
from pymongo import MongoClient
from pymongo.collection import Collection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from enums.Model import Model
from enums.ResamplingMethod import ResamplingMethod
from enums.ScalingMethod import ScalingMethod
from learning_methods.federated.logistic_regression import utils
from service.datasetservice.DatasetService import DatasetService
from service.exportservice.ExportService import ExportService


class Server:
    def __init__(self, number_of_rounds: int, base_path: Path):
        self._number_of_rounds = number_of_rounds
        self._base_path = base_path
        self._subject_nr = "server"

        dataset_service = DatasetService()
        self._export_service = ExportService(database="project_2_no_windows", collection="federated")

        self._x_train_all = dataset_service.load_training_features(which="all")
        self._x_test_all = dataset_service.load_testing_features(which="all")
        self._y_train_all = dataset_service.load_training_labels(which="all")
        self._y_test_all = dataset_service.load_testing_labels(which="all")

        self._x_train_all = pd.concat(self._x_train_all).to_numpy()
        self._y_train_all = pd.concat(self._y_train_all).to_numpy()

        self._x_test_all = pd.concat(self._x_test_all).to_numpy()
        self._y_test_all = pd.concat(self._y_test_all).to_numpy()

        # Replicate best performing pre-processing from centralized run
        scaler = StandardScaler()
        self._x_train_all = scaler.fit_transform(X=self._x_train_all)
        self._x_test_all = scaler.transform(X=self._x_test_all)
        resampler = SMOTEENN(random_state=42)
        self._x_train_all, self._y_train_all = resampler.fit_resample(X=self._x_train_all, y=self._y_train_all)

        self._collection: Collection = MongoClient().project_2_no_windows.federated
        params = {"C": 0.001, "solver": "saga", "penalty": "l1"}
        self._mongo_dict = {
            "subject_nr": self._subject_nr,
            "model": Model.LOGISTIC_REGRESSION,
            "pre-processing": {
                "resampling": {"method": ResamplingMethod.SMOTEENN},
                "scaling": {"method": ScalingMethod.STANDARDSCALER},
            },
            "params": params,
            "rounds": [],
        }
        self._mongo_id = self._collection.insert_one(self._mongo_dict).inserted_id

        # Define best performing model from centralized run
        self._model = LogisticRegression(
            random_state=42,
            C=params["C"],
            solver=params["solver"],
            penalty=params["penalty"],
        )

        # Setting initial parameters, akin to model.compile for keras models
        utils.set_initial_params(self._model)

    @staticmethod
    def fit_round(rnd: int) -> dict:
        """Send round number to client."""
        return {"rnd": rnd}

    def get_eval_fn(self, model: LogisticRegression):
        """Return an evaluation function for server-side evaluation."""

        # The `evaluate` function will be called after every round
        def evaluate(server_round: int, parameters: NDArrays) -> tuple[float, dict[str, Scalar]] | None:
            utils.set_model_params(model, parameters)
            loss = log_loss(self._y_test_all, model.predict_proba(self._x_test_all))
            print("Evaluate")
            pred_train = model.predict(self._x_train_all)
            scores_train, _ = utils.evaluate_prediction(pred=pred_train, y_true=self._y_train_all)
            pred_test = model.predict(self._x_test_all)
            scores_test, cm = utils.evaluate_prediction(pred=pred_test, y_true=self._y_test_all)

            scores = {"training_set": scores_train, "testing_set": scores_test}
            self._mongo_dict["rounds"].append(scores)

            self._collection.replace_one({"_id": self._mongo_id}, self._mongo_dict)

            # If last round export plots of final model
            if server_round == self._number_of_rounds:
                self._export_service.export_confusion_matrix_display(
                    cm=cm,
                    labels=["No-Stress", "Stress"],
                    mongo_id=str(self._mongo_id),
                    path=self._base_path,
                    which=self._subject_nr,
                )
                self._export_service.export_roc_display(
                    mongo_id=str(self._mongo_id),
                    x_test=self._x_test_all,
                    y_test=self._y_test_all,
                    path=self._base_path,
                    model=model,
                    which=self._subject_nr,
                )

            return loss, {
                "accuracy": scores_test["accuracy"],
                "precision": scores_test["precision"],
                "recall": scores_test["recall"],
                "f1": scores_test["f1"],
            }

        return evaluate

    def start(self) -> None:
        strategy = fl.server.strategy.FedAvg(
            min_available_clients=34,
            min_fit_clients=34,
            evaluate_fn=self.get_eval_fn(self._model),
            on_fit_config_fn=self.fit_round,
            on_evaluate_config_fn=self.fit_round,
            fraction_evaluate=1,
        )
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            strategy=strategy,
            config=ServerConfig(num_rounds=self._number_of_rounds),
        )
