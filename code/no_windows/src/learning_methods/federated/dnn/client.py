import warnings
from pathlib import Path

import flwr as fl
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


class Client:
    def __init__(self, subject_nr: int, number_of_rounds: int, base_path: Path):
        self._subject_nr = subject_nr
        self._number_of_rounds = number_of_rounds
        self._base_path = base_path

    def start(self) -> None:
        # Start Flower client
        print("Starting client...")
        fl.client.start_client(
            server_address="0.0.0.0:8080",
            client=StressClient(
                subject_nr=self._subject_nr, number_of_rounds=self._number_of_rounds, base_path=self._base_path
            ).to_client(),
        )


# Define Flower client
class StressClient(fl.client.NumPyClient):
    def __init__(self, subject_nr: int, number_of_rounds: int, base_path: Path):
        dataset_service = DatasetService()
        self._export_service = ExportService(database="project_2_no_windows", collection="federated")

        self._subject_nr = subject_nr
        self._number_of_rounds = number_of_rounds
        self._base_path = base_path

        self._collection: Collection = MongoClient().project_2_no_windows.federated
        params = {"C": 0.001, "solver": "saga", "penalty": "l1"}
        self._mongo_dict = {
            "subject_nr": subject_nr,
            "model": Model.LOGISTIC_REGRESSION,
            "pre-processing": {
                "resampling": {"method": ResamplingMethod.SMOTEENN},
                "scaling": {"method": ScalingMethod.STANDARDSCALER},
            },
            "params": params,
            "rounds": [],
        }
        self._mongo_id = self._collection.insert_one(self._mongo_dict).inserted_id
        self._x_train = dataset_service.load_training_features(which=subject_nr).to_numpy()
        self._x_test = dataset_service.load_testing_features(which=subject_nr).to_numpy()
        self._y_train = dataset_service.load_training_labels(which=subject_nr).to_numpy()
        self._y_test = dataset_service.load_testing_labels(which=subject_nr).to_numpy()

        # Replicate best performing pre-processing from centralized run
        scaler = StandardScaler()
        self._x_train = scaler.fit_transform(X=self._x_train)
        self._x_test = scaler.transform(X=self._x_test)
        resampler = SMOTEENN(random_state=42)
        self._x_train, self._y_train = resampler.fit_resample(X=self._x_train, y=self._y_train)

        # Define best performing model from centralized run
        self._model = LogisticRegression(
            random_state=42,
            C=params["C"],
            solver=params["solver"],
            penalty=params["penalty"],
            max_iter=1,
            warm_start=True,
        )

        # Setting initial parameters, akin to model.compile for keras models
        utils.set_initial_params(self._model)

    def get_parameters(self, config):  # type: ignore
        return utils.get_model_parameters(self._model)

    def fit(self, parameters, config):  # type: ignore
        utils.set_model_params(self._model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(self._x_train, self._y_train)
        print(f"Training finished for round {config['rnd']}")

        return list(utils.get_model_parameters(self._model)), len(self._x_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        utils.set_model_params(self._model, parameters)
        loss = log_loss(self._y_test, self._model.predict_proba(self._x_test))
        accuracy = self._model.score(self._x_test, self._y_test)
        print("Evaluate")
        pred_train = self._model.predict(self._x_train)
        scores_train, _ = utils.evaluate_prediction(pred=pred_train, y_true=self._y_train)
        pred_test = self._model.predict(self._x_test)
        scores_test, cm = utils.evaluate_prediction(pred=pred_test, y_true=self._y_test)

        scores = {"training_set": scores_train, "testing_set": scores_test}
        self._mongo_dict["rounds"].append(scores)

        self._collection.replace_one({"_id": self._mongo_id}, self._mongo_dict)

        # If last round export plots of final model
        if config["rnd"] == self._number_of_rounds:
            self._export_service.export_confusion_matrix_display(
                cm=cm,
                labels=["No-Stress", "Stress"],
                mongo_id=str(self._mongo_id),
                path=self._base_path,
                which=self._subject_nr,
            )
            self._export_service.export_roc_display(
                mongo_id=str(self._mongo_id),
                x_test=self._x_test,
                y_test=self._y_test,
                path=self._base_path,
                model=self._model,
                which=self._subject_nr,
            )

        return loss, len(self._x_test), {"accuracy": accuracy}
