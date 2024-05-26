import uuid
from pathlib import Path

import flwr as fl
import tensorflow as tf
import keras
from imblearn.under_sampling import RandomUnderSampler
from pymongo import MongoClient
from pymongo.collection import Collection
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import StandardScaler

from enums.Model import Model
from enums.ResamplingMethod import ResamplingMethod
from enums.ScalingMethod import ScalingMethod
from learning_methods.federated.dnn import utils
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
        tf.random.set_seed(42)
        keras.utils.set_random_seed(42)

        dataset_service = DatasetService()
        self._export_service = ExportService(database="project_2_no_windows", collection="federated")

        self._subject_nr = subject_nr
        self._number_of_rounds = number_of_rounds
        self._base_path = base_path

        self._collection: Collection = MongoClient().project_2_no_windows.federated

        unique_run_id = str(uuid.uuid4())
        log_dir = utils.get_log_dir(unique_run_id=unique_run_id)
        self._mongo_dict = {
            "subject_nr": self._subject_nr,
            "model": Model.DNN,
            "pre-processing": {
                "resampling": {"method": ResamplingMethod.UNDERSAMPLING},
                "scaling": {"method": ScalingMethod.STANDARDSCALER},
            },
            "rounds": [],
            "log_dir": str(log_dir),
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
        resampler = RandomUnderSampler(random_state=42)
        self._x_train, self._y_train = resampler.fit_resample(X=self._x_train, y=self._y_train)

        early_stopping_callback = keras.callbacks.EarlyStopping(patience=5)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self._model = KerasClassifier(
            model=utils.build_model(number_of_features=2),
            epochs=1,
            batch_size=32,
            verbose=1,
            validation_split=0.2,
            random_state=42,
            shuffle=True,
            callbacks=[early_stopping_callback, tensorboard_callback],
        )
        # Initialize model without fitting it
        self._model.initialize(self._x_train, self._y_train)

    def get_parameters(self, config):
        return self._model.model_.get_weights()

    def fit(self, parameters, config):
        self._model.model_.set_weights(parameters)
        self._model.fit(self._x_train, self._y_train)
        print(f"Training finished for round {config['rnd']}")
        return self._model.model_.get_weights(), len(self._x_train), {}

    def evaluate(self, parameters, config):
        self._model.model_.set_weights(parameters)
        loss, accuracy = self._model.model_.evaluate(self._x_test, self._y_test)

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

        return loss, len(self._x_test), {"accuracy": float(accuracy)}
