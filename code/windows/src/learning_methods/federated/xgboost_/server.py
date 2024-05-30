from logging import INFO
from pathlib import Path

import xgboost as xgb
import pandas as pd
import flwr as fl
from flwr.common import log, Parameters
from flwr.server import ServerConfig
from imblearn.under_sampling import RandomUnderSampler
from pymongo import MongoClient
from pymongo.collection import Collection
from sklearn.preprocessing import StandardScaler
from xgboost import Booster

from enums.Model import Model
from enums.ResamplingMethod import ResamplingMethod
from enums.ScalingMethod import ScalingMethod
from learning_methods.federated.xgboost_ import utils
from service.datasetservice.DatasetService import DatasetService
from service.exportservice.ExportService import ExportService


class Server:
    def __init__(self, number_of_rounds: int, base_path: Path):
        # Global model
        self.bst: Booster | None = None
        self._num_local_round = 1

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
        resampler = RandomUnderSampler(random_state=42)
        self._x_train_all, self._y_train_all = resampler.fit_resample(X=self._x_train_all, y=self._y_train_all)

        self._collection: Collection = MongoClient().project_2_no_windows.federated

        # Define best performing model params from centralized run
        self._params = {"max_depth": 10, "objective": "binary:logistic"}

        self._mongo_dict = {
            "subject_nr": self._subject_nr,
            "model": Model.XGBOOST,
            "pre-processing": {
                "resampling": {"method": ResamplingMethod.UNDERSAMPLING},
                "scaling": {"method": ScalingMethod.STANDARDSCALER},
            },
            "params": self._params,
            "rounds": [],
        }
        self._mongo_id = self._collection.insert_one(self._mongo_dict).inserted_id

        # Reformat data to DMatrix for xgboost
        self._train_dmatrix = utils.transform_dataset_to_dmatrix(x=self._x_train_all, y=self._y_train_all)
        self._test_dmatrix = utils.transform_dataset_to_dmatrix(x=self._x_test_all, y=self._y_test_all)

    @staticmethod
    def fit_round(rnd: int) -> dict:
        """Send round number to client."""
        return {"rnd": rnd}

    def get_eval_fn(self):
        """Return an evaluation function for server-side evaluation."""

        # The `evaluate` function will be called after every round
        def evaluate(server_round: int, parameters: Parameters, config: dict):
            # Build new Booster from client updates
            if not self.bst:
                # First round local training
                log(INFO, "Start training at round 1")
                empty_bst = xgb.train(
                    self._params,
                    self._train_dmatrix,
                    num_boost_round=self._num_local_round,
                    evals=[(self._test_dmatrix, "test"), (self._train_dmatrix, "train")],
                )
                self.config = empty_bst.save_config()
                self.bst = empty_bst
            else:
                for item in parameters.tensors:
                    global_model = bytearray(item)

                # Load global model into booster
                self.bst.load_model(global_model)
                self.bst.load_config(self.config)

            eval_results = self.bst.eval_set(
                evals=[(self._test_dmatrix, "test")],
                iteration=self.bst.num_boosted_rounds() - 1,
            )
            print("Evaluate")

            class_probs = self.bst.predict(self._train_dmatrix)
            # turns soft logit into class label
            pred_train = utils.get_class_labels_from_probs(probs=class_probs)
            scores_train, _ = utils.evaluate_prediction(pred=pred_train, y_true=self._y_train_all)

            class_probs = self.bst.predict(self._test_dmatrix)
            # turns soft logit into class label
            pred_test = utils.get_class_labels_from_probs(probs=class_probs)
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
                    which=self._subject_nr,
                    pred=pred_test,
                    estimator_name=Model.XGBOOST,
                )

            return round(float(eval_results.split("\t")[1].split(":")[1]), 4), {
                "accuracy": scores_test["accuracy"],
                "precision": scores_test["precision"],
                "recall": scores_test["recall"],
                "f1": scores_test["f1"],
            }

        return evaluate

    def start(self) -> None:
        # Define strategy
        strategy = fl.server.strategy.FedXgbBagging(
            fraction_fit=1,
            fraction_evaluate=1,
            min_fit_clients=34,
            min_available_clients=34,
            min_evaluate_clients=34,
            evaluate_function=self.get_eval_fn(),
            on_fit_config_fn=self.fit_round,
            on_evaluate_config_fn=self.fit_round,
        )
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            strategy=strategy,
            config=ServerConfig(num_rounds=self._number_of_rounds),
        )
