from pathlib import Path
from logging import INFO

import flwr as fl
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)
from flwr.common.logger import log
from imblearn.over_sampling import RandomOverSampler
from pymongo import MongoClient
from pymongo.collection import Collection
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import Booster

from enums.Model import Model
from enums.ResamplingMethod import ResamplingMethod
from enums.ScalingMethod import ScalingMethod
from learning_methods.federated.xgboost_ import utils
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
class StressClient(fl.client.Client):
    def __init__(self, subject_nr: int, number_of_rounds: int, base_path: Path):
        # Used by Flower
        self.bst: Booster | None = None
        self.config = None

        dataset_service = DatasetService()
        self._export_service = ExportService(database="project_2_windows", collection="federated")

        self._subject_nr = subject_nr
        self._number_of_rounds = number_of_rounds
        self._base_path = base_path

        self._collection: Collection = MongoClient().project_2_windows.federated

        # Define best performing model params from centralized run
        self._params = {"max_depth": 8, "objective": "binary:logistic"}
        self._num_local_round = 1

        self._mongo_dict = {
            "subject_nr": subject_nr,
            "model": Model.XGBOOST,
            "pre-processing": {
                "resampling": {"method": ResamplingMethod.OVERSAMPLING},
                "scaling": {"method": ScalingMethod.STANDARDSCALER},
            },
            "params": self._params,
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
        resampler = RandomOverSampler(random_state=42)
        self._x_train, self._y_train = resampler.fit_resample(X=self._x_train, y=self._y_train)

        # Reformat data to DMatrix for xgboost
        self._train_dmatrix = utils.transform_dataset_to_dmatrix(x=self._x_train, y=self._y_train)
        self._test_dmatrix = utils.transform_dataset_to_dmatrix(x=self._x_test, y=self._y_test)

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def fit(self, ins: FitIns) -> FitRes:
        if not self.bst:
            # First round local training
            log(INFO, "Start training at round 1")
            bst = xgb.train(
                self._params,
                self._train_dmatrix,
                num_boost_round=self._num_local_round,
                evals=[(self._test_dmatrix, "test"), (self._train_dmatrix, "train")],
            )
            self.config = bst.save_config()
            self.bst = bst
        else:
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            self.bst.load_model(global_model)
            self.bst.load_config(self.config)

            bst = self._local_boost()

        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        log(INFO, f"Training finished for round {ins.config['rnd']}")

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=len(self._x_train),
            metrics={},
        )

    def _local_boost(self):
        # Update trees based on local training data.
        for i in range(self._num_local_round):
            self.bst.update(self._train_dmatrix, self.bst.num_boosted_rounds())

        # Extract the last N=num_local_round trees for sever aggregation
        return self.bst[self.bst.num_boosted_rounds() - self._num_local_round : self.bst.num_boosted_rounds()]

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        eval_results = self.bst.eval_set(
            evals=[(self._test_dmatrix, "test")],
            iteration=self.bst.num_boosted_rounds() - 1,
        )
        print("Evaluate")

        class_probs = self.bst.predict(self._train_dmatrix)
        # turns soft logit into class label
        pred_train = utils.get_class_labels_from_probs(probs=class_probs)
        scores_train, _ = utils.evaluate_prediction(pred=pred_train, y_true=self._y_train)

        class_probs = self.bst.predict(self._test_dmatrix)
        # turns soft logit into class label
        pred_test = utils.get_class_labels_from_probs(probs=class_probs)
        scores_test, cm = utils.evaluate_prediction(pred=pred_test, y_true=self._y_test)

        scores = {"training_set": scores_train, "testing_set": scores_test}
        self._mongo_dict["rounds"].append(scores)

        self._collection.replace_one({"_id": self._mongo_id}, self._mongo_dict)

        # If last round export plots of final model
        if ins.config["rnd"] == self._number_of_rounds:
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
                which=self._subject_nr,
                pred=pred_test,
                estimator_name=Model.XGBOOST,
            )

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=round(float(eval_results.split("\t")[1].split(":")[1]), 4),
            num_examples=len(self._x_test),
            metrics={"accuracy": scores_test["accuracy"]},
        )
