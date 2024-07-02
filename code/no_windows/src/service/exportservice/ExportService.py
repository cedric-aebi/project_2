import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.collection import Collection

from service.visualizationservice.VisualizationService import VisualizationService


class ExportService:
    def __init__(self, database: str, collection: str):
        self.__client = MongoClient("localhost", 27017)
        self.__db = self.__client[database]
        self.__collection: Collection = self.__db[collection]

    def export_run_to_mongodb(self, run_info: dict) -> str | None:
        run_info["_id"] = self._dict_hash(dictionary=run_info)
        if not self.__collection.find_one({"_id": run_info["_id"]}):
            print(f"Exporting run with id {run_info['_id']}")
            self.__collection.insert_one(run_info)
            return run_info["_id"]
        else:
            print(f"Run with id {run_info['_id']} already exists. Not exporting.")

    def update_documents_with_average_scoring(self, collection: str) -> None:
        documents = self.__collection.find()

        match collection:
            case "centralized":
                for document in documents:
                    accs, precs, recs, f1s = [], [], [], []
                    for subject in document["individual_scoring"]:
                        accs.append(subject["testing_set"]["accuracy"])
                        precs.append(subject["testing_set"]["precision"])
                        recs.append(subject["testing_set"]["recall"])
                        f1s.append(subject["testing_set"]["f1"])

                    document["average_scoring"] = {
                        "mean_accuracy": statistics.fmean(accs),
                        "mean_precision": statistics.fmean(precs),
                        "mean_recall": statistics.fmean(recs),
                        "mean_f1": statistics.fmean(f1s),
                    }
                    self.__collection.update_one({"_id": document["_id"]}, {"$set": document})
            case "individual":
                for document in documents:
                    accs, precs, recs, f1s = [], [], [], []
                    for subject in document["subjects"]:
                        accs.append(subject["scores"]["testing_set"]["accuracy"])
                        precs.append(subject["scores"]["testing_set"]["precision"])
                        recs.append(subject["scores"]["testing_set"]["recall"])
                        f1s.append(subject["scores"]["testing_set"]["f1"])

                    document["average_scoring"] = {
                        "mean_accuracy": statistics.fmean(accs),
                        "mean_precision": statistics.fmean(precs),
                        "mean_recall": statistics.fmean(recs),
                        "mean_f1": statistics.fmean(f1s),
                    }

                    self.__collection.update_one({"_id": document["_id"]}, {"$set": document})
            case "federated":
                xboost_models = []
                logistic_regression_models = []
                dnn_models = []
                for document in documents:
                    if document["model"] == "XGBoost" and document["subject_nr"] != "server":
                        xboost_models.append(document)
                    if document["model"] == "Logistic Regression" and document["subject_nr"] != "server":
                        logistic_regression_models.append(document)
                    if document["model"] == "DNN" and document["subject_nr"] != "server":
                        dnn_models.append(document)

                # XBoost Models average
                accs, precs, recs, f1s = [], [], [], []
                for document in xboost_models:
                    accs.append(document["rounds"][-1]["testing_set"]["accuracy"])
                    precs.append(document["rounds"][-1]["testing_set"]["precision"])
                    recs.append(document["rounds"][-1]["testing_set"]["recall"])
                    f1s.append(document["rounds"][-1]["testing_set"]["f1"])

                document_to_insert = {
                    "model": xboost_models[0]["model"],
                    "subject_nr": "average",
                    "pre-processing": xboost_models[0]["pre-processing"],
                    "params": xboost_models[0]["params"],
                    "average_scoring": {
                        "mean_accuracy": statistics.fmean(accs),
                        "mean_precision": statistics.fmean(precs),
                        "mean_recall": statistics.fmean(recs),
                        "mean_f1": statistics.fmean(f1s),
                    },
                }
                self.__collection.insert_one(document_to_insert)

                # Logistic Regression Models average
                accs, precs, recs, f1s = [], [], [], []
                for document in logistic_regression_models:
                    accs.append(document["rounds"][-1]["testing_set"]["accuracy"])
                    precs.append(document["rounds"][-1]["testing_set"]["precision"])
                    recs.append(document["rounds"][-1]["testing_set"]["recall"])
                    f1s.append(document["rounds"][-1]["testing_set"]["f1"])

                document_to_insert = {
                    "model": logistic_regression_models[0]["model"],
                    "subject_nr": "average",
                    "pre-processing": logistic_regression_models[0]["pre-processing"],
                    "params": logistic_regression_models[0]["params"],
                    "average_scoring": {
                        "mean_accuracy": statistics.fmean(accs),
                        "mean_precision": statistics.fmean(precs),
                        "mean_recall": statistics.fmean(recs),
                        "mean_f1": statistics.fmean(f1s),
                    },
                }
                self.__collection.insert_one(document_to_insert)

                # DNN Models average
                accs, precs, recs, f1s = [], [], [], []
                for document in dnn_models:
                    accs.append(document["rounds"][-1]["testing_set"]["accuracy"])
                    precs.append(document["rounds"][-1]["testing_set"]["precision"])
                    recs.append(document["rounds"][-1]["testing_set"]["recall"])
                    f1s.append(document["rounds"][-1]["testing_set"]["f1"])

                document_to_insert = {
                    "model": dnn_models[0]["model"],
                    "subject_nr": "average",
                    "pre-processing": dnn_models[0]["pre-processing"],
                    "average_scoring": {
                        "mean_accuracy": statistics.fmean(accs),
                        "mean_precision": statistics.fmean(precs),
                        "mean_recall": statistics.fmean(recs),
                        "mean_f1": statistics.fmean(f1s),
                    },
                }
                self.__collection.insert_one(document_to_insert)

    def export_results_to_csv(self, collection: str, model: Model, base_path: Path) -> None:
        match collection:
            case "centralized":
                best = self.__collection.find({"model": model}).sort("average_scoring.mean_f1", -1)[0]

                rows = []

                # Individual scores
                idx = 2
                for subject in best["individual_scoring"]:
                    rows.append(
                        [
                            idx,
                            round(subject["testing_set"]["accuracy"], 4),
                            round(subject["testing_set"]["recall"], 4),
                            round(subject["testing_set"]["precision"], 4),
                            round(subject["testing_set"]["f1"], 4),
                        ]
                    )
                    idx += 1

                # Average scores
                rows.append(
                    [
                        "Average",
                        round(best["average_scoring"]["mean_accuracy"], 4),
                        round(best["average_scoring"]["mean_recall"], 4),
                        round(best["average_scoring"]["mean_precision"], 4),
                        round(best["average_scoring"]["mean_f1"], 4),
                    ]
                )

                # Centralized scores
                rows.append(
                    [
                        "Centralized",
                        round(best["centralized_scoring"]["testing_set"]["accuracy"], 4),
                        round(best["centralized_scoring"]["testing_set"]["recall"], 4),
                        round(best["centralized_scoring"]["testing_set"]["precision"], 4),
                        round(best["centralized_scoring"]["testing_set"]["f1"], 4),
                    ]
                )

                df = pd.DataFrame(data=rows, columns=["Subject", "Accuracy", "Recall", "Precision", "F1"])
                df.to_csv(base_path / collection / f"{model}_{best['_id']}.csv", index=False)
            case "individual":
                best = self.__collection.find({"model": model}).sort("average_scoring.mean_f1", -1)[0]

                rows = []

                # Individual scores
                for subject in best["subjects"]:
                    rows.append(
                        [
                            subject["subject"],
                            round(subject["scores"]["testing_set"]["accuracy"], 4),
                            round(subject["scores"]["testing_set"]["recall"], 4),
                            round(subject["scores"]["testing_set"]["precision"], 4),
                            round(subject["scores"]["testing_set"]["f1"], 4),
                        ]
                    )

                # Average scores
                rows.append(
                    [
                        "Average",
                        round(best["average_scoring"]["mean_accuracy"], 4),
                        round(best["average_scoring"]["mean_recall"], 4),
                        round(best["average_scoring"]["mean_precision"], 4),
                        round(best["average_scoring"]["mean_f1"], 4),
                    ]
                )

                df = pd.DataFrame(data=rows, columns=["Subject", "Accuracy", "Recall", "Precision", "F1"])
                df.to_csv(base_path / collection / f"{model}_{best['_id']}.csv", index=False)
            case "federated":
                rows = []

                # Individual scores
                for subject in range(2, 36):
                    document = self.__collection.find_one({"model": model, "subject_nr": subject})
                    rows.append(
                        [
                            document["subject_nr"],
                            round(document["rounds"][-1]["testing_set"]["accuracy"], 4),
                            round(document["rounds"][-1]["testing_set"]["recall"], 4),
                            round(document["rounds"][-1]["testing_set"]["precision"], 4),
                            round(document["rounds"][-1]["testing_set"]["f1"], 4),
                        ]
                    )

                # Average scores
                average = self.__collection.find_one({"model": model, "subject_nr": "average"})
                rows.append(
                    [
                        "Average",
                        round(average["average_scoring"]["mean_accuracy"], 4),
                        round(average["average_scoring"]["mean_recall"], 4),
                        round(average["average_scoring"]["mean_precision"], 4),
                        round(average["average_scoring"]["mean_f1"], 4),
                    ]
                )

                # Centralized Scoring
                server = self.__collection.find_one({"model": model, "subject_nr": "server"})
                rows.append(
                    [
                        "Centralized",
                        round(server["rounds"][-1]["testing_set"]["accuracy"], 4),
                        round(server["rounds"][-1]["testing_set"]["recall"], 4),
                        round(server["rounds"][-1]["testing_set"]["precision"], 4),
                        round(server["rounds"][-1]["testing_set"]["f1"], 4),
                    ]
                )

                df = pd.DataFrame(data=rows, columns=["Subject", "Accuracy", "Recall", "Precision", "F1"])
                df.to_csv(base_path / collection / f"{model}.csv", index=False)

    def export_pre_processing_comparison(self, base_path: Path) -> None:
        documents = []

        # Logistic Regression Models
        document = self.__collection.find_one(
            {
                "model": Model.LOGISTIC_REGRESSION,
                "pre-processing.resampling.method": None,
                "pre-processing.scaling.method": None,
            }
        )
        documents.append(document)
        document = self.__collection.find({"model": Model.LOGISTIC_REGRESSION}).sort(
            "centralized_scoring.testing_set.f1", -1
        )[0]
        documents.append(document)
        document = self.__collection.find(
            {
                "model": Model.LOGISTIC_REGRESSION,
                "pre-processing.resampling.method": {"$ne": document["pre-processing"]["resampling"]["method"]},
            }
        ).sort("centralized_scoring.testing_set.f1", -1)[0]
        documents.append(document)

        # XGBoost Models
        document = self.__collection.find_one(
            {"model": Model.XGBOOST, "pre-processing.resampling.method": None, "pre-processing.scaling.method": None}
        )
        documents.append(document)
        document = self.__collection.find({"model": Model.XGBOOST}).sort("centralized_scoring.testing_set.f1", -1)[0]
        documents.append(document)
        document = self.__collection.find(
            {
                "model": Model.XGBOOST,
                "pre-processing.resampling.method": {"$ne": document["pre-processing"]["resampling"]["method"]},
            }
        ).sort("centralized_scoring.testing_set.f1", -1)[0]
        documents.append(document)

        # DNN Models
        document = self.__collection.find_one(
            {"model": Model.DNN, "pre-processing.resampling.method": None, "pre-processing.scaling.method": None}
        )
        documents.append(document)
        document = self.__collection.find({"model": Model.DNN}).sort("centralized_scoring.testing_set.f1", -1)[0]
        documents.append(document)
        document = self.__collection.find(
            {
                "model": Model.DNN,
                "pre-processing.resampling.method": {"$ne": document["pre-processing"]["resampling"]["method"]},
            }
        ).sort("centralized_scoring.testing_set.f1", -1)[0]
        documents.append(document)

        rows = []
        for document in documents:
            resampling = (
                document["pre-processing"]["resampling"]["method"]
                if document["pre-processing"]["resampling"]["method"] is not None
                else "None"
            )
            scaling = (
                document["pre-processing"]["scaling"]["method"]
                if document["pre-processing"]["scaling"]["method"] is not None
                else "None"
            )
            rows.append(
                {
                    "Model": document["model"],
                    "Resampling/Normalization": f"{resampling}/{scaling}",
                    "Accuracy": round(document["centralized_scoring"]["testing_set"]["accuracy"], 4),
                    "Recall": round(document["centralized_scoring"]["testing_set"]["recall"], 4),
                    "Precision": round(document["centralized_scoring"]["testing_set"]["precision"], 4),
                    "F1": round(document["centralized_scoring"]["testing_set"]["f1"], 4),
                }
            )

        df = pd.DataFrame(
            data=rows, columns=["Model", "Resampling/Normalization", "Accuracy", "Recall", "Precision", "F1"]
        )
        df.to_csv(base_path / "comparison.csv", index=False)

    @staticmethod
    def _dict_hash(dictionary: dict[str, Any]) -> str:
        """MD5 hash of a dictionary."""
        dhash = hashlib.md5()
        # We need to sort arguments so {'a': 1, 'b': 2} is
        # the same as {'b': 2, 'a': 1}
        encoded = json.dumps(dictionary, sort_keys=True).encode()
        dhash.update(encoded)
        return dhash.hexdigest()

    @staticmethod
    def export_class_distribution_plot(dataset: pd.DataFrame, base_path: Path) -> None:
        VisualizationService.plot_class_distribution(dataset=dataset, path=base_path / "general")

    @staticmethod
    def export_confusion_matrix_display(
        mongo_id: str, which: str | int, cm: np.ndarray, labels: list[str], path: Path
    ) -> None:
        VisualizationService.plot_confusion_matrix(cm=cm, labels=labels, path=path / mongo_id / f"subject_{which}")

    @staticmethod
    def export_roc_display(
        mongo_id: str,
        which: str | int,
        x_test: np.ndarray,
        y_test: np.ndarray,
        path: Path,
        model: Any | None = None,
        pred: np.ndarray | None = None,
        estimator_name: str | None = None,
    ) -> None:
        VisualizationService.plot_roc(
            path=path / mongo_id / f"subject_{which}",
            x_test=x_test,
            y_test=y_test,
            model=model,
            pred=pred,
            estimator_name=estimator_name,
        )
