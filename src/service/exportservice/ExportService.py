import hashlib
import pickle
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.collection import Collection

from enums.Model import Model
from service.visualizationservice.VisualizationService import VisualizationService


class ExportService:
    def __init__(self, database: str | None = None, collection: str | None = None):
        self.__collection = None
        if database is not None and collection is not None:
            client = MongoClient("localhost", 27017)
            db = client[database]
            self.__collection: Collection = db[collection]

    def run_exists(self, run_id: str) -> bool:
        if self.__collection is None:
            raise ValueError("Collection not initialized.")
        return bool(self.__collection.find_one({"_id": run_id}))

    def run_is_finished(self, run_id: str) -> bool:
        if self.__collection is None:
            raise ValueError("Collection not initialized.")
        document = self.__collection.find_one({"_id": run_id})
        return "finished" in document

    def export_run_to_mongodb(self, run_info: dict) -> str:
        if self.__collection is None:
            raise ValueError("Collection not initialized.")

        if not self.__collection.find_one({"_id": run_info["_id"]}):
            print(f"Exporting run with id {run_info['_id']}")
            self.__collection.insert_one(run_info)
            return run_info["_id"]
        else:
            print(f"Run with id {run_info['_id']} already exists. Not exporting.")
            return run_info["_id"]

    def update_run(self, run_id, set_dict: dict) -> None:
        if self.__collection is None:
            raise ValueError("Collection not initialized.")
        self.__collection.update_one({"_id": run_id}, set_dict)

    def get_run(self, run_id: str) -> dict:
        if self.__collection is None:
            raise ValueError("Collection not initialized.")
        return self.__collection.find_one({"_id": run_id})

    def update_documents_with_average_scoring(self, collection: str) -> None:
        if self.__collection is None:
            raise ValueError("Collection not initialized.")

        documents = self.__collection.find()

        match collection:
            case "centralized":
                for document in documents:
                    accs, precs, recs, f1s = [], [], [], []
                    training_runs = document["training_runs"]

                    for run in training_runs:
                        accs.append(run["scores"]["testing_set"]["accuracy"])
                        precs.append(run["scores"]["testing_set"]["precision"])
                        recs.append(run["scores"]["testing_set"]["recall"])
                        f1s.append(run["scores"]["testing_set"]["f1"])

                    document["average_scoring"] = {
                        "mean_accuracy": statistics.fmean(accs),
                        "mean_precision": statistics.fmean(precs),
                        "mean_recall": statistics.fmean(recs),
                        "mean_f1": statistics.fmean(f1s),
                    }
                    self.__collection.update_one({"_id": document["_id"]}, {"$set": document})
            case "individual" | "federated_fine_tuned":
                for document in documents:
                    accs, precs, recs, f1s = [], [], [], []
                    for subject in document["participants"]:
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
                for document in documents:
                    server_accs, server_precs, server_recs, server_f1s = [], [], [], []
                    client_f1s = []
                    training_runs = document["training_runs"]
                    for run in training_runs:
                        clients = run["clients"]
                        server = clients["server"]

                        distributed_rounds = [r for r in server["distributed"]["round"].values()]
                        distributed_f1 = distributed_rounds[-1]["scores"]
                        client_f1s.append(distributed_f1)

                        centralized_rounds = [r for r in server["centralized"]["round"].values()]
                        centralized_f1 = centralized_rounds[-1]["scores"]["testing_set"]["f1"]
                        centralized_acc = centralized_rounds[-1]["scores"]["testing_set"]["accuracy"]
                        centralized_prec = centralized_rounds[-1]["scores"]["testing_set"]["precision"]
                        centralized_rec = centralized_rounds[-1]["scores"]["testing_set"]["recall"]
                        server_f1s.append(centralized_f1)
                        server_accs.append(centralized_acc)
                        server_precs.append(centralized_prec)
                        server_recs.append(centralized_rec)

                    document["average_client_scoring"] = {
                        "mean_f1": statistics.fmean(client_f1s),
                    }
                    document["average_server_scoring"] = {
                        "mean_accuracy": statistics.fmean(server_accs),
                        "mean_precision": statistics.fmean(server_precs),
                        "mean_recall": statistics.fmean(server_recs),
                        "mean_f1": statistics.fmean(server_f1s),
                    }
                    self.__collection.update_one({"_id": document["_id"]}, {"$set": document})

    def export_results_to_csv(self, collection: str, model: Model, base_path: Path) -> None:
        if self.__collection is None:
            raise ValueError("Collection not initialized.")

        match collection:
            case "centralized":
                best = self.__collection.find({"model": model.value}).sort("average_scoring.mean_f1", -1)[0]

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
                df.to_csv(base_path / collection / f"{model.value}_{best['_id']}.csv", index=False)
            case "individual":
                best = self.__collection.find({"model": model.value}).sort("average_scoring.mean_f1", -1)[0]

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
                df.to_csv(base_path / collection / f"{model.value}_{best['_id']}.csv", index=False)
            case "federated":
                rows = []

                # Individual scores
                for subject in range(2, 36):
                    document = self.__collection.find_one({"model": model.value, "subject_nr": subject})
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
                average = self.__collection.find_one({"model": model.value, "subject_nr": "average"})
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
                server = self.__collection.find_one({"model": model.value, "subject_nr": "server"})
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
                df.to_csv(base_path / collection / f"{model.value}.csv", index=False)

    def export_pre_processing_comparison(self, base_path: Path) -> None:
        if self.__collection is None:
            raise ValueError("Collection not initialized.")

        documents = []

        # Logistic Regression Models
        document = self.__collection.find_one(
            {
                "model": Model.LOGISTIC_REGRESSION.value,
                "pre-processing.resampling.method": None,
                "pre-processing.scaling.method": None,
            }
        )
        documents.append(document)
        document = self.__collection.find({"model": Model.LOGISTIC_REGRESSION.value}).sort(
            "centralized_scoring.testing_set.f1", -1
        )[0]
        documents.append(document)
        document = self.__collection.find(
            {
                "model": Model.LOGISTIC_REGRESSION.value,
                "pre-processing.resampling.method": {"$ne": document["pre-processing"]["resampling"]["method"]},
            }
        ).sort("centralized_scoring.testing_set.f1", -1)[0]
        documents.append(document)

        # XGBoost Models
        document = self.__collection.find_one(
            {
                "model": Model.XGBOOST.value,
                "pre-processing.resampling.method": None,
                "pre-processing.scaling.method": None,
            }
        )
        documents.append(document)
        document = self.__collection.find({"model": Model.XGBOOST.value}).sort(
            "centralized_scoring.testing_set.f1", -1
        )[0]
        documents.append(document)
        document = self.__collection.find(
            {
                "model": Model.XGBOOST.value,
                "pre-processing.resampling.method": {"$ne": document["pre-processing"]["resampling"]["method"]},
            }
        ).sort("centralized_scoring.testing_set.f1", -1)[0]
        documents.append(document)

        # DNN Models
        document = self.__collection.find_one(
            {
                "model": Model.SHALLOW_NN.value,
                "pre-processing.resampling.method": None,
                "pre-processing.scaling.method": None,
            }
        )
        documents.append(document)
        document = self.__collection.find({"model": Model.SHALLOW_NN.value}).sort(
            "centralized_scoring.testing_set.f1", -1
        )[0]
        documents.append(document)
        document = self.__collection.find(
            {
                "model": Model.SHALLOW_NN.value,
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
    def generate_unique_id(params: list[str]) -> str:
        # Combine the strings in a deterministic order
        combined = "|".join(sorted([(str(param) or "") for param in params]))
        # Use a hash function to generate a unique ID
        unique_id = hashlib.sha256(combined.encode()).hexdigest()
        return unique_id

    @staticmethod
    def export_file(path: Path, data: Any) -> None:
        file_to_store = open(path, "wb")
        pickle.dump(data, file_to_store)
        file_to_store.close()

    @staticmethod
    def export_class_distribution_plot(dataset: pd.DataFrame, base_path: Path) -> None:
        VisualizationService.plot_class_distribution(dataset=dataset, path=base_path / "general")

    @staticmethod
    def export_confusion_matrix_display(
        run_id: str, which: str | int, cm: np.ndarray, labels: list[str], path: Path
    ) -> None:
        VisualizationService.plot_confusion_matrix(cm=cm, labels=labels, path=path / run_id / f"subject_{which}")

    @staticmethod
    def export_roc_display(
        run_id: str,
        which: str | int,
        x_test: np.ndarray,
        y_test: np.ndarray,
        path: Path,
        model: Any | None = None,
        pred: np.ndarray | None = None,
        estimator_name: str | None = None,
    ) -> None:
        VisualizationService.plot_roc(
            path=path / run_id / f"subject_{which}",
            x_test=x_test,
            y_test=y_test,
            model=model,
            pred=pred,
            estimator_name=estimator_name,
        )
