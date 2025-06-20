import hashlib
import pickle
import statistics
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.collection import Collection

from enums.Dataset import Dataset
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

    def export_results_to_csv(
        self, collection: str, dataset: Dataset, with_features: bool, model: Model, base_path: Path
    ) -> None:
        if self.__collection is None:
            raise ValueError("Collection not initialized.")

        match collection:
            case "centralized":
                best = self.__collection.find({"model": model.value, "pre-processing.features": with_features}).sort(
                    "average_scoring.mean_f1", -1
                )[0]

                rows = []

                for run in best["training_runs"]:
                    rows.append(
                        [
                            run["participant_leave_out"],
                            round(run["scores"]["testing_set"]["accuracy"], 4),
                            round(run["scores"]["testing_set"]["recall"], 4),
                            round(run["scores"]["testing_set"]["precision"], 4),
                            round(run["scores"]["testing_set"]["f1"], 4),
                        ]
                    )

                # Get average scores from training runs on the training set
                accs, precs, recs, f1s = [], [], [], []
                for run in best["training_runs"]:
                    accs.append(run["scores"]["training_set"]["accuracy"])
                    precs.append(run["scores"]["training_set"]["precision"])
                    recs.append(run["scores"]["training_set"]["recall"])
                    f1s.append(run["scores"]["training_set"]["f1"])

                # Average training scores
                rows.append(
                    [
                        "Average Training Score",
                        round(statistics.fmean(accs), 4),
                        round(statistics.fmean(recs), 4),
                        round(statistics.fmean(precs), 4),
                        round(statistics.fmean(f1s), 4),
                    ]
                )

                # Average testing scores
                rows.append(
                    [
                        "Average LOSO Score",
                        round(best["average_scoring"]["mean_accuracy"], 4),
                        round(best["average_scoring"]["mean_recall"], 4),
                        round(best["average_scoring"]["mean_precision"], 4),
                        round(best["average_scoring"]["mean_f1"], 4),
                    ]
                )

                df = pd.DataFrame(data=rows, columns=["LOSO", "Accuracy", "Recall", "Precision", "F1"])

                if dataset == Dataset.STRESS:
                    export_path = base_path / "stress"
                elif dataset == Dataset.NURSE:
                    export_path = base_path / "nurse"
                else:
                    raise ValueError("Dataset not recognized.")

                if with_features:
                    export_path = export_path / collection / "with_features"
                else:
                    export_path = export_path / collection / "no_features"

                df.to_csv(export_path / f"{model.value}_{best['_id']}.csv", index=False)
            case "individual":
                best = self.__collection.find({"model": model.value, "pre-processing.features": with_features}).sort(
                    "average_scoring.mean_f1", -1
                )[0]

                rows = []

                # Individual scores
                accs, precs, recs, f1s = [], [], [], []
                for subject in best["participants"]:
                    rows.append(
                        [
                            subject["participant"],
                            round(subject["scores"]["testing_set"]["accuracy"], 4),
                            round(subject["scores"]["testing_set"]["recall"], 4),
                            round(subject["scores"]["testing_set"]["precision"], 4),
                            round(subject["scores"]["testing_set"]["f1"], 4),
                        ]
                    )
                    accs.append(subject["scores"]["training_set"]["accuracy"])
                    precs.append(subject["scores"]["training_set"]["precision"])
                    recs.append(subject["scores"]["training_set"]["recall"])
                    f1s.append(subject["scores"]["training_set"]["f1"])

                # Average training scores
                rows.append(
                    [
                        "Average Training Score",
                        round(statistics.fmean(accs), 4),
                        round(statistics.fmean(recs), 4),
                        round(statistics.fmean(precs), 4),
                        round(statistics.fmean(f1s), 4),
                    ]
                )

                # Average testing scores
                rows.append(
                    [
                        "Average Testing Score",
                        round(best["average_scoring"]["mean_accuracy"], 4),
                        round(best["average_scoring"]["mean_recall"], 4),
                        round(best["average_scoring"]["mean_precision"], 4),
                        round(best["average_scoring"]["mean_f1"], 4),
                    ]
                )

                if dataset == Dataset.STRESS:
                    export_path = base_path / "stress"
                elif dataset == Dataset.NURSE:
                    export_path = base_path / "nurse"
                else:
                    raise ValueError("Dataset not recognized.")

                if with_features:
                    export_path = export_path / collection / "with_features"
                else:
                    export_path = export_path / collection / "no_features"

                df = pd.DataFrame(data=rows, columns=["Participant", "Accuracy", "Recall", "Precision", "F1"])
                df.to_csv(export_path / f"{model.value}_{best['_id']}.csv", index=False)
            case "federated":
                best = self.__collection.find({"model": model.value, "pre-processing.features": with_features}).sort(
                    "average_client_scoring.mean_f1", -1
                )[0]

                rows = []

                clients: dict = best["training_runs"][0]["clients"]
                del clients["server"]
                # Sort clients by name
                if dataset == Dataset.NURSE:
                    sorted_clients = OrderedDict(sorted(clients.items(), key=lambda item: item[0]))
                else:
                    sorted_clients = OrderedDict(sorted(clients.items(), key=lambda item: int(item[0])))

                for client_name, client_data in sorted_clients.items():
                    rows.append(
                        [
                            client_name,
                            round(list(client_data["round"].values())[-1]["scores"]["testing_set"]["accuracy"], 4),
                            round(list(client_data["round"].values())[-1]["scores"]["testing_set"]["recall"], 4),
                            round(list(client_data["round"].values())[-1]["scores"]["testing_set"]["precision"], 4),
                            round(list(client_data["round"].values())[-1]["scores"]["testing_set"]["f1"], 4),
                        ]
                    )

                # Average training scores
                rows.append(
                    [
                        "Average Client Score",
                        "not measured",
                        "not measured",
                        "not measured",
                        round(best["average_client_scoring"]["mean_f1"], 4),
                    ]
                )

                # Average LOSO score
                rows.append(
                    [
                        "Average LOSO Score",
                        round(best["average_server_scoring"]["mean_accuracy"], 4),
                        round(best["average_server_scoring"]["mean_recall"], 4),
                        round(best["average_server_scoring"]["mean_precision"], 4),
                        round(best["average_server_scoring"]["mean_f1"], 4),
                    ]
                )

                df = pd.DataFrame(data=rows, columns=["Client", "Accuracy", "Recall", "Precision", "F1"])

                if dataset == Dataset.STRESS:
                    export_path = base_path / "stress"
                elif dataset == Dataset.NURSE:
                    export_path = base_path / "nurse"
                else:
                    raise ValueError("Dataset not recognized.")

                if with_features:
                    export_path = export_path / collection / "with_features"
                else:
                    export_path = export_path / collection / "no_features"

                df.to_csv(export_path / f"{model.value}_{best['_id']}.csv", index=False)

    def export_pre_processing_comparison(self, base_path: Path, with_features: bool) -> None:
        if self.__collection is None:
            raise ValueError("Collection not initialized.")

        documents = []

        # XGBoost Models
        xg_documents = self.__collection.find(
            {"model": Model.XGBOOST.value, "pre-processing.features": with_features}
        ).sort("average_scoring.mean_f1", -1)
        documents.append({"filler": "XGBoost"})
        documents.extend(xg_documents)

        # NN Models
        nn_documents = self.__collection.find(
            {"model": Model.SHALLOW_NN.value, "pre-processing.features": with_features}
        ).sort("average_scoring.mean_f1", -1)
        documents.append({"filler": "Neural Network"})
        documents.extend(nn_documents)

        # Logistic Regression Models
        lr_documents = self.__collection.find(
            {"model": Model.LOGISTIC_REGRESSION.value, "pre-processing.features": with_features}
        ).sort("average_scoring.mean_f1", -1)
        documents.append({"filler": "Logistic Regression"})
        documents.extend(lr_documents)

        rows = []
        for document in documents:
            if "filler" in document:
                rows.append(
                    {
                        "Resampling": document["filler"],
                        "Normalization": document["filler"],
                        "Accuracy": document["filler"],
                        "Recall": document["filler"],
                        "Precision": document["filler"],
                        "F1": document["filler"],
                    }
                )
                continue

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
                    "Resampling": resampling,
                    "Normalization": scaling,
                    "Accuracy": round(document["average_scoring"]["mean_accuracy"], 4),
                    "Recall": round(document["average_scoring"]["mean_recall"], 4),
                    "Precision": round(document["average_scoring"]["mean_precision"], 4),
                    "F1": round(document["average_scoring"]["mean_f1"], 4),
                }
            )

        df = pd.DataFrame(data=rows, columns=["Resampling", "Normalization", "Accuracy", "Recall", "Precision", "F1"])
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
