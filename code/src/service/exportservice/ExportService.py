import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.collection import Collection

from service.visualizationservice.VisualizationService import VisualizationService


class ExportService:
    def __init__(self, collection: str):
        self.__client = MongoClient("localhost", 27017)
        self.__db = self.__client.project_2
        self.__collection: Collection = self.__db[collection]

    def export_run_to_mongodb(self, run_info: dict) -> str | None:
        run_info["_id"] = self._dict_hash(dictionary=run_info)
        if not self.__collection.find_one({"_id": run_info["_id"]}):
            print(f"Exporting run with id {run_info['_id']}")
            self.__collection.insert_one(run_info)
            return run_info["_id"]
        else:
            print(f"Run with id {run_info['_id']} already exists. Not exporting.")

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
    def export_confusion_matrix_display(mongo_id: str, cm: np.ndarray, labels: list[str], path: Path) -> None:
        VisualizationService.plot_confusion_matrix(cm=cm, labels=labels, path=path / mongo_id)

    @staticmethod
    def export_roc_display(mongo_id: str, test_x: np.ndarray, test_y: np.ndarray, path: Path, model: Any) -> None:
        VisualizationService.plot_roc(path=path / mongo_id, test_x=test_x, test_y=test_y, model=model)
