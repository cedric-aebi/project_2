from pathlib import Path
import pandas as pd
from pymongo import MongoClient
from pymongo.collection import Collection
from enums.Model import Model

BASE_PATH = Path(__file__).parent.parent / "results" / "csv"
COLLECTION = "federated"
MODEL = Model.LOGISTIC_REGRESSION

if __name__ == "__main__":
    export_service = ExportService(database="project_2_windows", collection=COLLECTION)
    export_service.export_results_to_csv(collection=COLLECTION, model=MODEL, base_path=BASE_PATH)