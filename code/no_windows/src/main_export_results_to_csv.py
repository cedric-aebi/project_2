from pathlib import Path
from enums import Model
from service import ExportService

BASE_PATH = Path(__file__).parent.parent / "results" / "csv"
COLLECTION = "federated"
MODEL = Model.DNN

if __name__ == "__main__":
    export_service = ExportService(database="project_2_no_windows", collection=COLLECTION)
    export_service.export_results_to_csv(collection=COLLECTION, model=MODEL, base_path=BASE_PATH)
