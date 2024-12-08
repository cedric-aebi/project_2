from pathlib import Path
from enums.Model import Model
from service.argumentservice.ArgumentService import ArgumentService
from service.exportservice.ExportService import ExportService

BASE_PATH = Path(__file__).parent.parent / "results" / "csv"

if __name__ == "__main__":
    arg_service = ArgumentService()
    database = arg_service.get_database()
    collection = arg_service.get_collection()
    model = arg_service.get_model()

    export_service = ExportService(database=database, collection=collection)
    export_service.export_results_to_csv(collection=collection, model=model, base_path=BASE_PATH)
