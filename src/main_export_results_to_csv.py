from pathlib import Path
from service.argumentservice.ArgumentService import ArgumentService
from service.exportservice.ExportService import ExportService

BASE_PATH = Path(__file__).parent.parent / "results" / "csv"

# Exports a MongoDB collection to a CSV file
if __name__ == "__main__":
    arg_service = ArgumentService(database=True, collection=True, model=True)
    database = arg_service.get_database()
    collection = arg_service.get_collection()
    model = arg_service.get_model()

    export_service = ExportService(database=database, collection=collection)
    export_service.export_results_to_csv(collection=collection, model=model, base_path=BASE_PATH)
