from pathlib import Path

from service.argumentservice.ArgumentService import ArgumentService
from service.exportservice.ExportService import ExportService

if __name__ == "__main__":
    arg_service = ArgumentService(database=True)
    database = arg_service.get_database()

    base_path = Path(__file__).parent.parent / "results" / "csv" / "pre-processing"
    export_service = ExportService(database=database, collection="centralized")
    export_service.export_pre_processing_comparison(base_path=base_path)
