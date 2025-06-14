from pathlib import Path

from enums.Dataset import Dataset
from service.argumentservice.ArgumentService import ArgumentService
from service.exportservice.ExportService import ExportService

# Exports a csv with the pre-processing comparison results
if __name__ == "__main__":
    arg_service = ArgumentService(database=True)
    database = arg_service.get_database()
    dataset = arg_service.get_dataset()

    if dataset == Dataset.STRESS:
        base_path = Path(__file__).parent.parent / "results" / "csv" / "stress" / "pre-processing"
    else:
        base_path = Path(__file__).parent.parent / "results" / "csv" / "nurse" / "pre-processing"

    export_service = ExportService(database=database, collection="individual")
    export_service.export_pre_processing_comparison(base_path=base_path)
