from pathlib import Path

from service.exportservice.ExportService import ExportService

if __name__ == "__main__":
    base_path = Path(__file__).parent.parent / "results" / "csv" / "pre-processing"
    export_service = ExportService(database="project_2_no_windows", collection="centralized")
    export_service.export_pre_processing_comparison(base_path=base_path)
