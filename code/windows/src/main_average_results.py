from service.exportservice.ExportService import ExportService

DATABASE = "project_2_windows"
COLLECTION = "federated"

if __name__ == "__main__":
    export_service = ExportService(database=DATABASE, collection=COLLECTION)
    export_service.update_documents_with_average_scoring(collection=COLLECTION)
