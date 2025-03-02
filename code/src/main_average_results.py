from service.argumentservice.ArgumentService import ArgumentService
from service.exportservice.ExportService import ExportService

if __name__ == "__main__":
    arg_service = ArgumentService(database=True, collection=True)
    database = arg_service.get_database()
    collection = arg_service.get_collection()

    export_service = ExportService(database=database, collection=collection)
    export_service.update_documents_with_average_scoring(collection=collection)
