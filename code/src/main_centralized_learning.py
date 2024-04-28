from pathlib import Path

from enums.Model import Model
from enums.ResamplingMethod import ResamplingMethod
from model.XGBoostModel import XGBoostModel
from service.datasetservice.DatasetService import DatasetService
from service.exportservice.ExportService import ExportService
from service.trainingservice.TrainingService import TrainingService

# ************************ DEFINE CONFIGURATION *****************************
BASE_PATH = Path(__file__).parent.parent / "results" / "centralized"
EXPORT_CLASS_DISTRIBUTION = False
MODEL_CONFIGURATION = [{"name": Model.XGBOOST, "resampling_method": ResamplingMethod.SMOTE}]
# ***************************************************************************

if __name__ == "__main__":
    dataset_service = DatasetService()
    training_service = TrainingService()
    export_service = ExportService(collection="centralized")

    dataset = dataset_service.load_dataset()

    if EXPORT_CLASS_DISTRIBUTION:
        export_service.export_class_distribution_plot(dataset=dataset, base_path=BASE_PATH)

    dataset = dataset_service.remove_nan(dataset=dataset)
    x, y, labels = dataset_service.get_features_and_labels(dataset=dataset)

    # Execute machine learning pipeline for each configured model
    for config in MODEL_CONFIGURATION:
        # Keep track of what has been done
        run_info = {"model": config["name"], "pre-processing": {"resampling": {"method": config["resampling_method"]}}}

        x_resampled, y_resampled = dataset_service.resample(
            x=x,
            y=y,
            method=config["resampling_method"],
            run_info=run_info,
        )
        train_x, test_x, train_y, test_y = dataset_service.train_test_split(
            x=x_resampled, y=y_resampled, shuffle=True, run_info=run_info
        )

        match config["name"]:
            case Model.XGBOOST:
                model = XGBoostModel()
            case _:
                raise Exception(f"Could not initialize model {config['name']} for config")
        model.fit(train_x, train_y, grid_search=True, run_info=run_info)
        pred = model.predict(test_x=test_x)
        confusion_matrix = model.evaluate(pred=pred, test_y=test_y, run_info=run_info)

        # Export run configuration and results to mongodb
        mongo_id = export_service.export_run_to_mongodb(run_info=run_info)
        if mongo_id is not None:
            export_service.export_confusion_matrix_display(
                cm=confusion_matrix, labels=labels, mongo_id=mongo_id, path=BASE_PATH
            )
            export_service.export_roc_display(
                mongo_id=mongo_id, test_x=test_x, test_y=test_y, path=BASE_PATH, model=model.get_fitted_model()
            )
