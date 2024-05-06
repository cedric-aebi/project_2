from itertools import product
from pathlib import Path

from enums.Model import Model
from enums.ResamplingMethod import ResamplingMethod
from enums.ScalingMethod import ScalingMethod
from model.LogisticRegressionModel import LogisticRegressionModel
from model.XGBoostModel import XGBoostModel
from service.datasetservice.DatasetService import DatasetService
from service.exportservice.ExportService import ExportService

# ************************ DEFINE CONFIGURATION *****************************
BASE_PATH = Path(__file__).parent.parent / "results" / "centralized"
EXPORT_CLASS_DISTRIBUTION = False
MODELS = [Model.XGBOOST, Model.LOGISTIC_REGRESSION]
RESAMPLING_METHODS = [
    ResamplingMethod.SMOTEENN,
    ResamplingMethod.SMOTE,
    ResamplingMethod.TL,
    ResamplingMethod.OVERSAMPLING,
    ResamplingMethod.UNDERSAMPLING,
    None,
]
SCALING_METHODS = [ScalingMethod.STANDARDSCALER, ScalingMethod.MINMAXSCALER, None]
# ***************************************************************************

if __name__ == "__main__":
    dataset_service = DatasetService()
    export_service = ExportService(collection="centralized")

    dataset = dataset_service.load_dataset()

    if EXPORT_CLASS_DISTRIBUTION:
        export_service.export_class_distribution_plot(dataset=dataset, base_path=BASE_PATH)

    dataset = dataset_service.remove_nan(dataset=dataset)
    x, y, labels = dataset_service.get_features_and_labels(dataset=dataset)

    # Execute machine learning pipeline for each configured model
    for model, resampling_method, scaling_method in product(MODELS, RESAMPLING_METHODS, SCALING_METHODS):
        print(
            f"Executing run with: model={model}, resampling_method={resampling_method}, scaling_method={scaling_method}"
        )
        # Keep track of what has been done
        run_info = {
            "model": model.value,
            "pre-processing": {
                "resampling": {"method": resampling_method},
                "scaling": {"method": scaling_method},
            },
        }

        # 1. Split data
        train_x, test_x, train_y, test_y = dataset_service.train_test_split(x=x, y=y, shuffle=True, run_info=run_info)

        # 2. Get Scaler
        scaler = dataset_service.get_scaler(method=resampling_method)

        # 3. Get Resampler
        resampler = dataset_service.get_resampler(method=resampling_method)

        match model:
            case Model.XGBOOST:
                model = XGBoostModel(scaler=scaler, resampler=resampler)
            case Model.LOGISTIC_REGRESSION:
                model = LogisticRegressionModel(scaler=scaler, resampler=resampler)
            case _:
                raise Exception(f"Could not initialize model {model.value} for config")

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

        # Cleanup some memory
        del model
        del scaler
        del resampler
