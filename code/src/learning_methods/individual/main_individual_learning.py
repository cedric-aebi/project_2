from itertools import product
from pathlib import Path

from enums.Model import Model
from enums.ResamplingMethod import ResamplingMethod
from enums.ScalingMethod import ScalingMethod
from model.DNNModel import DNNModel
from model.LogisticRegressionModel import LogisticRegressionModel
from model.XGBoostModel import XGBoostModel
from service.datasetservice.DatasetService import DatasetService
from service.exportservice.ExportService import ExportService

# ************************ DEFINE CONFIGURATION *****************************
BASE_PATH = Path(__file__).parent.parent.parent.parent / "results" / "individual"
MODELS = [Model.DNN]
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
    export_service = ExportService(database="project_2_windows", collection="individual")

    # Execute machine learning pipeline for each configured model
    for model_enum, resampling_method, scaling_method in product(MODELS, RESAMPLING_METHODS, SCALING_METHODS):
        print(
            f"Executing run with: model={model_enum}, resampling_method={resampling_method}, "
            f"scaling_method={scaling_method}"
        )
        # Keep track of what has been done
        run_info = {
            "model": model_enum.value,
            "pre-processing": {
                "resampling": {"method": resampling_method},
                "scaling": {"method": scaling_method},
            },
            "subjects": [],
        }

        fitted_models = []
        idx = 0
        for subject in range(2, 36):
            run_info["subjects"].append({"subject": subject})

            x_train = dataset_service.load_training_features(which=subject).to_numpy()
            x_test = dataset_service.load_testing_features(which=subject).to_numpy()
            y_train = dataset_service.load_training_labels(which=subject).to_numpy().ravel()
            y_test = dataset_service.load_testing_labels(which=subject).to_numpy().ravel()

            scaler = dataset_service.get_scaler(method=scaling_method)
            resampler = dataset_service.get_resampler(method=resampling_method)

            match model_enum:
                case Model.XGBOOST:
                    model = XGBoostModel(scaler=scaler, resampler=resampler)
                case Model.LOGISTIC_REGRESSION:
                    model = LogisticRegressionModel(scaler=scaler, resampler=resampler)
                case Model.DNN:
                    model = DNNModel(scaler=scaler, resampler=resampler, number_of_features=120, run_info=run_info)
                case _:
                    raise Exception(f"Could not initialize model {model_enum.value} for config")

            model.fit(x_train=x_train, y_train=y_train, run_info=run_info["subjects"][idx])

            pred_train = model.predict(x=x_train)
            scores_train, _ = model.evaluate(pred=pred_train, y_true=y_train)
            pred_test = model.predict(x=x_test)
            scores_test, _ = model.evaluate(pred=pred_test, y_true=y_test)

            scores = {"training_set": scores_train, "testing_set": scores_test}
            run_info["subjects"][idx]["scores"] = scores
            fitted_models.append(model)
            idx += 1

            # Cleanup some memory
            del model
            del scaler
            del resampler

        # Export run configuration and results to mongodb
        mongo_id = export_service.export_run_to_mongodb(run_info=run_info)
        if mongo_id is not None:
            # Export individual results
            model_idx = 0
            for subject in range(2, 36):
                x_test = dataset_service.load_testing_features(which=subject).to_numpy()
                y_test = dataset_service.load_testing_labels(which=subject).to_numpy().ravel()

                model = fitted_models[model_idx]

                pred_test = model.predict(x=x_test)
                _, cm = model.evaluate(pred=pred_test, y_true=y_test)

                export_service.export_confusion_matrix_display(
                    cm=cm,
                    labels=["No-Stress", "Stress"],
                    mongo_id=mongo_id,
                    path=BASE_PATH,
                    which=subject,
                )
                export_service.export_roc_display(
                    mongo_id=mongo_id,
                    x_test=x_test,
                    y_test=y_test,
                    path=BASE_PATH,
                    model=model.get_fitted_model(),
                    which=subject,
                )

                model_idx += 1
