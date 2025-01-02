import json
from itertools import product
from pathlib import Path

import joblib

from enums.Model import Model
from model.DNNModel import DNNModel
from model.LogisticRegressionModel import LogisticRegressionModel
from model.XGBoostModel import XGBoostModel
from service.argumentservice.ArgumentService import ArgumentService
from service.datasetservice.DatasetService import DatasetService
from service.exportservice.ExportService import ExportService

# ************************ DEFINE CONFIGURATION *****************************
BASE_PATH = Path(__file__).parent.parent.parent.parent / "results" / "individual"
# ***************************************************************************

if __name__ == "__main__":
    arg_service = ArgumentService(model=True, resampling=True, scaling=True, database=True, features=True)
    model_enum = arg_service.get_model()
    resampling_methods = arg_service.get_resampling_methods()
    scaling_methods = arg_service.get_scaling_methods()
    database = arg_service.get_database()
    with_features = arg_service.get_features()

    dataset_service = DatasetService()
    export_service = ExportService(database=database, collection="individual")

    # Execute machine learning pipeline for each configured model
    for model_enum, resampling_method, scaling_method in product([model_enum], resampling_methods, scaling_methods):
        # 1. Initialize dummy model for hash calculation
        match model_enum:
            case Model.XGBOOST:
                dummy_model = XGBoostModel(scaler=None, resampler=None)
            case Model.LOGISTIC_REGRESSION:
                dummy_model = LogisticRegressionModel(scaler=None, resampler=None)
            case Model.DNN:
                dummy_model = DNNModel(scaler=None, resampler=None, number_of_features=120 if with_features else 2)
            case _:
                raise Exception(f"Could not initialize model {model_enum.value} for config")

        # 2. Create run configuration with the given parameters
        run_info = {
            "model": model_enum.value,
            "pre-processing": {
                "features": with_features,
                "resampling": {"method": resampling_method.value},
                "scaling": {"method": scaling_method.value},
            },
            "subjects": [],
            "hyperparameters": dummy_model.get_hyperparameter_grid(),
        }

        # 3. Create a has over the run_info dict and the current database and check if run already exists
        run_id = export_service.generate_unique_id([database, json.dumps(run_info)])

        if export_service.run_exists(run_id):
            print(f"Run with configuration: {run_info} on database {database} already exists")
            continue

        print(f"Executing run with configuration: {run_info} on database {database}")

        # 4. Set run id and fit the model on the centralized dataset
        run_info["_id"] = run_id

        # 5. Fit models
        for idx, subject in enumerate(range(2, 36)):
            run_info["subjects"].append({"subject": subject})

            x_train = dataset_service.load_training_features(which=subject, with_features=with_features).to_numpy()
            x_test = dataset_service.load_testing_features(which=subject, with_features=with_features).to_numpy()
            y_train = (
                dataset_service.load_training_labels(which=subject, with_features=with_features).to_numpy().ravel()
            )
            y_test = dataset_service.load_testing_labels(which=subject, with_features=with_features).to_numpy().ravel()

            scaler = dataset_service.get_scaler(method=scaling_method)
            resampler = dataset_service.get_resampler(method=resampling_method)

            match model_enum:
                case Model.XGBOOST:
                    model = XGBoostModel(scaler=scaler, resampler=resampler)
                case Model.LOGISTIC_REGRESSION:
                    model = LogisticRegressionModel(scaler=scaler, resampler=resampler)
                case Model.DNN:
                    model = DNNModel(scaler=scaler, resampler=resampler, number_of_features=120 if with_features else 2)
                case _:
                    raise Exception(f"Could not initialize model {model_enum.value} for config")

            model.fit(x_train=x_train, y_train=y_train, run_info=run_info["subjects"][idx])

            pred_train = model.predict(x=x_train)
            scores_train, _ = model.evaluate(pred=pred_train, y_true=y_train)
            pred_test = model.predict(x=x_test)
            scores_test, _ = model.evaluate(pred=pred_test, y_true=y_test)

            scores = {"training_set": scores_train, "testing_set": scores_test}
            run_info["subjects"][idx]["scores"] = scores
            joblib.dump(model, BASE_PATH / "models" / f"{run_id}.joblib", compress=3)

        # Export run configuration and results to mongodb
        mongo_id = export_service.export_run_to_mongodb(run_info=run_info)
