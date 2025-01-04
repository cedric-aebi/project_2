import json
import os
import sys
from itertools import product
from pathlib import Path

import joblib

from enums.Model import Model
from model.ShallowNNModel import ShallowNNModel
from model.LogisticRegressionModel import LogisticRegressionModel
from model.XGBoostModel import XGBoostModel
from service.argumentservice.ArgumentService import ArgumentService
from service.datasetservice.DatasetService import DatasetService
from service.exportservice.ExportService import ExportService

# ************************ DEFINE CONFIGURATION *****************************
EXPORT_PATH = Path(__file__).parent.parent.parent.parent / "results" / "centralized" / "models"
# ***************************************************************************

if __name__ == "__main__":
    arg_service = ArgumentService(model=True, resampling=True, scaling=True, database=True, features=True)
    model_enum = arg_service.get_model()
    resampling_methods = arg_service.get_resampling_methods()
    scaling_methods = arg_service.get_scaling_methods()
    database = arg_service.get_database()
    with_features = arg_service.get_features()

    dataset_service = DatasetService()
    export_service = ExportService(database=database, collection="centralized")

    x_all, y_all = dataset_service.get_subject_data(subject="all", with_features=with_features)
    x_train_all, x_test_all, y_train_all, y_test_all = dataset_service.train_test_split(x=x_all, y=y_all)

    # Execute machine learning pipeline for each configured model
    for model_enum, resampling_method, scaling_method in product([model_enum], resampling_methods, scaling_methods):
        # 1. Initialize model, scaler and resampler
        scaler = dataset_service.get_scaler(method=scaling_method)
        resampler = dataset_service.get_resampler(method=resampling_method)

        match model_enum:
            case Model.XGBOOST:
                model = XGBoostModel(scaler=scaler, resampler=resampler)
            case Model.LOGISTIC_REGRESSION:
                model = LogisticRegressionModel(scaler=scaler, resampler=resampler)
            case Model.SHALLOW_NN:
                model = ShallowNNModel(scaler=scaler, resampler=resampler, input_shape=144 if with_features else 2)
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
            "hyperparameters": model.get_hyperparameter_grid(),
        }

        # 3. Create a has over the run_info dict and the current database and check if run already exists
        run_id = export_service.generate_unique_id([database, json.dumps(run_info)])

        if export_service.run_exists(run_id):
            print(f"Run with configuration: {run_info} on database {database} already exists")
            continue

        print(f"Executing run with configuration: {run_info} on database {database}")

        # 4. Set run id and fit the model on the centralized dataset
        run_info["_id"] = run_id
        model.fit(x_train=x_train_all, y_train=y_train_all, run_info=run_info)

        # 5. Get training and testing results on centralized dataset
        pred_train_all = model.predict(x=x_train_all)
        scores_train_all, _ = model.evaluate(pred=pred_train_all, y_true=y_train_all)
        pred_test_all = model.predict(x=x_test_all)
        scores_test_all, cm_all = model.evaluate(pred=pred_test_all, y_true=y_test_all)
        run_info["centralized_scoring"] = {"training_set": scores_train_all, "testing_set": scores_test_all}

        # 6. Get training and testing results on individual datasets
        run_info["individual_scoring"] = []
        for subject in range(2, 36):
            x, y = dataset_service.get_subject_data(subject=subject, with_features=with_features)
            x_train, x_test, y_train, y_test = dataset_service.train_test_split(x=x, y=y)

            pred_train = model.predict(x=x_train)
            scores_train, _ = model.evaluate(pred=pred_train, y_true=y_train)
            pred_test = model.predict(x=x_test)
            scores_test, _ = model.evaluate(pred=pred_test, y_true=y_test)

            scores = {"training_set": scores_train, "testing_set": scores_test}
            run_info["individual_scoring"].append(scores)

        # 7. Export run configuration and results to mongodb
        if not os.path.exists(EXPORT_PATH):
            os.makedirs(EXPORT_PATH)
        joblib.dump(model, EXPORT_PATH / f"{run_id}.joblib", compress=3)
        mongo_id = export_service.export_run_to_mongodb(run_info=run_info)

        # 8. Cleanup some memory
        del model
        del scaler
        del resampler
