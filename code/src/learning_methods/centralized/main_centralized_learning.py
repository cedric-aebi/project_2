from itertools import product
from pathlib import Path

import pandas as pd

from enums.Model import Model
from model.DNNModel import DNNModel
from model.LogisticRegressionModel import LogisticRegressionModel
from model.XGBoostModel import XGBoostModel
from service.argumentservice.ArgumentService import ArgumentService
from service.datasetservice.DatasetService import DatasetService
from service.exportservice.ExportService import ExportService

# ************************ DEFINE CONFIGURATION *****************************
BASE_PATH = Path(__file__).parent.parent.parent.parent / "results" / "centralized"
# ***************************************************************************

if __name__ == "__main__":
    arg_service = ArgumentService(
        model=True, resampling=True, scaling=True, database=True, collection=True, features=True
    )
    model_enum = arg_service.get_model()
    resampling_methods = arg_service.get_resampling_methods()
    scaling_methods = arg_service.get_scaling_methods()
    database = arg_service.get_database()
    collection = arg_service.get_collection()
    with_features = arg_service.get_features()

    dataset_service = DatasetService()
    export_service = ExportService(database=database, collection=collection)

    x_train_all = dataset_service.load_training_features(which="all", with_features=with_features)
    x_test_all = dataset_service.load_testing_features(which="all", with_features=with_features)
    y_train_all = dataset_service.load_training_labels(which="all", with_features=with_features)
    y_test_all = dataset_service.load_testing_labels(which="all", with_features=with_features)

    x_train_all = pd.concat(x_train_all).to_numpy()
    y_train_all = pd.concat(y_train_all).to_numpy()

    x_test_all = pd.concat(x_test_all).to_numpy()
    y_test_all = pd.concat(y_test_all).to_numpy()

    # Execute machine learning pipeline for each configured model
    for model_enum, resampling_method, scaling_method in product([model_enum], resampling_methods, scaling_methods):
        run_id = export_service.generate_unique_id([model_enum, resampling_method, scaling_method, database])

        if export_service.run_exists(run_id):
            print(
                f"Run with: model={model_enum}, resampling_method={resampling_method}, scaling_method={scaling_method}"
                f" on database {database} already exists"
            )
            continue

        print(
            f"Executing run with: model={model_enum}, resampling_method={resampling_method}, "
            f"scaling_method={scaling_method} on database={database}"
        )

        # Keep track of what has been done
        run_info = {
            "_id": run_id,
            "model": model_enum.value,
            "pre-processing": {
                "resampling": {"method": resampling_method},
                "scaling": {"method": scaling_method},
            },
            "centralized_scoring": {},
            "individual_scoring": [],
        }

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

        model.fit(x_train=x_train_all, y_train=y_train_all, run_info=run_info)

        # Get training and testing results on centralized dataset
        pred_train_all = model.predict(x=x_train_all)
        scores_train_all, _ = model.evaluate(pred=pred_train_all, y_true=y_train_all)
        pred_test_all = model.predict(x=x_test_all)
        scores_test_all, cm_all = model.evaluate(pred=pred_test_all, y_true=y_test_all)

        run_info["centralized_scoring"] = {"training_set": scores_train_all, "testing_set": scores_test_all}

        # Get training and testing results on individual datasets
        for subject in range(2, 36):
            x_train = dataset_service.load_training_features(which=subject, with_features=with_features).to_numpy()
            x_test = dataset_service.load_testing_features(which=subject, with_features=with_features).to_numpy()
            y_train = dataset_service.load_training_labels(which=subject, with_features=with_features).to_numpy()
            y_test = dataset_service.load_testing_labels(which=subject, with_features=with_features).to_numpy()

            pred_train = model.predict(x=x_train)
            scores_train, _ = model.evaluate(pred=pred_train, y_true=y_train)
            pred_test = model.predict(x=x_test)
            scores_test, _ = model.evaluate(pred=pred_test, y_true=y_test)

            scores = {"training_set": scores_train, "testing_set": scores_test}

            run_info["individual_scoring"].append(scores)

        # Export run configuration and results to mongodb
        mongo_id = export_service.export_run_to_mongodb(run_info=run_info)
        if mongo_id is not None:
            model.save_model(path=BASE_PATH / "models" / f"{run_id}.joblib")

        # Cleanup some memory
        del model
        del scaler
        del resampler
