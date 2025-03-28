import json
import os
from itertools import product
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from enums.Dataset import Dataset
from enums.Model import Model
from model.ShallowNNModel import ShallowNNModel
from model.LogisticRegressionModel import LogisticRegressionModel
from model.XGBoostModel import XGBoostModel
from service.argumentservice.ArgumentService import ArgumentService
from service.exportservice.ExportService import ExportService
from utils import utils

# ************************ DEFINE CONFIGURATION *****************************
EXPORT_PATH = Path(__file__).parent.parent.parent.parent / "results" / "individual" / "models"
# ***************************************************************************


if __name__ == "__main__":
    arg_service = ArgumentService(model=True, resampling=True, scaling=True, database=True, features=True, dataset=True)
    models = arg_service.get_models()
    resampling_methods = arg_service.get_resampling_methods()
    scaling_methods = arg_service.get_scaling_methods()
    database = arg_service.get_database()
    features_list = arg_service.get_features()
    dataset = arg_service.get_dataset()

    export_service = ExportService(database=database, collection="individual")

    # Execute machine learning pipeline for each configured model
    for model_enum, resampling_method, scaling_method, with_features in product(
        models, resampling_methods, scaling_methods, features_list
    ):
        # 1. Initialize dummy model for hash calculation
        match model_enum:
            case Model.XGBOOST:
                dummy_model = XGBoostModel(scaler=None, resampler=None)
            case Model.LOGISTIC_REGRESSION:
                dummy_model = LogisticRegressionModel(scaler=None, resampler=None)
            case Model.SHALLOW_NN:
                dummy_model = ShallowNNModel(
                    scaler=None,
                    resampler=None,
                    input_shape=0,
                    epochs=0,
                    batch_size=0,
                )
            case _:
                raise Exception(f"Could not initialize model {model_enum.value} for config")

        # 2. Create run configuration with the given parameters
        run_info = {
            "model": model_enum.value,
            "pre-processing": {
                "features": with_features,
                "resampling": {"method": resampling_method.value if resampling_method is not None else None},
                "scaling": {"method": scaling_method.value if scaling_method is not None else None},
            },
            "participants": [],
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
        for idx, participant in enumerate(utils.get_list_of_participants(dataset=dataset)):
            run_info["participants"].append({"participant": participant})

            x, y = utils.load_data(dataset=dataset, which=participant, with_features=with_features)
            x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=42, stratify=y)

            scaler = utils.get_scaler(method=scaling_method)
            resampler = utils.get_resampler(method=resampling_method)

            match model_enum:
                case Model.XGBOOST:
                    model = XGBoostModel(scaler=scaler, resampler=resampler)
                case Model.LOGISTIC_REGRESSION:
                    model = LogisticRegressionModel(scaler=scaler, resampler=resampler)
                case Model.SHALLOW_NN:
                    model = ShallowNNModel(
                        scaler=scaler,
                        resampler=resampler,
                        input_shape=x_train.shape[1],
                        epochs=150 if dataset == Dataset.STRESS else 50,
                        batch_size=32 if dataset == Dataset.STRESS else 64,
                    )
                case _:
                    raise Exception(f"Could not initialize model {model_enum.value} for config")

            model.fit(x_train=x_train, y_train=y_train, run_info=run_info["participants"][idx])

            pred_train = model.predict(x=x_train)
            scores_train, _ = model.evaluate(pred=pred_train, y_true=y_train)
            pred_test = model.predict(x=x_test)
            scores_test, _ = model.evaluate(pred=pred_test, y_true=y_test)
            scores = {"training_set": scores_train, "testing_set": scores_test}
            run_info["participants"][idx]["scores"] = scores

            if not os.path.exists(EXPORT_PATH):
                os.makedirs(EXPORT_PATH)
            joblib.dump(model, EXPORT_PATH / f"{run_id}_participant_{participant}.joblib", compress=3)

            # Free up memory and garbage collect
            del scaler, resampler, model

        # Export run configuration and results to mongodb
        mongo_id = export_service.export_run_to_mongodb(run_info=run_info)

        del dummy_model

        # Restart the script to free up memory
        # os.execv(sys.executable, ["python"] + sys.argv)
