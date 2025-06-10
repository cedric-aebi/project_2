import json
import os
from itertools import product
from pathlib import Path

import joblib

from enums.Model import Model
from enums.ResamplingMethod import ResamplingMethod
from enums.ScalingMethod import ScalingMethod
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
    arg_service = ArgumentService(
        model=True, resampling=True, scaling=True, database=True, features=True, dataset=True, tiny=True
    )
    models = arg_service.get_models()
    resampling_methods = arg_service.get_resampling_methods()
    scaling_methods = arg_service.get_scaling_methods()
    database = arg_service.get_database()
    features_list = arg_service.get_features()
    dataset = arg_service.get_dataset()
    tiny_param = arg_service.get_tiny()

    export_service = ExportService(database=database, collection="individual")

    # Execute machine learning pipeline for each configured model
    for model_enum, resampling_method, scaling_method, with_features, tiny in product(
        models, resampling_methods, scaling_methods, features_list, tiny_param
    ):
        # 1. Create run configuration with the given parameters
        run_info = {
            "model": model_enum.value,
            "pre-processing": {
                "features": with_features,
                "resampling": {"method": resampling_method.value if resampling_method is not None else None},
                "scaling": {"method": scaling_method.value if scaling_method is not None else None},
            },
            "participants": [],
            "tiny": tiny,
        }

        # 2. Create a has over the run_info dict and the current database and check if run already exists
        run_id = export_service.generate_unique_id([database, json.dumps(run_info)])

        if export_service.run_exists(run_id):
            print(f"Run with configuration: {run_info} on database {database} already exists")
            continue

        print(f"Executing run with configuration: {run_info} on database {database}")

        # 3. Set run id and fit the model on the centralized dataset
        run_info["_id"] = run_id

        # 4. Fit models for each participant in the dataset
        for idx, participant in enumerate(utils.get_list_of_participants(dataset=dataset)):
            run_info["participants"].append({"participant": participant})

            df = utils.load_data(dataset=dataset, with_features=with_features, which=participant)
            x_train, x_val, x_test, y_train, y_val, y_test = utils.split_data(
                df=df, with_features=with_features, model=model_enum
            )

            scaler = utils.get_scaler(method=scaling_method)
            resampler = utils.get_resampler(method=resampling_method)

            match model_enum:
                case Model.XGBOOST:
                    model = XGBoostModel(
                        scaler=scaler, resampler=resampler, dataset=dataset, with_features=with_features
                    )
                case Model.LOGISTIC_REGRESSION:
                    model = LogisticRegressionModel(
                        scaler=scaler, resampler=resampler, dataset=dataset, with_features=with_features
                    )
                case Model.SHALLOW_NN:
                    # Scale validation data by hand
                    if scaler is not None:
                        _ = scaler.fit_transform(x_train)
                        x_val = scaler.transform(x_val)
                    model = ShallowNNModel(
                        scaler=scaler,
                        resampler=resampler,
                        input_shape=x_train.shape[1],
                        dataset=dataset,
                        with_features=with_features,
                        val_data=(x_val, y_val),
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
            del (
                df,
                x_train,
                x_val,
                x_test,
                y_train,
                y_val,
                y_test,
                scaler,
                resampler,
                model,
                pred_train,
                scores_train,
                pred_test,
                scores_test,
                scores,
            )

        # Export run configuration and results to mongodb
        mongo_id = export_service.export_run_to_mongodb(run_info=run_info)
