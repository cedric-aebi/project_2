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
EXPORT_PATH = Path(__file__).parent.parent.parent.parent / "results" / "centralized" / "models"
# ***************************************************************************

if __name__ == "__main__":
    arg_service = ArgumentService(model=True, resampling=True, scaling=True, database=True, features=True, dataset=True)
    models = arg_service.get_models()
    resampling_methods = arg_service.get_resampling_methods()
    scaling_methods = arg_service.get_scaling_methods()
    database = arg_service.get_database()
    features_list = arg_service.get_features()
    dataset = arg_service.get_dataset()

    export_service = ExportService(database=database, collection="centralized")

    # Find the specific methods you want
    oversampling_method = next((m for m in resampling_methods if m and m == ResamplingMethod.UNDERSAMPLING), None)
    standardscaling_method = next((m for m in scaling_methods if m and m == ScalingMethod.STANDARDSCALER), None)

    combinations = []

    for model_enum, with_features in product(models, features_list):
        # Both None
        combinations.append((model_enum, None, None, with_features))
        # Both oversampling and standardscaling
        if oversampling_method and standardscaling_method:
            combinations.append((model_enum, oversampling_method, standardscaling_method, with_features))

    # Execute machine learning pipeline for each configured model
    for model_enum, resampling_method, scaling_method, with_features in combinations:
        # 1. Create run configuration with the given parameters
        run_info = {
            "model": model_enum.value,
            "pre-processing": {
                "features": with_features,
                "resampling": {"method": resampling_method.value if resampling_method is not None else None},
                "scaling": {"method": scaling_method.value if scaling_method is not None else None},
            },
            "training_runs": [],
        }

        # 3. Create a has over the run_info dict and the current database and check if run already exists
        run_id = export_service.generate_unique_id([database, json.dumps(run_info)])

        if export_service.run_exists(run_id):
            print(f"Run with id: {run_id} on database {database} already exists")
            continue

        print(f"Executing run with configuration: {run_info} on database {database}")

        # 4. Set run id and fit the model on the centralized dataset
        run_info["_id"] = run_id

        # 5. Fit models
        for idx, participant_leave_out in enumerate(utils.get_list_of_lave_out_participants(dataset=dataset)):
            run_info["training_runs"].append({"participant_leave_out": str(participant_leave_out)})

            df = utils.load_data(dataset=dataset, which="all", with_features=with_features)
            # shuffle the data
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

            # Convert "Participant" column to string if it is not already
            df["Participant"] = df["Participant"].astype(str)

            train = df[df["Participant"] != str(participant_leave_out)]
            test = df[df["Participant"] == str(participant_leave_out)]
            x_train = train.drop(columns=["Participant", "Label"])
            y_train = train["Label"]
            x_test = test.drop(columns=["Participant", "Label"])
            y_test = test["Label"]

            if with_features:
                # If features are used, we need to drop the "Split" column
                x_train = x_train.drop(columns=["Split"])
                x_test = x_test.drop(columns=["Split"])

            # 1. Initialize model, scaler and resampler
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
                    model = ShallowNNModel(
                        scaler=scaler,
                        resampler=resampler,
                        input_shape=x_train.shape[1],
                        dataset=dataset,
                        with_features=with_features,
                        centralized=True,
                    )
                case _:
                    raise Exception(f"Could not initialize model {model_enum.value} for config")
            model.fit(x_train=x_train, y_train=y_train, run_info=run_info)

            # 5. Get training and testing results on centralized dataset
            pred_train = model.predict(x=x_train)
            scores_train, _ = model.evaluate(pred=pred_train, y_true=y_train)
            pred_test = model.predict(x=x_test)
            scores_test, cm = model.evaluate(pred=pred_test, y_true=y_test)
            scores = {"training_set": scores_train, "testing_set": scores_test}
            run_info["training_runs"][idx]["scores"] = scores

            # 7. Export run configuration and results to mongodb
            if not os.path.exists(EXPORT_PATH):
                os.makedirs(EXPORT_PATH)
            joblib.dump(model, EXPORT_PATH / f"{run_id}.joblib", compress=3)

            # 8. Cleanup memory
            del (
                df,
                train,
                test,
                x_train,
                y_train,
                x_test,
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

        mongo_id = export_service.export_run_to_mongodb(run_info=run_info)
