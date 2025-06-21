from pathlib import Path

from service.argumentservice.ArgumentService import ArgumentService
from service.exportservice.ExportService import ExportService
from enums.Model import Model
import joblib
from utils import utils
from enums.Dataset import Dataset
from keras.src.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
import xgboost as xgb

MODEL_ID = "170e083b41881e0114732a37b7c7292149589cad5c8a1d26be65b283c0575113"


def evaluate(pred: pd.DataFrame, y_true: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    scores = get_scores(pred=pred, y=y_true)
    tp, tn, fp, fn = get_classification_results(_cm=scores[4])
    results = {
        "accuracy": scores[0],
        "recall": scores[1],
        "precision": scores[2],
        "f1": scores[3],
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }
    # Return results and confusion matrix for later plotting
    return results, scores[4]


def get_scores(pred: pd.DataFrame, y: pd.DataFrame) -> tuple[float, float, float, float, pd.DataFrame]:
    acc = accuracy_score(pred, y)
    rec = recall_score(pred, y)
    prec = precision_score(pred, y)
    f1 = f1_score(pred, y)
    cm = confusion_matrix(y_true=y, y_pred=pred)
    return acc, rec, prec, f1, cm


def get_classification_results(_cm: pd.DataFrame) -> tuple[int, int, int, int]:
    tp = int(_cm[1][1])
    tn = int(_cm[0][0])
    fp = int(_cm[0][1])
    fn = int(_cm[1][0])
    return tp, tn, fp, fn


if __name__ == "__main__":
    arg_service = ArgumentService(model=True, resampling=True, scaling=True, database=True, features=True, dataset=True)
    model_enum = arg_service.get_models()[0]
    resampling_method = arg_service.get_resampling_methods()[0]
    scaling_method = arg_service.get_scaling_methods()[0]
    database = arg_service.get_database()
    with_features = arg_service.get_features()[0]
    dataset = arg_service.get_dataset()

    export_service = ExportService(collection="federated_fine_tuned", database=database)

    root_path = Path(__file__).parent.parent.parent.parent
    run_info = {
        "_id": MODEL_ID,
        "model": model_enum.value,
        "pre-processing": {
            "features": with_features,
            "resampling": {"method": resampling_method.value if resampling_method is not None else None},
            "scaling": {"method": scaling_method.value if scaling_method is not None else None},
        },
        "participants": [],
    }

    accs, f1s, precisions, recalls = [], [], [], []
    for idx, participant in enumerate(utils.get_list_of_participants(dataset=Dataset.NURSE)):
        run_info["participants"].append({"participant": participant})

        df = utils.load_data(dataset=dataset, with_features=with_features, which=participant)

        x_train, x_val, x_test, y_train, y_val, y_test = utils.split_data(
            df=df, with_features=with_features, model=model_enum
        )

        scaler = utils.get_scaler(method=scaling_method)
        resampler = utils.get_resampler(method=resampling_method)

        # scale
        if scaler is not None:
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_val = scaler.transform(x_val) if x_val is not None else None
            x_test = scaler.transform(x_test)
        # resample
        if resampler is not None:
            x_train, y_train = resampler.fit_resample(x_train, y_train)

        if model_enum == Model.XGBOOST:
            # Load the model
            model = xgb.Booster()
            model.load_model(root_path / "results" / "federated" / "models" / f"{MODEL_ID}_run_index_0_model.json")
            params = joblib.load(root_path / "results" / "federated" / "models" / f"{MODEL_ID}_run_index_0_params.pkl")

            # Local training
            bst = xgb.train(params, xgb.DMatrix(x_train, label=y_train), num_boost_round=100, xgb_model=model)

            y_pred = bst.predict(xgb.DMatrix(x_test, label=y_test))
            # Convert predictions to binary labels
            y_pred = (y_pred > 0.5).astype(int)
        elif model_enum == Model.SHALLOW_NN:
            model = joblib.load(
                Path(__file__).parent.parent.parent.parent
                / "results"
                / "federated"
                / "models"
                / f"{MODEL_ID}_run_index_0_model.pkl"
            )

            # Compile the model with a lower learning rate
            model.compile(
                optimizer=Adam(learning_rate=0.0001),  # Use a lower learning rate for fine-tuning
                loss="binary_crossentropy",  # Use the same loss function as in original training
                metrics=["accuracy"],
            )
            # Fine-tune the model
            history = model.fit(
                x_train, y_train, epochs=5, batch_size=128, validation_split=0.2, validation_data=(x_val, y_val)
            )

            # sklearn metrics: accuracy, f1, precision, recall
            y_pred = model.predict(x_test)
            # Convert predictions to binary labels
            y_pred = (y_pred > 0.5).astype(int)
        elif model_enum == Model.LOGISTIC_REGRESSION:
            model = joblib.load(
                Path(__file__).parent.parent.parent.parent
                / "results"
                / "federated"
                / "models"
                / f"{MODEL_ID}_run_index_0_model.pkl"
            )

            model.fit(x_train, y_train)

            # sklearn metrics: accuracy, f1, precision, recall
            y_pred = model.predict(x_test)
            # Convert predictions to binary labels
            y_pred = (y_pred > 0.5).astype(int)
        else:
            raise Exception(f"Could not initialize model {model_enum.value} for config")

        scores_test, cm = evaluate(y_true=y_test, pred=y_pred)
        scores = {"testing_set": scores_test}
        run_info["participants"][idx]["scores"] = scores

        del model, x_train, x_test, y_train, y_test, df, scaler

    export_service.export_run_to_mongodb(run_info=run_info)
