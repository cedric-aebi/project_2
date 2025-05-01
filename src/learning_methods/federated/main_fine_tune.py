from pathlib import Path

from enums.ResamplingMethod import ResamplingMethod
from enums.ScalingMethod import ScalingMethod
from service.exportservice.ExportService import ExportService
from enums.Model import Model
from sklearn.model_selection import train_test_split
import joblib
from utils import utils
from enums.Dataset import Dataset
from sklearn.preprocessing import StandardScaler
from keras.src.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
import xgboost as xgb

MODEL_ID = "f6da7034a9bd66ab7f34094d760a01c00e9f49e155523b0a2a23c6c8024a9ddc"
MODEL_TYPE = Model.SHALLOW_NN
SCALING: ScalingMethod | None = ScalingMethod.STANDARDSCALER
RESAMPLING: ResamplingMethod | None = None


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
    export_service = ExportService(collection="federated_fine_tuned", database="paper_2")

    root_path = Path(__file__).parent.parent.parent.parent
    run_info = {
        "_id": MODEL_ID,
        "model": MODEL_TYPE.value,
        "pre-processing": {
            "features": False,
            "resampling": {"method": RESAMPLING.value if RESAMPLING is not None else None},
            "scaling": {"method": SCALING.value},
        },
        "participants": [],
    }

    accs, f1s, precisions, recalls = [], [], [], []
    for idx, participant in enumerate(utils.get_list_of_participants(dataset=Dataset.NURSE)):
        run_info["participants"].append({"participant": participant})

        df = pd.read_pickle(root_path / "datasets" / "nurse" / "paper" / f"{participant}.pkl")
        x = df.drop(columns=["Label", "Participant"])
        y = df["Label"]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
        )

        # scale
        if SCALING is not None:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

        if MODEL_TYPE == Model.XGBOOST:
            # Load the model
            model = xgb.Booster()
            model.load_model(
                root_path / "results" / "bfh_server" / "federated" / "models" / f"{MODEL_ID}_run_index_0_model.json"
            )
            params = joblib.load(
                root_path / "results" / "bfh_server" / "federated" / "models" / f"{MODEL_ID}_run_index_0_params.pkl"
            )

            # Local training
            bst = xgb.train(params, xgb.DMatrix(x_train, label=y_train), num_boost_round=100, xgb_model=model)

            y_pred = bst.predict(xgb.DMatrix(x_test, label=y_test))
            # Convert predictions to binary labels
            y_pred = (y_pred > 0.5).astype(int)
        else:
            model = joblib.load(
                Path(__file__).parent.parent.parent.parent
                / "results"
                / "bfh_server"
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
            history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

            # sklearn metrics: accuracy, f1, precision, recall
            y_pred = model.predict(x_test)
            # Convert predictions to binary labels
            y_pred = (y_pred > 0.5).astype(int)

        scores_test, cm = evaluate(y_true=y_test, pred=y_pred)
        scores = {"testing_set": scores_test}
        run_info["participants"][idx]["scores"] = scores

        del model, x_train, x_test, y_train, y_test, df, scaler

    export_service.export_run_to_mongodb(run_info=run_info)
