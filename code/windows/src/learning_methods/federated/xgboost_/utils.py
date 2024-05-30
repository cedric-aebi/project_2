import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix


def transform_dataset_to_dmatrix(x: np.ndarray, y: np.ndarray) -> xgb.DMatrix:
    """Transform dataset to DMatrix format for xgboost."""
    new_data = xgb.DMatrix(x, label=y)
    return new_data


def get_class_labels_from_probs(probs: np.ndarray) -> np.ndarray:
    pred = np.repeat(0, probs.shape[0])
    pred[probs > 0.5] = 1
    return pred


def evaluate_prediction(pred: np.ndarray, y_true: np.ndarray) -> tuple[dict, np.ndarray]:
    scores = get_scores(pred=pred, y=y_true)
    tp, tn, fp, fn = get_classification_results(cm=scores[4])
    results = {
        "accuracy": scores[0],
        "recall": scores[1],
        "precision": scores[2],
        "f1": scores[3],
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }
    # Return results and confusion matrix for later plotting
    return results, scores[4]


def get_scores(pred: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, np.ndarray]:
    acc = accuracy_score(pred, y)
    rec = recall_score(pred, y)
    prec = precision_score(pred, y)
    f1 = f1_score(pred, y)
    cm = confusion_matrix(y_true=y, y_pred=pred)
    return acc, rec, prec, f1, cm


def get_classification_results(cm: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(cm[1][1])
    tn = int(cm[0][0])
    fp = int(cm[0][1])
    fn = int(cm[1][0])
    return tp, tn, fp, fn
