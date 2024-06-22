from pathlib import Path

import keras
import numpy as np
from keras import layers
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix


def build_model(number_of_features: int) -> keras.Sequential:
    # Define the model
    model = keras.Sequential()
    model.add(layers.Dense(512, input_dim=number_of_features, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(54, activation="relu"))
    model.add(layers.Dense(50, activation="relu"))
    # Output layer with 1 neuron, sigmoid activation for binary classification
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def get_log_dir(unique_run_id: str) -> Path:
    return Path(__file__).parent.parent.parent.parent.parent / "logs" / unique_run_id


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
