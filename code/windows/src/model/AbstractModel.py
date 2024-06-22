from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from imblearn.base import BaseSampler
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


class AbstractModel(ABC):
    def __init__(
        self, model: Any, scaler: BaseEstimator, resampler: BaseSampler, hyperparameter_grid: dict | None = None
    ) -> None:
        self._kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        self._model = model
        self._hyperparameter_grid = hyperparameter_grid
        self._scaler = scaler
        self._resampler = resampler
        self._pipeline = Pipeline([("scaler", self._scaler), ("resampler", self._resampler), ("model", self._model)])
        self._best_estimator = None

    def get_hyperparameter_grid(self) -> dict | None:
        return self._hyperparameter_grid

    @abstractmethod
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, run_info: dict) -> None:
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    def evaluate(self, pred: np.ndarray, y_true: np.ndarray) -> tuple[dict, np.ndarray]:
        scores = self.get_scores(pred=pred, y=y_true)
        tp, tn, fp, fn = self.get_classification_results(cm=scores[4])
        results = {
            "accuracy": scores[0],
            "recall": scores[1],
            "precision": scores[2],
            "f1": scores[3],
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        }
        # Return results and confusion matrix for later plotting
        return results, scores[4]

    @staticmethod
    def get_scores(pred: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, np.ndarray]:
        acc = accuracy_score(pred, y)
        rec = recall_score(pred, y)
        prec = precision_score(pred, y)
        f1 = f1_score(pred, y)
        cm = confusion_matrix(y_true=y, y_pred=pred)
        return acc, rec, prec, f1, cm

    @staticmethod
    def get_classification_results(cm: np.ndarray) -> tuple[int, int, int, int]:
        tp = int(cm[1][1])
        tn = int(cm[0][0])
        fp = int(cm[0][1])
        fn = int(cm[1][0])
        return tp, tn, fp, fn

    @abstractmethod
    def get_fitted_model(self) -> Pipeline | BaseEstimator:
        pass
