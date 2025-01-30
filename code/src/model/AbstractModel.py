from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
from imblearn.base import BaseSampler
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold


class AbstractModel(ABC):
    def __init__(
        self,
        clf: Any,
        scaler: BaseEstimator | None,
        resampler: BaseSampler | None,
        hyperparameter_grid: dict | None = None,
    ) -> None:
        self._cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self._clf = clf
        self._hyperparameter_grid = hyperparameter_grid
        self._scaler = scaler
        self._resampler = resampler
        self._pipeline = Pipeline([("scaler", self._scaler), ("resampler", self._resampler), ("clf", self._clf)])
        self._best_estimator = None

    def get_hyperparameter_grid(self) -> dict | None:
        return dict(sorted(self._hyperparameter_grid.items()))

    @abstractmethod
    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame, run_info: dict) -> None:
        pass

    @abstractmethod
    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        pass

    def evaluate(self, pred: pd.DataFrame, y_true: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
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
    def get_scores(pred: pd.DataFrame, y: pd.DataFrame) -> tuple[float, float, float, float, pd.DataFrame]:
        acc = accuracy_score(pred, y)
        rec = recall_score(pred, y)
        prec = precision_score(pred, y)
        f1 = f1_score(pred, y)
        cm = confusion_matrix(y_true=y, y_pred=pred)
        return acc, rec, prec, f1, cm

    @staticmethod
    def get_classification_results(cm: pd.DataFrame) -> tuple[int, int, int, int]:
        tp = int(cm[1][1])
        tn = int(cm[0][0])
        fp = int(cm[0][1])
        fn = int(cm[1][0])
        return tp, tn, fp, fn

    @abstractmethod
    def get_fitted_model(self) -> Pipeline:
        pass
