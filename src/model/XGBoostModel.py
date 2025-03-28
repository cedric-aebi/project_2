import os
import warnings

warnings.simplefilter("ignore", category=FutureWarning)
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

from imblearn.pipeline import Pipeline
import pandas as pd
from imblearn.base import BaseSampler
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from model.AbstractModel import AbstractModel


class XGBoostModel(AbstractModel):
    def __init__(self, scaler: BaseEstimator | None, resampler: BaseSampler | None):
        self._grid_search_cv = None
        hyperparameter_grid = {
            "clf__max_depth": [4, 6, 8, 10, 12],
        }
        super().__init__(
            clf=XGBClassifier(random_state=42),
            hyperparameter_grid=hyperparameter_grid,
            scaler=scaler,
            resampler=resampler,
        )

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame, run_info: dict) -> None:
        self._grid_search_cv = GridSearchCV(
            estimator=self._pipeline, param_grid=self._hyperparameter_grid, cv=self._cv, n_jobs=6, verbose=2
        )
        self._grid_search_cv.fit(x_train, y_train)
        self._best_estimator = self._grid_search_cv.best_estimator_
        run_info["best_params"] = self._grid_search_cv.best_params_

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return self._best_estimator.predict(x)

    def get_fitted_model(self) -> Pipeline:
        return self._best_estimator
