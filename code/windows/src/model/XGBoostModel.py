import numpy as np
from imblearn.base import BaseSampler
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from model.AbstractModel import AbstractModel


class XGBoostModel(AbstractModel):
    def __init__(self, scaler: BaseEstimator, resampler: BaseSampler):
        hyperparameter_grid = {
            "model__n_estimators": [10, 20, 50, 100, 150, 200, 250, 300, 350],
            "model__max_depth": range(2, 12, 2),
        }
        super().__init__(
            model=XGBClassifier(random_state=42),
            hyperparameter_grid=hyperparameter_grid,
            scaler=scaler,
            resampler=resampler,
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, run_info: dict) -> None:
        grid_search_cv = GridSearchCV(
            estimator=self._pipeline, param_grid=self._hyperparameter_grid, cv=self._kfold, n_jobs=-1
        )
        grid_result = grid_search_cv.fit(x_train, y_train)
        self._best_estimator = grid_result.best_estimator_
        run_info["best_params"] = grid_result.best_params_

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._best_estimator.predict(x)

    def get_fitted_model(self) -> BaseEstimator:
        return self._best_estimator
