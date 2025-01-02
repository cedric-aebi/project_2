import numpy as np
from imblearn.base import BaseSampler
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from model.AbstractModel import AbstractModel


class LogisticRegressionModel(AbstractModel):
    def __init__(self, scaler: BaseEstimator | None, resampler: BaseSampler | None):
        self._grid_search_cv = None
        hyperparameter_grid = {
            "clf__penalty": ["l2", "l1", None],
            "clf__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "clf__solver": ["lbfgs", "liblinear", "sag"],
        }
        super().__init__(
            clf=LogisticRegression(random_state=42, max_iter=1000),
            hyperparameter_grid=hyperparameter_grid,
            scaler=scaler,
            resampler=resampler,
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, run_info: dict) -> None:
        self._grid_search_cv = GridSearchCV(
            estimator=self._pipeline, param_grid=self._hyperparameter_grid, cv=self._cv, n_jobs=-1
        )
        self._grid_search_cv.fit(x_train, y_train.ravel())
        self._best_estimator = self._grid_search_cv.best_estimator_
        run_info["best_params"] = self._grid_search_cv.best_params_

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._best_estimator.predict(x)

    def get_fitted_model(self) -> Pipeline:
        return self._best_estimator
