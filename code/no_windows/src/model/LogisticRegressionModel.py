import numpy as np
from imblearn.base import BaseSampler
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from model.AbstractModel import AbstractModel


class LogisticRegressionModel(AbstractModel):
    def __init__(self, scaler: BaseEstimator, resampler: BaseSampler):
        hyperparameter_grid = {
            "model__penalty": ["l2", "l1", "elasticnet", None],
            "model__C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "model__solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
        }
        super().__init__(
            model=LogisticRegression(random_state=42),
            hyperparameter_grid=hyperparameter_grid,
            scaler=scaler,
            resampler=resampler,
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, run_info: dict) -> None:
        grid_search_cv = GridSearchCV(
            estimator=self._pipeline, param_grid=self._hyperparameter_grid, n_jobs=-1, cv=self._kfold
        )
        grid_result = grid_search_cv.fit(x_train, y_train.ravel())
        self._best_estimator = grid_result.best_estimator_
        run_info["best_params"] = grid_result.best_params_

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._best_estimator.predict(x)

    def get_fitted_model(self) -> BaseEstimator:
        return self._best_estimator
