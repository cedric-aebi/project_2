import os
import warnings

from enums.Dataset import Dataset

warnings.simplefilter("ignore", category=FutureWarning)
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

from imblearn.pipeline import Pipeline
import pandas as pd
from imblearn.base import BaseSampler
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from model.AbstractModel import AbstractModel


class LogisticRegressionModel(AbstractModel):
    def __init__(
        self, scaler: BaseEstimator | None, resampler: BaseSampler | None, dataset: Dataset, with_features: bool
    ):
        self._grid_search_cv = None
        self._dataset = dataset
        hyperparameter_grid = {
            "clf__penalty": ["l2", "l1"],  # l1 and l2 for regularization
            "clf__C": [1, 0.1, 0.01],
            "clf__solver": ["lbfgs", "saga"],
        }
        super().__init__(
            clf=LogisticRegression(random_state=42, max_iter=1000),
            hyperparameter_grid=hyperparameter_grid,
            scaler=scaler,
            resampler=resampler,
            dataset=dataset,
            with_features=with_features,
        )

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame, run_info: dict) -> None:
        self._grid_search_cv = GridSearchCV(
            estimator=self._pipeline,
            param_grid=self._hyperparameter_grid,
            cv=self._cv,
            n_jobs=self._get_number_of_jobs(),
            verbose=2,
        )
        self._grid_search_cv.fit(x_train, y_train.ravel())
        self._best_estimator = self._grid_search_cv.best_estimator_
        run_info["best_params"] = self._grid_search_cv.best_params_

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return self._best_estimator.predict(x)

    def get_fitted_model(self) -> Pipeline:
        return self._best_estimator
