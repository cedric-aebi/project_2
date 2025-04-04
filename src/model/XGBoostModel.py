import os
import warnings

from enums.Dataset import Dataset

warnings.simplefilter("ignore", category=FutureWarning)
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

from imblearn.pipeline import Pipeline
import pandas as pd
from imblearn.base import BaseSampler
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from xgboost import XGBClassifier

from model.AbstractModel import AbstractModel


class XGBoostModel(AbstractModel):
    def __init__(
        self, scaler: BaseEstimator | None, resampler: BaseSampler | None, dataset: Dataset, with_features: bool
    ):
        self._grid_search_cv = None
        self._dataset = dataset
        hyperparameter_grid = {
            "clf__max_depth": [4, 6, 8, 10, 12],
        }
        super().__init__(
            clf=XGBClassifier(random_state=42),
            hyperparameter_grid=hyperparameter_grid,
            scaler=scaler,
            resampler=resampler,
            dataset=dataset,
            with_features=with_features,
        )

    def fit(
        self, x_train: pd.DataFrame, y_train: pd.DataFrame, run_info: dict, groups: pd.DataFrame | None = None
    ) -> None:
        if groups is not None:
            cv = StratifiedGroupKFold(n_splits=len(set(groups)), shuffle=True, random_state=42)
            self._grid_search_cv = GridSearchCV(
                estimator=self._pipeline,
                param_grid=self._hyperparameter_grid,
                cv=cv,
                n_jobs=self._get_number_of_jobs(),
                verbose=2,
                scoring="f1_weighted",
            )
            self._grid_search_cv.fit(x_train, y_train, groups=groups)
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            self._grid_search_cv = GridSearchCV(
                estimator=self._pipeline,
                param_grid=self._hyperparameter_grid,
                cv=cv,
                n_jobs=self._get_number_of_jobs(),
                verbose=2,
                scoring="f1_weighted",
            )
            self._grid_search_cv.fit(x_train, y_train)
        self._best_estimator = self._grid_search_cv.best_estimator_
        selected_features = self._best_estimator.named_steps["feature_selection"].get_support(indices=True)
        feature_names = x_train.columns[selected_features]
        run_info["selected_features"] = list(feature_names)
        run_info["cv_best_score"] = self._grid_search_cv.best_score_
        run_info["cv_best_params"] = self._grid_search_cv.best_params_

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return self._best_estimator.predict(x)

    def get_fitted_model(self) -> Pipeline:
        return self._best_estimator
