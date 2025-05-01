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

from model.AbstractModel import AbstractModel


class LogisticRegressionModel(AbstractModel):
    def __init__(
        self, scaler: BaseEstimator | None, resampler: BaseSampler | None, dataset: Dataset, with_features: bool
    ):
        self._dataset = dataset
        super().__init__(
            clf=LogisticRegression(random_state=42, max_iter=1000),
            scaler=scaler,
            resampler=resampler,
            dataset=dataset,
            with_features=with_features,
        )

    def fit(
        self, x_train: pd.DataFrame, y_train: pd.DataFrame, run_info: dict, groups: pd.DataFrame | None = None
    ) -> None:
        self._pipeline.fit(x_train, y_train.ravel())

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return self._pipeline.predict(x)

    def get_fitted_model(self) -> Pipeline:
        return self._pipeline
