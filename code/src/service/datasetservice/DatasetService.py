from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.base import BaseSampler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from numpy import ndarray
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from enums.ResamplingMethod import ResamplingMethod
from enums.ScalingMethod import ScalingMethod


class DatasetService:
    def __init__(self):
        self.__path_to_dataset = (
            Path(__file__).parent.parent.parent.parent / "dataset" / "Improved_All_Combined_hr_rsp_binary.csv"
        )
        self.__path_to_individual_datasets = Path(__file__).parent.parent.parent.parent / "dataset" / "individual"
        self.__label_column = "Label"
        self.__labels = ["No Stress", "Stress"]
        self.__not_needed_columns = [self.__label_column, "Time(sec)", "Participant"]

    def load_dataset(self) -> pd.DataFrame:
        return pd.read_csv(self.__path_to_dataset, sep=",")

    def load_individual_dataset(self, participant: int) -> pd.DataFrame:
        return pd.read_csv(self.__path_to_individual_datasets / f"participant_{participant}.csv", sep=",")

    @staticmethod
    def remove_nan(dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset.dropna()

    def get_features_and_labels(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, ndarray, list[str]]:
        x = dataset.drop(columns=self.__not_needed_columns)
        y = dataset[self.__label_column].values
        return x, y, self.__labels

    @staticmethod
    def get_resampler(method: ResamplingMethod) -> BaseSampler | None:
        match method:
            case ResamplingMethod.SMOTE:
                resampler = SMOTE(random_state=42)
            case ResamplingMethod.OVERSAMPLING:
                resampler = RandomOverSampler(random_state=42)
            case ResamplingMethod.UNDERSAMPLING:
                resampler = RandomUnderSampler(random_state=42)
            case ResamplingMethod.TL:
                resampler = TomekLinks()
            case ResamplingMethod.SMOTEENN:
                resampler = SMOTEENN(random_state=42)
            case _:
                resampler = None

        return resampler

    @staticmethod
    def get_scaler(method: ScalingMethod) -> BaseEstimator | None:
        match method:
            case ScalingMethod.STANDARDSCALER:
                scaler = StandardScaler()
            case ScalingMethod.MINMAXSCALER:
                scaler = MinMaxScaler()
            case _:
                scaler = None
        return scaler

    @staticmethod
    def train_test_split(
        x: pd.DataFrame, y: np.ndarray, shuffle: bool, run_info: dict
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        split_ratio = 0.2
        train_x, test_x, train_y, test_y = train_test_split(
            x, y, test_size=split_ratio, shuffle=shuffle, random_state=42, stratify=y
        )
        run_info["data_splitting"] = {
            "split": split_ratio,
            "shuffle": shuffle,
            "train_size": len(train_x),
            "test_size": len(test_x),
        }
        return train_x, test_x, train_y, test_y
