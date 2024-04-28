from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from numpy import ndarray
from sklearn.model_selection import train_test_split

from enums.ResamplingMethod import ResamplingMethod


class DatasetService:
    def __init__(self):
        self.__path_to_dataset = (
            Path(__file__).parent.parent.parent.parent / "dataset" / "Improved_All_Combined_hr_rsp_binary.csv"
        )
        self.__label_column = "Label"
        self.__labels = ["No Stress", "Stress"]
        self.__not_needed_columns = [self.__label_column, "Time(sec)", "Participant"]

    def load_dataset(self) -> pd.DataFrame:
        return pd.read_csv(self.__path_to_dataset, sep=",")

    @staticmethod
    def remove_nan(dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset.dropna()

    def get_features_and_labels(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, ndarray, list[str]]:
        x = dataset.drop(columns=self.__not_needed_columns)
        y = dataset[self.__label_column].values
        return x, y, self.__labels

    def resample(self, x, y, method: ResamplingMethod, run_info: dict) -> tuple[np.ndarray, np.ndarray]:
        match method:
            case ResamplingMethod.SMOTE:
                x_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X=x, y=y)
            case ResamplingMethod.OVERSAMPLING:
                x_resampled, y_resampled = RandomOverSampler(random_state=42).fit_resample(X=x, y=y)
            case ResamplingMethod.UNDERSAMPLING:
                x_resampled, y_resampled = RandomUnderSampler(random_state=42).fit_resample(X=x, y=y)
            case _:
                x_resampled, y_resampled = None, None

        run_info["pre-processing"]["resampling"]["results"] = {
            "before": {self.__labels[0]: np.count_nonzero(y == 0), self.__labels[1]: np.count_nonzero(y == 1)},
            "after": {
                self.__labels[0]: np.count_nonzero(y_resampled == 0),
                self.__labels[1]: np.count_nonzero(y_resampled == 1),
            },
        }
        return x_resampled, y_resampled

    @staticmethod
    def train_test_split(
        x: np.ndarray, y: np.ndarray, shuffle: bool, run_info: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        split_ratio = 0.2
        train_x, test_x, train_y, test_y = train_test_split(
            x, y, test_size=split_ratio, shuffle=shuffle, random_state=42
        )
        run_info["data_splitting"] = {
            "split": split_ratio,
            "shuffle": shuffle,
            "train_size": len(train_x),
            "test:size": len(test_x),
        }
        return train_x, test_x, train_y, test_y
