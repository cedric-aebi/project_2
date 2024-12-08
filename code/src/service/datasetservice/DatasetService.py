import pickle
from pathlib import Path

import pandas as pd
from imblearn.base import BaseSampler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from enums.ResamplingMethod import ResamplingMethod
from enums.ScalingMethod import ScalingMethod


class DatasetService:
    def __init__(self):
        self.__path_to_datasets = Path(__file__).parent.parent.parent.parent / "dataset"
        self.__label_column = "Label"
        self.__labels = ["No Stress", "Stress"]
        self.__not_needed_columns = [self.__label_column, "Time(sec)", "Participant"]

    def load_training_features(self, which: str | int, with_features: bool) -> pd.DataFrame:
        if with_features:
            base = self.__path_to_datasets / "with_additional_features" / "features"
        else:
            base = self.__path_to_datasets / "no_additional_features" / "features"
        if which == "all":
            file_to_read = open(base / "all_training_features.pkl", "rb")
            x_train = pickle.load(file_to_read)
            file_to_read.close()
        else:
            file_to_read = open(base / f"training_features_{which}.pkl", "rb")
            x_train = pickle.load(file_to_read)
            file_to_read.close()
        return x_train

    def load_testing_features(self, which: str | int, with_features: bool) -> pd.DataFrame:
        if with_features:
            base = self.__path_to_datasets / "with_additional_features" / "features"
        else:
            base = self.__path_to_datasets / "no_additional_features" / "features"
        if which == "all":
            file_to_read = open(base / "all_testing_features.pkl", "rb")
            x_train = pickle.load(file_to_read)
            file_to_read.close()
        else:
            file_to_read = open(base / f"testing_features_{which}.pkl", "rb")
            x_train = pickle.load(file_to_read)
            file_to_read.close()
        return x_train

    def load_training_labels(self, which: str | int, with_features: bool) -> pd.DataFrame:
        if with_features:
            base = self.__path_to_datasets / "with_additional_features" / "features"
        else:
            base = self.__path_to_datasets / "no_additional_features" / "features"
        if which == "all":
            file_to_read = open(base / "all_training_labels.pkl", "rb")
            x_train = pickle.load(file_to_read)
            file_to_read.close()
        else:
            file_to_read = open(base / f"training_labels_{which}.pkl", "rb")
            x_train = pickle.load(file_to_read)
            file_to_read.close()
        return x_train

    def load_testing_labels(self, which: str | int, with_features: bool) -> pd.DataFrame:
        if with_features:
            base = self.__path_to_datasets / "with_additional_features" / "features"
        else:
            base = self.__path_to_datasets / "no_additional_features" / "features"
        if which == "all":
            file_to_read = open(base / "all_testing_labels.pkl", "rb")
            x_train = pickle.load(file_to_read)
            file_to_read.close()
        else:
            file_to_read = open(base / f"testing_labels_{which}.pkl", "rb")
            x_train = pickle.load(file_to_read)
            file_to_read.close()
        return x_train

    def load_dataset(self) -> pd.DataFrame:
        return pd.read_csv(self.__path_to_datasets / "Improved_All_Combined_hr_rsp_binary.csv", sep=",")

    @staticmethod
    def remove_nan(dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset.dropna()

    def get_features_and_labels(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        x = dataset.drop(columns=self.__not_needed_columns)
        y = dataset[self.__label_column].to_frame()
        return x, y, self.__labels

    @staticmethod
    def get_resampler(method: ResamplingMethod | None) -> BaseSampler | None:
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
            case None:
                resampler = None
            case _:
                raise Exception(f"Could not initialize resampler {method.value}")

        return resampler

    @staticmethod
    def get_scaler(method: ScalingMethod | None) -> BaseEstimator | None:
        match method:
            case ScalingMethod.STANDARDSCALER:
                scaler = StandardScaler()
            case ScalingMethod.MINMAXSCALER:
                scaler = MinMaxScaler()
            case None:
                scaler = None
            case _:
                raise Exception(f"Could not initialize scaler {method.value}")
        return scaler

    @staticmethod
    def train_test_split(
        x: pd.DataFrame, y: pd.DataFrame, shuffle: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        split_ratio = 0.2
        train_x, test_x, train_y, test_y = train_test_split(
            x, y, test_size=split_ratio, shuffle=shuffle, random_state=42, stratify=y
        )
        return train_x, test_x, train_y, test_y
