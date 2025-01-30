from pathlib import Path

import pandas as pd
from imblearn.base import BaseSampler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from enums.ResamplingMethod import ResamplingMethod
from enums.ScalingMethod import ScalingMethod


class DatasetService:
    def __init__(self):
        self.__path_to_datasets = Path(__file__).parent.parent.parent.parent / "dataset"
        self.__label_column = "Label"
        self.__dataset_with_features = pd.read_csv(self.__path_to_datasets / "dataset_with_features.csv", sep=",")
        self.__dataset_without_features = pd.read_csv(
            self.__path_to_datasets / "Improved_All_Combined_hr_rsp_binary.csv", sep=","
        )

    def get_subject_data(self, subject: str | int, with_features: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        if with_features:
            df = self.__dataset_with_features
            if subject != "all":
                df = df[df["Subject"] == subject]
            df = df.fillna(0)
            x = df.drop(columns=[self.__label_column, "Subject"])
            y = df[self.__label_column]
            return x, y
        else:
            df = self.__dataset_without_features
            if subject != "all":
                df = df[df["Participant"] == subject]
            df = df.ffill().bfill()
            x = df.drop(columns=[self.__label_column, "Participant", "Time(sec)"])
            y = df[self.__label_column]
            return x, y

    def load_original_dataset(self) -> pd.DataFrame:
        return pd.read_csv(self.__path_to_datasets / "Improved_All_Combined_hr_rsp_binary.csv", sep=",")

    @staticmethod
    def get_resampler(method: ResamplingMethod | None) -> BaseSampler | None:
        match method:
            case ResamplingMethod.SMOTE:
                resampler = SMOTE(random_state=42)
            case ResamplingMethod.OVERSAMPLING:
                resampler = RandomOverSampler(random_state=42)
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
        x: pd.DataFrame, y: pd.DataFrame, shuffle: bool = True, random_state: int | None = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        split_ratio = 0.2
        train_x, test_x, train_y, test_y = train_test_split(
            x, y, test_size=split_ratio, shuffle=shuffle, random_state=random_state, stratify=y
        )
        return train_x, test_x, train_y, test_y
