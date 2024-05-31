import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetService:
    def __init__(self):
        self.__path_to_datasets = Path(__file__).parent.parent.parent.parent / "dataset"
        self.__label_column = "Label"
        self.__labels = ["No Stress", "Stress"]
        self.__not_needed_columns = [self.__label_column, "Time(sec)", "Participant"]

    def load_training_features(self, which: str | int) -> pd.DataFrame:
        if which == "all":
            file_to_read = open(self.__path_to_datasets / "features" / "all_training_features.pkl", "rb")
            x_train = pickle.load(file_to_read)
            file_to_read.close()
        else:
            file_to_read = open(self.__path_to_datasets / "features" / f"training_features_{which}.pkl", "rb")
            x_train = pickle.load(file_to_read)
            file_to_read.close()
        return x_train

    def load_testing_features(self, which: str | int) -> pd.DataFrame:
        if which == "all":
            file_to_read = open(self.__path_to_datasets / "features" / "all_testing_features.pkl", "rb")
            x_train = pickle.load(file_to_read)
            file_to_read.close()
        else:
            file_to_read = open(self.__path_to_datasets / "features" / f"testing_features_{which}.pkl", "rb")
            x_train = pickle.load(file_to_read)
            file_to_read.close()
        return x_train

    def load_training_labels(self, which: str | int) -> pd.DataFrame:
        if which == "all":
            file_to_read = open(self.__path_to_datasets / "features" / "all_training_labels.pkl", "rb")
            x_train = pickle.load(file_to_read)
            file_to_read.close()
        else:
            file_to_read = open(self.__path_to_datasets / "features" / f"training_labels_{which}.pkl", "rb")
            x_train = pickle.load(file_to_read)
            file_to_read.close()
        return x_train

    def load_testing_labels(self, which: str | int) -> pd.DataFrame:
        if which == "all":
            file_to_read = open(self.__path_to_datasets / "features" / "all_testing_labels.pkl", "rb")
            x_train = pickle.load(file_to_read)
            file_to_read.close()
        else:
            file_to_read = open(self.__path_to_datasets / "features" / f"testing_labels_{which}.pkl", "rb")
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
    def train_test_split(
        x: pd.DataFrame, y: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
        return train_x, test_x, train_y, test_y
