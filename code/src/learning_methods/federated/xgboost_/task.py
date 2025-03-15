import warnings
from logging import INFO
from pathlib import Path

import pandas as pd
from flwr.common import log
import xgboost as xgb
from sklearn.model_selection import train_test_split

from service.datasetservice.DatasetService import DatasetService

dataset_service = DatasetService()

warnings.simplefilter(action="ignore", category=FutureWarning)

dataset = pd.read_csv(Path(__file__).parent.parent.parent.parent / "nurse" / "merged_data.csv", low_memory=False)
dataset = dataset.drop(columns=["datetime"])


def load_data(subject: int | str) -> tuple[xgb.DMatrix, xgb.DMatrix, int, int]:
    global dataset
    # Train/test splitting
    match subject:
        case "all":
            x = dataset.drop(columns=["id", "label"])
            y = dataset["label"]
        case 0:
            x = dataset[dataset["id"] == "15"].drop(columns=["id", "label"])
            y = dataset[dataset["id"] == "15"]["label"]
        case 1:
            x = dataset[dataset["id"] == "5C"].drop(columns=["id", "label"])
            y = dataset[dataset["id"] == "5C"]["label"]
        case 2:
            x = dataset[dataset["id"] == "6B"].drop(columns=["id", "label"])
            y = dataset[dataset["id"] == "6B"]["label"]
        case 3:
            x = dataset[dataset["id"] == "6D"].drop(columns=["id", "label"])
            y = dataset[dataset["id"] == "6D"]["label"]
        case 4:
            x = dataset[dataset["id"] == "7A"].drop(columns=["id", "label"])
            y = dataset[dataset["id"] == "7A"]["label"]
        case 5:
            x = dataset[dataset["id"] == "7E"].drop(columns=["id", "label"])
            y = dataset[dataset["id"] == "7E"]["label"]
        case 6:
            x = dataset[dataset["id"] == "8B"].drop(columns=["id", "label"])
            y = dataset[dataset["id"] == "8B"]["label"]
        case 7:
            x = dataset[dataset["id"] == "83"].drop(columns=["id", "label"])
            y = dataset[dataset["id"] == "83"]["label"]
        case 8:
            x = dataset[dataset["id"] == "94"].drop(columns=["id", "label"])
            y = dataset[dataset["id"] == "94"]["label"]
        case 9:
            x = dataset[dataset["id"] == "BG"].drop(columns=["id", "label"])
            y = dataset[dataset["id"] == "BG"]["label"]
        case 10:
            x = dataset[dataset["id"] == "CE"].drop(columns=["id", "label"])
            y = dataset[dataset["id"] == "CE"]["label"]
        case 11:
            x = dataset[dataset["id"] == "DF"].drop(columns=["id", "label"])
            y = dataset[dataset["id"] == "DF"]["label"]
        case 12:
            x = dataset[dataset["id"] == "E4"].drop(columns=["id", "label"])
            y = dataset[dataset["id"] == "E4"]["label"]
        case 13:
            x = dataset[dataset["id"] == "EG"].drop(columns=["id", "label"])
            y = dataset[dataset["id"] == "EG"]["label"]
        case 14:
            x = dataset[dataset["id"] == "F5"].drop(columns=["id", "label"])
            y = dataset[dataset["id"] == "F5"]["label"]
        case _:
            raise ValueError("Invalid subject number")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

    num_train = len(x_train)
    num_test = len(x_test)

    # Reformat data to DMatrix for xgboost
    log(INFO, "Reformatting data...")
    train_dmatrix = transform_dataset_to_dmatrix(x_train, y_train)
    test_dmatrix = transform_dataset_to_dmatrix(x_test, y_test)

    return train_dmatrix, test_dmatrix, num_train, num_test


def transform_dataset_to_dmatrix(x: pd.DataFrame, y: pd.DataFrame) -> xgb.DMatrix:
    """Transform dataset to DMatrix format for xgboost."""
    new_data = xgb.DMatrix(x, label=y)
    return new_data


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
