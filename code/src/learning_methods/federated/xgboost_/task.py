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

# dataset = pd.read_csv(Path(__file__).parent.parent.parent.parent / "nurse" / "merged_data.csv", low_memory=False)
# dataset = dataset.drop(columns=["datetime"])
# # Drop participants "CE" and "EG" due to lack of data
# dataset = dataset[~dataset["id"].isin(["CE", "EG"])]
# dataset.loc[dataset["label"] == 2, "label"] = 1

dataset = pd.read_csv(Path(__file__).parent.parent.parent.parent / "nurse" / "nurse_features.csv", engine="pyarrow")
# Drop participants "CE" and "EG" due to lack of data
dataset = dataset[~dataset["Participant"].isin(["CE", "EG"])]
dataset.loc[dataset["Label"] == 2, "Label"] = 1


def load_data(subject: int | str) -> tuple[xgb.DMatrix, xgb.DMatrix, int, int]:
    global dataset
    # Train/test splitting
    match subject:
        case "all":
            x = dataset.drop(columns=["Participant", "Label"])
            y = dataset["Label"]
        case 0:
            x = dataset[dataset["Participant"] == "15"].drop(columns=["Participant", "Label"])
            y = dataset[dataset["Participant"] == "15"]["Label"]
        case 1:
            x = dataset[dataset["Participant"] == "5C"].drop(columns=["Participant", "Label"])
            y = dataset[dataset["Participant"] == "5C"]["Label"]
        case 2:
            x = dataset[dataset["Participant"] == "6B"].drop(columns=["Participant", "Label"])
            y = dataset[dataset["Participant"] == "6B"]["Label"]
        case 3:
            x = dataset[dataset["Participant"] == "6D"].drop(columns=["Participant", "Label"])
            y = dataset[dataset["Participant"] == "6D"]["Label"]
        case 4:
            x = dataset[dataset["Participant"] == "7A"].drop(columns=["Participant", "Label"])
            y = dataset[dataset["Participant"] == "7A"]["Label"]
        case 5:
            x = dataset[dataset["Participant"] == "7E"].drop(columns=["Participant", "Label"])
            y = dataset[dataset["Participant"] == "7E"]["Label"]
        case 6:
            x = dataset[dataset["Participant"] == "8B"].drop(columns=["Participant", "Label"])
            y = dataset[dataset["Participant"] == "8B"]["Label"]
        case 7:
            x = dataset[dataset["Participant"] == "83"].drop(columns=["Participant", "Label"])
            y = dataset[dataset["Participant"] == "83"]["Label"]
        case 8:
            x = dataset[dataset["Participant"] == "94"].drop(columns=["Participant", "Label"])
            y = dataset[dataset["Participant"] == "94"]["Label"]
        case 9:
            x = dataset[dataset["Participant"] == "BG"].drop(columns=["Participant", "Label"])
            y = dataset[dataset["Participant"] == "BG"]["Label"]
        case 10:
            x = dataset[dataset["Participant"] == "DF"].drop(columns=["Participant", "Label"])
            y = dataset[dataset["Participant"] == "DF"]["Label"]
        case 11:
            x = dataset[dataset["Participant"] == "E4"].drop(columns=["Participant", "Label"])
            y = dataset[dataset["Participant"] == "E4"]["Label"]
        case 12:
            x = dataset[dataset["Participant"] == "F5"].drop(columns=["Participant", "Label"])
            y = dataset[dataset["Participant"] == "F5"]["Label"]
        case _:
            raise ValueError("Invalid subject number")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

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
