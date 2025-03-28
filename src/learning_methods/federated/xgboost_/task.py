import warnings
from logging import INFO

import pandas as pd
from flwr.common import log
import xgboost as xgb
from sklearn.model_selection import train_test_split

from enums.Dataset import Dataset
from enums.Participant import NurseParticipant
from utils import utils

warnings.simplefilter(action="ignore", category=FutureWarning)

DATASET = Dataset.NURSE


def load_data(which: int | str) -> tuple[xgb.DMatrix, xgb.DMatrix, int, int]:
    match which:
        case "all":
            x, y = utils.load_data(which=which, with_features=True, dataset=DATASET)
        case 0:
            x, y = utils.load_data(which=NurseParticipant.n_15, with_features=True, dataset=DATASET)
        case 1:
            x, y = utils.load_data(which=NurseParticipant.n_5C, with_features=True, dataset=DATASET)
        case 2:
            x, y = utils.load_data(which=NurseParticipant.n_6B, with_features=True, dataset=DATASET)
        case 3:
            x, y = utils.load_data(which=NurseParticipant.n_6D, with_features=True, dataset=DATASET)
        case 4:
            x, y = utils.load_data(which=NurseParticipant.n_7A, with_features=True, dataset=DATASET)
        case 5:
            x, y = utils.load_data(which=NurseParticipant.n_7E, with_features=True, dataset=DATASET)
        case 6:
            x, y = utils.load_data(which=NurseParticipant.n_8B, with_features=True, dataset=DATASET)
        case 7:
            x, y = utils.load_data(which=NurseParticipant.n_83, with_features=True, dataset=DATASET)
        case 8:
            x, y = utils.load_data(which=NurseParticipant.n_94, with_features=True, dataset=DATASET)
        case 9:
            x, y = utils.load_data(which=NurseParticipant.n_BG, with_features=True, dataset=DATASET)
        case 10:
            x, y = utils.load_data(which=NurseParticipant.n_DF, with_features=True, dataset=DATASET)
        case 11:
            x, y = utils.load_data(which=NurseParticipant.n_E4, with_features=True, dataset=DATASET)
        case 12:
            x, y = utils.load_data(which=NurseParticipant.n_F5, with_features=True, dataset=DATASET)
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
