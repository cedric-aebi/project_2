import warnings
from logging import INFO

import pandas as pd
from flwr.common import log
import xgboost as xgb
from imblearn.combine import SMOTEENN

from service.datasetservice.DatasetService import DatasetService

dataset_service = DatasetService()

warnings.simplefilter(action="ignore", category=FutureWarning)


def load_data(subject: int | str, centralised_eval_client: bool) -> tuple[xgb.DMatrix, xgb.DMatrix, int, int]:
    if centralised_eval_client:
        # Train/test splitting
        x_all, y_all = dataset_service.get_subject_data(subject="all", with_features=True)
        x_train_all, x_test_all, y_train_all, y_test_all = dataset_service.train_test_split(x=x_all, y=y_all)
        x, y = dataset_service.get_subject_data(subject=subject, with_features=True)
        x_train, x_test, y_train, y_test = dataset_service.train_test_split(x=x, y=y)

        x_train = x_train
        x_test = x_test_all
        y_train = y_train
        y_test = y_test_all
    else:
        # Train/test splitting
        x, y = dataset_service.get_subject_data(subject=subject, with_features=True)
        x_train, x_test, y_train, y_test = dataset_service.train_test_split(x=x, y=y)

    # Resample the data
    resampler = SMOTEENN(random_state=42)
    x_train, y_train = resampler.fit_resample(X=x_train, y=y_train)

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
