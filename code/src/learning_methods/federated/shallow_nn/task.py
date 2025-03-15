import os
import warnings
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler
from keras.src.regularizers import L1L2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from service.datasetservice.DatasetService import DatasetService

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.simplefilter(action="ignore", category=FutureWarning)


def load_model(
    input_shape: int,
    learning_rate: float,
    dropout: float | None,
    batch_normalization: bool,
    regularization: bool,
    optimizer: str,
) -> keras.Sequential:
    # Define the model
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_shape,)))

    model.add(keras.layers.Dense(1024, kernel_regularizer=L1L2() if regularization else None))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dense(512, kernel_regularizer=L1L2() if regularization else None))
    model.add(keras.layers.LeakyReLU())
    if dropout is not None:
        model.add(keras.layers.Dropout(dropout))
    if batch_normalization:
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(256, kernel_regularizer=L1L2() if regularization else None))
    model.add(keras.layers.LeakyReLU())
    if dropout is not None:
        model.add(keras.layers.Dropout(dropout))
    if batch_normalization:
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(128, kernel_regularizer=L1L2() if regularization else None))
    model.add(keras.layers.LeakyReLU())
    if dropout is not None:
        model.add(keras.layers.Dropout(dropout))
    if batch_normalization:
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(64, kernel_regularizer=L1L2() if regularization else None))
    model.add(keras.layers.LeakyReLU())
    if dropout is not None:
        model.add(keras.layers.Dropout(dropout))
    if batch_normalization:
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(50, kernel_regularizer=L1L2() if regularization else None))
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Dense(3, activation="softmax"))

    if optimizer == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "nadam":
        optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer}")

    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model


dataset = pd.read_csv(Path(__file__).parent.parent.parent.parent / "nurse" / "merged_data.csv", low_memory=False)
dataset = dataset.drop(columns=["datetime"])


def load_data(subject: int | str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

    resampler = RandomOverSampler()
    x_train, y_train = resampler.fit_resample(x_train, y_train)

    return x_train, x_test, y_train, y_test
