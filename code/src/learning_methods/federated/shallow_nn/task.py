import os
import warnings

import keras
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler
from keras.src.regularizers import L1L2
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

    # Output layer with 1 neuron, sigmoid activation for binary classification
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    if optimizer == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "nadam":
        optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer}")

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model


dataset_service = DatasetService()


def load_data(subject: int | str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x, y = dataset_service.get_subject_data(subject=subject, with_features=True)
    x_train, x_test, y_train, y_test = dataset_service.train_test_split(x=x, y=y)

    # Scale the data
    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)

    # Resample the data
    # resampler = RandomOverSampler(random_state=42)
    # x_train, y_train = resampler.fit_resample(X=x_train, y=y_train)

    return x_train, x_test, y_train, y_test
