import os
import warnings

import keras
import pandas as pd
from imblearn.combine import SMOTEENN
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
    model.add(keras.layers.Dense(512, activation="relu", kernel_regularizer=L1L2() if regularization else None))
    if dropout is not None:
        model.add(keras.layers.Dropout(dropout))
    if batch_normalization:
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(256, activation="relu", kernel_regularizer=L1L2() if regularization else None))
    if dropout is not None:
        model.add(keras.layers.Dropout(dropout))
    if batch_normalization:
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(128, activation="relu", kernel_regularizer=L1L2() if regularization else None))
    if dropout is not None:
        model.add(keras.layers.Dropout(dropout))
    if batch_normalization:
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(64, activation="relu", kernel_regularizer=L1L2() if regularization else None))
    if dropout is not None:
        model.add(keras.layers.Dropout(dropout))
    if batch_normalization:
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(32, activation="relu", kernel_regularizer=L1L2() if regularization else None))
    if dropout is not None:
        model.add(keras.layers.Dropout(dropout))
    if batch_normalization:
        model.add(keras.layers.BatchNormalization())

    # Output layer with 1 neuron, sigmoid activation for binary classification
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    if optimizer == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer}")

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model


dataset_service = DatasetService()


def load_data(subject: int | str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x, y = dataset_service.get_subject_data(subject="all", with_features=True)
    x_train, x_test, y_train, y_test = dataset_service.train_test_split(x=x, y=y)

    # Scale the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Resample the data
    resampler = SMOTEENN(random_state=42)
    x_train, y_train = resampler.fit_resample(X=x_train, y=y_train)

    if subject == "all":
        return x_train, x_test, y_train, y_test

    # Filter the data for the specific subject range
    if isinstance(subject, int) and 0 <= subject <= 35:
        # Assuming the dataset can be sliced by ranges for subjects
        subject_range_start = subject * len(x_train) // 36
        subject_range_end = (subject + 1) * len(x_train) // 36

        x_train = x_train[subject_range_start:subject_range_end]
        y_train = y_train[subject_range_start:subject_range_end]

        subject_range_start = subject * len(x_test) // 36
        subject_range_end = (subject + 1) * len(x_test) // 36

        x_test = x_test[subject_range_start:subject_range_end]
        y_test = y_test[subject_range_start:subject_range_end]

        return x_train, x_test, y_train, y_test

    raise ValueError("Invalid subject value. It must be 'all' or an integer between 0 and 35.")
