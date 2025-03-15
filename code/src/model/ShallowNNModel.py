import os
import warnings

from enums.Dataset import Dataset

warnings.simplefilter("ignore", category=FutureWarning)
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

from imblearn.pipeline import Pipeline
import pandas as pd
import keras
from keras.src.regularizers import L1L2
import tensorflow as tf
from imblearn.base import BaseSampler
from scikeras.wrappers import KerasClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from model.AbstractModel import AbstractModel


class ShallowNNModel(AbstractModel):
    def __init__(
        self, scaler: BaseEstimator | None, resampler: BaseSampler | None, input_shape: int, dataset_name: Dataset
    ):
        tf.random.set_seed(42)
        keras.utils.set_random_seed(42)
        self._grid_search_cv = None
        self._dataset_name = dataset_name
        hyperparameter_grid = {
            "clf__model__optimizer": ["adam"],
            "clf__model__learning_rate": [0.001],
            "clf__model__dropout": [None],
            "clf__model__batch_normalization": [False],
            "clf__model__regularization": [False],
            "clf__model__dataset_name": [dataset_name],
        }
        early_stopping_callback = keras.callbacks.EarlyStopping(patience=10, monitor="val_loss", min_delta=0.001)
        reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(patience=7, monitor="val_loss", factor=0.2)
        clf = KerasClassifier(
            model=self._build_model,
            model__input_shape=input_shape,
            epochs=150,
            batch_size=32,
            verbose=2,
            random_state=42,
            validation_split=0.2,
            callbacks=[early_stopping_callback, reduce_lr_callback],
        )
        super().__init__(clf=clf, hyperparameter_grid=hyperparameter_grid, scaler=scaler, resampler=resampler)

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame, run_info: dict) -> None:
        self._grid_search_cv = GridSearchCV(
            estimator=self._pipeline, param_grid=self._hyperparameter_grid, cv=self._cv, n_jobs=-1, verbose=2
        )
        self._grid_search_cv.fit(x_train, y_train)
        self._best_estimator = self._grid_search_cv.best_estimator_
        run_info["best_params"] = self._grid_search_cv.best_params_

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return self._best_estimator.predict(x)

    @staticmethod
    def _build_model(
        input_shape: int,
        optimizer: str,
        learning_rate: float,
        dropout: float | None,
        batch_normalization: bool,
        regularization: bool,
        dataset_name: Dataset,
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

        if dataset_name == Dataset.NURSE:
            # output layer for multi-class classification (3 classes)
            model.add(keras.layers.Dense(3, activation="softmax"))
        else:
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

        if dataset_name == Dataset.NURSE:
            model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        else:
            model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        return model

    def get_fitted_model(self) -> Pipeline:
        return self._best_estimator
