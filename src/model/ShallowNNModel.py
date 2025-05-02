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

from model.AbstractModel import AbstractModel


class ShallowNNModel(AbstractModel):
    def __init__(
        self,
        scaler: BaseEstimator | None,
        resampler: BaseSampler | None,
        input_shape: int,
        dataset: Dataset,
        with_features: bool,
        centralized=False,
    ) -> None:
        tf.random.set_seed(42)
        keras.utils.set_random_seed(42)
        early_stopping_callback = keras.callbacks.EarlyStopping(
            patience=5, monitor="loss" if centralized else "val_loss", min_delta=0.001
        )
        clf = KerasClassifier(
            model=self._build_model,
            model__input_shape=input_shape,
            model__dropout=None,
            model__batch_normalization=False,
            model__regularization=False,
            epochs=150 if dataset == Dataset.STRESS else 50,
            batch_size=32 if dataset == Dataset.STRESS else 128,
            verbose=2,
            random_state=42,
            validation_split=0.0 if centralized else 0.2,
            callbacks=[early_stopping_callback],
        )
        super().__init__(
            clf=clf,
            scaler=scaler,
            resampler=resampler,
            dataset=dataset,
            with_features=with_features,
        )

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame, run_info: dict) -> None:
        self._pipeline.fit(x_train, y_train)

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return self._pipeline.predict(x)

    @staticmethod
    def _build_model(
        input_shape: int,
        dropout: float | None,
        batch_normalization: bool,
        regularization: bool,
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

        optimizer = keras.optimizers.Adam(learning_rate=0.001)

        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        return model

    def get_fitted_model(self) -> Pipeline:
        return self._pipeline
