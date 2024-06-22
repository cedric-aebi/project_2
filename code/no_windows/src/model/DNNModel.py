import uuid
from pathlib import Path

import numpy as np
import keras
from imblearn.base import BaseSampler
from imblearn.pipeline import Pipeline
from keras import layers
import tensorflow as tf
from keras.src.optimizers import SGD
from scikeras.wrappers import KerasClassifier
from sklearn.base import BaseEstimator

from model.AbstractModel import AbstractModel


class DNNModel(AbstractModel):
    def __init__(self, scaler: BaseEstimator, resampler: BaseSampler, number_of_features: int, run_info: dict):
        tf.random.set_seed(42)
        keras.utils.set_random_seed(42)
        build_model = self._build_model(number_of_features=number_of_features)
        early_stopping_callback = keras.callbacks.EarlyStopping(patience=5)
        self._unique_run_id = str(uuid.uuid4())
        log_dir = Path(__file__).parent.parent.parent / "logs" / self._unique_run_id
        run_info["log_dir"] = str(log_dir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        model = KerasClassifier(
            model=build_model,
            epochs=500,
            batch_size=32,
            verbose=1,
            validation_split=0.2,
            random_state=42,
            shuffle=True,
            callbacks=[early_stopping_callback, tensorboard_callback],
            loss="binary_crossentropy",
            optimizer=SGD(learning_rate=0.001),
            metrics=["accuracy"],
        )
        super().__init__(model=model, scaler=scaler, resampler=resampler)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self._pipeline.fit(x_train, y_train)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._pipeline.predict(x)

    @staticmethod
    def _build_model(number_of_features: int) -> keras.Sequential:
        # Define the model
        # For the non-window approach use a simpler model
        model = keras.Sequential()
        model.add(layers.Dense(16, input_dim=number_of_features, activation="relu"))
        model.add(layers.Dense(8, activation="relu"))
        model.add(layers.Dense(4, activation="relu"))
        # Output layer with 1 neuron, sigmoid activation for binary classification
        model.add(layers.Dense(1, activation="sigmoid"))

        return model

    def get_fitted_model(self) -> Pipeline:
        return self._pipeline
