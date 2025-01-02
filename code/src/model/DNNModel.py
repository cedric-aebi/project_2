import warnings

from imblearn.pipeline import Pipeline

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import keras
import tensorflow as tf
from imblearn.base import BaseSampler
from scikeras.wrappers import KerasClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from model.AbstractModel import AbstractModel


class DNNModel(AbstractModel):
    def __init__(self, scaler: BaseEstimator | None, resampler: BaseSampler | None, number_of_features: int):
        tf.random.set_seed(42)
        keras.utils.set_random_seed(42)
        self._grid_search_cv = None
        hyperparameter_grid = {
            "clf__model__optimizer": ["adam"],
            "clf__model__learning_rate": [0.001, 0.01, 0.1],
            "clf__model__dropout": [0.2],
        }
        early_stopping_callback = keras.callbacks.EarlyStopping(patience=5, monitor="loss")
        clf = KerasClassifier(
            model=self._build_model,
            model__number_of_features=number_of_features,
            epochs=50,
            batch_size=32,
            verbose=False,
            random_state=42,
            callbacks=[early_stopping_callback],
        )
        super().__init__(clf=clf, hyperparameter_grid=hyperparameter_grid, scaler=scaler, resampler=resampler)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, run_info: dict) -> None:
        self._grid_search_cv = GridSearchCV(
            estimator=self._pipeline, param_grid=self._hyperparameter_grid, cv=self._cv, n_jobs=-1
        )
        self._grid_search_cv.fit(x_train, y_train)
        self._best_estimator = self._grid_search_cv.best_estimator_
        run_info["best_params"] = self._grid_search_cv.best_params_

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._best_estimator.predict(x)

    @staticmethod
    def _build_model(
        number_of_features: int, optimizer: str, learning_rate: float, dropout: float | None
    ) -> keras.Sequential:
        # Define the model
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(number_of_features,)))
        model.add(keras.layers.Dense(512, activation="relu"))
        if dropout is not None:
            model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(256, activation="relu"))
        if dropout is not None:
            model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(128, activation="relu"))
        if dropout is not None:
            model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(64, activation="relu"))
        if dropout is not None:
            model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(54, activation="relu"))
        if dropout is not None:
            model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(50, activation="relu"))
        if dropout is not None:
            model.add(keras.layers.Dropout(dropout))
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

    def get_fitted_model(self) -> Pipeline:
        return self._best_estimator
