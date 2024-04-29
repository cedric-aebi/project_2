import numpy as np
import keras
from keras import layers

from model.AbstractModel import AbstractModel


class DNNModel(AbstractModel):
    def __init__(self):
        model = self._build_model()
        super().__init__(model=model)

    def fit(self, train_x: np.ndarray, train_y: np.ndarray, grid_search: bool, run_info: dict) -> None:
        self._model = self._model.fit(train_x, train_y, epochs=50, batch_size=32, validation_split=0.2)
        pred = self._model.predict(train_x)
        scores = self.get_scores(pred=pred, y=train_y)
        tp, tn, fp, fn = self.get_classification_results(cm=scores[4])
        run_info["training"] = {
            "fitted_model": {
                "params": self._model.get_params(),
                "scores": {
                    "accuracy": scores[0],
                    "recall": scores[1],
                    "precision": scores[2],
                    "f1": scores[3],
                    "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
                },
            },
        }

    def _build_model(self) -> keras.Sequential:
        # Define the model
        model = keras.Sequential()
        model.add(layers.Dense(4, input_dim=2, activation="relu"))
        model.add(layers.Dense(4, activation="relu"))
        # Output layer with 1 neuron, sigmoid activation for binary classification
        model.add(layers.Dense(1, activation="sigmoid"))

        # Compile the model
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model
