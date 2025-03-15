import keras
import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from sklearn.metrics import f1_score

from task import load_data, load_model


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(
        self,
        learning_rate,
        data,
        epochs,
        batch_size,
        optimizer,
        regularization,
        batch_normalization,
        dropout,
        verbose,
    ):
        self.x_train, self.x_test, self.y_train, self.y_test = data
        self.model = load_model(
            learning_rate=learning_rate,
            input_shape=self.x_train.shape[1],
            dropout=dropout,
            batch_normalization=batch_normalization,
            regularization=regularization,
            optimizer=optimizer,
        )
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        early_stopping_callback = keras.callbacks.EarlyStopping(patience=10, monitor="val_loss", min_delta=0.001)
        reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(patience=7, monitor="val_loss", factor=0.2)
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_split=0.2,
            callbacks=[early_stopping_callback, reduce_lr_callback],
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        y_pred = np.argmax(self.model.predict(self.x_test), axis=-1)
        f1 = f1_score(self.y_test, y_pred, average="weighted")

        return loss, len(self.x_test), {"f1": f1}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Ensure a new session is started
    keras.backend.clear_session()

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    data = load_data(partition_id)

    # Read run_config to fetch hyperparameters relevant to this run
    epochs = context.run_config["local-epochs"]
    regularization = context.run_config["regularization"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")
    learning_rate = context.run_config["learning-rate"]
    optimizer = context.run_config["optimizer"]
    batch_normalization = context.run_config["batch-normalization"]
    dropout = None if context.run_config["dropout"] == False else context.run_config["dropout"]

    # Return Client instance
    return FlowerClient(
        learning_rate, data, epochs, batch_size, optimizer, regularization, batch_normalization, dropout, verbose
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
