from typing import Callable

import keras
from flwr.client import NumPyClient
from flwr.common import Context

from enums.Participant import NurseParticipant
from enums.ResamplingMethod import ResamplingMethod
from enums.ScalingMethod import ScalingMethod
from learning_methods.federated.shallow_nn.task import load_model, load_data, evaluate
from service.exportservice.ExportService import ExportService


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
        participant,
        mongo_id: str,
        run_index: int,
        database: str,
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
        self.participant = participant
        self.mongo_id = mongo_id
        self.run_index = run_index
        self.export_service = ExportService(collection="federated", database=database)

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        self.model.set_weights(parameters)

        self.model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

        # print confusion matrix
        y_pred = self.model.predict(self.x_train)
        y_pred = (y_pred > 0.5).astype(int)
        scores, _ = evaluate(pred=y_pred, y_true=self.y_train)
        print(scores["confusion_matrix"])

        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        global_round = int(config["global_round"])
        self.model.set_weights(parameters)
        training_loss, _ = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        test_loss, _ = self.model.evaluate(self.x_test, self.y_test, verbose=0)

        y_pred_train = self.model.predict(self.x_train)
        y_pred_train = (y_pred_train > 0.5).astype(int)
        scores_train, _ = evaluate(pred=y_pred_train, y_true=self.y_train)
        scores_train["loss"] = training_loss

        y_pred_test = self.model.predict(self.x_test)
        y_pred_test = (y_pred_test > 0.5).astype(int)
        scores_test, _ = evaluate(pred=y_pred_test, y_true=self.y_test)
        scores_test["loss"] = test_loss

        scores = {"training_set": scores_train, "testing_set": scores_test}

        self.export_service.update_run(
            run_id=self.mongo_id,
            set_dict={
                "$set": {
                    f"training_runs.{self.run_index}.clients.{self.participant}.round.{str(global_round)}.scores": scores
                }
            },
        )

        return test_loss, len(self.x_test), {"f1": scores_test["f1"], "server_round": global_round}


def get_client_fn(
    cfg: dict,
    mongo_id: str,
    run_index: int,
    scaling_method: ScalingMethod | None,
    resampling_method: ResamplingMethod | None,
    database: str,
    participant_leave_out: NurseParticipant,
) -> Callable:
    def client_fn(context: Context):
        """Construct a Client that will be run in a ClientApp."""

        # Ensure a new session is started
        keras.backend.clear_session()

        # Read the node_config to fetch data partition associated to this node
        partition_id = context.node_config["partition-id"]
        x_train, x_test, y_train, y_test, participant = load_data(
            which=partition_id,
            scaling_method=scaling_method,
            resampling_method=resampling_method,
            participant_leave_out=participant_leave_out,
        )

        # Read run_config to fetch hyperparameters relevant to this run
        params = cfg["params"]
        epochs = cfg["local_epochs"]
        regularization = params["regularization"]
        batch_size = params["batch_size"]
        verbose = params["verbose"]
        learning_rate = params["learning_rate"]
        optimizer = params["optimizer"]
        batch_normalization = params["batch_normalization"]
        dropout = None if params["dropout"] == False else params["dropout"]

        # Return Client instance
        return FlowerClient(
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            regularization=regularization,
            batch_normalization=batch_normalization,
            dropout=dropout,
            verbose=verbose,
            data=(x_train, x_test, y_train, y_test),
            participant=participant,
            mongo_id=mongo_id,
            run_index=run_index,
            database=database,
        ).to_client()

    return client_fn
