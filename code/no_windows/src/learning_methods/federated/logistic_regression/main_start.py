import sys
from pathlib import Path

from learning_methods.federated.logistic_regression.client import Client
from learning_methods.federated.logistic_regression.server import Server

if __name__ == "__main__":
    BASE_PATH = Path(__file__).parent.parent.parent.parent.parent / "results" / "federated" / "logistic_regression"
    NUMBER_OF_ROUNDS = 30

    if sys.argv[1] == "server":
        server = Server(number_of_rounds=NUMBER_OF_ROUNDS, base_path=BASE_PATH)
        server.start()
    elif sys.argv[1] == "client":
        client = Client(subject_nr=int(sys.argv[2]), number_of_rounds=NUMBER_OF_ROUNDS, base_path=BASE_PATH)
        client.start()
    else:
        raise ValueError("Invalid argument")
