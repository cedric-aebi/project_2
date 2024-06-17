import sys
from pathlib import Path

from enums.Model import Model
from learning_methods.federated.xgboost_.client import Client as XGBoostClient
from learning_methods.federated.xgboost_.server import Server as XGBoostServer
from learning_methods.federated.logistic_regression.client import Client as LogisticRegressionClient
from learning_methods.federated.logistic_regression.server import Server as LogisticRegressionServer
from learning_methods.federated.dnn.client import Client as DNNClient
from learning_methods.federated.dnn.server import Server as DNNServer

if __name__ == "__main__":
    MODEL = Model.LOGISTIC_REGRESSION
    NUMBER_OF_ROUNDS = 30

    match MODEL:
        case Model.LOGISTIC_REGRESSION:
            BASE_PATH = Path(__file__).parent.parent.parent.parent / "results" / "federated" / "logistic_regression"
        case Model.XGBOOST:
            BASE_PATH = Path(__file__).parent.parent.parent.parent / "results" / "federated" / "xgboost"
        case Model.DNN:
            BASE_PATH = Path(__file__).parent.parent.parent.parent / "results" / "federated" / "dnn"
        case _:
            raise Exception("Invalid model")

    if sys.argv[1] == "server":
        match MODEL:
            case Model.LOGISTIC_REGRESSION:
                server = LogisticRegressionServer(number_of_rounds=NUMBER_OF_ROUNDS, base_path=BASE_PATH)
            case Model.XGBOOST:
                server = XGBoostServer(number_of_rounds=NUMBER_OF_ROUNDS, base_path=BASE_PATH)
            case Model.DNN:
                server = DNNServer(number_of_rounds=NUMBER_OF_ROUNDS, base_path=BASE_PATH)
            case _:
                raise Exception("Invalid model")
        server.start()
    elif sys.argv[1] == "client":
        match MODEL:
            case Model.LOGISTIC_REGRESSION:
                client = LogisticRegressionClient(
                    subject_nr=int(sys.argv[2]), number_of_rounds=NUMBER_OF_ROUNDS, base_path=BASE_PATH
                )
            case Model.XGBOOST:
                client = XGBoostClient(
                    subject_nr=int(sys.argv[2]), number_of_rounds=NUMBER_OF_ROUNDS, base_path=BASE_PATH
                )
            case Model.DNN:
                client = DNNClient(subject_nr=int(sys.argv[2]), number_of_rounds=NUMBER_OF_ROUNDS, base_path=BASE_PATH)
            case _:
                raise Exception("Invalid model")
        client.start()
    else:
        raise ValueError("Invalid argument")
