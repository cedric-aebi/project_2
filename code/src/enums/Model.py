from enum import Enum


class Model(Enum):
    LOGISTIC_REGRESSION = "Logistic Regression"
    XGBOOST = "XGBoost"
    DNN = "DNN"

    def __str__(self):
        return str(self.value)
