from enum import Enum


class Model(Enum):
    LOGISTIC_REGRESSION = "Logistic Regression"
    XGBOOST = "XGBoost"
    SHALLOW_NN = "Shallow NN"

    def __str__(self):
        return str(self.value)
