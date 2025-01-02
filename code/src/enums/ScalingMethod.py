from enum import Enum


class ScalingMethod(Enum):
    STANDARDSCALER = "StandardScaler"
    MINMAXSCALER = "MinMaxScaler"

    def __str__(self):
        return str(self.value)
