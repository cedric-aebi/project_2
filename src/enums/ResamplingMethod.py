from enum import Enum


class ResamplingMethod(Enum):
    SMOTE = "SMOTE"
    UNDERSAMPLING = "UNDERSAMPLING"
    OVERSAMPLING = "OVERSAMPLING"

    def __str__(self):
        return str(self.value)
