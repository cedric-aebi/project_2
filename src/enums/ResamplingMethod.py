from enum import Enum


class ResamplingMethod(Enum):
    SMOTE = "SMOTE"
    TL = "TOMELINKS"
    OVERSAMPLING = "OVERSAMPLING"

    def __str__(self):
        return str(self.value)
