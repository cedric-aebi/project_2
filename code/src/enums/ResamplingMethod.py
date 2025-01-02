from enum import Enum


class ResamplingMethod(Enum):
    SMOTE = "SMOTE"
    TL = "TOMELINKS"
    UNDERSAMPLING = "UNDERSAMPLING"
    OVERSAMPLING = "OVERSAMPLING"
    SMOTEENN = "SMOTEENN"

    def __str__(self):
        return str(self.value)
