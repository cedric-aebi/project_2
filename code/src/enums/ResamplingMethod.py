from enum import StrEnum


class ResamplingMethod(StrEnum):
    SMOTE = "SMOTE"
    UNDERSAMPLING = "UNDERSAMPLING"
    OVERSAMPLING = "OVERSAMPLING"
