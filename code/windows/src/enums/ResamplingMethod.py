from enum import StrEnum


class ResamplingMethod(StrEnum):
    SMOTE = "SMOTE"
    TL = "TOMELINKS"
    UNDERSAMPLING = "UNDERSAMPLING"
    OVERSAMPLING = "OVERSAMPLING"
    SMOTEENN = "SMOTEENN"
