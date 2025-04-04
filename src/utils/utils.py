from pathlib import Path

import pandas as pd
from imblearn.base import BaseSampler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import TomekLinks
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from enums.Dataset import Dataset
from enums.Participant import NurseParticipant, StressParticipant
from enums.ResamplingMethod import ResamplingMethod
from enums.ScalingMethod import ScalingMethod


def get_resampler(method: ResamplingMethod | None) -> BaseSampler | None:
    match method:
        case ResamplingMethod.SMOTE:
            resampler = SMOTE(random_state=42)
        case ResamplingMethod.OVERSAMPLING:
            resampler = RandomOverSampler(random_state=42)
        case ResamplingMethod.TL:
            resampler = TomekLinks()
        case ResamplingMethod.SMOTEENN:
            resampler = SMOTEENN(random_state=42)
        case None:
            resampler = None
        case _:
            raise Exception(f"Could not initialize resampler {method.value}")

    return resampler


def get_scaler(method: ScalingMethod | None) -> BaseEstimator | None:
    match method:
        case ScalingMethod.STANDARDSCALER:
            scaler = StandardScaler()
        case ScalingMethod.MINMAXSCALER:
            scaler = MinMaxScaler()
        case None:
            scaler = None
        case _:
            raise Exception(f"Could not initialize scaler {method.value}")
    return scaler


def load_data(dataset: Dataset, with_features: bool, which: str | NurseParticipant | StressParticipant) -> pd.DataFrame:
    PATH_TO_DATASETS = Path(__file__).parent.parent.parent / "datasets"
    if dataset == Dataset.NURSE:
        if with_features:
            df = pd.read_pickle(PATH_TO_DATASETS / "nurse" / "processed" / "with_features" / f"{which}.pkl")
        else:
            df = pd.read_pickle(PATH_TO_DATASETS / "nurse" / "processed" / "no_features" / f"{which}.pkl")
    elif dataset == Dataset.STRESS:
        if with_features:
            df = pd.read_pickle(PATH_TO_DATASETS / "stress" / "processed" / "with_features" / f"{which}.pkl")
        else:
            df = pd.read_pickle(PATH_TO_DATASETS / "stress" / "processed" / "no_features" / f"{which}.pkl")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return df


def get_list_of_participants(dataset: Dataset) -> list[str] | list[int]:
    if dataset == Dataset.NURSE:
        participants = [str(e.value) for e in NurseParticipant]
    elif dataset == Dataset.STRESS:
        participants = [e.value for e in StressParticipant]
    else:
        raise ValueError(f"Dataset {dataset} not recognized")

    return participants
