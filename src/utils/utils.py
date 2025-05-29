from pathlib import Path

import pandas as pd
from imblearn.base import BaseSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from enums.Dataset import Dataset
from enums.Model import Model
from enums.Participant import NurseParticipant, StressParticipant
from enums.ResamplingMethod import ResamplingMethod
from enums.ScalingMethod import ScalingMethod


def get_resampler(method: ResamplingMethod | None) -> BaseSampler | None:
    match method:
        case ResamplingMethod.SMOTE:
            resampler = SMOTE(random_state=42)
        case ResamplingMethod.OVERSAMPLING:
            resampler = RandomOverSampler(random_state=42)
        case ResamplingMethod.UNDERSAMPLING:
            resampler = RandomUnderSampler(random_state=42)
        case None:
            resampler = None
        case _:
            raise Exception(f"Could not initialize resampler {method.value}")

    return resampler


def get_scaler(method: ScalingMethod | None) -> StandardScaler | MinMaxScaler | None:
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


def split_data(
    df: pd.DataFrame, model: Model | None, with_features: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if with_features:
        if model == Model.SHALLOW_NN:
            train_data = df[df["Split"] == "train"]
            val_data = df[df["Split"] == "val"]
            test_data = df[df["Split"] == "test"]

            # Shuffle the data
            train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
            val_data = val_data.sample(frac=1, random_state=42).reset_index(drop=True)
            test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)

            x_train = train_data.drop(columns=["Label", "Participant", "Split"])
            y_train = train_data["Label"]
            x_val = val_data.drop(columns=["Label", "Participant", "Split"])
            y_val = val_data["Label"]
            x_test = test_data.drop(columns=["Label", "Participant", "Split"])
            y_test = test_data["Label"]
        else:
            train_data = df[(df["Split"] == "train") | (df["Split"] == "val")]
            test_data = df[df["Split"] == "test"]

            # Shuffle the data
            train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
            test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)

            x_train = train_data.drop(columns=["Label", "Participant", "Split"])
            y_train = train_data["Label"]
            x_test = test_data.drop(columns=["Label", "Participant", "Split"])
            y_test = test_data["Label"]
            x_val, y_val = None, None
    else:
        x = df.drop(columns=["Label", "Participant"])
        y = df["Label"]

        x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, shuffle=True, random_state=42, stratify=y)

        if model == Model.SHALLOW_NN:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train_val,
                y_train_val,
                shuffle=True,
                random_state=42,
                stratify=y_train_val,
                test_size=0.2,
            )
        else:
            x_train, y_train = x_train_val, y_train_val
            x_val, y_val = None, None

    return x_train, x_val, x_test, y_train, y_val, y_test


def get_list_of_participants(dataset: Dataset) -> list[str] | list[int]:
    if dataset == Dataset.NURSE:
        participants = [str(e.value) for e in NurseParticipant]
    elif dataset == Dataset.STRESS:
        participants = [e.value for e in StressParticipant]
    else:
        raise ValueError(f"Dataset {dataset} not recognized")

    return participants


def get_list_of_lave_out_participants(dataset: Dataset) -> list[NurseParticipant] | list[StressParticipant]:
    if dataset == Dataset.NURSE:
        participants = [
            NurseParticipant.n_F5,
            NurseParticipant.n_E4,
            NurseParticipant.n_DF,
        ]
    elif dataset == Dataset.STRESS:
        participants = [
            StressParticipant.s_33,
            StressParticipant.s_34,
            StressParticipant.s_35,
        ]
    else:
        raise ValueError(f"Dataset {dataset} not recognized")

    return participants
