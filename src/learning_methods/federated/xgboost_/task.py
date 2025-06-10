import warnings
from logging import INFO
from pathlib import Path

import numpy as np
import pandas as pd
from flwr.common import log
import xgboost as xgb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from enums.Participant import NurseParticipant, StressParticipant
from enums.ResamplingMethod import ResamplingMethod
from enums.ScalingMethod import ScalingMethod
from utils import utils

warnings.simplefilter(action="ignore", category=FutureWarning)


def load_data_nurse(
    which: int | str | NurseParticipant | StressParticipant,
    with_features: bool,
    scaling_method: ScalingMethod | None,
    resampling_method: ResamplingMethod | None,
    participant_leave_out: NurseParticipant | None = None,
) -> tuple[xgb.DMatrix, xgb.DMatrix, int, int, str | NurseParticipant]:
    if with_features:
        base_path = (
            Path(__file__).parent.parent.parent.parent.parent / "datasets" / "nurse" / "processed" / "with_features"
        )
    else:
        base_path = (
            Path(__file__).parent.parent.parent.parent.parent / "datasets" / "nurse" / "processed" / "no_features"
        )

    # Adjust the participant number based on leave out condition
    if (participant_leave_out == NurseParticipant.n_DF and which == 10) or (
        participant_leave_out == NurseParticipant.n_E4 and which == 11
    ):
        which = 12
    match which:
        case "all":
            df = pd.read_pickle(base_path / "all.pkl")
            participant = "server"
        case 0:
            df = pd.read_pickle(base_path / f"{NurseParticipant.n_15}.pkl")
            participant = NurseParticipant.n_15
        case 1:
            df = pd.read_pickle(base_path / f"{NurseParticipant.n_5C}.pkl")
            participant = NurseParticipant.n_5C
        case 2:
            df = pd.read_pickle(base_path / f"{NurseParticipant.n_6B}.pkl")
            participant = NurseParticipant.n_6B
        case 3:
            df = pd.read_pickle(base_path / f"{NurseParticipant.n_6D}.pkl")
            participant = NurseParticipant.n_6D
        case 4:
            df = pd.read_pickle(base_path / f"{NurseParticipant.n_7A}.pkl")
            participant = NurseParticipant.n_7A
        case 5:
            df = pd.read_pickle(base_path / f"{NurseParticipant.n_7E}.pkl")
            participant = NurseParticipant.n_7E
        case 6:
            df = pd.read_pickle(base_path / f"{NurseParticipant.n_8B}.pkl")
            participant = NurseParticipant.n_8B
        case 7:
            df = pd.read_pickle(base_path / f"{NurseParticipant.n_83}.pkl")
            participant = NurseParticipant.n_83
        case 8:
            df = pd.read_pickle(base_path / f"{NurseParticipant.n_94}.pkl")
            participant = NurseParticipant.n_94
        case 9:
            df = pd.read_pickle(base_path / f"{NurseParticipant.n_BG}.pkl")
            participant = NurseParticipant.n_BG
        case 10:
            df = pd.read_pickle(base_path / f"{NurseParticipant.n_DF}.pkl")
            participant = NurseParticipant.n_DF
        case 11:
            df = pd.read_pickle(base_path / f"{NurseParticipant.n_E4}.pkl")
            participant = NurseParticipant.n_E4
        case 12:
            df = pd.read_pickle(base_path / f"{NurseParticipant.n_F5}.pkl")
            participant = NurseParticipant.n_F5
        case _:
            raise ValueError("Invalid subject number")

    x_train, _, x_test, y_train, _, y_test = utils.split_data(df=df, model=None, with_features=with_features)

    scaler = utils.get_scaler(method=scaling_method)
    if scaler is not None:
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    resampler = utils.get_resampler(method=resampling_method)
    if resampler is not None:
        x_train, y_train = resampler.fit_resample(x_train, y_train)

    num_train = len(x_train)
    num_test = len(x_test)

    # Reformat data to DMatrix for xgboost
    log(INFO, "Reformatting data...")
    train_dmatrix = transform_dataset_to_dmatrix(x_train, y_train)
    test_dmatrix = transform_dataset_to_dmatrix(x_test, y_test)

    return train_dmatrix, test_dmatrix, num_train, num_test, participant


def load_data_stress(
    which: int | str | StressParticipant,
    with_features: bool,
    scaling_method: ScalingMethod | None,
    resampling_method: ResamplingMethod | None,
    participant_leave_out: StressParticipant | None = None,
) -> tuple[xgb.DMatrix, xgb.DMatrix, int, int, str | StressParticipant]:
    if with_features:
        base_path = (
            Path(__file__).parent.parent.parent.parent.parent / "datasets" / "stress" / "processed" / "with_features"
        )
    else:
        base_path = (
            Path(__file__).parent.parent.parent.parent.parent / "datasets" / "stress" / "processed" / "no_features"
        )

    # Adjust the participant number based on leave out condition
    if (participant_leave_out == StressParticipant.s_35 and which == 30) or (
        participant_leave_out == StressParticipant.s_34 and which == 31
    ):
        which = 32
    match which:
        case "all":
            df = pd.read_pickle(base_path / "all.pkl")
            participant = "server"
        case 0:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_2}.pkl")
            participant = StressParticipant.s_2
        case 1:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_3}.pkl")
            participant = StressParticipant.s_3
        case 2:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_4}.pkl")
            participant = StressParticipant.s_4
        case 3:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_5}.pkl")
            participant = StressParticipant.s_5
        case 4:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_6}.pkl")
            participant = StressParticipant.s_6
        case 5:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_7}.pkl")
            participant = StressParticipant.s_7
        case 6:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_8}.pkl")
            participant = StressParticipant.s_8
        case 7:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_9}.pkl")
            participant = StressParticipant.s_9
        case 8:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_10}.pkl")
            participant = StressParticipant.s_10
        case 9:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_11}.pkl")
            participant = StressParticipant.s_11
        case 10:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_12}.pkl")
            participant = StressParticipant.s_12
        case 11:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_14}.pkl")
            participant = StressParticipant.s_14
        case 12:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_15}.pkl")
            participant = StressParticipant.s_15
        case 13:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_16}.pkl")
            participant = StressParticipant.s_16
        case 14:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_17}.pkl")
            participant = StressParticipant.s_17
        case 15:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_18}.pkl")
            participant = StressParticipant.s_18
        case 16:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_19}.pkl")
            participant = StressParticipant.s_19
        case 17:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_20}.pkl")
            participant = StressParticipant.s_20
        case 18:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_21}.pkl")
            participant = StressParticipant.s_21
        case 19:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_22}.pkl")
            participant = StressParticipant.s_22
        case 20:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_23}.pkl")
            participant = StressParticipant.s_23
        case 21:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_24}.pkl")
            participant = StressParticipant.s_24
        case 22:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_25}.pkl")
            participant = StressParticipant.s_25
        case 23:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_26}.pkl")
            participant = StressParticipant.s_26
        case 24:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_27}.pkl")
            participant = StressParticipant.s_27
        case 25:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_28}.pkl")
            participant = StressParticipant.s_28
        case 26:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_29}.pkl")
            participant = StressParticipant.s_29
        case 27:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_30}.pkl")
            participant = StressParticipant.s_30
        case 28:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_31}.pkl")
            participant = StressParticipant.s_31
        case 29:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_32}.pkl")
            participant = StressParticipant.s_32
        case 30:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_33}.pkl")
            participant = StressParticipant.s_33
        case 31:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_34}.pkl")
            participant = StressParticipant.s_34
        case 32:
            df = pd.read_pickle(base_path / f"{StressParticipant.s_35}.pkl")
            participant = StressParticipant.s_35
        case _:
            raise ValueError("Invalid subject number")

    x_train, _, x_test, y_train, _, y_test = utils.split_data(df=df, model=None, with_features=with_features)

    scaler = utils.get_scaler(method=scaling_method)
    if scaler is not None:
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    resampler = utils.get_resampler(method=resampling_method)
    if resampler is not None:
        x_train, y_train = resampler.fit_resample(x_train, y_train)

    num_train = len(x_train)
    num_test = len(x_test)

    # Reformat data to DMatrix for xgboost
    log(INFO, "Reformatting data...")
    train_dmatrix = transform_dataset_to_dmatrix(x_train, y_train)
    test_dmatrix = transform_dataset_to_dmatrix(x_test, y_test)

    return train_dmatrix, test_dmatrix, num_train, num_test, participant


def transform_dataset_to_dmatrix(x: pd.DataFrame, y: pd.DataFrame) -> xgb.DMatrix:
    """Transform dataset to DMatrix format for xgboost."""
    new_data = xgb.DMatrix(x, label=y)
    return new_data


def evaluate(pred: pd.DataFrame | np.ndarray, y_true: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    scores = get_scores(pred=pred, y=y_true)
    tp, tn, fp, fn = get_classification_results(cm=scores[4])
    results = {
        "accuracy": scores[0],
        "recall": scores[1],
        "precision": scores[2],
        "f1": scores[3],
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }
    # Return results and confusion matrix for later plotting
    return results, scores[4]


def get_scores(pred: pd.DataFrame, y: pd.DataFrame) -> tuple[float, float, float, float, pd.DataFrame]:
    acc = accuracy_score(y_true=y, y_pred=pred)
    rec = recall_score(y_true=y, y_pred=pred)
    prec = precision_score(y_true=y, y_pred=pred)
    f1 = f1_score(y_true=y, y_pred=pred)

    cm = confusion_matrix(y_true=y, y_pred=pred)
    return acc, rec, prec, f1, cm


def get_classification_results(cm: pd.DataFrame) -> tuple[int, int, int, int]:
    tp = int(cm[1][1])
    tn = int(cm[0][0])
    fp = int(cm[0][1])
    fn = int(cm[1][0])

    return tp, tn, fp, fn
