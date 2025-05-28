from pathlib import Path

import numpy as np
import pandas as pd
from flwr.common import NDArrays
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from enums.Participant import NurseParticipant
from enums.ResamplingMethod import ResamplingMethod
from enums.ScalingMethod import ScalingMethod
from utils import utils

# This information is needed to create a correct scikit-learn model
NUM_UNIQUE_LABELS = 2


def get_model_parameters(model: LogisticRegression) -> NDArrays:
    """Returns the parameters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(model: LogisticRegression, params: NDArrays) -> None:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]


def set_initial_params(model: LogisticRegression, num_features: int) -> None:
    """Sets initial parameters as zeros Required since model params are uninitialized
    until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    model.classes_ = np.arange(NUM_UNIQUE_LABELS)

    model.coef_ = np.zeros((NUM_UNIQUE_LABELS, num_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((NUM_UNIQUE_LABELS,))


def create_log_reg_and_instantiate_parameters(penalty: str, max_iter: int, num_features: int) -> LogisticRegression:
    """Helper function to create a LogisticRegression model."""
    model = LogisticRegression(
        penalty=penalty,
        max_iter=max_iter,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting,
    )
    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model, num_features=num_features)
    return model


def load_data(
    which: int | str,
    scaling_method: ScalingMethod | None,
    resampling_method: ResamplingMethod | None,
    participant_leave_out: NurseParticipant | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, NurseParticipant | str]:
    # TODO: rewrite
    base_path = Path(__file__).parent.parent.parent.parent.parent / "datasets" / "nurse" / "paper"

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

    x = df.drop(columns=["Label", "Participant"])
    y = df["Label"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

    scaler = utils.get_scaler(method=scaling_method)
    if scaler is not None:
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    resampler = utils.get_resampler(method=resampling_method)
    if resampler is not None:
        x_train, y_train = resampler.fit_resample(x_train, y_train)

    return x_train, x_test, y_train, y_test, participant


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
