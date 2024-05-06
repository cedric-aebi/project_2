from pathlib import Path
import os
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


class VisualizationService:
    @staticmethod
    def plot_class_distribution(dataset: pd.DataFrame, path: Path) -> None:
        fig = plt.figure()
        dataset["Label"].value_counts().plot(kind="barh", title="Class Distribution")
        os.makedirs(path, exist_ok=True)
        fig.savefig(path / "class_distribution.png")
        plt.close()

    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, labels: list[str], path: Path) -> None:
        os.makedirs(path, exist_ok=True)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot().figure_.savefig(path / "confusion_matrix.png")
        plt.close()

    @staticmethod
    def plot_roc(path: Path, test_x: np.ndarray, test_y: np.ndarray, model: Any) -> None:
        os.makedirs(path, exist_ok=True)
        disp = RocCurveDisplay.from_estimator(estimator=model, X=test_x, y=test_y)
        disp.plot().figure_.savefig(path / "roc.png")
        plt.close()
