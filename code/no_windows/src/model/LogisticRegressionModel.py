import numpy as np
from imblearn.base import BaseSampler
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from model.AbstractModel import AbstractModel


class LogisticRegressionModel(AbstractModel):
    def __init__(self, scaler: BaseEstimator, resampler: BaseSampler):
        hyperparameter_grid = {
            "model__penalty": ["l2", "l1", "elasticnet", None],
            "model__C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "model__solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
        }
        super().__init__(
            model=LogisticRegression(random_state=42, max_iter=10000),
            hyperparameter_grid=hyperparameter_grid,
            scaler=scaler,
            resampler=resampler,
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, grid_search: bool, run_info: dict) -> None:
        if grid_search:
            grid_search_cv = GridSearchCV(
                estimator=self._pipeline, param_grid=self._hyperparameter_grid, n_jobs=-1, cv=self._kfold
            )
            grid_result = grid_search_cv.fit(x_train, y_train)
            self._best_estimator = grid_result.best_estimator_
        else:
            pass
            # TODO: Handle non-grid searches
            # self._model = self._model.fit(train_x, train_y)
            # pred = self._model.predict(train_x)
            # scores = self.get_scores(pred=pred, y=train_y)
            # tp, tn, fp, fn = self.get_classification_results(cm=scores[4])
            # run_info["training"] = {
            #     "fitted_model": {
            #         "params": self._model.get_params(),
            #         "scores": {
            #             "accuracy": scores[0],
            #             "recall": scores[1],
            #             "precision": scores[2],
            #             "f1": scores[3],
            #             "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            #         },
            #     },
            # }

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._best_estimator.predict(x)
