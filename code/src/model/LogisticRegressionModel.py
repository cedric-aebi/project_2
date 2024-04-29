import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from model.AbstractModel import AbstractModel


class LogisticRegressionModel(AbstractModel):
    def __init__(self):
        hyperparameter_grid = {
            "penalty": ["l2", "l1", "elasticnet", None],
            "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
        }
        super().__init__(
            model=LogisticRegression(random_state=42, max_iter=500), hyperparameter_grid=hyperparameter_grid
        )

    def fit(self, train_x: np.ndarray, train_y: np.ndarray, grid_search: bool, run_info: dict) -> None:
        if grid_search:
            grid_search = GridSearchCV(self._model, self._hyperparameter_grid, n_jobs=-1, cv=self._kfold)
            grid_result = grid_search.fit(train_x, train_y)
            self._model = grid_result.best_estimator_
            pred = self._model.predict(train_x)
            scores = self.get_scores(pred=pred, y=train_y)
            tp, tn, fp, fn = self.get_classification_results(cm=scores[4])
            run_info["training"] = {
                "grid_search": {
                    "scoring_method": "accuracy",
                    "best_params": grid_result.best_params_,
                    "best_validation_score": grid_result.best_score_,
                },
                "fitted_model": {
                    "params": self._model.get_params(),
                    "scores": {
                        "accuracy": scores[0],
                        "recall": scores[1],
                        "precision": scores[2],
                        "f1": scores[3],
                        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
                    },
                },
            }
        else:
            self._model = self._model.fit(train_x, train_y)
            pred = self._model.predict(train_x)
            scores = self.get_scores(pred=pred, y=train_y)
            tp, tn, fp, fn = self.get_classification_results(cm=scores[4])
            run_info["training"] = {
                "fitted_model": {
                    "params": self._model.get_params(),
                    "scores": {
                        "accuracy": scores[0],
                        "recall": scores[1],
                        "precision": scores[2],
                        "f1": scores[3],
                        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
                    },
                },
            }
