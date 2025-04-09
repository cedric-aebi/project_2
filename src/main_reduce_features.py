import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut

from enums.Dataset import Dataset
from utils import utils

if __name__ == "__main__":
    dataset = Dataset.NURSE

    df = utils.load_data(dataset=dataset, which="all", with_features=True)

    features = [col for col in df.columns if col not in ["Label", "Participant"]]
    X = df[features].values
    y = df["Label"].values
    groups = df["Participant"].values

    # Leave-One-Subject-Out CV
    logo = LeaveOneGroupOut()

    # Store mean absolute SHAP values for each feature across folds
    shap_values_list = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, y_train = X[train_idx], y[train_idx]

        # Train XGBoost model
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train)

        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        # Take mean absolute SHAP values per feature
        shap_mean = np.abs(shap_values).mean(axis=0)
        shap_values_list.append(shap_mean)

    # ---- Aggregate SHAP values across folds ----
    mean_shap_importance = np.mean(shap_values_list, axis=0)

    # Create a DataFrame for easy handling
    feature_importance_df = pd.DataFrame({"feature": features, "mean_abs_shap": mean_shap_importance})

    # Sort by importance
    feature_importance_df.sort_values(by="mean_abs_shap", ascending=False, inplace=True)

    n_top = len(features) // 2
    top_features = feature_importance_df.head(n_top)["feature"].tolist()

    # Filter your dataframe to keep only those features
    df_reduced = df[top_features + ["Label", "Participant"]]

    # Save the dataset
    df_reduced.to_pickle("../../datasets/nurse/processed/with_features/reduced/all.pkl")

    for participant in df_reduced["Participant"].unique():
        participant_data = df_reduced[df_reduced["Participant"] == participant]
        participant_data.to_pickle(f"../../datasets/nurse/processed/with_features/reduced/{participant}.pkl")
