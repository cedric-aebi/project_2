from pathlib import Path

import pandas as pd

from enums.Dataset import Dataset

# **************************** CONFIGURATION ****************************
DATASET = Dataset.STRESS
WITH_FEATURES = False
# *************************************************************************

PATH_TO_DATASETS = Path(__file__).parent.parent / "datasets"

if __name__ == "__main__":
    if DATASET == Dataset.NURSE:
        if WITH_FEATURES:
            data = pd.read_csv(PATH_TO_DATASETS / "nurse" / "raw" / "nurse_features.csv", engine="pyarrow")
            data = data.fillna(0)
            # Remove ids "CE" and "EG" due to lack of data
            data = data[~data["Participant"].isin(["CE", "EG"])]
            # Binary classification
            data.loc[data["Label"] == 2, "Label"] = 1

            # Save the dataset
            data.to_pickle(PATH_TO_DATASETS / "nurse" / "processed" / "with_features" / "all.pkl")

            for participant in data["Participant"].unique():
                participant_data = data[data["Participant"] == participant]
                participant_data.to_pickle(
                    PATH_TO_DATASETS / "nurse" / "processed" / "with_features" / f"{participant}.pkl"
                )
        else:
            data = pd.read_csv(PATH_TO_DATASETS / "nurse" / "raw" / "nurse_no_features.csv", engine="pyarrow")
            # Remove ids "CE" and "EG" due to lack of data
            data = data[~data["id"].isin(["CE", "EG"])]
            # Binary classification
            data.loc[data["label"] == 2, "label"] = 1
            # drop column datetime
            data = data.drop(columns=["datetime"])
            # rename id column to Participant
            data = data.rename(columns={"id": "Participant"})
            # rename label column to Label
            data = data.rename(columns={"label": "Label"})

            # Save the dataset
            data.to_pickle(PATH_TO_DATASETS / "nurse" / "processed" / "no_features" / "all.pkl")

            for participant in data["Participant"].unique():
                participant_data = data[data["Participant"] == participant]
                participant_data.to_pickle(
                    PATH_TO_DATASETS / "nurse" / "processed" / "no_features" / f"{participant}.pkl"
                )
    if DATASET == Dataset.STRESS:
        if WITH_FEATURES:
            data = pd.read_csv(PATH_TO_DATASETS / "stress" / "raw" / "stress_features.csv", sep=",")
            data = data.fillna(0)

            # Rename "Subject" column to "Participant"
            data = data.rename(columns={"Subject": "Participant"})

            # Save the dataset
            data.to_pickle(PATH_TO_DATASETS / "stress" / "processed" / "with_features" / "all.pkl")

            for participant in data["Participant"].unique():
                participant_data = data[data["Participant"] == participant]
                participant_data.to_pickle(
                    PATH_TO_DATASETS / "stress" / "processed" / "with_features" / f"{participant}.pkl"
                )
        else:
            data = pd.read_csv(PATH_TO_DATASETS / "stress" / "raw" / "stress_no_features.csv", sep=",")
            data = data.ffill().bfill()
            # drop column "Time(sec)"
            data = data.drop(columns=["Time(sec)"])

            # Save the dataset
            data.to_pickle(PATH_TO_DATASETS / "stress" / "processed" / "no_features" / "all.pkl")

            for participant in data["Participant"].unique():
                participant_data = data[data["Participant"] == participant]
                participant_data.to_pickle(
                    PATH_TO_DATASETS / "stress" / "processed" / "no_features" / f"{participant}.pkl"
                )
