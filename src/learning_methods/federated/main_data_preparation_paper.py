from pathlib import Path

import pandas as pd


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent.parent.parent / "datasets" / "nurse"
    original_df = pd.read_csv(base_path / "raw" / "nurse_no_features.csv", low_memory=False)
    original_df = original_df[original_df["EDA"] != 0]

    new_df = original_df
    # Ensure the datetime column is in datetime format
    new_df["datetime"] = pd.to_datetime(new_df["datetime"])

    # Get time of day and date of year as separate features. time of day in hours
    new_df["time_of_day"] = new_df["datetime"].dt.hour + new_df["datetime"].dt.minute / 60
    new_df["date_of_year"] = new_df["datetime"].dt.dayofyear
    new_df = new_df.drop(columns=["datetime"])
    # relabel 2 as 1
    new_df["label"] = new_df["label"].replace(2, 1)
    # rename id column to Participant
    new_df = new_df.rename(columns={"id": "Participant"})
    # rename label column to Label
    new_df = new_df.rename(columns={"label": "Label"})

    # Shuffle
    new_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the dataset
    new_df.to_pickle(base_path / "paper" / "all.pkl")

    for participant in new_df["Participant"].unique():
        participant_data = new_df[new_df["Participant"] == participant]
        participant_data.to_pickle(base_path / "paper" / f"{participant}.pkl")
