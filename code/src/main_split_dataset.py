import os
from pathlib import Path

from service.datasetservice.DatasetService import DatasetService

# ************************ DEFINE CONFIGURATION *****************************
BASE_PATH = Path(__file__).parent.parent / "dataset" / "individual"
# ***************************************************************************
if __name__ == "__main__":
    dataset_service = DatasetService()
    dataset = dataset_service.load_dataset()

    for participant in range(2, 36):
        split = dataset[dataset["Participant"] == participant]
        split.to_csv(BASE_PATH / f"participant_{participant}.csv", index=False)
