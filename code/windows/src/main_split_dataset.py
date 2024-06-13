from pathlib import Path
import pickle

import pandas as pd

from service.datasetservice.DatasetService import DatasetService

# ************************ DEFINE CONFIGURATION *****************************
DATASET_PATH = Path(__file__).parent.parent / "dataset"
FEATURES_PATH = DATASET_PATH / "features"
# ***************************************************************************
if __name__ == "__main__":
    file_to_read = open(DATASET_PATH / "all_features.pkl", "rb")
    loaded_features = pickle.load(file_to_read)
    file_to_read.close()

    file_to_read = open(DATASET_PATH / "all_label.pkl", "rb")
    loaded_labels = pickle.load(file_to_read)
    file_to_read.close()

    all_training_features = []
    all_training_labels = []
    all_testing_features = []
    all_testing_labels = []

    idx = 0
    for feature in loaded_features:
        df_feature = pd.DataFrame(feature)
        df_feature = df_feature.reset_index(drop=True)

        # Fill in nan values
        df_feature = df_feature.ffill().bfill()

        new_label = pd.DataFrame(loaded_labels[idx])

        x_train, x_test, y_train, y_test = DatasetService().train_test_split(x=df_feature, y=new_label, shuffle=True)

        file_to_store = open(FEATURES_PATH / f"training_features_{str(idx + 2)}.pkl", "wb")
        pickle.dump(x_train, file_to_store)
        file_to_store.close()

        file_to_store = open(FEATURES_PATH / f"training_labels_{str(idx + 2)}.pkl", "wb")
        pickle.dump(y_train, file_to_store)
        file_to_store.close()

        file_to_store = open(FEATURES_PATH / f"testing_features_{str(idx + 2)}.pkl", "wb")
        pickle.dump(x_test, file_to_store)
        file_to_store.close()

        file_to_store = open(FEATURES_PATH / f"testing_labels_{str(idx + 2)}.pkl", "wb")
        pickle.dump(y_test, file_to_store)
        file_to_store.close()

        all_training_features.append(x_train)
        all_training_labels.append(y_train)
        all_testing_features.append(x_test)
        all_testing_labels.append(y_test)

        idx = idx + 1

    file_to_store = open(FEATURES_PATH / "all_training_features.pkl", "wb")
    pickle.dump(all_training_features, file_to_store)
    file_to_store.close()

    file_to_store = open(FEATURES_PATH / "all_training_labels.pkl", "wb")
    pickle.dump(all_training_labels, file_to_store)
    file_to_store.close()

    file_to_store = open(FEATURES_PATH / "all_testing_features.pkl", "wb")
    pickle.dump(all_testing_features, file_to_store)
    file_to_store.close()

    file_to_store = open(FEATURES_PATH / "all_testing_labels.pkl", "wb")
    pickle.dump(all_testing_labels, file_to_store)
    file_to_store.close()
