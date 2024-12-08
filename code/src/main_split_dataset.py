import os
from pathlib import Path
import pickle
import pandas as pd

from service.argumentservice.ArgumentService import ArgumentService
from service.datasetservice.DatasetService import DatasetService
from service.exportservice.ExportService import ExportService

# ************************ DEFINE CONFIGURATION *****************************
DATASET_PATH = Path(__file__).parent.parent / "dataset"
WITH_FEATURES_EXPORT_PATH = DATASET_PATH / "with_additional_features" / "features"
NO_FEATURES_EXPORT_PATH = DATASET_PATH / "no_additional_features" / "features"
# ***************************************************************************
if __name__ == "__main__":
    dataset_service = DatasetService()
    argument_service = ArgumentService(features=True)
    with_features = argument_service.get_features()
    export_service = ExportService()

    all_training_features = []
    all_training_labels = []
    all_testing_features = []
    all_testing_labels = []

    if not os.path.exists(WITH_FEATURES_EXPORT_PATH):
        os.makedirs(WITH_FEATURES_EXPORT_PATH)
    if not os.path.exists(NO_FEATURES_EXPORT_PATH):
        os.makedirs(NO_FEATURES_EXPORT_PATH)

    if with_features:
        file_to_read = open(DATASET_PATH / "with_additional_features" / "all_features.pkl", "rb")
        loaded_features = pickle.load(file_to_read)
        file_to_read.close()

        file_to_read = open(DATASET_PATH / "with_additional_features" / "all_label.pkl", "rb")
        loaded_labels = pickle.load(file_to_read)
        file_to_read.close()

        idx = 0
        for feature in loaded_features:
            df_feature = pd.DataFrame(feature)
            df_feature = df_feature.reset_index(drop=True)

            # Fill in nan values
            df_feature = df_feature.ffill().bfill()

            new_label = pd.DataFrame(loaded_labels[idx])

            x_train, x_test, y_train, y_test = DatasetService().train_test_split(
                x=df_feature, y=new_label, shuffle=True
            )

            export_service.export_file(
                path=WITH_FEATURES_EXPORT_PATH / f"training_features_{str(idx + 2)}.pkl", data=x_train
            )
            export_service.export_file(
                path=WITH_FEATURES_EXPORT_PATH / f"training_labels_{str(idx + 2)}.pkl", data=y_train
            )
            export_service.export_file(
                path=WITH_FEATURES_EXPORT_PATH / f"testing_features_{str(idx + 2)}.pkl", data=x_test
            )
            export_service.export_file(
                path=WITH_FEATURES_EXPORT_PATH / f"testing_labels_{str(idx + 2)}.pkl", data=y_test
            )

            all_training_features.append(x_train)
            all_training_labels.append(y_train)
            all_testing_features.append(x_test)
            all_testing_labels.append(y_test)

            idx = idx + 1

        export_service.export_file(
            path=WITH_FEATURES_EXPORT_PATH / "all_training_features.pkl", data=all_training_features
        )
        export_service.export_file(path=WITH_FEATURES_EXPORT_PATH / "all_training_labels.pkl", data=all_training_labels)
        export_service.export_file(
            path=WITH_FEATURES_EXPORT_PATH / "all_testing_features.pkl", data=all_testing_features
        )
        export_service.export_file(path=WITH_FEATURES_EXPORT_PATH / "all_testing_labels.pkl", data=all_testing_labels)
    else:
        dataset = dataset_service.load_dataset()
        dataset = dataset_service.remove_nan(dataset=dataset)

        for participant in range(2, 36):
            subject_data = dataset[dataset["Participant"] == participant]
            x, y, labels = dataset_service.get_features_and_labels(dataset=subject_data)
            train_x, test_x, train_y, test_y = dataset_service.train_test_split(x=x, y=y, shuffle=True)

            export_service.export_file(
                path=NO_FEATURES_EXPORT_PATH / f"training_features_{participant}.pkl", data=train_x
            )
            export_service.export_file(
                path=NO_FEATURES_EXPORT_PATH / f"training_labels_{participant}.pkl", data=train_y
            )
            export_service.export_file(
                path=NO_FEATURES_EXPORT_PATH / f"testing_features_{participant}.pkl", data=test_x
            )
            export_service.export_file(path=NO_FEATURES_EXPORT_PATH / f"testing_labels_{participant}.pkl", data=test_y)

            all_training_features.append(train_x)
            all_training_labels.append(train_y)
            all_testing_features.append(test_x)
            all_testing_labels.append(test_y)

        export_service.export_file(
            path=NO_FEATURES_EXPORT_PATH / "all_training_features.pkl", data=all_training_features
        )
        export_service.export_file(path=NO_FEATURES_EXPORT_PATH / "all_training_labels.pkl", data=all_training_labels)
        export_service.export_file(path=NO_FEATURES_EXPORT_PATH / "all_testing_features.pkl", data=all_testing_features)
        export_service.export_file(path=NO_FEATURES_EXPORT_PATH / "all_testing_labels.pkl", data=all_testing_labels)
