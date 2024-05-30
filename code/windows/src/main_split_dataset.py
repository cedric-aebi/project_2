from pathlib import Path
import pickle

from service.datasetservice.DatasetService import DatasetService

# ************************ DEFINE CONFIGURATION *****************************
EXPORT_PATH = Path(__file__).parent.parent / "dataset" / "features"
# ***************************************************************************
if __name__ == "__main__":
    dataset_service = DatasetService()
    dataset = dataset_service.load_dataset()
    dataset = dataset_service.remove_nan(dataset=dataset)

    print(len(dataset))

    all_training_features = []
    all_training_labels = []
    all_testing_features = []
    all_testing_labels = []

    for participant in range(2, 36):
        subject_data = dataset[dataset["Participant"] == participant]
        x, y, labels = dataset_service.get_features_and_labels(dataset=subject_data)
        train_x, test_x, train_y, test_y = dataset_service.train_test_split(x=x, y=y, shuffle=True)

        file_to_store = open(EXPORT_PATH / f"training_features_{participant}.pkl", "wb")
        pickle.dump(train_x, file_to_store)
        file_to_store.close()

        file_to_store = open(EXPORT_PATH / f"training_labels_{participant}.pkl", "wb")
        pickle.dump(train_y, file_to_store)
        file_to_store.close()

        file_to_store = open(EXPORT_PATH / f"testing_features_{participant}.pkl", "wb")
        pickle.dump(test_x, file_to_store)
        file_to_store.close()

        file_to_store = open(EXPORT_PATH / f"testing_labels_{participant}.pkl", "wb")
        pickle.dump(test_y, file_to_store)
        file_to_store.close()

        all_training_features.append(train_x)
        all_training_labels.append(train_y)
        all_testing_features.append(test_x)
        all_testing_labels.append(test_y)

    file_to_store = open(EXPORT_PATH / "all_training_features.pkl", "wb")
    pickle.dump(all_training_features, file_to_store)
    file_to_store.close()

    file_to_store = open(EXPORT_PATH / "all_training_labels.pkl", "wb")
    pickle.dump(all_training_labels, file_to_store)
    file_to_store.close()

    file_to_store = open(EXPORT_PATH / "all_testing_features.pkl", "wb")
    pickle.dump(all_testing_features, file_to_store)
    file_to_store.close()

    file_to_store = open(EXPORT_PATH / "all_testing_labels.pkl", "wb")
    pickle.dump(all_testing_labels, file_to_store)
    file_to_store.close()
