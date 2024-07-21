import pickle
from pathlib import Path

import numpy as np
import pywt

from service.datasetservice.DatasetService import DatasetService
from service.featureservice.FeatureService import FeatureService

EXPORT_PATH = Path(__file__).parent.parent / "dataset"

if __name__ == "__main__":
    dataset_service = DatasetService()
    feature_service = FeatureService()

    dataset = dataset_service.load_dataset()

    all_features = []
    all_label = []
    for subject in range(2, 36):
        subject_data = dataset[dataset["Participant"] == subject]

        # Fill the NaN values with the previous and next values
        subject_data = subject_data.ffill().bfill()

        no_stress_1, stress_1, no_stress_2, stress_2, no_stress_3, stress_3, no_stress_4 = (
            feature_service.extract_different_timeseries(subject_data=subject_data)
        )

        hr_data_1 = no_stress_1["HR"].to_numpy()
        hr_data_2 = stress_1["HR"].to_numpy()
        hr_data_3 = no_stress_2["HR"].to_numpy()
        hr_data_4 = stress_2["HR"].to_numpy()
        hr_data_5 = no_stress_3["HR"].to_numpy()
        hr_data_6 = stress_3["HR"].to_numpy()
        hr_data_7 = no_stress_4["HR"].to_numpy()

        respr_data_1 = no_stress_1["respr"].to_numpy()
        respr_data_2 = stress_1["respr"].to_numpy()
        respr_data_3 = no_stress_2["respr"].to_numpy()
        respr_data_4 = stress_2["respr"].to_numpy()
        respr_data_5 = no_stress_3["respr"].to_numpy()
        respr_data_6 = stress_3["respr"].to_numpy()
        respr_data_7 = no_stress_4["respr"].to_numpy()

        HR_1 = feature_service.get_windows(data=hr_data_1, window_length=60, step_size=1)
        HR_2 = feature_service.get_windows(data=hr_data_2, window_length=60, step_size=1)
        HR_3 = feature_service.get_windows(data=hr_data_3, window_length=60, step_size=1)
        HR_4 = feature_service.get_windows(data=hr_data_4, window_length=60, step_size=1)
        HR_5 = feature_service.get_windows(data=hr_data_5, window_length=60, step_size=1)
        HR_6 = feature_service.get_windows(data=hr_data_6, window_length=60, step_size=1)
        HR_7 = feature_service.get_windows(data=hr_data_7, window_length=60, step_size=1)

        RESPR_1 = feature_service.get_windows(data=respr_data_1, window_length=60, step_size=1)
        RESPR_2 = feature_service.get_windows(data=respr_data_2, window_length=60, step_size=1)
        RESPR_3 = feature_service.get_windows(data=respr_data_3, window_length=60, step_size=1)
        RESPR_4 = feature_service.get_windows(data=respr_data_4, window_length=60, step_size=1)
        RESPR_5 = feature_service.get_windows(data=respr_data_5, window_length=60, step_size=1)
        RESPR_6 = feature_service.get_windows(data=respr_data_6, window_length=60, step_size=1)
        RESPR_7 = feature_service.get_windows(data=respr_data_7, window_length=60, step_size=1)

        HR = np.concatenate((HR_1, HR_2, HR_3, HR_4, HR_5, HR_6, HR_7), axis=0)
        RESPR = np.concatenate((RESPR_1, RESPR_2, RESPR_3, RESPR_4, RESPR_5, RESPR_6, RESPR_7), axis=0)

        label_1 = [0] * len(HR_1)  # No stress
        label_2 = [1] * len(HR_2)  # Stress
        label_3 = [0] * len(HR_3)  # No stress
        label_4 = [1] * len(HR_4)  # Stress
        label_5 = [0] * len(HR_5)  # No Stress
        label_6 = [1] * len(HR_6)  # Stress
        label_7 = [0] * len(HR_7)  # No Stress

        LABEL = np.concatenate((label_1, label_2, label_3, label_4, label_5, label_6, label_7), axis=0)

        all_label.append(LABEL)

        # Construct the feature matrix, 60 HR features and 60 RESP features = 120 features in total.

        length = len(LABEL)
        features = np.zeros((length, 120))

        for i in range(length):
            if i % 500 == 0:
                print(i)

            deriv_HR, second_deriv_HR = feature_service.get_derivatives(data=HR[i, :])
            deriv_RESPR, second_deriv_RESPR = feature_service.get_derivatives(data=RESPR[i, :])

            _, HR_cD_3, HR_cD_2, HR_cD_1 = pywt.wavedec(HR[i, :], "Haar", level=3)  # 3 = 1Hz, 2 = 2Hz, 1=4Hz
            _, RESPR_cD_3, RESPR_cD_2, RESPR_cD_1 = pywt.wavedec(RESPR[i, :], "Haar", level=3)

            # ----- HR features -----
            # HR statistical features:
            features[i, 0:10] = feature_service.get_statistics(data=HR[i, :])
            features[i, 10:20] = feature_service.get_statistics(data=deriv_HR)
            features[i, 20:30] = feature_service.get_statistics(data=second_deriv_HR)
            # HR wavelet features:
            features[i, 30:40] = feature_service.get_statistics(data=HR_cD_3)
            features[i, 40:50] = feature_service.get_statistics(data=HR_cD_2)
            features[i, 50:60] = feature_service.get_statistics(data=HR_cD_1)

            # ----- RESPR features -----
            # RESPR statistical features:
            features[i, 60:70] = feature_service.get_statistics(data=RESPR[i, :])
            features[i, 70:80] = feature_service.get_statistics(data=deriv_RESPR)
            features[i, 80:90] = feature_service.get_statistics(data=second_deriv_RESPR)
            # RESPR wavelet features:
            features[i, 90:100] = feature_service.get_statistics(data=RESPR_cD_3)
            features[i, 100:110] = feature_service.get_statistics(data=RESPR_cD_2)
            features[i, 110:120] = feature_service.get_statistics(data=RESPR_cD_1)

        all_features.append(features)

    file_to_store = open(EXPORT_PATH / "all_features.pkl", "wb")
    pickle.dump(all_features, file_to_store)
    file_to_store.close()

    file_to_store = open(EXPORT_PATH / "all_label.pkl", "wb")
    pickle.dump(all_label, file_to_store)
    file_to_store.close()
