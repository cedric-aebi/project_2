import os
from pathlib import Path

import numpy as np
import pandas as pd
import pywt

from service.featureservice.FeatureService import FeatureService

DATASETS_PATH = Path(__file__).parent.parent / "datasets" / "stress" / "raw"

if __name__ == "__main__":
    feature_service = FeatureService()

    dataset = pd.read_csv(DATASETS_PATH / "stress_no_features.csv", sep=",")

    final_dataframe = pd.DataFrame()
    for subject in range(2, 36):
        print(f"Subject: {subject}")

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

        # Construct the feature matrix, 72 HR features and 72 RESP features = 144 features in total.
        length = len(LABEL)
        rows = []
        for i in range(length):
            if i % 500 == 0:
                print(i)

            deriv_HR, second_deriv_HR = feature_service.get_derivatives(data=HR[i, :])
            deriv_RESPR, second_deriv_RESPR = feature_service.get_derivatives(data=RESPR[i, :])

            _, HR_cD_3, HR_cD_2, HR_cD_1 = pywt.wavedec(HR[i, :], "Haar", level=3)  # 3 = 1Hz, 2 = 2Hz, 1=4Hz
            _, RESPR_cD_3, RESPR_cD_2, RESPR_cD_1 = pywt.wavedec(RESPR[i, :], "Haar", level=3)

            feature_dict = {"Subject": subject, "Label": LABEL[i]}

            # ----- HR features -----
            # HR statistical features:
            (
                feature_dict["HR_mean"],
                feature_dict["HR_median"],
                feature_dict["HR_min"],
                feature_dict["HR_max"],
                feature_dict["HR_max_amp"],
                feature_dict["HR_range"],
                feature_dict["HR_var"],
                feature_dict["HR_std_dev"],
                feature_dict["HR_abs_dev"],
                feature_dict["HR_rms"],
                feature_dict["HR_kurtosis"],
                feature_dict["HR_skew"],
            ) = feature_service.get_statistics(data=HR[i, :])
            (
                feature_dict["Deriv_HR_mean"],
                feature_dict["Deriv_HR_median"],
                feature_dict["Deriv_HR_min"],
                feature_dict["Deriv_HR_max"],
                feature_dict["Deriv_HR_max_amp"],
                feature_dict["Deriv_HR_range"],
                feature_dict["Deriv_HR_var"],
                feature_dict["Deriv_HR_std_dev"],
                feature_dict["Deriv_HR_abs_dev"],
                feature_dict["Deriv_HR_rms"],
                feature_dict["Deriv_HR_kurtosis"],
                feature_dict["Deriv_HR_skew"],
            ) = feature_service.get_statistics(data=deriv_HR)
            (
                feature_dict["Deriv_2_HR_mean"],
                feature_dict["Deriv_2_HR_median"],
                feature_dict["Deriv_2_HR_min"],
                feature_dict["Deriv_2_HR_max"],
                feature_dict["Deriv_2_HR_max_amp"],
                feature_dict["Deriv_2_HR_range"],
                feature_dict["Deriv_2_HR_var"],
                feature_dict["Deriv_2_HR_std_dev"],
                feature_dict["Deriv_2_HR_abs_dev"],
                feature_dict["Deriv_2_HR_rms"],
                feature_dict["Deriv_2_HR_kurtosis"],
                feature_dict["Deriv_2_HR_skew"],
            ) = feature_service.get_statistics(data=second_deriv_HR)
            # HR wavelet features:
            (
                feature_dict["Wavelet_1Hz_HR_mean"],
                feature_dict["Wavelet_1Hz_HR_median"],
                feature_dict["Wavelet_1Hz_HR_min"],
                feature_dict["Wavelet_1Hz_HR_max"],
                feature_dict["Wavelet_1Hz_HR_max_amp"],
                feature_dict["Wavelet_1Hz_HR_range"],
                feature_dict["Wavelet_1Hz_HR_var"],
                feature_dict["Wavelet_1Hz_HR_std_dev"],
                feature_dict["Wavelet_1Hz_HR_abs_dev"],
                feature_dict["Wavelet_1Hz_HR_rms"],
                feature_dict["Wavelet_1Hz_HR_kurtosis"],
                feature_dict["Wavelet_1Hz_HR_skew"],
            ) = feature_service.get_statistics(data=HR_cD_3)
            (
                feature_dict["Wavelet_2Hz_HR_mean"],
                feature_dict["Wavelet_2Hz_HR_median"],
                feature_dict["Wavelet_2Hz_HR_min"],
                feature_dict["Wavelet_2Hz_HR_max"],
                feature_dict["Wavelet_2Hz_HR_max_amp"],
                feature_dict["Wavelet_2Hz_HR_range"],
                feature_dict["Wavelet_2Hz_HR_var"],
                feature_dict["Wavelet_2Hz_HR_std_dev"],
                feature_dict["Wavelet_2Hz_HR_abs_dev"],
                feature_dict["Wavelet_2Hz_HR_rms"],
                feature_dict["Wavelet_2Hz_HR_kurtosis"],
                feature_dict["Wavelet_2Hz_HR_skew"],
            ) = feature_service.get_statistics(data=HR_cD_2)
            (
                feature_dict["Wavelet_4Hz_HR_mean"],
                feature_dict["Wavelet_4Hz_HR_median"],
                feature_dict["Wavelet_4Hz_HR_min"],
                feature_dict["Wavelet_4Hz_HR_max"],
                feature_dict["Wavelet_4Hz_HR_max_amp"],
                feature_dict["Wavelet_4Hz_HR_range"],
                feature_dict["Wavelet_4Hz_HR_var"],
                feature_dict["Wavelet_4Hz_HR_std_dev"],
                feature_dict["Wavelet_4Hz_HR_abs_dev"],
                feature_dict["Wavelet_4Hz_HR_rms"],
                feature_dict["Wavelet_4Hz_HR_kurtosis"],
                feature_dict["Wavelet_4Hz_HR_skew"],
            ) = feature_service.get_statistics(data=HR_cD_1)

            # ----- RESPR features -----
            # RESPR statistical features:
            (
                feature_dict["RESPR_mean"],
                feature_dict["RESPR_median"],
                feature_dict["RESPR_min"],
                feature_dict["RESPR_max"],
                feature_dict["RESPR_max_amp"],
                feature_dict["RESPR_range"],
                feature_dict["RESPR_var"],
                feature_dict["RESPR_std_dev"],
                feature_dict["RESPR_abs_dev"],
                feature_dict["RESPR_rms"],
                feature_dict["RESPR_kurtosis"],
                feature_dict["RESPR_skew"],
            ) = feature_service.get_statistics(data=RESPR[i, :])
            (
                feature_dict["Deriv_RESPR_mean"],
                feature_dict["Deriv_RESPR_median"],
                feature_dict["Deriv_RESPR_min"],
                feature_dict["Deriv_RESPR_max"],
                feature_dict["Deriv_RESPR_max_amp"],
                feature_dict["Deriv_RESPR_range"],
                feature_dict["Deriv_RESPR_var"],
                feature_dict["Deriv_RESPR_std_dev"],
                feature_dict["Deriv_RESPR_abs_dev"],
                feature_dict["Deriv_RESPR_rms"],
                feature_dict["Deriv_RESPR_kurtosis"],
                feature_dict["Deriv_RESPR_skew"],
            ) = feature_service.get_statistics(data=deriv_RESPR)
            (
                feature_dict["Deriv_2_RESPR_mean"],
                feature_dict["Deriv_2_RESPR_median"],
                feature_dict["Deriv_2_RESPR_min"],
                feature_dict["Deriv_2_RESPR_max"],
                feature_dict["Deriv_2_RESPR_max_amp"],
                feature_dict["Deriv_2_RESPR_range"],
                feature_dict["Deriv_2_RESPR_var"],
                feature_dict["Deriv_2_RESPR_std_dev"],
                feature_dict["Deriv_2_RESPR_abs_dev"],
                feature_dict["Deriv_2_RESPR_rms"],
                feature_dict["Deriv_2_RESPR_kurtosis"],
                feature_dict["Deriv_2_RESPR_skew"],
            ) = feature_service.get_statistics(data=second_deriv_RESPR)
            # RESPR wavelet features:
            (
                feature_dict["Wavelet_1Hz_RESPR_mean"],
                feature_dict["Wavelet_1Hz_RESPR_median"],
                feature_dict["Wavelet_1Hz_RESPR_min"],
                feature_dict["Wavelet_1Hz_RESPR_max"],
                feature_dict["Wavelet_1Hz_RESPR_max_amp"],
                feature_dict["Wavelet_1Hz_RESPR_range"],
                feature_dict["Wavelet_1Hz_RESPR_var"],
                feature_dict["Wavelet_1Hz_RESPR_std_dev"],
                feature_dict["Wavelet_1Hz_RESPR_abs_dev"],
                feature_dict["Wavelet_1Hz_RESPR_rms"],
                feature_dict["Wavelet_1Hz_RESPR_kurtosis"],
                feature_dict["Wavelet_1Hz_RESPR_skew"],
            ) = feature_service.get_statistics(data=RESPR_cD_3)
            (
                feature_dict["Wavelet_2Hz_RESPR_mean"],
                feature_dict["Wavelet_2Hz_RESPR_median"],
                feature_dict["Wavelet_2Hz_RESPR_min"],
                feature_dict["Wavelet_2Hz_RESPR_max"],
                feature_dict["Wavelet_2Hz_RESPR_max_amp"],
                feature_dict["Wavelet_2Hz_RESPR_range"],
                feature_dict["Wavelet_2Hz_RESPR_var"],
                feature_dict["Wavelet_2Hz_RESPR_std_dev"],
                feature_dict["Wavelet_2Hz_RESPR_abs_dev"],
                feature_dict["Wavelet_2Hz_RESPR_rms"],
                feature_dict["Wavelet_2Hz_RESPR_kurtosis"],
                feature_dict["Wavelet_2Hz_RESPR_skew"],
            ) = feature_service.get_statistics(data=RESPR_cD_2)
            (
                feature_dict["Wavelet_4Hz_RESPR_mean"],
                feature_dict["Wavelet_4Hz_RESPR_median"],
                feature_dict["Wavelet_4Hz_RESPR_min"],
                feature_dict["Wavelet_4Hz_RESPR_max"],
                feature_dict["Wavelet_4Hz_RESPR_max_amp"],
                feature_dict["Wavelet_4Hz_RESPR_range"],
                feature_dict["Wavelet_4Hz_RESPR_var"],
                feature_dict["Wavelet_4Hz_RESPR_std_dev"],
                feature_dict["Wavelet_4Hz_RESPR_abs_dev"],
                feature_dict["Wavelet_4Hz_RESPR_rms"],
                feature_dict["Wavelet_4Hz_RESPR_kurtosis"],
                feature_dict["Wavelet_4Hz_RESPR_skew"],
            ) = feature_service.get_statistics(data=RESPR_cD_1)

            rows.append(feature_dict)

        df = pd.DataFrame(rows)
        final_dataframe = pd.concat([final_dataframe, df])

    if not os.path.exists(DATASETS_PATH):
        os.makedirs(DATASETS_PATH)

    final_dataframe.to_csv(DATASETS_PATH / "stress_features.csv", index=False)
