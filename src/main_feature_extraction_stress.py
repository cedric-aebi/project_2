import os
from pathlib import Path

import numpy as np
import pandas as pd
import pywt

from service.featureservice.FeatureService import FeatureService

DATASETS_PATH = Path(__file__).parent.parent / "datasets" / "stress" / "raw"
WINDOW_LENGTH = 45
STEP_SIZE = 1

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

        train_hr_1, val_hr_1, test_hr_1 = feature_service.split_and_window(
            data=hr_data_1, window_length=WINDOW_LENGTH, step_size=STEP_SIZE
        )
        train_hr_2, val_hr_2, test_hr_2 = feature_service.split_and_window(
            data=hr_data_2, window_length=WINDOW_LENGTH, step_size=STEP_SIZE
        )
        train_hr_3, val_hr_3, test_hr_3 = feature_service.split_and_window(
            data=hr_data_3, window_length=WINDOW_LENGTH, step_size=STEP_SIZE
        )
        train_hr_4, val_hr_4, test_hr_4 = feature_service.split_and_window(
            data=hr_data_4, window_length=WINDOW_LENGTH, step_size=STEP_SIZE
        )
        train_hr_5, val_hr_5, test_hr_5 = feature_service.split_and_window(
            data=hr_data_5, window_length=WINDOW_LENGTH, step_size=STEP_SIZE
        )
        train_hr_6, val_hr_6, test_hr_6 = feature_service.split_and_window(
            data=hr_data_6, window_length=WINDOW_LENGTH, step_size=STEP_SIZE
        )
        train_hr_7, val_hr_7, test_hr_7 = feature_service.split_and_window(
            data=hr_data_7, window_length=WINDOW_LENGTH, step_size=STEP_SIZE
        )

        train_respr_1, val_respr_1, test_respr_1 = feature_service.split_and_window(
            data=respr_data_1, window_length=WINDOW_LENGTH, step_size=STEP_SIZE
        )
        train_respr_2, val_respr_2, test_respr_2 = feature_service.split_and_window(
            data=respr_data_2, window_length=WINDOW_LENGTH, step_size=STEP_SIZE
        )
        train_respr_3, val_respr_3, test_respr_3 = feature_service.split_and_window(
            data=respr_data_3, window_length=WINDOW_LENGTH, step_size=STEP_SIZE
        )
        train_respr_4, val_respr_4, test_respr_4 = feature_service.split_and_window(
            data=respr_data_4, window_length=WINDOW_LENGTH, step_size=STEP_SIZE
        )
        train_respr_5, val_respr_5, test_respr_5 = feature_service.split_and_window(
            data=respr_data_5, window_length=WINDOW_LENGTH, step_size=STEP_SIZE
        )
        train_respr_6, val_respr_6, test_respr_6 = feature_service.split_and_window(
            data=respr_data_6, window_length=WINDOW_LENGTH, step_size=STEP_SIZE
        )
        train_respr_7, val_respr_7, test_respr_7 = feature_service.split_and_window(
            data=respr_data_7, window_length=WINDOW_LENGTH, step_size=STEP_SIZE
        )

        label_train_1, label_val_1, label_test_1 = (
            [0] * len(train_hr_1) if train_hr_1 is not None else None,
            [0] * len(val_hr_1) if val_hr_1 is not None else None,
            [0] * len(test_hr_1) if test_hr_1 is not None else None,
        )  # No stress
        label_train_2, label_val_2, label_test_2 = (
            [1] * len(train_hr_2) if train_hr_2 is not None else None,
            [1] * len(val_hr_2) if val_hr_2 is not None else None,
            [1] * len(test_hr_2) if test_hr_2 is not None else None,
        )  # Stress
        label_train_3, label_val_3, label_test_3 = (
            [0] * len(train_hr_3) if train_hr_3 is not None else None,
            [0] * len(val_hr_3) if val_hr_3 is not None else None,
            [0] * len(test_hr_3) if test_hr_3 is not None else None,
        )  # No Stress
        label_train_4, label_val_4, label_test_4 = (
            [1] * len(train_hr_4) if train_hr_4 is not None else None,
            [1] * len(val_hr_4) if val_hr_4 is not None else None,
            [1] * len(test_hr_4) if test_hr_4 is not None else None,
        )  # Stress
        label_train_5, label_val_5, label_test_5 = (
            [0] * len(train_hr_5) if train_hr_5 is not None else None,
            [0] * len(val_hr_5) if val_hr_5 is not None else None,
            [0] * len(test_hr_5) if test_hr_5 is not None else None,
        )  # No Stress
        label_train_6, label_val_6, label_test_6 = (
            [1] * len(train_hr_6) if train_hr_6 is not None else None,
            [1] * len(val_hr_6) if val_hr_6 is not None else None,
            [1] * len(test_hr_6) if test_hr_6 is not None else None,
        )  # Stress
        label_train_7, label_val_7, label_test_7 = (
            [0] * len(train_hr_7) if train_hr_7 is not None else None,
            [0] * len(val_hr_7) if val_hr_7 is not None else None,
            [0] * len(test_hr_7) if test_hr_7 is not None else None,
        )  # No Stress

        split_train_1, split_val_1, split_test_1 = (
            ["train"] * len(train_hr_1) if train_hr_1 is not None else None,
            ["val"] * len(val_hr_1) if val_hr_1 is not None else None,
            ["test"] * len(test_hr_1) if test_hr_1 is not None else None,
        )
        split_train_2, split_val_2, split_test_2 = (
            ["train"] * len(train_hr_2) if train_hr_2 is not None else None,
            ["val"] * len(val_hr_2) if val_hr_2 is not None else None,
            ["test"] * len(test_hr_2) if test_hr_2 is not None else None,
        )
        split_train_3, split_val_3, split_test_3 = (
            ["train"] * len(train_hr_3) if train_hr_3 is not None else None,
            ["val"] * len(val_hr_3) if val_hr_3 is not None else None,
            ["test"] * len(test_hr_3) if test_hr_3 is not None else None,
        )
        split_train_4, split_val_4, split_test_4 = (
            ["train"] * len(train_hr_4) if train_hr_4 is not None else None,
            ["val"] * len(val_hr_4) if val_hr_4 is not None else None,
            ["test"] * len(test_hr_4) if test_hr_4 is not None else None,
        )
        split_train_5, split_val_5, split_test_5 = (
            ["train"] * len(train_hr_5) if train_hr_5 is not None else None,
            ["val"] * len(val_hr_5) if val_hr_5 is not None else None,
            ["test"] * len(test_hr_5) if test_hr_5 is not None else None,
        )
        split_train_6, split_val_6, split_test_6 = (
            ["train"] * len(train_hr_6) if train_hr_6 is not None else None,
            ["val"] * len(val_hr_6) if val_hr_6 is not None else None,
            ["test"] * len(test_hr_6) if test_hr_6 is not None else None,
        )
        split_train_7, split_val_7, split_test_7 = (
            ["train"] * len(train_hr_7) if train_hr_7 is not None else None,
            ["val"] * len(val_hr_7) if val_hr_7 is not None else None,
            ["test"] * len(test_hr_7) if test_hr_7 is not None else None,
        )

        # Concatenate all HR windows
        hr_arrays = [
            train_hr_1,
            val_hr_1,
            test_hr_1,
            train_hr_2,
            val_hr_2,
            test_hr_2,
            train_hr_3,
            val_hr_3,
            test_hr_3,
            train_hr_4,
            val_hr_4,
            test_hr_4,
            train_hr_5,
            val_hr_5,
            test_hr_5,
            train_hr_6,
            val_hr_6,
            test_hr_6,
            train_hr_7,
            val_hr_7,
            test_hr_7,
        ]
        HR = np.concatenate([arr for arr in hr_arrays if arr is not None], axis=0)

        # Concatenate all RESPR windows
        respr_arrays = [
            train_respr_1,
            val_respr_1,
            test_respr_1,
            train_respr_2,
            val_respr_2,
            test_respr_2,
            train_respr_3,
            val_respr_3,
            test_respr_3,
            train_respr_4,
            val_respr_4,
            test_respr_4,
            train_respr_5,
            val_respr_5,
            test_respr_5,
            train_respr_6,
            val_respr_6,
            test_respr_6,
            train_respr_7,
            val_respr_7,
            test_respr_7,
        ]
        RESPR = np.concatenate([arr for arr in respr_arrays if arr is not None], axis=0)

        # Concatenate all labels
        label_arrays = [
            label_train_1,
            label_val_1,
            label_test_1,
            label_train_2,
            label_val_2,
            label_test_2,
            label_train_3,
            label_val_3,
            label_test_3,
            label_train_4,
            label_val_4,
            label_test_4,
            label_train_5,
            label_val_5,
            label_test_5,
            label_train_6,
            label_val_6,
            label_test_6,
            label_train_7,
            label_val_7,
            label_test_7,
        ]
        LABEL = np.concatenate([arr for arr in label_arrays if arr is not None], axis=0)

        # Concatenate all splits
        split_arrays = [
            split_train_1,
            split_val_1,
            split_test_1,
            split_train_2,
            split_val_2,
            split_test_2,
            split_train_3,
            split_val_3,
            split_test_3,
            split_train_4,
            split_val_4,
            split_test_4,
            split_train_5,
            split_val_5,
            split_test_5,
            split_train_6,
            split_val_6,
            split_test_6,
            split_train_7,
            split_val_7,
            split_test_7,
        ]
        SPLIT = np.concatenate([arr for arr in split_arrays if arr is not None], axis=0)

        # Construct the feature matrix, 72 HR features and 72 RESP features = 144 features in total.
        length = len(LABEL)
        rows = []
        for i in range(length):
            if i % 500 == 0:
                print(i)

            feature_dict = {"Subject": subject, "Label": LABEL[i], "Split": SPLIT[i]}

            deriv_HR, second_deriv_HR = feature_service.get_derivatives(data=HR[i, :])
            deriv_RESPR, second_deriv_RESPR = feature_service.get_derivatives(data=RESPR[i, :])

            _, HR_cD_3, HR_cD_2, HR_cD_1 = pywt.wavedec(HR[i, :], "Haar", level=3)  # 3 = 1Hz, 2 = 2Hz, 1=4Hz
            _, RESPR_cD_3, RESPR_cD_2, RESPR_cD_1 = pywt.wavedec(RESPR[i, :], "Haar", level=3)

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

        # convert float64 to float32
        for col in final_dataframe.select_dtypes(include=["float64"]).columns:
            final_dataframe[col] = final_dataframe[col].astype(np.float32)

    if not os.path.exists(DATASETS_PATH):
        os.makedirs(DATASETS_PATH)

    final_dataframe.to_csv(DATASETS_PATH / "stress_features.csv", index=False)
