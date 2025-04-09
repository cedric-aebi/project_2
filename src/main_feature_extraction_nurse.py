import os
import warnings
from pathlib import Path

from neurokit2 import NeuroKitWarning

from service.featureservice.FeatureService import FeatureService

warnings.simplefilter("ignore", NeuroKitWarning)

import neurokit2 as nk
import pandas as pd
from tqdm import tqdm
import pywt

DATASETS_PATH = Path(__file__).parent.parent / "datasets" / "nurse" / "raw"

if __name__ == "__main__":
    feature_service = FeatureService()

    original_df = pd.read_csv(DATASETS_PATH / "nurse_no_features.csv", low_memory=False)
    original_df = original_df[original_df["EDA"] != 0]
    participants = original_df["id"].unique()

    new_df = original_df
    # Ensure the datetime column is in datetime format
    new_df["datetime"] = pd.to_datetime(new_df["datetime"])

    # Initialize an empty list to store the results
    all_segments = []

    # Group the data by "id"
    grouped = new_df.groupby("id")

    # Define the threshold for identifying gaps (e.g., 0.1 seconds)
    threshold = 1.0

    # Process each group separately
    for _, group in grouped:
        group = group.sort_values(by="datetime").copy()

        # Calculate the time difference between consecutive rows
        group["time_diff"] = group["datetime"].diff().dt.total_seconds()

        # Handle first row which will have NaN time_diff
        group["time_diff"] = group["time_diff"].fillna(0)

        # Identify gaps
        group["gap"] = group["time_diff"] > threshold

        # Create a segment ID based on gaps
        group["segment_id"] = group["gap"].cumsum()

        # Drop unnecessary columns
        group = group.drop(columns=["time_diff", "gap"])

        # Group by segment_id and assign labels
        segments = []
        for seg_id, segment in group.groupby("segment_id"):
            segment_copy = segment.copy()
            segment_copy["label"] = segment_copy["label"].iloc[0]
            segments.append(segment_copy)

        # Append the processed segments to the list
        all_segments.append(pd.concat(segments))

    # Combine all segments back into a single dataframe
    df_with_time_series_index = pd.concat(all_segments)
    df_with_time_series_index.reset_index(drop=True, inplace=True)

    final_data = []  # Store rows in a list instead of repeatedly concatenating DataFrames

    for participant in tqdm(participants, desc=" Participants", position=0):
        participant_df = df_with_time_series_index[df_with_time_series_index["id"] == participant]

        for segment_id in tqdm(participant_df["segment_id"].unique(), desc=" Segments", position=1, leave=False):
            segment_df = participant_df[participant_df["segment_id"] == segment_id]
            label = segment_df["label"].iloc[0]

            if len(segment_df) < 1920:
                continue

            window_length = 1920  # 60s at 32Hz
            step_size = 1920  # 60s at 32Hz

            eda_windows = feature_service.get_windows(segment_df["EDA"].values, window_length, step_size)
            temp_windows = feature_service.get_windows(segment_df["TEMP"].values, window_length, step_size)
            hr_windows = feature_service.get_windows(segment_df["HR"].values, window_length, step_size)
            acc_x_windows = feature_service.get_windows(segment_df["X"].values, window_length, step_size)
            acc_y_windows = feature_service.get_windows(segment_df["Y"].values, window_length, step_size)
            acc_z_windows = feature_service.get_windows(segment_df["Z"].values, window_length, step_size)

            # Iterate over all the windows
            for i in tqdm(range(len(eda_windows)), desc=" Windows", position=2, leave=False):
                deriv_EDA, second_deriv_EDA = feature_service.get_derivatives(data=eda_windows[i])
                deriv_HR, second_deriv_HR = feature_service.get_derivatives(data=hr_windows[i])
                deriv_TEMP, second_deriv_TEMP = feature_service.get_derivatives(data=temp_windows[i])
                deriv_ACC_X, second_deriv_ACC_X = feature_service.get_derivatives(data=acc_x_windows[i])
                deriv_ACC_Y, second_deriv_ACC_Y = feature_service.get_derivatives(data=acc_y_windows[i])
                deriv_ACC_Z, second_deriv_ACC_Z = feature_service.get_derivatives(data=acc_z_windows[i])

                _, EDA_cD_3, EDA_cD_2, EDA_cD_1 = pywt.wavedec(
                    eda_windows[i], "Haar", level=3
                )  # 3 = 1Hz, 2 = 2Hz, 1=4Hz
                _, HR_cD_3, HR_cD_2, HR_cD_1 = pywt.wavedec(hr_windows[i], "Haar", level=3)  # 3 = 1Hz, 2 = 2Hz, 1=4Hz
                _, TEMP_cD_3, TEMP_cD_2, TEMP_cD_1 = pywt.wavedec(
                    temp_windows[i], "Haar", level=3
                )  # 3 = 1Hz, 2 = 2Hz, 1=4Hz
                _, ACC_X_cD_3, ACC_X_cD_2, ACC_X_cD_1 = pywt.wavedec(
                    acc_x_windows[i], "Haar", level=3
                )  # 3 = 1Hz, 2 = 2Hz, 1=4Hz
                _, ACC_Y_cD_3, ACC_Y_cD_2, ACC_Y_cD_1 = pywt.wavedec(
                    acc_y_windows[i], "Haar", level=3
                )  # 3 = 1Hz, 2 = 2Hz, 1=4Hz
                _, ACC_Z_cD_3, ACC_Z_cD_2, ACC_Z_cD_1 = pywt.wavedec(
                    acc_z_windows[i], "Haar", level=3
                )  # 3 = 1Hz, 2 = 2Hz, 1=4Hz

                # Analyze EDA with NeuroKit2
                processed_data, info = nk.eda_process(eda_windows[i], sampling_rate=4)
                results = nk.eda_analyze(processed_data, sampling_rate=4)

                # Store data as dictionary (faster than creating DataFrame inside loop)
                row = {"Participant": participant, "Label": label, **results.to_dict(orient="records")[0]}

                # ----- EDA features -----
                # EDA statistical features:
                (
                    row["EDA_mean"],
                    row["EDA_median"],
                    row["EDA_min"],
                    row["EDA_max"],
                    row["EDA_max_amp"],
                    row["EDA_range"],
                    row["EDA_var"],
                    row["EDA_std_dev"],
                    row["EDA_abs_dev"],
                    row["EDA_rms"],
                    row["EDA_kurtosis"],
                    row["EDA_skew"],
                ) = feature_service.get_statistics(data=eda_windows[i])
                (
                    row["Deriv_EDA_mean"],
                    row["Deriv_EDA_median"],
                    row["Deriv_EDA_min"],
                    row["Deriv_EDA_max"],
                    row["Deriv_EDA_max_amp"],
                    row["Deriv_EDA_range"],
                    row["Deriv_EDA_var"],
                    row["Deriv_EDA_std_dev"],
                    row["Deriv_EDA_abs_dev"],
                    row["Deriv_EDA_rms"],
                    row["Deriv_EDA_kurtosis"],
                    row["Deriv_EDA_skew"],
                ) = feature_service.get_statistics(data=deriv_EDA)
                (
                    row["Deriv_2_EDA_mean"],
                    row["Deriv_2_EDA_median"],
                    row["Deriv_2_EDA_min"],
                    row["Deriv_2_EDA_max"],
                    row["Deriv_2_EDA_max_amp"],
                    row["Deriv_2_EDA_range"],
                    row["Deriv_2_EDA_var"],
                    row["Deriv_2_EDA_std_dev"],
                    row["Deriv_2_EDA_abs_dev"],
                    row["Deriv_2_EDA_rms"],
                    row["Deriv_2_EDA_kurtosis"],
                    row["Deriv_2_EDA_skew"],
                ) = feature_service.get_statistics(data=second_deriv_EDA)
                # EDA wavelet features:
                (
                    row["Wavelet_1Hz_EDA_mean"],
                    row["Wavelet_1Hz_EDA_median"],
                    row["Wavelet_1Hz_EDA_min"],
                    row["Wavelet_1Hz_EDA_max"],
                    row["Wavelet_1Hz_EDA_max_amp"],
                    row["Wavelet_1Hz_EDA_range"],
                    row["Wavelet_1Hz_EDA_var"],
                    row["Wavelet_1Hz_EDA_std_dev"],
                    row["Wavelet_1Hz_EDA_abs_dev"],
                    row["Wavelet_1Hz_EDA_rms"],
                    row["Wavelet_1Hz_EDA_kurtosis"],
                    row["Wavelet_1Hz_EDA_skew"],
                ) = feature_service.get_statistics(data=EDA_cD_3)
                (
                    row["Wavelet_2Hz_EDA_mean"],
                    row["Wavelet_2Hz_EDA_median"],
                    row["Wavelet_2Hz_EDA_min"],
                    row["Wavelet_2Hz_EDA_max"],
                    row["Wavelet_2Hz_EDA_max_amp"],
                    row["Wavelet_2Hz_EDA_range"],
                    row["Wavelet_2Hz_EDA_var"],
                    row["Wavelet_2Hz_EDA_std_dev"],
                    row["Wavelet_2Hz_EDA_abs_dev"],
                    row["Wavelet_2Hz_EDA_rms"],
                    row["Wavelet_2Hz_EDA_kurtosis"],
                    row["Wavelet_2Hz_EDA_skew"],
                ) = feature_service.get_statistics(data=EDA_cD_2)
                (
                    row["Wavelet_4Hz_EDA_mean"],
                    row["Wavelet_4Hz_EDA_median"],
                    row["Wavelet_4Hz_EDA_min"],
                    row["Wavelet_4Hz_EDA_max"],
                    row["Wavelet_4Hz_EDA_max_amp"],
                    row["Wavelet_4Hz_EDA_range"],
                    row["Wavelet_4Hz_EDA_var"],
                    row["Wavelet_4Hz_EDA_std_dev"],
                    row["Wavelet_4Hz_EDA_abs_dev"],
                    row["Wavelet_4Hz_EDA_rms"],
                    row["Wavelet_4Hz_EDA_kurtosis"],
                    row["Wavelet_4Hz_EDA_skew"],
                ) = feature_service.get_statistics(data=EDA_cD_1)

                # ----- HR features -----
                # HR statistical features:
                (
                    row["HR_mean"],
                    row["HR_median"],
                    row["HR_min"],
                    row["HR_max"],
                    row["HR_max_amp"],
                    row["HR_range"],
                    row["HR_var"],
                    row["HR_std_dev"],
                    row["HR_abs_dev"],
                    row["HR_rms"],
                    row["HR_kurtosis"],
                    row["HR_skew"],
                ) = feature_service.get_statistics(data=hr_windows[i])
                (
                    row["Deriv_HR_mean"],
                    row["Deriv_HR_median"],
                    row["Deriv_HR_min"],
                    row["Deriv_HR_max"],
                    row["Deriv_HR_max_amp"],
                    row["Deriv_HR_range"],
                    row["Deriv_HR_var"],
                    row["Deriv_HR_std_dev"],
                    row["Deriv_HR_abs_dev"],
                    row["Deriv_HR_rms"],
                    row["Deriv_HR_kurtosis"],
                    row["Deriv_HR_skew"],
                ) = feature_service.get_statistics(data=deriv_HR)
                (
                    row["Deriv_2_HR_mean"],
                    row["Deriv_2_HR_median"],
                    row["Deriv_2_HR_min"],
                    row["Deriv_2_HR_max"],
                    row["Deriv_2_HR_max_amp"],
                    row["Deriv_2_HR_range"],
                    row["Deriv_2_HR_var"],
                    row["Deriv_2_HR_std_dev"],
                    row["Deriv_2_HR_abs_dev"],
                    row["Deriv_2_HR_rms"],
                    row["Deriv_2_HR_kurtosis"],
                    row["Deriv_2_HR_skew"],
                ) = feature_service.get_statistics(data=second_deriv_HR)
                # HR wavelet features:
                (
                    row["Wavelet_1Hz_HR_mean"],
                    row["Wavelet_1Hz_HR_median"],
                    row["Wavelet_1Hz_HR_min"],
                    row["Wavelet_1Hz_HR_max"],
                    row["Wavelet_1Hz_HR_max_amp"],
                    row["Wavelet_1Hz_HR_range"],
                    row["Wavelet_1Hz_HR_var"],
                    row["Wavelet_1Hz_HR_std_dev"],
                    row["Wavelet_1Hz_HR_abs_dev"],
                    row["Wavelet_1Hz_HR_rms"],
                    row["Wavelet_1Hz_HR_kurtosis"],
                    row["Wavelet_1Hz_HR_skew"],
                ) = feature_service.get_statistics(data=HR_cD_3)
                (
                    row["Wavelet_2Hz_HR_mean"],
                    row["Wavelet_2Hz_HR_median"],
                    row["Wavelet_2Hz_HR_min"],
                    row["Wavelet_2Hz_HR_max"],
                    row["Wavelet_2Hz_HR_max_amp"],
                    row["Wavelet_2Hz_HR_range"],
                    row["Wavelet_2Hz_HR_var"],
                    row["Wavelet_2Hz_HR_std_dev"],
                    row["Wavelet_2Hz_HR_abs_dev"],
                    row["Wavelet_2Hz_HR_rms"],
                    row["Wavelet_2Hz_HR_kurtosis"],
                    row["Wavelet_2Hz_HR_skew"],
                ) = feature_service.get_statistics(data=HR_cD_2)
                (
                    row["Wavelet_4Hz_HR_mean"],
                    row["Wavelet_4Hz_HR_median"],
                    row["Wavelet_4Hz_HR_min"],
                    row["Wavelet_4Hz_HR_max"],
                    row["Wavelet_4Hz_HR_max_amp"],
                    row["Wavelet_4Hz_HR_range"],
                    row["Wavelet_4Hz_HR_var"],
                    row["Wavelet_4Hz_HR_std_dev"],
                    row["Wavelet_4Hz_HR_abs_dev"],
                    row["Wavelet_4Hz_HR_rms"],
                    row["Wavelet_4Hz_HR_kurtosis"],
                    row["Wavelet_4Hz_HR_skew"],
                ) = feature_service.get_statistics(data=HR_cD_1)

                # ----- TEMP features -----
                # TEMP statistical features:
                (
                    row["TEMP_mean"],
                    row["TEMP_median"],
                    row["TEMP_min"],
                    row["TEMP_max"],
                    row["TEMP_max_amp"],
                    row["TEMP_range"],
                    row["TEMP_var"],
                    row["TEMP_std_dev"],
                    row["TEMP_abs_dev"],
                    row["TEMP_rms"],
                    row["TEMP_kurtosis"],
                    row["TEMP_skew"],
                ) = feature_service.get_statistics(data=temp_windows[i])
                (
                    row["Deriv_TEMP_mean"],
                    row["Deriv_TEMP_median"],
                    row["Deriv_TEMP_min"],
                    row["Deriv_TEMP_max"],
                    row["Deriv_TEMP_max_amp"],
                    row["Deriv_TEMP_range"],
                    row["Deriv_TEMP_var"],
                    row["Deriv_TEMP_std_dev"],
                    row["Deriv_TEMP_abs_dev"],
                    row["Deriv_TEMP_rms"],
                    row["Deriv_TEMP_kurtosis"],
                    row["Deriv_TEMP_skew"],
                ) = feature_service.get_statistics(data=deriv_TEMP)
                (
                    row["Deriv_2_TEMP_mean"],
                    row["Deriv_2_TEMP_median"],
                    row["Deriv_2_TEMP_min"],
                    row["Deriv_2_TEMP_max"],
                    row["Deriv_2_TEMP_max_amp"],
                    row["Deriv_2_TEMP_range"],
                    row["Deriv_2_TEMP_var"],
                    row["Deriv_2_TEMP_std_dev"],
                    row["Deriv_2_TEMP_abs_dev"],
                    row["Deriv_2_TEMP_rms"],
                    row["Deriv_2_TEMP_kurtosis"],
                    row["Deriv_2_TEMP_skew"],
                ) = feature_service.get_statistics(data=second_deriv_TEMP)

                # TEMP wavelet features:
                (
                    row["Wavelet_1Hz_TEMP_mean"],
                    row["Wavelet_1Hz_TEMP_median"],
                    row["Wavelet_1Hz_TEMP_min"],
                    row["Wavelet_1Hz_TEMP_max"],
                    row["Wavelet_1Hz_TEMP_max_amp"],
                    row["Wavelet_1Hz_TEMP_range"],
                    row["Wavelet_1Hz_TEMP_var"],
                    row["Wavelet_1Hz_TEMP_std_dev"],
                    row["Wavelet_1Hz_TEMP_abs_dev"],
                    row["Wavelet_1Hz_TEMP_rms"],
                    row["Wavelet_1Hz_TEMP_kurtosis"],
                    row["Wavelet_1Hz_TEMP_skew"],
                ) = feature_service.get_statistics(data=TEMP_cD_3)
                (
                    row["Wavelet_2Hz_TEMP_mean"],
                    row["Wavelet_2Hz_TEMP_median"],
                    row["Wavelet_2Hz_TEMP_min"],
                    row["Wavelet_2Hz_TEMP_max"],
                    row["Wavelet_2Hz_TEMP_max_amp"],
                    row["Wavelet_2Hz_TEMP_range"],
                    row["Wavelet_2Hz_TEMP_var"],
                    row["Wavelet_2Hz_TEMP_std_dev"],
                    row["Wavelet_2Hz_TEMP_abs_dev"],
                    row["Wavelet_2Hz_TEMP_rms"],
                    row["Wavelet_2Hz_TEMP_kurtosis"],
                    row["Wavelet_2Hz_TEMP_skew"],
                ) = feature_service.get_statistics(data=TEMP_cD_2)
                (
                    row["Wavelet_4Hz_TEMP_mean"],
                    row["Wavelet_4Hz_TEMP_median"],
                    row["Wavelet_4Hz_TEMP_min"],
                    row["Wavelet_4Hz_TEMP_max"],
                    row["Wavelet_4Hz_TEMP_max_amp"],
                    row["Wavelet_4Hz_TEMP_range"],
                    row["Wavelet_4Hz_TEMP_var"],
                    row["Wavelet_4Hz_TEMP_std_dev"],
                    row["Wavelet_4Hz_TEMP_abs_dev"],
                    row["Wavelet_4Hz_TEMP_rms"],
                    row["Wavelet_4Hz_TEMP_kurtosis"],
                    row["Wavelet_4Hz_TEMP_skew"],
                ) = feature_service.get_statistics(data=TEMP_cD_1)

                # ----- ACC features -----
                # ACC statistical features:
                (
                    row["ACC_X_mean"],
                    row["ACC_X_median"],
                    row["ACC_X_min"],
                    row["ACC_X_max"],
                    row["ACC_X_max_amp"],
                    row["ACC_X_range"],
                    row["ACC_X_var"],
                    row["ACC_X_std_dev"],
                    row["ACC_X_abs_dev"],
                    row["ACC_X_rms"],
                    row["ACC_X_kurtosis"],
                    row["ACC_X_skew"],
                ) = feature_service.get_statistics(data=acc_x_windows[i])
                (
                    row["Deriv_ACC_X_mean"],
                    row["Deriv_ACC_X_median"],
                    row["Deriv_ACC_X_min"],
                    row["Deriv_ACC_X_max"],
                    row["Deriv_ACC_X_max_amp"],
                    row["Deriv_ACC_X_range"],
                    row["Deriv_ACC_X_var"],
                    row["Deriv_ACC_X_std_dev"],
                    row["Deriv_ACC_X_abs_dev"],
                    row["Deriv_ACC_X_rms"],
                    row["Deriv_ACC_X_kurtosis"],
                    row["Deriv_ACC_X_skew"],
                ) = feature_service.get_statistics(data=deriv_ACC_X)
                (
                    row["Deriv_2_ACC_X_mean"],
                    row["Deriv_2_ACC_X_median"],
                    row["Deriv_2_ACC_X_min"],
                    row["Deriv_2_ACC_X_max"],
                    row["Deriv_2_ACC_X_max_amp"],
                    row["Deriv_2_ACC_X_range"],
                    row["Deriv_2_ACC_X_var"],
                    row["Deriv_2_ACC_X_std_dev"],
                    row["Deriv_2_ACC_X_abs_dev"],
                    row["Deriv_2_ACC_X_rms"],
                    row["Deriv_2_ACC_X_kurtosis"],
                    row["Deriv_2_ACC_X_skew"],
                ) = feature_service.get_statistics(data=second_deriv_ACC_X)
                # ACC wavelet features:
                (
                    row["Wavelet_1Hz_ACC_X_mean"],
                    row["Wavelet_1Hz_ACC_X_median"],
                    row["Wavelet_1Hz_ACC_X_min"],
                    row["Wavelet_1Hz_ACC_X_max"],
                    row["Wavelet_1Hz_ACC_X_max_amp"],
                    row["Wavelet_1Hz_ACC_X_range"],
                    row["Wavelet_1Hz_ACC_X_var"],
                    row["Wavelet_1Hz_ACC_X_std_dev"],
                    row["Wavelet_1Hz_ACC_X_abs_dev"],
                    row["Wavelet_1Hz_ACC_X_rms"],
                    row["Wavelet_1Hz_ACC_X_kurtosis"],
                    row["Wavelet_1Hz_ACC_X_skew"],
                ) = feature_service.get_statistics(data=ACC_X_cD_3)
                (
                    row["Wavelet_2Hz_ACC_X_mean"],
                    row["Wavelet_2Hz_ACC_X_median"],
                    row["Wavelet_2Hz_ACC_X_min"],
                    row["Wavelet_2Hz_ACC_X_max"],
                    row["Wavelet_2Hz_ACC_X_max_amp"],
                    row["Wavelet_2Hz_ACC_X_range"],
                    row["Wavelet_2Hz_ACC_X_var"],
                    row["Wavelet_2Hz_ACC_X_std_dev"],
                    row["Wavelet_2Hz_ACC_X_abs_dev"],
                    row["Wavelet_2Hz_ACC_X_rms"],
                    row["Wavelet_2Hz_ACC_X_kurtosis"],
                    row["Wavelet_2Hz_ACC_X_skew"],
                ) = feature_service.get_statistics(data=ACC_X_cD_2)
                (
                    row["Wavelet_4Hz_ACC_X_mean"],
                    row["Wavelet_4Hz_ACC_X_median"],
                    row["Wavelet_4Hz_ACC_X_min"],
                    row["Wavelet_4Hz_ACC_X_max"],
                    row["Wavelet_4Hz_ACC_X_max_amp"],
                    row["Wavelet_4Hz_ACC_X_range"],
                    row["Wavelet_4Hz_ACC_X_var"],
                    row["Wavelet_4Hz_ACC_X_std_dev"],
                    row["Wavelet_4Hz_ACC_X_abs_dev"],
                    row["Wavelet_4Hz_ACC_X_rms"],
                    row["Wavelet_4Hz_ACC_X_kurtosis"],
                    row["Wavelet_4Hz_ACC_X_skew"],
                ) = feature_service.get_statistics(data=ACC_X_cD_1)

                (
                    row["ACC_Y_mean"],
                    row["ACC_Y_median"],
                    row["ACC_Y_min"],
                    row["ACC_Y_max"],
                    row["ACC_Y_max_amp"],
                    row["ACC_Y_range"],
                    row["ACC_Y_var"],
                    row["ACC_Y_std_dev"],
                    row["ACC_Y_abs_dev"],
                    row["ACC_Y_rms"],
                    row["ACC_Y_kurtosis"],
                    row["ACC_Y_skew"],
                ) = feature_service.get_statistics(data=acc_y_windows[i])
                (
                    row["Deriv_ACC_Y_mean"],
                    row["Deriv_ACC_Y_median"],
                    row["Deriv_ACC_Y_min"],
                    row["Deriv_ACC_Y_max"],
                    row["Deriv_ACC_Y_max_amp"],
                    row["Deriv_ACC_Y_range"],
                    row["Deriv_ACC_Y_var"],
                    row["Deriv_ACC_Y_std_dev"],
                    row["Deriv_ACC_Y_abs_dev"],
                    row["Deriv_ACC_Y_rms"],
                    row["Deriv_ACC_Y_kurtosis"],
                    row["Deriv_ACC_Y_skew"],
                ) = feature_service.get_statistics(data=deriv_ACC_Y)
                (
                    row["Deriv_2_ACC_Y_mean"],
                    row["Deriv_2_ACC_Y_median"],
                    row["Deriv_2_ACC_Y_min"],
                    row["Deriv_2_ACC_Y_max"],
                    row["Deriv_2_ACC_Y_max_amp"],
                    row["Deriv_2_ACC_Y_range"],
                    row["Deriv_2_ACC_Y_var"],
                    row["Deriv_2_ACC_Y_std_dev"],
                    row["Deriv_2_ACC_Y_abs_dev"],
                    row["Deriv_2_ACC_Y_rms"],
                    row["Deriv_2_ACC_Y_kurtosis"],
                    row["Deriv_2_ACC_Y_skew"],
                ) = feature_service.get_statistics(data=second_deriv_ACC_Y)
                # ACC wavelet features:
                (
                    row["Wavelet_1Hz_ACC_Y_mean"],
                    row["Wavelet_1Hz_ACC_Y_median"],
                    row["Wavelet_1Hz_ACC_Y_min"],
                    row["Wavelet_1Hz_ACC_Y_max"],
                    row["Wavelet_1Hz_ACC_Y_max_amp"],
                    row["Wavelet_1Hz_ACC_Y_range"],
                    row["Wavelet_1Hz_ACC_Y_var"],
                    row["Wavelet_1Hz_ACC_Y_std_dev"],
                    row["Wavelet_1Hz_ACC_Y_abs_dev"],
                    row["Wavelet_1Hz_ACC_Y_rms"],
                    row["Wavelet_1Hz_ACC_Y_kurtosis"],
                    row["Wavelet_1Hz_ACC_Y_skew"],
                ) = feature_service.get_statistics(data=ACC_Y_cD_3)
                (
                    row["Wavelet_2Hz_ACC_Y_mean"],
                    row["Wavelet_2Hz_ACC_Y_median"],
                    row["Wavelet_2Hz_ACC_Y_min"],
                    row["Wavelet_2Hz_ACC_Y_max"],
                    row["Wavelet_2Hz_ACC_Y_max_amp"],
                    row["Wavelet_2Hz_ACC_Y_range"],
                    row["Wavelet_2Hz_ACC_Y_var"],
                    row["Wavelet_2Hz_ACC_Y_std_dev"],
                    row["Wavelet_2Hz_ACC_Y_abs_dev"],
                    row["Wavelet_2Hz_ACC_Y_rms"],
                    row["Wavelet_2Hz_ACC_Y_kurtosis"],
                    row["Wavelet_2Hz_ACC_Y_skew"],
                ) = feature_service.get_statistics(data=ACC_Y_cD_2)
                (
                    row["Wavelet_4Hz_ACC_Y_mean"],
                    row["Wavelet_4Hz_ACC_Y_median"],
                    row["Wavelet_4Hz_ACC_Y_min"],
                    row["Wavelet_4Hz_ACC_Y_max"],
                    row["Wavelet_4Hz_ACC_Y_max_amp"],
                    row["Wavelet_4Hz_ACC_Y_range"],
                    row["Wavelet_4Hz_ACC_Y_var"],
                    row["Wavelet_4Hz_ACC_Y_std_dev"],
                    row["Wavelet_4Hz_ACC_Y_abs_dev"],
                    row["Wavelet_4Hz_ACC_Y_rms"],
                    row["Wavelet_4Hz_ACC_Y_kurtosis"],
                    row["Wavelet_4Hz_ACC_Y_skew"],
                ) = feature_service.get_statistics(data=ACC_Y_cD_1)

                (
                    row["ACC_Z_mean"],
                    row["ACC_Z_median"],
                    row["ACC_Z_min"],
                    row["ACC_Z_max"],
                    row["ACC_Z_max_amp"],
                    row["ACC_Z_range"],
                    row["ACC_Z_var"],
                    row["ACC_Z_std_dev"],
                    row["ACC_Z_abs_dev"],
                    row["ACC_Z_rms"],
                    row["ACC_Z_kurtosis"],
                    row["ACC_Z_skew"],
                ) = feature_service.get_statistics(data=acc_z_windows[i])
                (
                    row["Deriv_ACC_Z_mean"],
                    row["Deriv_ACC_Z_median"],
                    row["Deriv_ACC_Z_min"],
                    row["Deriv_ACC_Z_max"],
                    row["Deriv_ACC_Z_max_amp"],
                    row["Deriv_ACC_Z_range"],
                    row["Deriv_ACC_Z_var"],
                    row["Deriv_ACC_Z_std_dev"],
                    row["Deriv_ACC_Z_abs_dev"],
                    row["Deriv_ACC_Z_rms"],
                    row["Deriv_ACC_Z_kurtosis"],
                    row["Deriv_ACC_Z_skew"],
                ) = feature_service.get_statistics(data=deriv_ACC_Z)
                (
                    row["Deriv_2_ACC_Z_mean"],
                    row["Deriv_2_ACC_Z_median"],
                    row["Deriv_2_ACC_Z_min"],
                    row["Deriv_2_ACC_Z_max"],
                    row["Deriv_2_ACC_Z_max_amp"],
                    row["Deriv_2_ACC_Z_range"],
                    row["Deriv_2_ACC_Z_var"],
                    row["Deriv_2_ACC_Z_std_dev"],
                    row["Deriv_2_ACC_Z_abs_dev"],
                    row["Deriv_2_ACC_Z_rms"],
                    row["Deriv_2_ACC_Z_kurtosis"],
                    row["Deriv_2_ACC_Z_skew"],
                ) = feature_service.get_statistics(data=second_deriv_ACC_Z)
                # ACC wavelet features:
                (
                    row["Wavelet_1Hz_ACC_Z_mean"],
                    row["Wavelet_1Hz_ACC_Z_median"],
                    row["Wavelet_1Hz_ACC_Z_min"],
                    row["Wavelet_1Hz_ACC_Z_max"],
                    row["Wavelet_1Hz_ACC_Z_max_amp"],
                    row["Wavelet_1Hz_ACC_Z_range"],
                    row["Wavelet_1Hz_ACC_Z_var"],
                    row["Wavelet_1Hz_ACC_Z_std_dev"],
                    row["Wavelet_1Hz_ACC_Z_abs_dev"],
                    row["Wavelet_1Hz_ACC_Z_rms"],
                    row["Wavelet_1Hz_ACC_Z_kurtosis"],
                    row["Wavelet_1Hz_ACC_Z_skew"],
                ) = feature_service.get_statistics(data=ACC_Z_cD_3)
                (
                    row["Wavelet_2Hz_ACC_Z_mean"],
                    row["Wavelet_2Hz_ACC_Z_median"],
                    row["Wavelet_2Hz_ACC_Z_min"],
                    row["Wavelet_2Hz_ACC_Z_max"],
                    row["Wavelet_2Hz_ACC_Z_max_amp"],
                    row["Wavelet_2Hz_ACC_Z_range"],
                    row["Wavelet_2Hz_ACC_Z_var"],
                    row["Wavelet_2Hz_ACC_Z_std_dev"],
                    row["Wavelet_2Hz_ACC_Z_abs_dev"],
                    row["Wavelet_2Hz_ACC_Z_rms"],
                    row["Wavelet_2Hz_ACC_Z_kurtosis"],
                    row["Wavelet_2Hz_ACC_Z_skew"],
                ) = feature_service.get_statistics(data=ACC_Z_cD_2)
                (
                    row["Wavelet_4Hz_ACC_Z_mean"],
                    row["Wavelet_4Hz_ACC_Z_median"],
                    row["Wavelet_4Hz_ACC_Z_min"],
                    row["Wavelet_4Hz_ACC_Z_max"],
                    row["Wavelet_4Hz_ACC_Z_max_amp"],
                    row["Wavelet_4Hz_ACC_Z_range"],
                    row["Wavelet_4Hz_ACC_Z_var"],
                    row["Wavelet_4Hz_ACC_Z_std_dev"],
                    row["Wavelet_4Hz_ACC_Z_abs_dev"],
                    row["Wavelet_4Hz_ACC_Z_rms"],
                    row["Wavelet_4Hz_ACC_Z_kurtosis"],
                    row["Wavelet_4Hz_ACC_Z_skew"],
                ) = feature_service.get_statistics(data=ACC_Z_cD_1)

                final_data.append(row)

    # Create DataFrame once at the end
    final_df = pd.DataFrame(final_data)

    if not os.path.exists(DATASETS_PATH):
        os.makedirs(DATASETS_PATH)

    final_df.to_csv(DATASETS_PATH / "nurse_features.csv", index=False)
