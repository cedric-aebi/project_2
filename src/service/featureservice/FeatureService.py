from typing import Any
import pandas as pd
import numpy as np
from biosppy.signals import tools


class FeatureService:
    @staticmethod
    def extract_different_timeseries(
        subject_data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        subject_data.reset_index(inplace=True)

        all_list = []
        current_list = []
        current_label = 0
        for idx in subject_data.index:
            if subject_data["Label"][idx] == current_label:
                # If the label hasn't changed, append the row to the current series
                current_list.append(subject_data.iloc[idx])
            else:
                # If the label has changed, start a new series
                all_list.append(pd.DataFrame(current_list))
                current_list = [subject_data.iloc[idx]]
                if current_label == 0:
                    current_label = 1
                else:
                    current_label = 0

        # Remember to add the last series
        all_list.append(pd.DataFrame(current_list))

        # index 0 = 0, index 1 = 1, index 2 = 0, index 3 = 1, index 4 = 0, index 5 = 1, index 6 = 0
        return all_list[0], all_list[1], all_list[2], all_list[3], all_list[4], all_list[5], all_list[6]

    def split_and_window(
        self, data: np.ndarray, window_length: int, step_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(data)
        train_end = int(n * 0.6)  # 60% train
        val_end = int(n * 0.8)  # 20% validation

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        train_windows = self._get_windows(data=train_data, window_length=window_length, step_size=step_size)
        val_windows = self._get_windows(data=val_data, window_length=window_length, step_size=step_size)
        test_windows = self._get_windows(data=test_data, window_length=window_length, step_size=step_size)
        return train_windows, val_windows, test_windows

    @staticmethod
    def _get_windows(data: np.ndarray, window_length: int, step_size: float) -> np.ndarray | None:
        if data.size < window_length:
            return None
        nrows = ((data.size - window_length) // step_size) + 1
        n = data.strides[0]
        return np.lib.stride_tricks.as_strided(data, shape=(nrows, window_length), strides=(step_size * n, n))

    @staticmethod
    def get_derivatives(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Get the first and second derivatives of the data
        deriv = (data[1:-1] + data[2:]) / 2.0 - (data[1:-1] + data[:-2]) / 2.0
        second_deriv = data[2:] - 2 * data[1:-1] + data[:-2]
        return deriv, second_deriv

    @staticmethod
    def get_statistics(
        data: np.ndarray,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
        s_mean, s_median, s_min, s_max, s_max_amp, s_range, s_var, s_std_dev, s_abs_dev, s_rms, s_kurtosis, s_skew = (
            tools.signal_stats(data)
        )
        return (
            s_mean,
            s_median,
            s_min,
            s_max,
            s_max_amp,
            s_range,
            s_var,
            s_std_dev,
            s_abs_dev,
            s_rms,
            s_kurtosis,
            s_skew,
        )
