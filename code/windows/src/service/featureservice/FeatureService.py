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

        # Don't forget to add the last series
        all_list.append(pd.DataFrame(current_list))

        # index 0 = 0, index 1 = 1, index 2 = 0, index 3 = 1, index 4 = 0, index 5 = 1, index 6 = 0
        return all_list[0], all_list[1], all_list[2], all_list[3], all_list[4], all_list[5], all_list[6]

    @staticmethod
    def get_windows(data: pd.DataFrame, window_length: int, step_size: float) -> np.ndarray:
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
    def get_statistics(data: np.ndarray) -> tuple[float, float, float, float, float, float, float, float, float, float]:
        avg = np.mean(data)
        sd = np.std(data)
        maxm = max(data)
        minm = min(data)
        s_mean, s_med, _, _, s_max, _, s_var, s_std_dev, s_abs_dev, _, s_kurtois, s_skew = tools.signal_stats(data)
        return avg, sd, maxm, minm, s_med, s_max, s_var, s_abs_dev, s_kurtois, s_skew
