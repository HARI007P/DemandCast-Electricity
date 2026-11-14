"""
DemandCast - Time-Series Sequence & Dataset Generator
Handles:
- Strict chronological dataset partitioning (Train / Val / Test)
- Data leakage prevention: Scalers fit strictly on Training partition only
- Sliding window sequence creation for arbitrary window lengths (6, 12, 24, 48, 72, 168 hours)
- Multi-feature and Univariate sequence generation
- Scaler serialization for FastAPI inference and evaluation
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DemandCast.SeqGen")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_FILE = os.path.join(BASE_DIR, "data", "processed", "featured_demand_data.parquet")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")

# Standard feature sets
UNIVARIATE_FEATURES = ["demand_mw"]
MULTIVARIATE_FEATURES = [
    "demand_mw",
    "solar_mw",
    "wind_mw",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    "is_weekend"
]
EXTENDED_FEATURES = MULTIVARIATE_FEATURES + [
    "renewable_penetration",
    "demand_roll_mean_6h",
    "demand_roll_std_6h",
    "demand_roll_mean_24h",
    "demand_roll_std_24h"
]


class TimeSeriesDatasetBuilder:
    def __init__(self,
                 feature_cols: Optional[List[str]] = None,
                 target_col: str = "demand_mw",
                 train_ratio: float = 0.70,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15):
        self.feature_cols = feature_cols if feature_cols is not None else MULTIVARIATE_FEATURES
        self.target_col = target_col
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False

        self.split_indices = {}

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Chronologically splits dataframe into Train, Validation, and Test sets.
        Strictly preserves temporal sequence to eliminate lookahead bias / data leakage.
        """
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_df = df.iloc[:train_end].copy().reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
        test_df = df.iloc[val_end:].copy().reset_index(drop=True)

        self.split_indices = {
            "train": (0, train_end, str(train_df["timestamp"].iloc[0]), str(train_df["timestamp"].iloc[-1])),
            "val": (train_end, val_end, str(val_df["timestamp"].iloc[0]), str(val_df["timestamp"].iloc[-1])),
            "test": (val_end, n, str(test_df["timestamp"].iloc[0]), str(test_df["timestamp"].iloc[-1]))
        }

        logger.info("Chronological split completed:")
        logger.info(" - Train: %d samples (%s -> %s)", len(train_df), self.split_indices["train"][2], self.split_indices["train"][3])
        logger.info(" - Val  : %d samples (%s -> %s)", len(val_df), self.split_indices["val"][2], self.split_indices["val"][3])
        logger.info(" - Test : %d samples (%s -> %s)", len(test_df), self.split_indices["test"][2], self.split_indices["test"][3])

        return train_df, val_df, test_df

    def fit_scalers(self, train_df: pd.DataFrame):
        """
        Fits scalers EXCLUSIVELY on the training partition.
        """
        self.feature_scaler.fit(train_df[self.feature_cols].values)
        self.target_scaler.fit(train_df[[self.target_col]].values)
        self.is_fitted = True
        logger.info("Scalers fitted strictly on training partition with %d features.", len(self.feature_cols))

    def save_scalers(self, save_dir: str = MODELS_DIR, prefix: str = "default"):
        os.makedirs(save_dir, exist_ok=True)
        scaler_payload = {
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "split_indices": self.split_indices
        }
        path = os.path.join(save_dir, f"{prefix}_scalers.joblib")
        joblib.dump(scaler_payload, path)
        logger.info("Saved scalers payload to: %s", path)
        return path

    @staticmethod
    def load_scalers(filepath: str):
        payload = joblib.load(filepath)
        return payload

    def create_sliding_sequences(self,
                                 features: np.ndarray,
                                 targets: np.ndarray,
                                 seq_length: int,
                                 horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates sliding window sequences:
        X[i] = features[i : i + seq_length]
        y[i] = targets[i + seq_length + horizon - 1]
        """
        num_samples = len(features) - seq_length - horizon + 1
        if num_samples <= 0:
            raise ValueError(f"Dataset length ({len(features)}) is too small for seq_length={seq_length} + horizon={horizon}")

        X = np.empty((num_samples, seq_length, features.shape[1]), dtype=np.float32)
        y = np.empty((num_samples,), dtype=np.float32)

        for i in range(num_samples):
            X[i] = features[i: i + seq_length]
            y[i] = targets[i + seq_length + horizon - 1, 0]

        return X, y

    def build_datasets_for_seq_length(self,
                                     df: pd.DataFrame,
                                     seq_length: int,
                                     horizon: int = 1):
        """
        End-to-end dataset builder for a given sequence length.
        Maintains continuity at partition boundaries by borrowing lookback context from previous split.
        """
        train_df, val_df, test_df = self.split_data(df)

        if not self.is_fitted:
            self.fit_scalers(train_df)

        # Scale features and targets
        train_f_scaled = self.feature_scaler.transform(train_df[self.feature_cols].values)
        train_t_scaled = self.target_scaler.transform(train_df[[self.target_col]].values)

        val_f_scaled = self.feature_scaler.transform(val_df[self.feature_cols].values)
        val_t_scaled = self.target_scaler.transform(val_df[[self.target_col]].values)

        test_f_scaled = self.feature_scaler.transform(test_df[self.feature_cols].values)
        test_t_scaled = self.target_scaler.transform(test_df[[self.target_col]].values)

        # Create training sequences purely from train data
        X_train, y_train = self.create_sliding_sequences(train_f_scaled, train_t_scaled, seq_length, horizon)

        # For validation, prepend the last seq_length timesteps of train to allow full prediction coverage
        val_f_padded = np.vstack([train_f_scaled[-seq_length:], val_f_scaled])
        val_t_padded = np.vstack([train_t_scaled[-seq_length:], val_t_scaled])
        X_val, y_val = self.create_sliding_sequences(val_f_padded, val_t_padded, seq_length, horizon)

        # For test, prepend the last seq_length timesteps of val
        test_f_padded = np.vstack([val_f_scaled[-seq_length:], test_f_scaled])
        test_t_padded = np.vstack([val_t_scaled[-seq_length:], test_t_scaled])
        X_test, y_test = self.create_sliding_sequences(test_f_padded, test_t_padded, seq_length, horizon)

        # Keep true unscaled targets for realistic test evaluation
        y_test_unscaled = test_df[self.target_col].values[:len(y_test)]
        test_timestamps = test_df["timestamp"].iloc[:len(y_test)].values

        logger.info("Constructed sliding sequences (L=%d, H=%d):", seq_length, horizon)
        logger.info(" - X_train: %s, y_train: %s", X_train.shape, y_train.shape)
        logger.info(" - X_val  : %s, y_val  : %s", X_val.shape, y_val.shape)
        logger.info(" - X_test : %s, y_test : %s", X_test.shape, y_test.shape)

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "y_test_unscaled": y_test_unscaled,
            "test_timestamps": test_timestamps,
            "target_scaler": self.target_scaler,
            "feature_scaler": self.feature_scaler,
            "feature_cols": self.feature_cols
        }


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED_FILE)
    builder = TimeSeriesDatasetBuilder()
    data_dict = builder.build_datasets_for_seq_length(df, seq_length=24)
    builder.save_scalers(prefix="multivariate")
