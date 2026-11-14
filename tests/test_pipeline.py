"""
DemandCast - Automated Test Suite
Verifies data pipeline, continuity, sequence generation, metrics, and models.
"""

import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.dataset_generator import TimeSeriesDatasetBuilder, UNIVARIATE_FEATURES, MULTIVARIATE_FEATURES
from src.evaluate import compute_metrics
from src.models import get_model


def test_processed_data_integrity():
    clean_path = os.path.join(BASE_DIR, "data", "processed", "cleaned_demand_data.parquet")
    assert os.path.exists(clean_path), "Cleaned parquet data file must exist"

    df = pd.read_parquet(clean_path)
    assert len(df) == 12984, f"Expected 12,984 hourly readings, got {len(df)}"
    assert not df.isna().any().any(), "Data must have zero null values"
    assert (df["solar_mw"] >= 0).all(), "Solar generation must not have negative values"
    assert (df["wind_mw"] >= 0).all(), "Wind generation must not have negative values"

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    assert df["timestamp"].is_monotonic_increasing, "Timestamps must be strictly chronological"
    diffs = df["timestamp"].diff().dropna()
    assert (diffs == pd.Timedelta(hours=1)).all(), "Every step must be exactly 1 hour"


def test_feature_engineering_bounds():
    feat_path = os.path.join(BASE_DIR, "data", "processed", "featured_demand_data.parquet")
    assert os.path.exists(feat_path), "Featured data must exist"

    df = pd.read_parquet(feat_path)
    for col in ["hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"]:
        assert col in df.columns, f"Missing cyclical feature {col}"
        assert df[col].min() >= -1.0 - 1e-5, f"{col} minimum out of bounds"
        assert df[col].max() <= 1.0 + 1e-5, f"{col} maximum out of bounds"


def test_sliding_window_shapes():
    df = pd.read_parquet(os.path.join(BASE_DIR, "data", "processed", "featured_demand_data.parquet"))
    builder = TimeSeriesDatasetBuilder(feature_cols=MULTIVARIATE_FEATURES)

    for seq_len in [6, 12, 24]:
        dataset = builder.build_datasets_for_seq_length(df, seq_len, horizon=1)
        assert dataset["X_train"].shape[1] == seq_len, f"Sequence length mismatch for L={seq_len}"
        assert dataset["X_train"].shape[2] == len(MULTIVARIATE_FEATURES), "Feature count mismatch"
        assert len(dataset["X_train"]) == len(dataset["y_train"]), "X and y count must match"
        assert len(dataset["X_test"]) == len(dataset["y_test_unscaled"]), "Test set lengths must align"


def test_metrics_calculation():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 300.0, 420.0])

    m = compute_metrics(y_true, y_pred)
    assert m["MAE_MW"] == 10.0
    assert m["MAPE_percent"] == 5.0
    assert m["R2_Score"] > 0.98


def test_model_instantiation():
    for arch in ["simplernn", "single_lstm", "stacked_lstm"]:
        model = get_model(arch, input_shape=(24, 10))
        assert model is not None
        assert model.input_shape == (None, 24, 10)
        assert model.output_shape == (None, 1)


if __name__ == "__main__":
    test_processed_data_integrity()
    test_feature_engineering_bounds()
    test_sliding_window_shapes()
    test_metrics_calculation()
    test_model_instantiation()
    print("All unit and integration tests passed successfully!")
