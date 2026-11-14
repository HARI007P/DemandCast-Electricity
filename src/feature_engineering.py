"""
DemandCast - Feature Engineering Module
Transforms cleaned time-series data into feature-rich representations:
- Cyclical temporal embeddings (Hour, Day of Week, Month via Sin/Cos transformations)
- Calendar flags (Weekend)
- Exogenous renewable generation features (Solar MW, Wind MW, Renewable Penetration Ratio)
- Rolling window statistical moments (6-hour and 24-hour mean, std)
- Lag features (1h, 2h, 24h, 168h) for baseline/analysis
"""

import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DemandCast.FeatureEng")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_demand_data.parquet")
FEATURE_ENGINEERED_PATH = os.path.join(BASE_DIR, "data", "processed", "featured_demand_data.parquet")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cyclical time features, calendar indicators, renewable indicators, and rolling statistics.
    """
    logger.info("Starting feature engineering on dataframe of shape %s...", df.shape)
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # 1. Temporal cyclical encodings
    # Hour of day (0-23) -> cyclical sin and cos
    hour = data["timestamp"].dt.hour
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # Day of week (0=Monday, 6=Sunday) -> cyclical sin and cos
    dow = data["timestamp"].dt.dayofweek
    data["day_sin"] = np.sin(2 * np.pi * dow / 7.0)
    data["day_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # Month of year (1-12) -> cyclical sin and cos
    month = data["timestamp"].dt.month
    data["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)

    # Weekend binary indicator
    data["is_weekend"] = dow.isin([5, 6]).astype(float)

    # 2. Exogenous Renewable Generation Metrics
    # Total renewable generation
    data["total_renewable_mw"] = data["solar_mw"] + data["wind_mw"]
    # Renewable penetration ratio (avoid divide by zero)
    data["renewable_penetration"] = (data["total_renewable_mw"] / (data["demand_mw"] + 1e-5)).clip(0.0, 1.0)

    # 3. Rolling window statistics on demand (purely backward-looking to prevent future data leakage)
    data["demand_roll_mean_6h"] = data["demand_mw"].rolling(window=6, min_periods=1).mean()
    data["demand_roll_std_6h"] = data["demand_mw"].rolling(window=6, min_periods=1).std().fillna(0.0)
    data["demand_roll_mean_24h"] = data["demand_mw"].rolling(window=24, min_periods=1).mean()
    data["demand_roll_std_24h"] = data["demand_mw"].rolling(window=24, min_periods=1).std().fillna(0.0)

    # 4. Key temporal lag signals (1h, 24h, 168h)
    data["demand_lag_1h"] = data["demand_mw"].shift(1).bfill()
    data["demand_lag_24h"] = data["demand_mw"].shift(24).bfill()
    data["demand_lag_168h"] = data["demand_mw"].shift(168).bfill()

    logger.info("Feature engineering complete. Output shape: %s", data.shape)
    logger.info("Engineered feature columns: %s", [c for c in data.columns if c not in df.columns])
    return data


def run_feature_pipeline():
    if not os.path.exists(PROCESSED_CSV_PATH):
        raise FileNotFoundError(f"Cleaned dataset not found at {PROCESSED_CSV_PATH}. Run preprocess.py first.")
    df = pd.read_parquet(PROCESSED_CSV_PATH)
    featured_df = engineer_features(df)
    featured_df.to_parquet(FEATURE_ENGINEERED_PATH, index=False)
    featured_df.to_csv(FEATURE_ENGINEERED_PATH.replace(".parquet", ".csv"), index=False)
    logger.info("Saved featured dataset to %s", FEATURE_ENGINEERED_PATH)
    return featured_df


if __name__ == "__main__":
    run_feature_pipeline()
