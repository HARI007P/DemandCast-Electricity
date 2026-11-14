"""
DemandCast - Data Cleaning and Preprocessing Pipeline
Processes raw Indian electricity demand data from Mendeley repository:
- Cleans metadata and summary footer rows
- Standardizes hourly datetime stamps
- Validates data integrity (monotonicity, duplicates, missing timestamps)
- Rectifies sensor drift (e.g. negative night solar readings)
- Exports clean tabular data for EDA, feature engineering, and model training.
"""

import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DemandCast.Preprocess")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Jan2024_June2025.xlsx")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
CLEANED_CSV_PATH = os.path.join(PROCESSED_DIR, "cleaned_demand_data.csv")
CLEANED_PARQUET_PATH = os.path.join(PROCESSED_DIR, "cleaned_demand_data.parquet")


def load_raw_dataset(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Loads raw Excel file from the Mendeley dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at {filepath}. Run download_dataset.py first.")
    logger.info("Loading raw dataset from %s...", filepath)
    df = pd.read_excel(filepath, sheet_name="Report")
    logger.info("Raw dataset shape: %s", df.shape)
    return df


def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw data, removes non-temporal footer rows, standardizes types and checks continuity.
    """
    logger.info("Original columns: %s", df.columns.tolist())

    # Step 1: Parse timestamps with dayfirst=True to handle Indian DD-MM-YYYY format
    ts_parsed = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    valid_mask = ts_parsed.notna()
    invalid_count = (~valid_mask).sum()
    logger.info("Identified %d invalid/footer rows (e.g., summary statistics) to remove.", invalid_count)

    clean_df = df[valid_mask].copy()
    clean_df["timestamp"] = ts_parsed[valid_mask]

    # Step 2: Standardize column names
    col_mapping = {
        "Demand (MW)": "demand_mw",
        "Wind (MW)": "wind_mw",
        "Solar (MW)": "solar_mw",
        "Total Generation (MW)": "total_gen_mw"
    }
    clean_df = clean_df.rename(columns=col_mapping)
    required_cols = ["timestamp", "demand_mw", "wind_mw", "solar_mw", "total_gen_mw"]
    clean_df = clean_df[required_cols]

    # Step 3: Convert numeric columns
    for col in ["demand_mw", "wind_mw", "solar_mw", "total_gen_mw"]:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # Step 4: Verify physical boundaries and sensor noise
    # Solar cannot physically be negative (sensor baseline drift at night)
    neg_solar = (clean_df["solar_mw"] < 0).sum()
    if neg_solar > 0:
        logger.info("Clipping %d negative solar values (sensor zero-drift) to 0.0 MW", neg_solar)
        clean_df["solar_mw"] = clean_df["solar_mw"].clip(lower=0.0)

    neg_wind = (clean_df["wind_mw"] < 0).sum()
    if neg_wind > 0:
        logger.info("Clipping %d negative wind values to 0.0 MW", neg_wind)
        clean_df["wind_mw"] = clean_df["wind_mw"].clip(lower=0.0)

    # Step 5: Chronological sorting and continuity audit
    clean_df = clean_df.sort_values("timestamp").reset_index(drop=True)
    duplicates = clean_df["timestamp"].duplicated().sum()
    if duplicates > 0:
        logger.warning("Found %d duplicate timestamps. Dropping duplicates keeping first.", duplicates)
        clean_df = clean_df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)

    # Time delta check
    time_diffs = clean_df["timestamp"].diff().dropna()
    step_counts = (time_diffs.dt.total_seconds() / 3600.0).value_counts()
    logger.info("Timestamp step distribution (in hours):\n%s", step_counts.to_string())

    # Step 6: Missing values audit
    null_summary = clean_df.isna().sum()
    logger.info("Missing value audit across columns:\n%s", null_summary.to_string())
    if clean_df.isna().any().any():
        logger.info("Imputing any minor NaNs using time-based linear interpolation...")
        clean_df = clean_df.interpolate(method="linear").bfill().ffill()

    logger.info("Cleaned dataset details:")
    logger.info("Start Time : %s", clean_df["timestamp"].min())
    logger.info("End Time   : %s", clean_df["timestamp"].max())
    logger.info("Total Rows : %d hours", len(clean_df))
    logger.info("Demand stats (MW) - Min: %.1f, Mean: %.1f, Max: %.1f, Std: %.1f",
                clean_df["demand_mw"].min(),
                clean_df["demand_mw"].mean(),
                clean_df["demand_mw"].max(),
                clean_df["demand_mw"].std())

    return clean_df


def save_processed_data(df: pd.DataFrame, csv_path: str = CLEANED_CSV_PATH, parquet_path: str = CLEANED_PARQUET_PATH):
    """Saves cleaned DataFrame to CSV and Parquet formats."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    logger.info("Successfully exported clean dataset to:")
    logger.info(" - %s (%d rows)", csv_path, len(df))
    logger.info(" - %s", parquet_path)


def run_pipeline():
    raw_df = load_raw_dataset()
    clean_df = clean_and_validate(raw_df)
    save_processed_data(clean_df)
    return clean_df


if __name__ == "__main__":
    run_pipeline()
