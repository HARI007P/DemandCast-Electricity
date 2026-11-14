"""
DemandCast - Train All Univariate Models Across Architectures and Sequence Lengths
Ensures every sequence length (6, 12, 24, 48, 72, 168) is fully trained and available
for SimpleRNN, Single LSTM, and Stacked LSTM under univariate feature mode.
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.environ["MPLCONFIGDIR"] = os.path.join(BASE_DIR, "artifacts", ".mpl_cache")

import logging
import pandas as pd
from src.train import train_single_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DemandCast.TrainUnivariate")

PROCESSED_FILE = os.path.join(BASE_DIR, "data", "processed", "featured_demand_data.parquet")
METRICS_DIR = os.path.join(BASE_DIR, "artifacts", "metrics")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")

def run():
    logger.info("Loading processed dataset...")
    df = pd.read_parquet(PROCESSED_FILE)

    architectures = ["single_lstm", "stacked_lstm", "simplernn"]
    seq_lengths = [6, 12, 24, 48, 72, 168]

    # Load existing metrics
    csv_path = os.path.join(METRICS_DIR, "all_experiments.csv")
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
    else:
        existing_df = pd.DataFrame()

    new_results = []

    for arch in architectures:
        for seq_len in seq_lengths:
            model_file = f"{arch}_univariate_{seq_len}h_h1.keras"
            model_path = os.path.join(MODELS_DIR, model_file)

            if os.path.exists(model_path):
                logger.info("Model %s already exists, skipping.", model_file)
                continue

            logger.info(">>> Training missing model: %s <<<", model_file)
            try:
                metrics, _ = train_single_experiment(
                    df=df,
                    model_arch=arch,
                    seq_length=seq_len,
                    feature_mode="univariate",
                    horizon=1,
                    epochs=12,
                    batch_size=64
                )
                new_results.append(metrics)
            except Exception as e:
                logger.error("Failed to train %s: %s", model_file, e, exc_info=True)

    if new_results:
        new_df = pd.DataFrame(new_results)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Drop duplicates if any
        combined_df.drop_duplicates(subset=["model", "seq_length", "feature_mode", "horizon"], keep="last", inplace=True)
        combined_df.to_csv(csv_path, index=False)
        combined_df.to_json(os.path.join(METRICS_DIR, "all_experiments.json"), orient="records", indent=2)
        logger.info("Successfully updated %s with %d newly trained univariate models!", csv_path, len(new_results))
    else:
        logger.info("All univariate models are already up to date.")

if __name__ == "__main__":
    run()
