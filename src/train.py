"""
DemandCast - Deep Learning Training & Experimentation Pipeline
Executes systematic empirical evaluations across:
- Architectures: SimpleRNN, Single-layer LSTM, Two-layer Stacked LSTM
- Sequence Lookback Windows: 6, 12, 24, 48, 72, and 168 hours
- Feature Sets: Multivariate (Demand + Solar + Wind + Cyclical Time) vs Univariate
- Forecast Horizons: 1-hour, 6-hour, and 24-hour ahead forecasting
- Complete evaluation with MAE, RMSE, MAPE, R², loss curves, and actual-vs-predicted graphs.
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.environ["MPLCONFIGDIR"] = os.path.join(BASE_DIR, "artifacts", ".mpl_cache")

import json
import time
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.dataset_generator import TimeSeriesDatasetBuilder, UNIVARIATE_FEATURES, MULTIVARIATE_FEATURES
from src.models import get_model
from src.evaluate import compute_metrics, plot_actual_vs_predicted, plot_loss_curves, plot_sequence_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DemandCast.Train")

PROCESSED_FILE = os.path.join(BASE_DIR, "data", "processed", "featured_demand_data.parquet")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")
FIGURES_DIR = os.path.join(BASE_DIR, "artifacts", "figures")
METRICS_DIR = os.path.join(BASE_DIR, "artifacts", "metrics")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "artifacts", "predictions")


def train_single_experiment(df: pd.DataFrame,
                             model_arch: str,
                             seq_length: int,
                             feature_mode: str = "multivariate",
                             horizon: int = 1,
                             epochs: int = 25,
                             batch_size: int = 64):
    """
    Trains a single model instance for a specific architecture and sequence length.
    Strictly avoids data leakage and tracks full telemetry.
    """
    logger.info("=================================================================")
    logger.info("Starting Run: Model=%s | SeqLen=%dh | Features=%s | Horizon=%dh",
                model_arch, seq_length, feature_mode, horizon)
    logger.info("=================================================================")

    feature_cols = UNIVARIATE_FEATURES if feature_mode == "univariate" else MULTIVARIATE_FEATURES

    builder = TimeSeriesDatasetBuilder(feature_cols=feature_cols)
    dataset = builder.build_datasets_for_seq_length(df, seq_length=seq_length, horizon=horizon)

    X_train, y_train = dataset["X_train"], dataset["y_train"]
    X_val, y_val = dataset["X_val"], dataset["y_val"]
    X_test, y_test = dataset["X_test"], dataset["y_test"]
    y_test_unscaled = dataset["y_test_unscaled"]
    test_timestamps = dataset["test_timestamps"]
    target_scaler = dataset["target_scaler"]

    # Save scalers for this feature mode
    builder.save_scalers(save_dir=MODELS_DIR, prefix=feature_mode)

    # Instantiate model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = get_model(model_arch, input_shape=input_shape)

    run_id = f"{model_arch}_{feature_mode}_{seq_length}h_h{horizon}"
    model_save_path = os.path.join(MODELS_DIR, f"{run_id}.keras")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0),
        ModelCheckpoint(filepath=model_save_path, monitor="val_loss", save_best_only=True, verbose=0)
    ]

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    training_time = round(time.time() - start_time, 2)
    logger.info("Training concluded in %.2fs across %d epochs.", training_time, len(history.history["loss"]))

    # Test set evaluation
    logger.info("Evaluating on test set (%d samples)...", len(X_test))
    y_pred_scaled = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred_unscaled = target_scaler.inverse_transform(y_pred_scaled).ravel()

    # Calculate metrics in MW
    metrics = compute_metrics(y_test_unscaled, y_pred_unscaled)
    metrics.update({
        "model": model_arch,
        "seq_length": seq_length,
        "feature_mode": feature_mode,
        "horizon": horizon,
        "epochs_trained": len(history.history["loss"]),
        "training_time_sec": training_time,
        "total_parameters": int(model.count_params()),
        "model_file": f"{run_id}.keras"
    })
    logger.info("Test Metrics: MAE=%.1f MW | RMSE=%.1f MW | MAPE=%.2f%% | R²=%.4f",
                metrics["MAE_MW"], metrics["RMSE_MW"], metrics["MAPE_percent"], metrics["R2_Score"])

    # Save loss curve plot
    loss_plot_path = os.path.join(FIGURES_DIR, f"loss_{run_id}.png")
    plot_loss_curves(history, model_arch, seq_length, loss_plot_path)

    # Save actual vs predicted plot
    pred_plot_path = os.path.join(FIGURES_DIR, f"forecast_{run_id}.png")
    plot_actual_vs_predicted(test_timestamps, y_test_unscaled, y_pred_unscaled, model_arch, seq_length, pred_plot_path)

    # Export test predictions dataframe for the interactive dashboard
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    pred_df = pd.DataFrame({
        "timestamp": test_timestamps,
        "actual_demand_mw": y_test_unscaled,
        "predicted_demand_mw": y_pred_unscaled,
        "error_mw": y_test_unscaled - y_pred_unscaled
    })
    pred_csv_path = os.path.join(PREDICTIONS_DIR, f"pred_{run_id}.csv")
    pred_df.to_csv(pred_csv_path, index=False)

    return metrics, pred_df


def run_all_experiments(seq_lengths: list = None,
                        models: list = None,
                        epochs: int = 20,
                        batch_size: int = 64):
    """
    Runs systematic comparative experiments across all sequence lengths, architectures,
    and feature configurations.
    """
    if seq_lengths is None:
        seq_lengths = [6, 12, 24, 48, 72, 168]
    if models is None:
        models = ["simplernn", "single_lstm", "stacked_lstm"]

    logger.info("Loading processed dataset from %s...", PROCESSED_FILE)
    df = pd.read_parquet(PROCESSED_FILE)

    all_results = []

    # 1. Main comparison across Sequence Lengths & Architectures (Multivariate, 1-step ahead)
    for model_name in models:
        for seq_len in seq_lengths:
            try:
                res, _ = train_single_experiment(
                    df,
                    model_arch=model_name,
                    seq_length=seq_len,
                    feature_mode="multivariate",
                    horizon=1,
                    epochs=epochs,
                    batch_size=batch_size
                )
                all_results.append(res)
            except Exception as e:
                logger.error("Error running %s (L=%d): %s", model_name, seq_len, e, exc_info=True)

    # 2. Feature Ablation: Univariate vs Multivariate (Lookback 24h)
    for model_name in ["single_lstm", "stacked_lstm"]:
        try:
            res_uni, _ = train_single_experiment(
                df,
                model_arch=model_name,
                seq_length=24,
                feature_mode="univariate",
                horizon=1,
                epochs=epochs,
                batch_size=batch_size
            )
            all_results.append(res_uni)
        except Exception as e:
            logger.error("Error running univariate ablation %s: %s", model_name, e)

    # 3. Horizon Ablation: Multi-step forecasting horizons (6h and 24h ahead with Stacked LSTM 24h lookback)
    for horizon in [6, 24]:
        try:
            res_h, _ = train_single_experiment(
                df,
                model_arch="stacked_lstm",
                seq_length=24,
                feature_mode="multivariate",
                horizon=horizon,
                epochs=epochs,
                batch_size=batch_size
            )
            all_results.append(res_h)
        except Exception as e:
            logger.error("Error running horizon ablation H=%d: %s", horizon, e)

    # Consolidate results
    results_df = pd.DataFrame(all_results)
    os.makedirs(METRICS_DIR, exist_ok=True)

    csv_path = os.path.join(METRICS_DIR, "all_experiments.csv")
    json_path = os.path.join(METRICS_DIR, "all_experiments.json")
    results_df.to_csv(csv_path, index=False)
    results_df.to_json(json_path, orient="records", indent=2)

    logger.info("Saved all experiment results to:")
    logger.info(" - %s", csv_path)
    logger.info(" - %s", json_path)

    # Generate sequence length comparison chart for multivariate runs
    multivariate_results = results_df[(results_df["feature_mode"] == "multivariate") & (results_df["horizon"] == 1)]
    if not multivariate_results.empty:
        comp_chart_path = os.path.join(FIGURES_DIR, "model_comparison_sequence_lengths.png")
        plot_sequence_comparison(multivariate_results, comp_chart_path)

    logger.info("All experiments completed successfully.")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DemandCast Model Training")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seq_lengths", nargs="+", type=int, default=[6, 12, 24, 48, 72, 168], help="Sequence lengths")
    parser.add_argument("--models", nargs="+", type=str, default=["simplernn", "single_lstm", "stacked_lstm"], help="Models")
    args = parser.parse_args()

    run_all_experiments(
        seq_lengths=args.seq_lengths,
        models=args.models,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
