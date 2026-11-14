"""
DemandCast - Model Evaluation & Performance Metrics
Computes:
- Mean Absolute Error (MAE) in MW
- Root Mean Squared Error (RMSE) in MW
- Mean Absolute Percentage Error (MAPE) in %
- Coefficient of Determination (R²)
- Generates high-fidelity visual comparisons, residual distributions, and test vs predicted plots.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DemandCast.Evaluate")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "artifacts", "figures")
METRICS_DIR = os.path.join(BASE_DIR, "artifacts", "metrics")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression time-series forecasting metrics.
    All inputs are expected in original physical units (Megawatts - MW).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))
    mape = float(np.mean(np.abs(errors / (y_true + 1e-6))) * 100.0)

    # R^2 Score
    ss_res = np.sum(sq_errors)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-10)))

    nrmse = float(rmse / (np.mean(y_true) + 1e-10))
    mbe = float(np.mean(errors))  # Mean Bias Error

    return {
        "MAE_MW": round(mae, 2),
        "RMSE_MW": round(rmse, 2),
        "MAPE_percent": round(mape, 3),
        "R2_Score": round(r2, 4),
        "NRMSE": round(nrmse, 4),
        "MBE_MW": round(mbe, 2),
        "max_error_MW": round(float(np.max(abs_errors)), 2)
    }


def plot_actual_vs_predicted(timestamps,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             model_name: str,
                             seq_length: int,
                             save_path: str = None,
                             zoom_days: int = 14):
    """
    Plots full test set comparison and a zoomed-in 14-day view to examine diurnal peak tracking.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

    # Panel 1: Full Test Window
    ax1.plot(timestamps, y_true, label="Actual Grid Demand (MW)", color="#1e293b", linewidth=1.4, alpha=0.85)
    ax1.plot(timestamps, y_pred, label=f"Predicted ({model_name})", color="#2563eb", linewidth=1.2, linestyle="--", alpha=0.9)
    ax1.set_title(f"Test Set Demand Forecasting: Actual vs Predicted [{model_name} | Lookback: {seq_length}h]")
    ax1.set_ylabel("Demand (MW)")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Zoomed view (last `zoom_days` days)
    zoom_pts = min(len(timestamps), zoom_days * 24)
    ax2.plot(timestamps[-zoom_pts:], y_true[-zoom_pts:], label="Actual Demand (MW)", color="#0f172a", linewidth=2.0, marker="o", markersize=3)
    ax2.plot(timestamps[-zoom_pts:], y_pred[-zoom_pts:], label="Predicted Demand (MW)", color="#dc2626", linewidth=1.8, linestyle="--", marker="x", markersize=4)
    ax2.set_title(f"Detailed 14-Day Trajectory Tracking & Peak Alignment (Last {zoom_days} Days)")
    ax2.set_ylabel("Demand (MW)")
    ax2.set_xlabel("Timestamp")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved forecast comparison plot to: %s", save_path)
    plt.close()


def plot_loss_curves(history, model_name: str, seq_length: int, save_path: str):
    """
    Plots training vs validation loss over training epochs.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history.history["loss"]) + 1)

    ax.plot(epochs, history.history["loss"], label="Training Loss (MSE)", color="#2563eb", linewidth=2.0)
    if "val_loss" in history.history:
        ax.plot(epochs, history.history["val_loss"], label="Validation Loss (MSE)", color="#ef4444", linewidth=2.0, linestyle="--")

    ax.set_title(f"Convergence Loss Curve [{model_name} | Lookback: {seq_length}h]")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error (Scaled)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info("Saved loss curve plot to: %s", save_path)
    plt.close()


def plot_sequence_comparison(results_df: pd.DataFrame, save_path: str):
    """
    Plots multi-metric comparison across sequence lengths and model architectures.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [("MAE_MW", "MAE (MW)", axes[0, 0]),
               ("RMSE_MW", "RMSE (MW)", axes[0, 1]),
               ("MAPE_percent", "MAPE (%)", axes[1, 0]),
               ("R2_Score", "R² Score", axes[1, 1])]

    models = results_df["model"].unique()
    markers = ["o", "s", "^", "D"]
    colors = ["#2563eb", "#059669", "#dc2626", "#7c3aed"]

    for col, label, ax in metrics:
        for idx, m in enumerate(models):
            sub = results_df[results_df["model"] == m].sort_values("seq_length")
            ax.plot(sub["seq_length"], sub[col], marker=markers[idx % len(markers)],
                    color=colors[idx % len(colors)], linewidth=2.0, label=m)
        ax.set_title(f"Model Comparison: {label} vs Sequence Lookback Length")
        ax.set_xlabel("Lookback Sequence Length (Hours)")
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info("Saved sequence length comparison plot to: %s", save_path)
    plt.close()
