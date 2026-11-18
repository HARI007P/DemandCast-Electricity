"""
DemandCast - FastAPI Backend API
Exposes:
- Dataset statistics & EDA summaries
- Model evaluation metrics across sequence lengths (6, 12, 24, 48, 72, 168)
- Test set actual-vs-predicted trajectory overlays
- Live interactive forecasting inference endpoint using trained Keras models
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.environ["MPLCONFIGDIR"] = os.path.join(BASE_DIR, "artifacts", ".mpl_cache")

# Crucial for Windows: import TensorFlow first to initialize OpenMP / runtime DLLs before NumPy/Pandas
import tensorflow as tf

import json
import joblib
import logging
import numpy as np
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DemandCast.API")

app = FastAPI(
    title="DemandCast API",
    description="LSTM & Recurrent Neural Network Electricity Demand Forecasting for Indian National Grid",
    version="1.0.0"
)

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")
FIGURES_DIR = os.path.join(BASE_DIR, "artifacts", "figures")
METRICS_DIR = os.path.join(BASE_DIR, "artifacts", "metrics")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "artifacts", "predictions")
DATA_FILE = os.path.join(BASE_DIR, "data", "processed", "featured_demand_data.parquet")

# Mount static figures
if os.path.exists(FIGURES_DIR):
    app.mount("/static/figures", StaticFiles(directory=FIGURES_DIR), name="figures")

# Mount production React build if available
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend", "dist")
if os.path.exists(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")
    from starlette.responses import FileResponse
    @app.get("/")
    def serve_frontend_root():
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))

# Cache loaded models
LOADED_MODELS = {}
LOADED_SCALERS = {}


def get_model(run_id: str):
    if run_id not in LOADED_MODELS:
        model_path = os.path.join(MODELS_DIR, f"{run_id}.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        logger.info("Loading model: %s", model_path)
        LOADED_MODELS[run_id] = tf.keras.models.load_model(model_path)
    return LOADED_MODELS[run_id]


def get_scalers(prefix: str = "multivariate"):
    if prefix not in LOADED_SCALERS:
        scaler_path = os.path.join(MODELS_DIR, f"{prefix}_scalers.joblib")
        if not os.path.exists(scaler_path):
            # fallback to default
            scaler_path = os.path.join(MODELS_DIR, "default_scalers.joblib")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scalers not found at {scaler_path}")
        LOADED_SCALERS[prefix] = joblib.load(scaler_path)
    return LOADED_SCALERS[prefix]


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "DemandCast Forecasting API",
        "dataset": "Indian Electricity Demand (Mendeley y58jknpgs8)",
        "framework": "TensorFlow / Keras",
        "tf_version": tf.__version__
    }


@app.get("/api/eda/summary")
def get_eda_summary():
    eda_path = os.path.join(METRICS_DIR, "eda_summary.json")
    if not os.path.exists(eda_path):
        raise HTTPException(status_code=404, detail="EDA summary not found. Run eda.py first.")
    with open(eda_path, "r") as f:
        data = json.load(f)
    return data


@app.get("/api/eda/timeseries")
def get_timeseries_sample(hours: int = Query(default=168, ge=24, le=1000)):
    if not os.path.exists(DATA_FILE):
        raise HTTPException(status_code=404, detail="Processed dataset not found.")
    df = pd.read_parquet(DATA_FILE)
    tail_df = df.tail(hours).copy()
    tail_df["timestamp"] = tail_df["timestamp"].astype(str)
    return tail_df.to_dict(orient="records")


@app.get("/api/experiments")
def get_experiment_metrics():
    metrics_path = os.path.join(METRICS_DIR, "all_experiments.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Experiment metrics not found. Training in progress or run train.py first.")
    with open(metrics_path, "r") as f:
        data = json.load(f)
    return data


@app.get("/api/predictions/{model_arch}/{seq_length}")
def get_predictions(model_arch: str, seq_length: int, horizon: int = 1, limit: int = 336):
    run_id = f"{model_arch}_multivariate_{seq_length}h_h{horizon}"
    pred_path = os.path.join(PREDICTIONS_DIR, f"pred_{run_id}.csv")
    if not os.path.exists(pred_path):
        raise HTTPException(status_code=404, detail=f"Predictions for {run_id} not found.")
    df = pd.read_csv(pred_path)
    df["timestamp"] = df["timestamp"].astype(str)
    tail_df = df.tail(limit).copy()
    tail_df["actual"] = tail_df["actual_demand_mw"]
    tail_df["predicted"] = tail_df["predicted_demand_mw"]
    return tail_df.to_dict(orient="records")


class ForecastRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_arch: str = "stacked_lstm"
    seq_length: int = 24
    feature_mode: str = "multivariate"
    horizon: int = 24
    offset_hours: int = 0  # 0 means latest available window


@app.post("/api/forecast")
def generate_forecast(req: ForecastRequest):
    """
    Performs multi-step forecasting: takes the lookback window of sequence length L,
    runs inference through the selected neural network, and returns forecasted hourly values.
    """
    run_id = f"{req.model_arch}_{req.feature_mode}_{req.seq_length}h_h1"
    try:
        model = get_model(run_id)
        scalers_payload = get_scalers(req.feature_mode)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    feature_scaler = scalers_payload["feature_scaler"]
    target_scaler = scalers_payload["target_scaler"]
    feature_cols = scalers_payload["feature_cols"]

    # Load recent dataset rows
    df = pd.read_parquet(DATA_FILE)
    if req.offset_hours > 0:
        window_df = df.iloc[-(req.seq_length + req.offset_hours):-req.offset_hours].copy()
    else:
        window_df = df.tail(req.seq_length).copy()

    if len(window_df) < req.seq_length:
        raise HTTPException(status_code=400, detail="Not enough history for lookback sequence.")

    # Scale window
    X_input = feature_scaler.transform(window_df[feature_cols].values)
    current_seq = np.expand_dims(X_input, axis=0)  # shape (1, seq_length, num_features)

    forecast_results = []
    last_timestamp = pd.to_datetime(window_df["timestamp"].iloc[-1])

    # Iterative autoregressive forecasting for multi-hour horizons
    simulated_seq = current_seq.copy()
    for step in range(1, req.horizon + 1):
        pred_scaled = model.predict(simulated_seq, verbose=0)
        pred_mw = float(target_scaler.inverse_transform(pred_scaled)[0, 0])

        step_time = last_timestamp + pd.Timedelta(hours=step)

        # Estimate confidence interval (± 1.96 * typical model RMSE ~ 3.5%)
        ci_lower = max(0.0, pred_mw * 0.965)
        ci_upper = pred_mw * 1.035

        forecast_results.append({
            "step": step,
            "timestamp": str(step_time),
            "forecast_mw": round(pred_mw, 2),
            "ci_lower_mw": round(ci_lower, 2),
            "ci_upper_mw": round(ci_upper, 2)
        })

        # Roll the sequence forward for the next step
        # Create next timestep feature vector
        next_vec = simulated_seq[0, -1, :].copy()
        # update target feature (index 0 is demand_mw)
        next_vec[0] = pred_scaled[0, 0]
        # update cyclical time features for step_time
        hour = step_time.hour
        dow = step_time.dayofweek
        if "hour_sin" in feature_cols:
            idx = feature_cols.index("hour_sin")
            next_vec[idx] = np.sin(2 * np.pi * hour / 24.0)
        if "hour_cos" in feature_cols:
            idx = feature_cols.index("hour_cos")
            next_vec[idx] = np.cos(2 * np.pi * hour / 24.0)
        if "day_sin" in feature_cols:
            idx = feature_cols.index("day_sin")
            next_vec[idx] = np.sin(2 * np.pi * dow / 7.0)
        if "day_cos" in feature_cols:
            idx = feature_cols.index("day_cos")
            next_vec[idx] = np.cos(2 * np.pi * dow / 7.0)

        # Append and slide window
        simulated_seq = np.concatenate([
            simulated_seq[:, 1:, :],
            np.expand_dims(np.expand_dims(next_vec, axis=0), axis=0)
        ], axis=1)

    return {
        "model": req.model_arch,
        "seq_length": req.seq_length,
        "feature_mode": req.feature_mode,
        "horizon": req.horizon,
        "history_start": str(window_df["timestamp"].iloc[0]),
        "history_end": str(window_df["timestamp"].iloc[-1]),
        "forecast": forecast_results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
