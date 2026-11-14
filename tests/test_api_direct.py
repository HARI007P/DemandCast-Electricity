"""
DemandCast - Direct API Logic & Forecasting Inference Test Suite
Tests FastAPI endpoint handlers directly to verify data access, model inference,
confidence bounds, and edge cases without network/socket dependencies.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.main import (
    health_check,
    get_eda_summary,
    get_timeseries_sample,
    get_experiment_metrics,
    get_predictions,
    generate_forecast,
    ForecastRequest
)

def test_api_handlers():
    print("\n--- 1. Testing health_check() ---")
    h = health_check()
    assert h["status"] == "healthy"
    print(f"Health check OK: {h}")

    print("\n--- 2. Testing get_eda_summary() ---")
    eda = get_eda_summary()
    assert eda["period"]["total_hours"] == 12984
    assert eda["demand_mw"]["max"] > 240000
    print(f"EDA Summary OK: Records={eda['period']['total_hours']}, Max Demand={eda['demand_mw']['max']} MW")

    print("\n--- 3. Testing get_timeseries_sample(hours=48) ---")
    ts = get_timeseries_sample(hours=48)
    assert len(ts) == 48
    assert "demand_mw" in ts[0]
    assert "solar_mw" in ts[0]
    print(f"Timeseries Sample OK: 48 hours returned (first={ts[0]['timestamp']}, last={ts[-1]['timestamp']})")

    print("\n--- 4. Testing get_experiment_metrics() ---")
    exps = get_experiment_metrics()
    assert len(exps) >= 18
    print(f"Experiment Metrics OK: {len(exps)} models evaluated.")

    print("\n--- 5. Testing get_predictions() across architectures ---")
    for arch in ["stacked_lstm", "single_lstm", "simplernn"]:
        preds = get_predictions(model_arch=arch, seq_length=24, horizon=1, limit=12)
        assert len(preds) == 12
        assert "actual" in preds[0] and "predicted" in preds[0]
        print(f"Predictions OK for {arch}_multivariate_24h_h1 (Sample Act={preds[0]['actual']} MW, Pred={preds[0]['predicted']} MW)")

    print("\n--- 6. Testing generate_forecast() multi-step inference ---")
    test_cases = [
        ("stacked_lstm", 24, 24, "multivariate"),
        ("single_lstm", 48, 12, "multivariate"),
        ("simplernn", 12, 6, "multivariate"),
        ("stacked_lstm", 168, 24, "multivariate"),
    ]

    for arch, seq_len, horizon, fmode in test_cases:
        req = ForecastRequest(
            model_arch=arch,
            seq_length=seq_len,
            feature_mode=fmode,
            horizon=horizon,
            offset_hours=0
        )
        res = generate_forecast(req)
        assert res["model"] == arch
        assert res["seq_length"] == seq_len
        assert len(res["forecast"]) == horizon
        step1 = res["forecast"][0]
        step_last = res["forecast"][-1]
        assert step1["ci_lower_mw"] <= step1["forecast_mw"] <= step1["ci_upper_mw"]
        assert step_last["ci_lower_mw"] <= step_last["forecast_mw"] <= step_last["ci_upper_mw"]
        print(f"Inference OK: {arch} (L={seq_len}h, H={horizon}h) -> Step 1: {step1['forecast_mw']} MW, Step {horizon}: {step_last['forecast_mw']} MW")

    print("\n=======================================================")
    print("ALL API ENDPOINT & MODEL INFERENCE TESTS COMPLETED 100%!")
    print("=======================================================")

if __name__ == "__main__":
    test_api_handlers()
