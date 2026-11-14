"""
DemandCast - Test API Endpoints
Verifies all FastAPI routes, serialization, forecasting inference, and error handling.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from starlette.testclient import TestClient
from api.main import app

def test_all_api_routes():
    client = TestClient(app)

    print("\n--- 1. Testing /api/health ---")
    res = client.get("/api/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    health_data = res.json()
    assert health_data["status"] == "healthy"
    print(f"Health check OK: {health_data}")

    print("\n--- 2. Testing /api/eda/summary ---")
    res = client.get("/api/eda/summary")
    assert res.status_code == 200, f"EDA summary failed: {res.text}"
    eda_data = res.json()
    assert "total_records" in eda_data
    assert eda_data["total_records"] == 12984
    print(f"EDA summary OK: total_records={eda_data['total_records']}, peak_demand={eda_data['demand']['max_mw']} MW")

    print("\n--- 3. Testing /api/eda/timeseries ---")
    res = client.get("/api/eda/timeseries?hours=72")
    assert res.status_code == 200, f"Timeseries failed: {res.text}"
    ts_data = res.json()
    assert len(ts_data) == 72
    assert "demand_mw" in ts_data[0]
    print(f"Timeseries OK: retrieved {len(ts_data)} hourly records.")

    print("\n--- 4. Testing /api/experiments ---")
    res = client.get("/api/experiments")
    assert res.status_code == 200, f"Experiments failed: {res.text}"
    exp_data = res.json()
    assert len(exp_data) >= 18
    print(f"Experiments OK: retrieved {len(exp_data)} benchmark configurations.")

    print("\n--- 5. Testing /api/predictions for models ---")
    for arch in ["stacked_lstm", "single_lstm", "simplernn"]:
        res = client.get(f"/api/predictions/{arch}/24?horizon=1&limit=24")
        assert res.status_code == 200, f"Prediction retrieval failed for {arch}: {res.text}"
        preds = res.json()
        assert len(preds) == 24
        assert "actual" in preds[0] and "predicted" in preds[0]
        print(f"Prediction retrieval OK for {arch}_multivariate_24h_h1 (sample: act={preds[0]['actual']}, pred={preds[0]['predicted']})")

    print("\n--- 6. Testing POST /api/forecast (Live Multi-Step Inference) ---")
    test_cases = [
        {"model_arch": "stacked_lstm", "seq_length": 24, "horizon": 24, "feature_mode": "multivariate"},
        {"model_arch": "single_lstm", "seq_length": 48, "horizon": 12, "feature_mode": "multivariate"},
        {"model_arch": "simplernn", "seq_length": 12, "horizon": 6, "feature_mode": "multivariate"},
        {"model_arch": "stacked_lstm", "seq_length": 168, "horizon": 24, "feature_mode": "multivariate"},
    ]

    for tc in test_cases:
        res = client.post("/api/forecast", json=tc)
        assert res.status_code == 200, f"Forecast failed for {tc}: {res.text}"
        forecast_res = res.json()
        assert "forecast" in forecast_res
        assert len(forecast_res["forecast"]) == tc["horizon"]
        first_step = forecast_res["forecast"][0]
        assert "forecast_mw" in first_step
        assert "ci_lower_mw" in first_step
        assert "ci_upper_mw" in first_step
        assert first_step["ci_lower_mw"] <= first_step["forecast_mw"] <= first_step["ci_upper_mw"]
        print(f"Forecast OK for {tc['model_arch']} (L={tc['seq_length']}, H={tc['horizon']}): Step 1 = {first_step['forecast_mw']} MW (CI: [{first_step['ci_lower_mw']}, {first_step['ci_upper_mw']}])")

    print("\n--- 7. Testing Frontend Root / ---")
    res = client.get("/")
    assert res.status_code == 200, f"Frontend root failed: {res.status_code}"
    assert "DemandCast" in res.text or "<div id=\"root\">" in res.text
    print("Frontend root serving OK!")

    print("\n==========================================")
    print("ALL API ENDPOINT & INFERENCE TESTS PASSED!")
    print("==========================================")

if __name__ == "__main__":
    test_all_api_routes()
