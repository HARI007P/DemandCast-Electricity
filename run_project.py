"""
DemandCast - Master Project CLI & Server Orchestrator
Usage:
    python run_project.py --pipeline     # Runs data download, cleaning, EDA, feature engineering, and model training
    python run_project.py --server       # Launches FastAPI server serving REST API + React Dashboard
    python run_project.py --eval         # Prints consolidated benchmark summary table
    python run_project.py --all          # Runs full pipeline and launches server
"""

import os
import sys
import argparse
import logging
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DemandCast.Master")


def run_pipeline():
    logger.info("=== STEP 1: DOWNLOADING DATASET (MENDELEY y58jknpgs8) ===")
    from src.download_dataset import download_mendeley_dataset
    download_mendeley_dataset()

    logger.info("=== STEP 2: DATA CLEANING & CONTINUITY AUDIT ===")
    from src.preprocess import run_pipeline as run_preprocess
    run_preprocess()

    logger.info("=== STEP 3: EXPLORATORY DATA ANALYSIS (EDA) & PLOTS ===")
    from src.eda import run_eda
    run_eda()

    logger.info("=== STEP 4: FEATURE ENGINEERING (CYCLICAL, RENEWABLE, ROLLING) ===")
    from src.feature_engineering import run_feature_pipeline
    run_feature_pipeline()

    logger.info("=== STEP 5: DEEP LEARNING MODEL TRAINING & EXPERIMENTATION ===")
    from src.train import run_all_experiments
    run_all_experiments(epochs=15, batch_size=64)

    logger.info("DemandCast pipeline executed successfully!")


def print_eval_summary():
    import pandas as pd
    metrics_path = os.path.join(BASE_DIR, "artifacts", "metrics", "all_experiments.csv")
    if not os.path.exists(metrics_path):
        logger.error("No benchmark metrics found. Run --pipeline first.")
        return

    df = pd.read_csv(metrics_path)
    logger.info("\n" + "="*80)
    logger.info("DEMANDCAST BENCHMARK RESULTS SUMMARY (MAE, RMSE, MAPE, R²)")
    logger.info("="*80)
    display_cols = ["model", "seq_length", "feature_mode", "horizon", "MAE_MW", "RMSE_MW", "MAPE_percent", "R2_Score"]
    print(df[display_cols].to_string(index=False))
    logger.info("="*80)


def start_server(host: str = "127.0.0.1", port: int = 8000):
    logger.info("Starting DemandCast FastAPI server on http://%s:%d...", host, port)
    logger.info("Interactive Dashboard UI: http://%s:%d/", host, port)
    logger.info("Swagger API Documentation: http://%s:%d/docs", host, port)
    import uvicorn
    uvicorn.run("api.main:app", host=host, port=port, reload=False)


def main():
    parser = argparse.ArgumentParser(description="DemandCast Master CLI")
    parser.add_argument("--pipeline", action="store_true", help="Run end-to-end data and modeling pipeline")
    parser.add_argument("--server", action="store_true", help="Start FastAPI web server and UI")
    parser.add_argument("--eval", action="store_true", help="Display benchmark evaluation summary")
    parser.add_argument("--all", action="store_true", help="Run full pipeline and start server")
    parser.add_argument("--port", type=int, default=8000, help="Port for web server (default: 8000)")
    args = parser.parse_args()

    if args.pipeline:
        run_pipeline()
    elif args.eval:
        print_eval_summary()
    elif args.server:
        start_server(port=args.port)
    elif args.all:
        run_pipeline()
        start_server(port=args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
