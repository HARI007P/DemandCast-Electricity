# DemandCast: LSTM-Based Electricity Consumption Forecasting

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.21+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React 18](https://img.shields.io/badge/React-18.3+-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4+-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com)
[![DOI](https://img.shields.io/badge/Mendeley_Data-10.17632%2Fy58jknpgs8.2-orange?style=for-the-badge)](https://doi.org/10.17632/y58jknpgs8.2)

---

## 1. Project Overview

**DemandCast** is a complete, production-ready, research-grade Machine Learning and Deep Learning system designed to forecast future electrical energy demand across the synchronous **Indian National Power Grid** using historical hourly time-series observations.

### Core Objectives
1. **Predict Future Electricity Demand:** Model non-linear temporal dependencies, diurnal peaks, weekend industrial load drops, and seasonal surges across the synchronized national grid.
2. **Official Dataset Integration:** Built using the latest official Indian electricity dataset from **Mendeley Data** (`DOI: 10.17632/y58jknpgs8.2`), sourced directly from the North-Eastern Regional Load Despatch Centre (**NERLDC**) / **Grid-India** under the Ministry of Power, Government of India.
3. **Rigorous Machine Learning Protocol:** Enforces **zero data leakage** via strictly chronological partitioning (70% Train, 15% Validation, 15% Test) and training-isolated feature/target MinMax scaling.
4. **Architectural Comparison:** Compares three recurrent neural network architectures:
   - Traditional **SimpleRNN**
   - Single-layer **LSTM**
   - Two-layer **Stacked LSTM**
5. **Sequence Length Sensitivity Analysis:** Systematically evaluates sliding lookback sequence windows of **6, 12, 24, 48, 72, and 168 hours**.
6. **Exogenous Feature Engineering:** Leverages cyclical trigonometric time embeddings (`hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `month_sin`, `month_cos`), calendar flags (`is_weekend`), and synchronized utility-scale **Solar PV** and **Wind Generation** telemetry.
7. **Interactive Dashboard:** Modern full-stack application featuring a high-performance **FastAPI** backend and a responsive **React + Vite + Tailwind CSS** frontend equipped with live autoregressive inference simulator and interactive Recharts visualizations.

---

## 2. Repository Folder Structure

```
DemandCast/
├── api/
│   ├── __init__.py
│   └── main.py                     # FastAPI REST API & SPA static server
├── artifacts/
│   ├── figures/                    # Generated high-resolution plots (EDA, loss, forecasts)
│   ├── metrics/                    # JSON and CSV benchmark results & EDA metrics
│   ├── models/                     # Saved Keras models (.keras) and fitted scalers (.joblib)
│   └── predictions/                # Test-set prediction overlays (actual vs forecast)
├── data/
│   ├── raw/                        # Downloaded Excel dataset from Mendeley Data
│   └── processed/                  # Cleaned CSV and Parquet datasets with engineered features
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx          # Header navigation & connection status
│   │   │   ├── Overview.jsx        # Executive overview, KPI cards & methodology
│   │   │   ├── EDAPanel.jsx        # Diurnal, weekly, monthly & correlation analysis
│   │   │   ├── ModelComparison.jsx # Benchmark table, sequence sensitivity & ablation charts
│   │   │   ├── LiveSimulator.jsx   # Real-time multi-step forecasting sandbox
│   │   │   └── AcademicReport.jsx  # Complete academic paper & research findings
│   │   ├── App.jsx                 # Main React layout component
│   │   ├── main.jsx                # React root mount
│   │   └── index.css               # Tailwind CSS directives
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.js
├── src/
│   ├── __init__.py
│   ├── download_dataset.py         # Automated Mendeley Data API downloader
│   ├── preprocess.py               # Data cleaning, timestamp parsing, sensor drift rectification
│   ├── eda.py                      # Exploratory Data Analysis & publication figure generator
│   ├── feature_engineering.py      # Cyclical encodings, renewable ratios, rolling stats
│   ├── dataset_generator.py        # Sliding window sequence builder & chronological splitter
│   ├── models.py                   # SimpleRNN, Single LSTM, and Stacked LSTM Keras builders
│   ├── train.py                    # Multi-experiment runner, callbacks, and evaluator
│   └── evaluate.py                 # MAE, RMSE, MAPE, R², loss curves, comparison plots
├── run_project.py                  # Master CLI orchestrator
├── requirements.txt                # Python dependencies
└── README.md                       # Comprehensive project documentation
```

---

## 3. Dataset Description & Statistics

The dataset was curated by Debanjan Mukherjee, Karuna Kalita, and Subhash Kumar (IIT Guwahati & Power Grid Corporation of India Ltd), published under Mendeley Data `DOI: 10.17632/y58jknpgs8.2`.

* **Temporal Horizon:** January 1, 2024 to June 24, 2025 (12,984 continuous hours / 541 days)
* **Sampling Rate:** Exactly 1 hour interval (zero duplicate or missing timestamps)
* **National Demand (MW):** Instantaneous load across Northern, Western, Southern, Eastern, and North-Eastern regional grids.
  - Mean Load: **194,010.0 MW (~194.0 GW)**
  - Peak Recorded Load: **248,688.2 MW (~248.7 GW)**
  - Minimum Recorded Load: **97,065.3 MW (~97.1 GW)**
  - Standard Deviation: **20,227.7 MW**
* **Solar PV Generation:** Mean: 15,624 MW | Max: **64,701 MW** (Diurnal solar peak at 12:00 PM IST)
* **Wind Generation:** Mean: 8,894 MW | Max: **29,305 MW**
* **Diurnal Dynamics:** National peak demand occurs at **10:00 AM IST** (mean 210,549 MW), with night minimum trough at **03:00 AM IST** (mean 172,853 MW).
* **Weekend Drop:** National weekend average dips by **4,508 MW (-2.31%)** due to commercial and industrial shutdowns.
* **Renewable Co-variance:** Demand vs. Solar PV Pearson correlation is **+0.470**, reflecting how midday solar generation aligns with cooling and industrial demand peaks.

---

## 4. Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Node.js 18+ and npm 9+
- Git

### Step 1: Clone Repository & Setup Virtual Environment
```bash

cd DemandCast

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Install Frontend Dependencies
```bash
cd frontend
npm install
npm run build
cd ..
```

---

## 5. End-to-End Execution Guide

You can run each stage independently or orchestrate the full pipeline using `run_project.py`.

### Option A: Complete Automated Pipeline
```bash
python run_project.py --pipeline
```

### Option B: Step-by-Step Execution

#### Step 1: Download Mendeley Dataset
```bash
python src/download_dataset.py
```
*Queries the Mendeley Data API (`y58jknpgs8`), locates the latest consolidated hourly Excel file, and streams it to `data/raw/Jan2024_June2025.xlsx`.*

#### Step 2: Data Cleaning & Preprocessing
```bash
python src/preprocess.py
```
*Parses DD-MM-YYYY timestamps, drops Excel footer summary rows, clips nocturnal negative solar sensor drift to 0.0 MW, validates hourly continuity, and outputs `data/processed/cleaned_demand_data.parquet`.*

#### Step 3: Exploratory Data Analysis (EDA)
```bash
python src/eda.py
```
*Generates publication-quality charts in `artifacts/figures/` (diurnal profiles, weekly patterns, seasonal trends, correlation heatmap) and exports `artifacts/metrics/eda_summary.json`.*

#### Step 4: Feature Engineering
```bash
python src/feature_engineering.py
```
*Generates cyclical trigonometric embeddings (`hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `month_sin`, `month_cos`), calendar flags (`is_weekend`), renewable ratios, and rolling moments.*

#### Step 5: Model Training & Experimentation
```bash
python src/train.py --epochs 15 --batch_size 64
```
*Trains SimpleRNN, Single LSTM, and Stacked LSTM models across sequence lookback windows (6, 12, 24, 48, 72, 168 hours), performs feature and horizon ablations, generates loss curves and test set actual-vs-predicted graphs, and serializes models to `artifacts/models/`.*

#### Step 6: Print Evaluation Metrics Summary
```bash
python run_project.py --eval
```

---

## 6. Launching the Web Application

### Standalone Production Server (FastAPI + React Dashboard)
FastAPI serves both the REST API and the compiled React single-page application at one address:
```bash
python run_project.py --server --port 8000
```
- **Dashboard UI:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Development Mode (with Hot Module Replacement)
1. **Start Backend:**
   ```bash
   uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
        
   ```
2. **Start Frontend:**
   ```bash
   cd frontend
   npm run dev
   ```
   Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 7. Mathematical Formulations & Architectures

### 1. Traditional SimpleRNN
Hidden state update at timestep $t$:
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
Output projection:
$$\hat{y}_{t+1} = W_{hy} h_t + b_y$$

### 2. Single-Layer LSTM
Governed by input ($i_t$), forget ($f_t$), and output ($o_t$) gating mechanisms:
$$\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\
\tilde{C}_t &= \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\
h_t &= o_t \odot \tanh(C_t)
\end{aligned}$$

### 3. Two-Layer Stacked LSTM
- **Layer 1:** Preserves full hidden state trajectory across all lookback steps:
  $$H^{(1)} = [h_1^{(1)}, h_2^{(1)}, \dots, h_L^{(1)}] \in \mathbb{R}^{L \times d_1}$$
- **Layer 2:** Processes the sequential output $H^{(1)}$ and extracts high-level representations into a final summary vector $h_L^{(2)} \in \mathbb{R}^{d_2}$.
- **Dense Head:** Maps $h_L^{(2)} \to \text{ReLU} \to \text{Linear} \to \hat{y}_{t+1}$.

### 4. Evaluation Metrics
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i| \quad (\text{MW})$$
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2} \quad (\text{MW})$$
$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right| \quad (\%)$$
$$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

---

## 8. Summary of Empirical Results

| Architecture | Sequence Length ($L$) | Feature Set | Forecast Horizon | MAE (MW) | RMSE (MW) | MAPE (%) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stacked LSTM** | **24h** | **Multivariate** | **+1h** | **6,392.9** | **7,907.3** | **3.13%** | **0.6587** |
| Stacked LSTM | 12h | Multivariate | +1h | 6,493.0 | 8,076.2 | 3.19% | 0.6439 |
| Single LSTM | 24h | Multivariate | +1h | 6,854.2 | 8,421.1 | 3.38% | 0.6124 |
| SimpleRNN | 12h | Multivariate | +1h | 7,612.4 | 9,340.5 | 3.75% | 0.5230 |
| SimpleRNN | 168h | Multivariate | +1h | 10,845.0 | 13,290.1 | 5.32% | 0.0340 |
| Stacked LSTM | 24h | Univariate (Demand only) | +1h | 7,298.5 | 8,980.4 | 3.61% | 0.5601 |
| Stacked LSTM | 24h | Multivariate | +24h | 11,432.0 | 14,102.5 | 5.62% | -0.0820 |

### Key Empirical Findings:
1. **Superiority of Stacked LSTM:** Stacking two LSTM layers reduces test MAPE from 3.75% (SimpleRNN) to **3.13%**, confirming that hierarchical temporal representations effectively decouple diurnal fluctuations from weekly drift.
2. **Optimal Sequence Lookback:** A lookback of **24 hours** offers the optimal trade-off. Longer lookbacks (168 hours / 1 week) suffer from gradient dissipation in SimpleRNN ($R^2$ drops to 0.034), whereas LSTM remains robust.
3. **Value of Exogenous Renewables:** Adding Solar PV, Wind, and cyclical calendar features provides a **~12.4% error reduction** compared to demand-only univariate models.
4. **Horizon Decay:** Autoregressive recursive multi-step forecasting retains high fidelity through 12 hours ahead (MAPE < 4.5%), with anticipated variance increase at the 24-hour mark (MAPE ~5.6%).

---

   
## 9. License
This project is licensed under the MIT License. See `LICENSE` for details.
