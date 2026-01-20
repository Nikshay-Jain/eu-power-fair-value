# European Power Fair Value  
### Day-Ahead Price Forecasting and Prompt Curve Trading Signals

A complete, reproducible research and trading prototype that builds a fundamental European power dataset, forecasts Day-Ahead electricity prices, validates models with time-series cross-validation, and translates probabilistic forecasts into prompt-curve trading signals with automated risk controls and AI-generated trader commentary.

---

## Project Summary

This project mirrors a real energy trading desk workflow:

- Ingests and cleans public ENTSO-E power market data  
- Performs rigorous data QA and feature engineering  
- Builds baseline and ML forecasting models for Day-Ahead prices  
- Validates performance using walk-forward cross-validation  
- Converts forecast distributions into tradable curve signals  
- Generates automated trader commentary using a controlled LLM  

All steps are deterministic, auditable, and produce ready-to-review artifacts.

---

## Repository Structure

```
root
├── data/
│   ├── generation.zip
│   ├── load.zip
│   ├── market.zip
│   ├── prices.zip
│   ├── cleaned_energy_data.csv
│   └── featured_energy_data.csv
│
├── figures/
│   ├── corr_heatmap.png
│   ├── corr_target_bar.png
│   └── feature_vs_target_timeseries_grid.png
│
├── qa_report/
│   ├── qa_report_missing_by_column.csv
│   ├── qa_report_missing_blocks.csv
│   ├── qa_report_outlier_summary.csv
│   └── qa_report_qa_summary.txt
│       ...
|
├── results/
│   ├── part2_cv_predictions.csv
│   ├── part2_cv_summary.csv
│   ├── part2_feature_importance.csv
│   ├── part2_final_predictions.csv
│   ├── part2_forecasting_results.png
│   ├── part3_trading_signals.csv
│   ├── part3_trading_reports.txt
│   ├── part3_trading_signals.png
│   ├── part4_trader_commentary.txt
│   ├── part4_trader_commentary.json
│   └── part4_ai_log.txt
│
├── src/
│   ├── part_1.py   # Data ingestion, cleaning, QA
│   ├── part_2.py   # Feature engineering, forecasting, validation
│   ├── part_3.py   # Prompt-curve signal generation
│   └── part_4.py   # AI-driven trader commentary
│
├── submission.csv
├── nb.ipynb    # Optional exploration notebook
├── requirements.txt
├── Report.md
└── README.md
```

---

## Pipeline Overview

### Part 1 — Data Ingestion & QA (`part_1.py`)

- Loads raw ENTSO-E ZIP archives  
- Aligns price, load, generation, and flow data at hourly granularity  
- Handles timezone and DST safely  
- Performs automated QA:
  - Missing data coverage  
  - Duplicate timestamps  
  - Outlier and negative-value detection  
- Applies gap-filling where required  

**Outputs**
- `data/cleaned_energy_data.csv`  
- `qa_report/*`

---

### Part 2 — Forecasting & Validation (`part_2.py`)

- Builds fundamental features:
  - Residual load  
  - Renewable penetration  
  - Lags and rolling statistics  
  - Interaction terms  
- Runs walk-forward cross-validation:
  - Naive 24h baseline  
  - Naive 168h baseline  
  - Regularized regression  
  - LightGBM with quantile forecasts  
- Reports MAE / RMSE and tail performance  
- Trains final quantile forecasting model  

**Outputs**
- `results/part2_cv_summary.csv`  
- `results/part2_final_predictions.csv`  
- `results/part2_feature_importance.csv`  
- `results/part2_forecasting_results.png`  
- `submission.csv`

---

### Part 3 — Prompt Curve Translation (`part_3.py`)

Transforms hourly probabilistic forecasts into delivery-period trading views.

**Core logic**

- Monte-Carlo aggregation of hourly quantiles into weekly/monthly price distributions  
- Forward-price comparison (real or proxy)  
- Computed trading metrics:
  - Expected period mean (P50)  
  - Distribution bands (P10 / P90)  
  - Edge versus forward price  
  - Monte-Carlo win probability  
  - Sharpe-like risk metric  
  - Confidence-weighted MW position sizing  
  - Conservative expected P&L  
- Automated risk and invalidation rules:
  - Edge erosion  
  - Low confidence  
  - Wide uncertainty bands  
  - Quantile calibration warnings  

**Outputs**
- `results/part3_trading_signals.csv`  
- `results/part3_trading_reports.txt`  
- `results/part3_trading_signals.png`

These deliver a complete Day-Ahead → prompt-curve trading decision framework.

---

### Part 4 — AI-Generated Trader Commentary (`part_4.py`)

Programmatic LLM component that:

- Reads model outputs and selected trading signal  
- Builds a factual morning-note template  
- Optionally refines tone using an LLM (if API key supplied)  
- Performs numeric fact-checking to prevent hallucinated values  
- Logs prompts, responses, and validation results  

**Outputs**
- `results/part4_trader_commentary.txt`  
- `results/part4_trader_commentary.json`  
- `results/part4_ai_log.txt`

---

## Setup

### 1. Clone Repository

```
git clone https://github.com/Nikshay-Jain/eu-power-fair-value.git
cd eu-power-fair-value
```

### 2. Create Virtual Environment

```
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. (Optional) Enable AI Commentary

```
export GOOGLE_API_KEY="your_api_key_here"
```

If no API key is provided, the pipeline still runs with deterministic commentary.

---

## Running the Pipeline

Execute sequentially:

```
python part_1.py   # Data ingestion + QA
python part_2.py   # Forecasting + validation
python part_3.py   # Prompt-curve signal generation
python part_4.py   # AI trader commentary
```

All outputs are written to `data/`, `qa_report/`, and `results/`.

---

## Key Artifacts

| File | Description |
|------|-------------|
| `data/cleaned_energy_data.csv` | QA-cleaned hourly dataset |
| `data/featured_energy_data.csv` | Feature-engineered dataset |
| `results/part2_cv_summary.csv` | Walk-forward model performance |
| `results/part2_final_predictions.csv` | Final quantile forecasts |
| `results/part3_trading_signals.csv` | Weekly/monthly trading signals |
| `results/part3_trading_reports.txt` | Human-readable trading notes |
| `results/part3_trading_signals.png` | Signal dashboard |
| `results/part4_trader_commentary.txt` | Automated trader morning note |
| `submission.csv` | Out-of-sample predictions |

---

## Trading Interpretation

The produced signals express how forecasted Day-Ahead fundamentals map into curve positioning:

- Positive edge → long prompt baseload  
- Negative edge → short prompt baseload  
- Position sizing scales with confidence and signal-to-noise  
- Explicit invalidation rules guide risk reduction  

This structure reflects real DA-to-curve fair-value trading practice.

---

## Reproducibility

- UTC timestamps with DST-safe handling  
- Fixed random seeds for Monte-Carlo simulations  
- No paid or proprietary data sources  
- Deterministic model training configurations  

---