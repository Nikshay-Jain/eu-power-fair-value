```markdown
# European Power Fair Value  
**Day-Ahead Price Forecasting and Prompt Curve Trading Signals**

End-to-end prototype pipeline that builds a fundamental European power dataset, forecasts Day-Ahead electricity prices, validates models with walk-forward cross-validation, and translates forecast distributions into prompt-curve trading signals with automated risk controls and AI-generated trader commentary.

---

## Overview

The project implements a realistic energy trading research workflow:

- Public ENTSO-E data ingestion and cleaning  
- Robust data QA and imputation  
- Feature engineering grounded in power system fundamentals  
- Baseline and ML-based Day-Ahead price forecasting  
- Walk-forward time-series validation and performance reporting  
- Probabilistic forecast → prompt-curve signal translation  
- Automated trader-style commentary using an LLM with fact-checking  

The pipeline is fully reproducible and produces all tables, figures, and signal reports used for trading interpretation.

---

## Repository Structure

```

.
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
│
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
├── submission.csv
├── part_1.py   # Data ingestion, stacking, QA, cleaning
├── part_2.py   # Feature engineering, forecasting, validation
├── part_3.py   # Prompt-curve signal generation
├── part_4.py   # AI-driven trader commentary
├── nb.ipynb    # Optional exploration notebook
├── requirements.txt
└── README.md

```

---

## Pipeline Stages

### Part 1 — Data Ingestion & QA (`part_1.py`)

- Loads raw ENTSO-E ZIP archives  
- Enforces strict hourly granularity and DST-safe timestamps  
- Merges price, load, generation, and flow data  
- Generates automated QA reports:
  - Missingness and coverage
  - Duplicate timestamps
  - Outliers and negative-value checks  
- Imputes remaining gaps using time-series aware methods  
- Outputs:
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
  - Ridge regression
  - LightGBM (point + quantile forecasts)  
- Reports:
  - MAE / RMSE
  - Tail MAE (P90 hours)  
- Trains final quantile LightGBM model  
- Outputs:
  - `results/part2_cv_summary.csv`
  - `results/part2_final_predictions.csv`
  - `results/part2_feature_importance.csv`
  - `results/part2_forecasting_results.png`
  - `submission.csv`

---

### Part 3 — Prompt Curve Translation (`part_3.py`)

Transforms hourly probabilistic forecasts into tradable delivery-period views.

Key components:

- Monte-Carlo aggregation of hourly quantiles into weekly/monthly price distributions  
- Forward-price comparison (real or fallback proxy)  
- Computed metrics:
  - Expected period mean (P50)
  - Distribution bands (P10 / P90)
  - Edge vs forward
  - Win probability (MC-based, uncertainty-penalized)
  - Sharpe-like score
  - Confidence-weighted position sizing (MW)
  - Conservative expected P&L  
- Risk and signal invalidation logic:
  - Edge erosion
  - Low confidence
  - Wide forecast bands
  - Quantile calibration warnings  

Outputs:

- `results/part3_trading_signals.csv`
- `results/part3_trading_reports.txt`
- `results/part3_trading_signals.png`

These provide a complete DA → curve-trading decision framework.

---

### Part 4 — AI-Generated Trader Commentary (`part_4.py`)

Programmatic LLM component that:

- Reads model outputs and selected trading signal  
- Builds a deterministic factual morning note  
- Optionally refines tone using an LLM (if API key provided)  
- Performs numeric fact-checking to prevent hallucinated values  
- Logs prompts, outputs, and validation results  

Outputs:

- `results/part4_trader_commentary.txt`
- `results/part4_trader_commentary.json`
- `results/part4_ai_log.txt`

---

## Setup

### 1. Clone Repository

```

git clone [https://github.com/Nikshay-Jain/eu-power-fair-value.git](https://github.com/Nikshay-Jain/eu-power-fair-value.git)
cd eu-power-fair-value

```

### 2. Create Environment

```

python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

```

### 3. Install Dependencies

```

pip install -r requirements.txt

```

### 4. (Optional) Enable AI Commentary

If using the LLM commentary module, set:

```

export GOOGLE_API_KEY="your_api_key_here"

```

No API key is required to run the rest of the pipeline.

---

## Running the Full Pipeline

Execute in sequence:

```

python part_1.py   # Data ingestion + QA + cleaning
python part_2.py   # Feature engineering + forecasting + validation
python part_3.py   # Prompt-curve trading signals
python part_4.py   # AI-generated trader commentary

```

All intermediate and final artifacts are written to `/data`, `/qa_report`, and `/results`.

---

## Key Outputs

| File | Description |
|------|-------------|
| `data/cleaned_energy_data.csv` | QA-cleaned hourly dataset |
| `data/featured_energy_data.csv` | Feature-engineered dataset |
| `results/part2_cv_summary.csv` | Walk-forward model performance |
| `results/part2_final_predictions.csv` | Final quantile forecasts |
| `results/part3_trading_signals.csv` | Weekly/monthly trading signals |
| `results/part3_trading_reports.txt` | Human-readable trading reports |
| `results/part3_trading_signals.png` | Signal visualization dashboard |
| `results/part4_trader_commentary.txt` | Automated trader morning note |
| `submission.csv` | Out-of-sample DA price predictions |

---

## Reproducibility Notes

- All timestamps handled in UTC with explicit DST-safe parsing  
- Random seeds fixed for Monte-Carlo simulations  
- No external paid data required  
- All model training uses deterministic configurations  

---

## Trading Interpretation Summary

The produced signals express how forecasted Day-Ahead fundamentals map into prompt-curve positioning:

- Positive edge → long prompt baseload  
- Negative edge → short prompt baseload  
- Position size scales with confidence and signal-to-noise  
- Explicit invalidation triggers guide risk reduction  

This mirrors real desk workflows for DA-to-curve fair-value trading.

---

## Requirements

- Python ≥ 3.9  
- pandas, numpy, scikit-learn  
- lightgbm  
- matplotlib, seaborn  
- scipy  
- python-dotenv  
- langchain-google-genai (optional)

See `requirements.txt` for exact versions.

---

## Author

Nikshay Jain  
Energy Markets & Applied ML
```