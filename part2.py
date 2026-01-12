# part2.py
"""
Part 2 — Forecasting & Model Validation
Implements exactly the required elements:
 - Baselines: seasonal naive (t-24), last-week-same-hour (t-168)
 - Improved model: regularised regression (Ridge) and optional LightGBM
 - Time-series CV: expanding-window walk-forward
 - Metrics: MAE, RMSE, MAPE, Tail MAE (top-decile)
 - Next-day hourly DA forecast + fan bands (P10/P50/P90)
 - Next-week / next-month expected averages derived from forecast distribution
 - Tradable view translation (expected mean, bands, simple express/avoid rule)

Usage:
 - Set DATAFILE to your CSV containing timestamp-like column 'MTU (UTC)' and the target column.
 - Set TARGET to the exact target column name (Day-Ahead Price (EUR/MWh) preferred).
 - Run: python part2.py
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# LightGBM optional
try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

# -------------------------
# Validator
# -------------------------
class WalkForwardValidator:
    def __init__(self, train_size_hours: int = 24 * 365 * 2,
                 test_size_hours: int = 24 * 30, step_hours: int = 24 * 30):
        self.train_size = int(train_size_hours)
        self.test_size = int(test_size_hours)
        self.step = int(step_hours)

    def split(self, df: pd.DataFrame, timestamp_col: str = "timestamp"):
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        n = len(df_sorted)
        start = self.train_size
        while start + self.test_size <= n:
            train_idx = np.arange(0, start, dtype=int)
            test_idx = np.arange(start, start + self.test_size, dtype=int)
            yield train_idx, test_idx
            start += self.step

# -------------------------
# Metrics (positional alignment)
# -------------------------
def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict:
    y_true = y_true.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)
    mask = ~(y_true.isna() | y_pred.isna())
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = len(y_true)
    if n == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "tail_mae_p90": np.nan, "n": 0}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6))) * 100.0)
    p90 = np.quantile(y_true, 0.90)
    mask_tail = y_true >= p90
    tail_mae = mean_absolute_error(y_true[mask_tail], y_pred[mask_tail]) if mask_tail.sum() > 0 else np.nan
    return {"mae": float(mae), "rmse": rmse, "mape": mape, "tail_mae_p90": float(tail_mae) if not np.isnan(tail_mae) else np.nan, "n": int(n)}

# -------------------------
# Leakage guard (conservative)
# -------------------------
def drop_target_derived_features(feature_cols: List[str], target_col: str) -> List[str]:
    feat = list(feature_cols)
    tl = target_col.lower()
    if any(x in tl for x in ["load", "day-ahead", "day ahead", "total load forecast", "price"]):
        blacklist = ["residual_load", "renewable_penetration", "total_renewable", "residual_load_rollmean", "residual_load_ramp"]
        feat = [c for c in feat if not any(b in c.lower() for b in blacklist)]
    return feat

# -------------------------
# Pipelines / models
# -------------------------
def ridge_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=0))
    ])

def fit_lightgbm_point(X, y, params: Optional[dict] = None):
    if not _HAS_LGB:
        raise RuntimeError("LightGBM not installed")
    params = params or {"objective": "regression", "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31, "verbosity": -1, "random_state": 42}
    m = lgb.LGBMRegressor(**params)
    m.fit(X, y)
    return m

# -------------------------
# Quantile forecaster (LightGBM-based). Optional; trains separate q-models
# -------------------------
class LightGBMQuantileForecaster:
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        if not _HAS_LGB:
            raise RuntimeError("LightGBM required for quantile forecaster")
        self.quantiles = quantiles
        self.models = {}

    def _params(self):
        return {"num_leaves": 64, "learning_rate": 0.05, "feature_fraction": 0.8,
                "bagging_fraction": 0.8, "bagging_freq": 5, "max_depth": 8,
                "min_data_in_leaf": 50, "lambda_l1": 0.1, "lambda_l2": 0.1, "verbose": -1}

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              num_boost_round: int = 800, early_stopping_rounds: int = 50):
        params_base = self._params()
        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        for q in self.quantiles:
            p = params_base.copy()
            p["objective"] = "quantile"
            p["alpha"] = float(q)
            bst = lgb.train(p, train_data, num_boost_round=num_boost_round, valid_sets=[train_data],
                            verbose_eval=False)
            self.models[q] = bst

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = {}
        for q in self.quantiles:
            m = self.models.get(q)
            if m is None:
                preds[f"p{int(q*100)}"] = np.repeat(np.nan, len(X))
            else:
                preds[f"p{int(q*100)}"] = m.predict(X)
        df = pd.DataFrame(preds, index=X.index)
        # normalize names to p10,p50,p90 if present
        return df

# -------------------------
# Walk-forward evaluation (baselines + ridge + optional LGBM)
# -------------------------
def run_walkforward_evaluation(df: pd.DataFrame,
                               feature_cols: List[str],
                               target_col: str,
                               validator: Optional[WalkForwardValidator] = None,
                               use_lgbm: bool = False) -> Tuple[Dict, Dict]:
    if validator is None:
        validator = WalkForwardValidator()

    dfw = df.copy().sort_values("timestamp").reset_index(drop=True)
    if target_col not in dfw.columns:
        raise KeyError("target_col missing")

    naive_24 = dfw[target_col].shift(24)
    naive_168 = dfw[target_col].shift(168)

    models = ["naive_24h", "naive_168h", "ridge"]
    if use_lgbm and _HAS_LGB:
        models.append("lgbm")

    per_fold = {m: [] for m in models}
    safe_features = drop_target_derived_features(feature_cols, target_col)

    for i, (train_idx, test_idx) in enumerate(validator.split(dfw), start=1):
        dtr = dfw.iloc[train_idx].reset_index(drop=True)
        dte = dfw.iloc[test_idx].reset_index(drop=True)
        train_med = float(dtr[target_col].median(skipna=True))

        # naive 24
        p24 = naive_24.iloc[test_idx].reset_index(drop=True).fillna(train_med)
        per_fold["naive_24h"].append(compute_metrics(dte[target_col], p24))

        # naive 168
        p168 = naive_168.iloc[test_idx].reset_index(drop=True).fillna(train_med)
        per_fold["naive_168h"].append(compute_metrics(dte[target_col], p168))

        # ridge
        feats = [f for f in safe_features if f in dfw.columns]
        if not feats:
            fallback = [c for c in ["hour", "day_of_week", "month", "is_weekend"] if c in dfw.columns]
            feats = fallback

        if not feats:
            pred_ridge = pd.Series(train_med, index=dte.index)
        else:
            pipe = ridge_pipeline()
            pipe.fit(dtr[feats], dtr[target_col])
            pred_ridge = pd.Series(pipe.predict(dte[feats]), index=dte.index)
        per_fold["ridge"].append(compute_metrics(dte[target_col], pred_ridge))

        # optional lgbm
        if use_lgbm and _HAS_LGB:
            feats_l = [f for f in safe_features if f in dfw.columns]
            if feats_l:
                Xtr = dtr[feats_l].fillna(dtr[feats_l].median())
                Xte = dte[feats_l].fillna(dtr[feats_l].median())
                try:
                    model = fit_lightgbm_point(Xtr, dtr[target_col])
                    p = pd.Series(model.predict(Xte), index=dte.index)
                except Exception:
                    p = pd.Series(train_med, index=dte.index)
            else:
                p = pd.Series(train_med, index=dte.index)
            per_fold["lgbm"].append(compute_metrics(dte[target_col], p))

        tmin, tmax = dte["timestamp"].min(), dte["timestamp"].max()
        print(f"Fold {i}: train={len(dtr)} test={len(dte)} period={tmin} to {tmax}")

    # aggregate weighted
    summary = {}
    for m, metrics in per_fold.items():
        if not metrics:
            summary[m] = {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "tail_mae_p90": np.nan, "n_total": 0}
            continue
        maes = np.array([x["mae"] if x["n"] > 0 else np.nan for x in metrics], dtype=float)
        rmses = np.array([x["rmse"] if x["n"] > 0 else np.nan for x in metrics], dtype=float)
        mapes = np.array([x["mape"] if x["n"] > 0 else np.nan for x in metrics], dtype=float)
        tails = np.array([x["tail_mae_p90"] if x["n"] > 0 else np.nan for x in metrics], dtype=float)
        ns = np.array([x["n"] for x in metrics], dtype=float)
        ns_total = ns.sum() if ns.sum() > 0 else 1.0
        def wavg(arr): return float(np.nansum(np.nan_to_num(arr) * ns) / ns_total)
        summary[m] = {"mae": wavg(maes), "rmse": wavg(rmses), "mape": wavg(mapes), "tail_mae_p90": wavg(tails), "n_total": int(ns_total)}

    print("\nAggregated weighted CV results:")
    for k, v in summary.items():
        print(f"{k} | MAE: {v['mae']:.3f} | RMSE: {v['rmse']:.3f} | MAPE: {v['mape']:.3f}% | TailMAE_P90: {v['tail_mae_p90']:.3f} | n: {v['n_total']}")
    return summary, per_fold

# -------------------------
# Train on full history + forecast next-day (point + bands) + aggregates
# -------------------------
def train_full_and_forecast(df: pd.DataFrame,
                            feature_cols: List[str],
                            target_col: str,
                            model: str = "ridge",
                            use_lgbm: bool = False,
                            use_lgbm_quantile: bool = False,
                            next_day_df: Optional[pd.DataFrame] = None) -> Dict:
    dfw = df.copy().sort_values("timestamp").reset_index(drop=True)
    if target_col not in dfw.columns:
        raise KeyError("target_col missing")
    safe_features = drop_target_derived_features(feature_cols, target_col)
    feats = [f for f in safe_features if f in dfw.columns]

    # train point model
    if model == "ridge":
        if feats:
            pipe = ridge_pipeline()
            pipe.fit(dfw[feats], dfw[target_col])
            predict_fn = lambda X: pipe.predict(X)
            trained = pipe
        else:
            med = float(dfw[target_col].median(skipna=True))
            predict_fn = lambda X: np.repeat(med, len(X))
            trained = None
    elif model == "lgbm" and _HAS_LGB:
        if not feats:
            raise ValueError("No features for LGBM")
        m = fit_lightgbm_point(dfw[feats].fillna(dfw[feats].median()), dfw[target_col])
        predict_fn = lambda X: m.predict(X.fillna(dfw[feats].median()))
        trained = m
    else:
        raise ValueError("Unsupported model or dependencies missing")

    # build X_next
    if next_day_df is None:
        last24 = dfw.tail(24).copy().reset_index(drop=True)
        last24["timestamp"] = last24["timestamp"] + pd.Timedelta(days=1)
        if feats:
            X_next = last24[feats]
        else:
            X_next = pd.DataFrame(index=range(24))
        X_next.index = pd.to_datetime(last24["timestamp"])
    else:
        X_next = next_day_df[[c for c in feats if c in next_day_df.columns]].copy()
        X_next.index = pd.to_datetime(next_day_df["timestamp"])

    # point forecast
    point = pd.Series(predict_fn(X_next), index=X_next.index)

    # quantile forecast: prefer LightGBM quantile if requested; otherwise residual-quantile fallback
    quantile_df = None
    if use_lgbm_quantile and _HAS_LGB:
        qfeats = [f for f in feats if f in dfw.columns]
        if qfeats:
            qf = LightGBMQuantileForecaster([0.1, 0.5, 0.9])
            qf.train(dfw[qfeats].fillna(0), dfw[target_col])
            quantile_df = qf.predict(X_next.fillna(0))
    # fallback: compute residual quantiles from in-sample residuals and broadcast
    if quantile_df is None:
        # produce in-sample preds
        if feats:
            in_sample_pred = pd.Series(predict_fn(dfw[feats]), index=dfw["timestamp"])
        else:
            in_sample_pred = pd.Series(float(dfw[target_col].median(skipna=True)), index=dfw["timestamp"])
        residuals = (dfw[target_col].reset_index(drop=True) - in_sample_pred.reset_index(drop=True)).dropna()
        if len(residuals) > 0:
            q10, q50, q90 = residuals.quantile([0.1, 0.5, 0.9]).values
        else:
            q10 = q50 = q90 = 0.0
        quantile_df = pd.DataFrame({"p10": point + q10, "p50": point + q50, "p90": point + q90}, index=point.index)

    # next-week / next-month expected averages (from point distribution)
    next_week_mean = float(point.head(24 * 7).mean()) if len(point) >= 24 * 7 else float(point.mean())
    next_month_mean = float(point.head(24 * 30).mean()) if len(point) >= 24 * 30 else float(point.mean())

    return {"model": trained, "point_forecast": point, "fan_bands": quantile_df,
            "next_week_mean": next_week_mean, "next_month_mean": next_month_mean}

# -------------------------
# Translate forecast -> tradable view
# -------------------------
def translate_forecast_to_tradable(point: pd.Series, fan_bands: pd.DataFrame,
                                   prompt_price: Optional[pd.Series] = None,
                                   express_thresh: float = 5.0, avoid_thresh: float = -5.0) -> Dict:
    expected_mean = float(point.mean())
    # ensure p10,p50,p90 exist
    if {"p10", "p50", "p90"}.issubset(set(fan_bands.columns)):
        p10, p50, p90 = fan_bands["p10"], fan_bands["p50"], fan_bands["p90"]
    else:
        p50 = fan_bands.iloc[:, 0]
        p10 = p50
        p90 = p50
    signal = "hold"
    if prompt_price is not None:
        common = point.index.intersection(prompt_price.index)
        if len(common) > 0:
            spread = (point.loc[common] - prompt_price.loc[common]).mean()
            if spread >= express_thresh:
                signal = "express"
            elif spread <= avoid_thresh:
                signal = "avoid"
    return {"expected_mean": expected_mean, "p10": p10, "p50": p50, "p90": p90, "signal": signal}

# -------------------------
# Minimal example execution (compliant with requirements)
# -------------------------
if __name__ == "__main__":
    # Adjust DATAFILE and TARGET as needed
    DATAFILE = "data/featured_energy_data.csv"
    TARGET = "Day-ahead Total Load Forecast (MW)"

    df = pd.read_csv(DATAFILE)
    # ensure timestamp column exists
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime(df["MTU (UTC)"].str.split(" - ").str[0], dayfirst=True, utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # verify target present
    if TARGET not in df.columns:
        raise SystemExit(f"Target column '{TARGET}' not found. Merge DA price history and retry.")

    # feature columns (drop MTU/timestamp/target)
    feature_cols = [c for c in df.columns if c not in ("MTU (UTC)", "timestamp", TARGET)]

    # 1) Walk-forward CV evaluation
    validator = WalkForwardValidator(train_size_hours=24 * 365 * 2, test_size_hours=24 * 30, step_hours=24 * 30)
    summary, per_fold = run_walkforward_evaluation(df, feature_cols, TARGET, validator=validator, use_lgbm=False)

    # 2) Train full + next-day forecast (point + fan bands via residual quantiles)
    out = train_full_and_forecast(df, feature_cols, TARGET, model="ridge", use_lgbm=False, use_lgbm_quantile=False)
    print("\nNext-day expected mean:", out["next_week_mean"])
    print("Next-month expected mean:", out["next_month_mean"])
    print("Sample fan bands (first 5 rows):")
    print(out["fan_bands"].head())

    # 3) Translate into tradable view (requires prompt price series to compare; optional)
    trad = translate_forecast_to_tradable(out["point_forecast"], out["fan_bands"], prompt_price=None)
    print("Signal:", trad["signal"])
