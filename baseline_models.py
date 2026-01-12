"""
baseline_models.py (updated, leakage-safe)

Fixes:
 - Automatic removal of target-derived features when target is a load/forecast
 - Robust positional alignment in metric computation
 - Robust in-sample residual computation for uncertainty (no NaN fan-bands)
 - Same API as before; improved diagnostics and safety guards
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# sklearn components
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# optional LightGBM
try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

# ---------------------------
# Walk-forward validator
# ---------------------------
class WalkForwardValidator:
    def __init__(self,
                 train_size_hours: int = 24 * 365 * 2,
                 test_size_hours: int = 24 * 30,
                 step_hours: int = 24 * 30):
        self.train_size = int(train_size_hours)
        self.test_size = int(test_size_hours)
        self.step = int(step_hours)

    def split(self, df: pd.DataFrame, timestamp_col: str = "timestamp"):
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        n = len(df_sorted)
        start_test = self.train_size
        fold = 0
        while start_test + self.test_size <= n:
            fold += 1
            train_idx = np.arange(0, start_test, dtype=int)
            test_idx = np.arange(start_test, start_test + self.test_size, dtype=int)
            yield train_idx, test_idx
            start_test += self.step

# ---------------------------
# Metrics helpers
# ---------------------------
def _compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict:
    """
    Compute metrics. Resets indices to ensure positional alignment before any boolean masking.
    Returns dict with mae, rmse, mape, tail_mae_p90, n
    """
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

    # tail metric (top 10% true values)
    p90 = np.quantile(y_true, 0.90)
    mask_tail = y_true >= p90
    tail_mae = mean_absolute_error(y_true[mask_tail], y_pred[mask_tail]) if mask_tail.sum() > 0 else np.nan

    return {"mae": float(mae), "rmse": rmse, "mape": mape, "tail_mae_p90": float(tail_mae) if not np.isnan(tail_mae) else np.nan, "n": int(n)}

# ---------------------------
# Models
# ---------------------------
class Models:
    @staticmethod
    def naive_shift(df: pd.DataFrame, target_col: str, shift_hours: int) -> pd.Series:
        return df[target_col].shift(shift_hours)

    @staticmethod
    def ridge_pipeline():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=0))
        ])
        return pipe

    @staticmethod
    def fit_lgbm(X_train: pd.DataFrame, y_train: pd.Series, params: Optional[dict] = None):
        if not _HAS_LGB:
            raise RuntimeError("LightGBM not installed")
        params = params or {
            "objective": "regression",
            "n_estimators": 800,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbosity": -1,
            "random_state": 42
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train)], eval_metric="mae", verbose=False)
        return model

def drop_target_derived_features(feature_cols: List[str], target_col: str) -> List[str]:
    """
    Heuristic removal of columns that leak target when the chosen target is a 'load forecast'
    These heuristics are conservative: only remove features that are very likely derived from the
    target at the same timestamp (residuals/penetration/roll stats using target).
    """
    feat = list(feature_cols)
    target_lower = target_col.lower()
    # If target is a load/forecast-like column, remove known target-derived names
    if any(x in target_lower for x in ["load", "day-ahead", "day ahead", "total load forecast"]):
        blacklist_substr = [
            "residual_load", "renewable_penetration", "total_renewable_mw_forecast_or_fallback",
            "residual_load_rollmean", "residual_load_ramp", "residual_load_mw_actual",
            "total_renewable_mw", "total_renewable_mw_actual"
        ]
        feat = [c for c in feat if not any(sub in c.lower() for sub in blacklist_substr)]
    return feat

# ---------------------------
# Runner (fixes + improvements)
# ---------------------------
def run_walkforward_evaluation(df: pd.DataFrame,
                                feature_cols: List[str],
                                target_col: str,
                                validator: Optional[WalkForwardValidator] = None,
                                use_lgbm: bool = False) -> Tuple[Dict[str, Dict], Dict[str, List[Dict]]]:
    if validator is None:
        validator = WalkForwardValidator()

    df_work = df.copy().sort_values("timestamp").reset_index(drop=True)
    if target_col not in df_work.columns:
        raise KeyError(f"target_col '{target_col}' not found in dataframe")

    # Precompute naive shifts
    naive_24 = Models.naive_shift(df_work, target_col, 24)
    naive_168 = Models.naive_shift(df_work, target_col, 168)

    # We'll evaluate these models
    models = ["naive_24h", "naive_168h", "ridge_features"]
    if use_lgbm and _HAS_LGB:
        models.append("lgbm_features")

    per_fold_results = {m: [] for m in models}
    fold_no = 0

    # If user provided a broad feature list, remove likely leakage columns here (global safe filter)
    safe_feature_cols = drop_target_derived_features(feature_cols, target_col)

    for train_idx, test_idx in validator.split(df_work, timestamp_col="timestamp"):
        fold_no += 1
        df_train = df_work.iloc[train_idx].reset_index(drop=True)
        df_test = df_work.iloc[test_idx].reset_index(drop=True)

        n_train_nonnull = int(df_train[target_col].notna().sum())
        n_test_nonnull = int(df_test[target_col].notna().sum())
        if n_train_nonnull < 24 * 7:
            print(f"WARNING fold {fold_no}: small non-null target in train ({n_train_nonnull})")

        train_median = float(df_train[target_col].median(skipna=True))

        # --- Naive 24h ---
        pred_24 = naive_24.iloc[test_idx].reset_index(drop=True).fillna(train_median)
        metrics_24 = _compute_metrics(df_test[target_col], pred_24)
        per_fold_results["naive_24h"].append(metrics_24)

        # --- Naive 168h ---
        pred_168 = naive_168.iloc[test_idx].reset_index(drop=True).fillna(train_median)
        metrics_168 = _compute_metrics(df_test[target_col], pred_168)
        per_fold_results["naive_168h"].append(metrics_168)

        # --- Ridge features ---
        feat_avail = [f for f in safe_feature_cols if f in df_work.columns]
        # fallback to simple calendar if necessary
        if not feat_avail:
            fallback = [c for c in ["hour", "day_of_week", "month", "is_weekend"] if c in df_work.columns]
            feat_avail = fallback

        if not feat_avail:
            # naive constant
            pred_ridge = pd.Series(train_median, index=df_test.index)
            metrics_ridge = _compute_metrics(df_test[target_col], pred_ridge)
            per_fold_results["ridge_features"].append(metrics_ridge)
        else:
            pipe = Models.ridge_pipeline()
            X_tr = df_train[feat_avail]
            y_tr = df_train[target_col]
            X_te = df_test[feat_avail]

            # Fit on train only (imputer/stat scaling fitted on train)
            pipe.fit(X_tr, y_tr)
            pred_ridge_vals = pipe.predict(X_te)
            pred_ridge = pd.Series(pred_ridge_vals, index=df_test.index)
            metrics_ridge = _compute_metrics(df_test[target_col], pred_ridge)
            per_fold_results["ridge_features"].append(metrics_ridge)

        # --- Optional LightGBM ---
        if use_lgbm and _HAS_LGB:
            feat_for_lgb = [f for f in safe_feature_cols if f in df_work.columns]
            if feat_for_lgb:
                X_tr_l = df_train[feat_for_lgb].fillna(df_train[feat_for_lgb].median())
                y_tr_l = df_train[target_col]
                X_te_l = df_test[feat_for_lgb].fillna(df_train[feat_for_lgb].median())
                try:
                    lgbm_model = Models.fit_lgbm(X_tr_l, y_tr_l)
                    pred_lgb = pd.Series(lgbm_model.predict(X_te_l), index=df_test.index)
                except Exception as e:
                    print(f"Fold {fold_no}: LightGBM training failed: {e}")
                    pred_lgb = pd.Series(train_median, index=df_test.index)
                metrics_lgb = _compute_metrics(df_test[target_col], pred_lgb)
                per_fold_results["lgbm_features"].append(metrics_lgb)
            else:
                per_fold_results["lgbm_features"].append({"mae": np.nan, "rmse": np.nan, "mape": np.nan, "tail_mae_p90": np.nan, "n": 0})

        tmin = df_test["timestamp"].min()
        tmax = df_test["timestamp"].max()
        print(f"Fold {fold_no}: train_rows={len(df_train)}, train_nonnull={n_train_nonnull}, test_rows={len(df_test)}, test_nonnull={n_test_nonnull}, test_period={tmin} to {tmax}")

    # Aggregate weighted
    summary = {}
    for model_name, folds in per_fold_results.items():
        if not folds:
            summary[model_name] = {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "tail_mae_p90": np.nan, "n_total": 0}
            continue
        maes, rmses, mapes, tails, ns = [], [], [], [], []
        for m in folds:
            maes.append(m.get("mae", np.nan))
            rmses.append(m.get("rmse", np.nan))
            mapes.append(m.get("mape", np.nan))
            tails.append(m.get("tail_mae_p90", np.nan))
            ns.append(m.get("n", 0))
        ns = np.array(ns, dtype=float)
        ns_total = ns.sum() if ns.sum() > 0 else 1.0
        def wavg(vals):
            vals_arr = np.array([0.0 if np.isnan(v) else v for v in vals], dtype=float)
            return float(np.dot(vals_arr, ns) / ns_total)
        summary[model_name] = {
            "mae": wavg(maes),
            "rmse": wavg(rmses),
            "mape": wavg(mapes),
            "tail_mae_p90": wavg(tails),
            "n_total": int(ns.sum())
        }

    print("\nAggregated weighted CV results:")
    for m, v in summary.items():
        print(f"Model: {m} | MAE: {v['mae']:.3f} | RMSE: {v['rmse']:.3f} | MAPE: {v['mape']:.3f}% | TailMAE_P90: {v['tail_mae_p90']:.3f} | n_total: {v['n_total']}")

    return summary, per_fold_results

# ---------------------------
# Train full model + next-day forecast (robust)
# ---------------------------
def train_full_model_and_forecast_next_day(df: pd.DataFrame,
                                           feature_cols: List[str],
                                           target_col: str,
                                           model_name: str = "ridge",
                                           use_lgbm: bool = False,
                                           next_day_df: Optional[pd.DataFrame] = None) -> Dict:
    df_work = df.sort_values("timestamp").reset_index(drop=True)
    safe_feature_cols = drop_target_derived_features(feature_cols, target_col)
    feat_avail = [f for f in safe_feature_cols if f in df_work.columns]

    # Build X,y
    if feat_avail:
        X = df_work[feat_avail]
    else:
        X = pd.DataFrame(index=df_work.index)  # empty

    y = df_work[target_col]

    # Train model
    if model_name == "ridge":
        if feat_avail:
            pipe = Models.ridge_pipeline()
            pipe.fit(X, y)
            trained = pipe
            predict_func = lambda Xp: pipe.predict(Xp)
        else:
            # no features, fallback to median predictor
            trained = None
            predict_func = lambda Xp: np.repeat(float(y.median(skipna=True)), len(Xp))
    elif model_name == "lgbm" and use_lgbm and _HAS_LGB:
        if not feat_avail:
            raise ValueError("No features available for LightGBM")
        Ximp = X.fillna(X.median())
        model = Models.fit_lgbm(Ximp, y)
        trained = model
        predict_func = lambda Xp: model.predict(Xp.fillna(X.median()))
    else:
        raise ValueError("Unsupported model_name or LightGBM not available")

    # Build X_next
    if next_day_df is None:
        last_24 = df_work.tail(24).copy().reset_index(drop=True)
        last_24["timestamp"] = last_24["timestamp"] + pd.Timedelta(days=1)
        if feat_avail:
            X_next = last_24[feat_avail]
        else:
            X_next = pd.DataFrame(index=range(24))
        X_next.index = pd.to_datetime(last_24["timestamp"])
    else:
        X_next = next_day_df[[c for c in feat_avail if c in next_day_df.columns]].copy()
        X_next.index = pd.to_datetime(next_day_df["timestamp"])

    # Predict next-day
    preds = pd.Series(predict_func(X_next), index=X_next.index)

    # In-sample predictions for residuals (handle no-features case)
    if feat_avail:
        in_sample_preds = pd.Series(predict_func(X), index=df_work["timestamp"])
    else:
        in_sample_preds = pd.Series([float(y.median(skipna=True))] * len(df_work), index=df_work["timestamp"])

    # compute residuals safe
    residuals = (y.reset_index(drop=True) - in_sample_preds.reset_index(drop=True))
    residuals = residuals.dropna()
    if len(residuals) == 0:
        resid_q = {0.1: 0.0, 0.5: 0.0, 0.9: 0.0}
        resid_std = 0.0
    else:
        resid_q = residuals.quantile([0.1, 0.5, 0.9]).to_dict()
        resid_std = float(residuals.std(skipna=True))

    uncertainty = {"resid_q": resid_q, "resid_std": resid_std}

    return {"model": trained, "forecast": preds, "uncertainty": uncertainty}

# ---------------------------
# translate forecast to tradable view
# ---------------------------
def translate_forecast_to_tradable_view(forecast: pd.Series,
                                        uncertainty: Dict,
                                        prompt_price: Optional[pd.Series] = None,
                                        trade_rule: Optional[Dict] = None) -> Dict:
    if trade_rule is None:
        trade_rule = {"express_threshold_eur": 5.0, "avoid_threshold_eur": -5.0}

    q10 = uncertainty["resid_q"].get(0.1, 0.0)
    q50 = uncertainty["resid_q"].get(0.5, 0.0)
    q90 = uncertainty["resid_q"].get(0.9, 0.0)

    p10 = forecast + q10
    p50 = forecast + q50
    p90 = forecast + q90

    expected_mean = float(forecast.mean())

    signal = "hold"
    if prompt_price is not None:
        common_index = forecast.index.intersection(prompt_price.index)
        if len(common_index) > 0:
            df_cmp = pd.DataFrame({"forecast": forecast.loc[common_index], "prompt": prompt_price.loc[common_index]})
            spread = (df_cmp["forecast"] - df_cmp["prompt"]).mean()
            if spread >= trade_rule["express_threshold_eur"]:
                signal = "express"
            elif spread <= trade_rule["avoid_threshold_eur"]:
                signal = "avoid"
            else:
                signal = "hold"

    fan_bands = pd.DataFrame({"p10": p10, "p50": p50, "p90": p90})
    tradable = {"expected_mean": expected_mean, "fan_bands": fan_bands, "signal": signal, "signal_rule": trade_rule}
    return tradable

# ---------------------------
# Minimal script example
# ---------------------------
if __name__ == "__main__":
    df_feat = pd.read_csv("data/featured_energy_data.csv")
    if "timestamp" not in df_feat.columns:
        df_feat["timestamp"] = pd.to_datetime(df_feat["MTU (UTC)"].str.split(" - ").str[0], dayfirst=True, utc=True)
    else:
        df_feat["timestamp"] = pd.to_datetime(df_feat["timestamp"], utc=True)

    # Example target selected from available columns
    TARGET = "Day-ahead Total Load Forecast (MW)"  # user chose this earlier

    # Build feature list excluding MTU/timestamp/target
    feat_cols = [c for c in df_feat.columns if c not in ("MTU (UTC)", "timestamp", TARGET)]

    # Run evaluation
    summary, per_fold = run_walkforward_evaluation(df_feat, feat_cols, target_col=TARGET, use_lgbm=False)

    # Train full + next day
    model_info = train_full_model_and_forecast_next_day(df_feat, feat_cols, TARGET, model_name="ridge")
    forecast_next_day = model_info["forecast"]
    uncertainty = model_info["uncertainty"]

    tradable = translate_forecast_to_tradable_view(forecast_next_day, uncertainty, prompt_price=None)
    print("\nNext-day expected mean:", tradable["expected_mean"])
    print("Signal:", tradable["signal"])
    print(tradable["fan_bands"].head())
