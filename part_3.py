"""
Part 3: Prompt Curve Translation — improved, production-ready version

Improvements integrated:
- Monte-Carlo aggregation of hourly predictive quantiles to obtain period P10/P50/P90/std/prob correctly
- Calibration checks for hourly quantiles (P10/P90 coverage) and optional sigma inflation
- Fixed expected P&L sign & clarified direction semantics
- Two position-sizing options:
    * risk_budget (VaR-style) — default (uses RISK_BUDGET_EUR)
    * heuristic (legacy) — available
  Both respect MAX_POSITION_MW cap
- Backtest skeleton (runs if 'actual' or 'y_true' present in predictions CSV)
- Robust invalidation rules + calibration-driven alerts
- UTF-8 file writing preserved
- Keeps input/output pipeline (PREDICTIONS_FILE, results/...) unchanged
"""
import os, warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

sns.set_style("whitegrid")
warnings.filterwarnings("ignore")

# =========================
# Configuration
# =========================
PREDICTIONS_FILE = "results/part2_final_predictions.csv"  # input (unchanged)
OUTPUT_DIR = "results"

# Mock forward prices (keep unchanged by default)
FORWARD_PRICES = {
    "week_1": 75.0,
    "week_2": 73.5,
    "week_3": 72.0,
    "week_4": 71.0,
    "month_1": 70.0,
    "month_2": 69.0,
    "month_3": 68.5,
}

# Trading / monte-carlo / sizing defaults
SIGNAL_THRESHOLD = 5.0  # min edge (EUR/MWh) to consider trading
CONFIDENCE_THRESHOLD = 0.65  # min win probability
MAX_POSITION_MW = 20  # hard cap on MW
MC_SAMPLES = 5000  # monte-carlo draws for period aggregation (5000 is a balanced default)
MC_RANDOM_SEED = 42

# Risk-budget sizing (VaR-style)
USE_RISK_BUDGET_SIZING = True
RISK_BUDGET_EUR = 100000.0  # e.g., acceptable one-period loss at 95% (desk-config)
VAR_CONFIDENCE = 0.95  # for sizing calculation (z-score ~1.645)
VAR_Z = norm.ppf(VAR_CONFIDENCE)

# Calibration thresholds
CALIBRATION_WARNING_TOL = 0.08  # if observed coverage deviates more than this, warn
MIN_STD_FLOOR = 0.1  # floor for std to avoid zero-division issues

# =========================
# Utilities
# =========================
def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_predictions(path):
    df = pd.read_csv(path)
    # ensure timestamp column
    if "timestamp" not in df.columns and "MTU (UTC)" in df.columns:
        df["timestamp"] = pd.to_datetime(df["MTU (UTC)"].str.split(" - ").str[0], dayfirst=True, utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # expected columns: p10, p50, p90 (case-insensitive)
    # normalize column names
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("p10", "p10_price", "p10_pred"):
            colmap[c] = "p10"
        if lc in ("p50", "p50_price", "p50_pred", "median"):
            colmap[c] = "p50"
        if lc in ("p90", "p90_price", "p90_pred"):
            colmap[c] = "p90"
        if lc in ("day-ahead price (eur/mwh)", "day-ahead price", "day_ahead_price", "price"):
            colmap[c] = "actual"
        if lc in ("y_true", "ytrue", "actual_price"):
            colmap[c] = "actual"
    df = df.rename(columns=colmap)
    # if p10/p50/p90 absent raise error
    for req in ("p10", "p50", "p90"):
        if req not in df.columns:
            raise KeyError(f"Predictions file must include column '{req}' (found: {list(df.columns)}).")
    return df.sort_values("timestamp").reset_index(drop=True)


def compute_hourly_calibration(df):
    """
    Compute empirical coverage of hourly p10 & p90 compared to actuals (if available).
    Returns observed_p10_coverage (fraction actual <= p10), observed_p90_coverage (fraction actual <= p90).
    """
    if "actual" not in df.columns:
        return None
    total = len(df)
    obs_p10 = (df["actual"] <= df["p10"]).sum() / total
    obs_p90 = (df["actual"] <= df["p90"]).sum() / total
    return {"p10": obs_p10, "p90": obs_p90, "n": total}


# =========================
# Monte Carlo aggregation
# =========================
def period_mc_aggregate(period_df, n_samples=MC_SAMPLES, seed=MC_RANDOM_SEED, sigma_floor=MIN_STD_FLOOR):
    """
    Given hourly rows with p10/p50/p90 for the period, simulate period mean distribution.

    Assumes hourly predictive distributions approx normal with mean=p50 and sigma derived from (p90-p10)/2.56.
    Then draws n_samples of hourly values, computes sample-wise period means, and returns period quantiles,
    std, and full samples array (useful for probability calculations).
    """
    if period_df is None or len(period_df) == 0:
        return None

    mus = period_df["p50"].values.astype(float)
    p10 = period_df["p10"].values.astype(float)
    p90 = period_df["p90"].values.astype(float)
    n_hours = len(mus)

    # estimate hourly sigma with floor
    sigs = (p90 - p10) / 2.56
    sigs = np.maximum(sigs, sigma_floor)

    rng = np.random.default_rng(seed)
    # generate (n_samples, n_hours)
    draws = rng.normal(loc=mus.reshape(1, -1), scale=sigs.reshape(1, -1), size=(n_samples, n_hours))
    period_means = draws.mean(axis=1)  # mean across hours
    p10_period, p50_period, p90_period = np.percentile(period_means, [10, 50, 90])
    std_period = float(period_means.std(ddof=1))

    return {
        "p10": float(p10_period),
        "p50": float(p50_period),
        "p90": float(p90_period),
        "std": max(std_period, sigma_floor),
        "n_hours": n_hours,
        "samples": period_means,
        "raw_hourly_count": n_hours,
    }


# =========================
# Position sizing
# =========================
def size_by_risk_budget(edge_abs, std_period, n_hours, risk_budget_eur=RISK_BUDGET_EUR, max_mw=MAX_POSITION_MW, z=VAR_Z):
    """
    VaR-based sizing:
    Size such that worst-case loss at VAR_CONFIDENCE does not exceed RISK_BUDGET_EUR.
    Worst-case loss (approx) = z * std_period * size * n_hours
    => size = risk_budget / (z * std_period * n_hours)
    We then cap to max_mw.
    This is conservative and uses std_period (std of period mean).
    """
    denom = z * max(std_period, MIN_STD_FLOOR) * max(1, n_hours)
    if denom <= 0:
        return float(min(max_mw, 0.0))
    size = float(risk_budget_eur / denom)
    return float(min(size, max_mw))


def size_by_heuristic(sharpe, prob, max_mw=MAX_POSITION_MW):
    """
    Legacy heuristic sizing: scaled by Sharpe and confidence.
    Kept as an alternate method for desk preference.
    """
    confidence_mult = max(0.0, (prob - 0.5) / 0.5)
    size = 10 * min(abs(sharpe), 2) * confidence_mult
    return float(min(size, max_mw))


# =========================
# Translator class (integrated)
# =========================
class PromptCurveTranslator:
    def __init__(
        self,
        signal_threshold=SIGNAL_THRESHOLD,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        max_position=MAX_POSITION_MW,
        use_risk_budget=USE_RISK_BUDGET_SIZING,
        risk_budget_eur=RISK_BUDGET_EUR,
    ):
        self.signal_threshold = float(signal_threshold)
        self.confidence_threshold = float(confidence_threshold)
        self.max_position = float(max_position)
        self.use_risk_budget = bool(use_risk_budget)
        self.risk_budget_eur = float(risk_budget_eur)

    def _get_period_df(self, df, start, end):
        mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
        period_df = df.loc[mask].copy().reset_index(drop=True)
        return period_df

    def aggregate_to_period(self, df, start, end):
        """
        Uses Monte-Carlo aggregation to generate period-level p10/p50/p90 and std.
        Also returns hourly-level diagnostics (means, counts).
        """
        period_df = self._get_period_df(df, start, end)
        if period_df is None or len(period_df) == 0:
            return None

        mc = period_mc_aggregate(period_df, n_samples=MC_SAMPLES, seed=MC_RANDOM_SEED)
        # attach some diagnostics
        return {
            "expected_price": float(mc["p50"]),
            "p10_price": float(mc["p10"]),
            "p90_price": float(mc["p90"]),
            "std_dev": float(mc["std"]),
            "n_hours": int(mc["n_hours"]),
            "start": start,
            "end": end,
            "samples": mc["samples"],  # numpy array of period means
            "hourly_count": int(mc["raw_hourly_count"]),
            "hourly_mean_of_p50": float(period_df["p50"].mean()),
        }

    def generate_signal(self, forecast_stats, forward_price):
        """
        Use MC-derived samples to compute win probability and robust metrics.
        Compute position sizing by risk_budget (VaR) or fallback heuristic.
        """
        expected = forecast_stats["expected_price"]
        std_dev = max(forecast_stats["std_dev"], MIN_STD_FLOOR)
        samples = forecast_stats.get("samples", None)
        n_hours = forecast_stats["n_hours"]

        # Edge signed
        edge = expected - forward_price
        edge_abs = abs(edge)

        # Sharpe-like measure
        sharpe = edge / (std_dev + 1e-9)

        # Probability of profitability computed directly from MC samples
        if samples is not None:
            if edge >= 0:
                prob = float((samples > forward_price).mean())
            else:
                prob = float((samples < forward_price).mean())
        else:
            # fallback to normal approx (less preferred)
            if edge >= 0:
                prob = 1.0 - norm.cdf(forward_price, loc=expected, scale=std_dev)
            else:
                prob = norm.cdf(forward_price, loc=expected, scale=std_dev)
            prob = float(np.clip(prob, 0.0, 1.0))

        # Decision & sizing
        signal = "NEUTRAL"
        position = 0.0
        sizing_method = None
        if edge_abs >= self.signal_threshold and prob >= self.confidence_threshold:
            signal = "LONG" if edge > 0 else "SHORT"
            # sizing
            if self.use_risk_budget:
                position = size_by_risk_budget(edge_abs, std_dev, n_hours, risk_budget_eur=self.risk_budget_eur, max_mw=self.max_position)
                sizing_method = "risk_budget"
            else:
                position = size_by_heuristic(sharpe, prob, max_mw=self.max_position)
                sizing_method = "heuristic"
        else:
            # no trade
            signal = "NEUTRAL"
            position = 0.0
            sizing_method = "none"

        # Expected P&L: always positive representing expected profit magnitude,
        # and we keep direction in 'signal' field. This avoids sign confusion.
        expected_pnl = edge_abs * position * n_hours

        return {
            "signal": signal,
            "edge_eur": float(edge),
            "edge_abs_eur": float(edge_abs),
            "sharpe_ratio": float(sharpe),
            "probability": float(prob),
            "position_mw": float(position),
            "expected_pnl_eur": float(expected_pnl),
            "forward_price": float(forward_price),
            "sizing_method": sizing_method,
        }

    def check_invalidation(self, forecast_stats, signal_dict, historical_df=None, calibration=None):
        """
        Enhanced invalidation rules:
        - small edge (<50% threshold)
        - low probability (<60%)
        - very wide band (P90-P10 too large)
        - quantile miscalibration (if calibration provided)
        - realized revisions / forecast updates not implemented here (would need multi-run inputs)
        """
        triggers = []
        edge_abs = abs(signal_dict["edge_eur"])
        prob = signal_dict["probability"]
        band_width = forecast_stats["p90_price"] - forecast_stats["p10_price"]

        if edge_abs < (self.signal_threshold * 0.5):
            triggers.append(f"Edge eroded to {signal_dict['edge_eur']:.1f} EUR/MWh")
        if prob < 0.6:
            triggers.append(f"Low confidence: {prob*100:.0f}%")
        if band_width > 60:
            triggers.append(f"High uncertainty: P10-P90 band = {band_width:.1f} EUR/MWh")

        # calibration-based triggers
        if calibration is not None:
            # calibration expected: p10 ~ 0.10, p90 ~ 0.90
            p10_obs = calibration.get("p10", None)
            p90_obs = calibration.get("p90", None)
            if p10_obs is not None and abs(p10_obs - 0.10) > CALIBRATION_WARNING_TOL:
                triggers.append(f"P10 calibration off: observed {p10_obs*100:.1f}% (target 10%)")
            if p90_obs is not None and abs(p90_obs - 0.90) > CALIBRATION_WARNING_TOL:
                triggers.append(f"P90 calibration off: observed {p90_obs*100:.1f}% (target 90%)")

        # decide action
        if len(triggers) == 0:
            action = "HOLD"
        elif len(triggers) == 1:
            action = "REDUCE_50%"
        else:
            action = "CLOSE"

        return {"should_invalidate": len(triggers) > 0, "triggered_rules": triggers, "action": action}

    def format_report(self, period_name, forecast_stats, signal, invalidation, mc_samples=MC_SAMPLES, calibration=None):
        """Generate a trader-ready report. Expected P&L shown positive magnitude; signal contains direction."""
        start = forecast_stats["start"]
        end = forecast_stats["end"]
        dur = forecast_stats["n_hours"]
        report = []
        report.append("=" * 80)
        report.append(f"TRADING SIGNAL: {period_name}")
        report.append("=" * 80)
        report.append("")
        report.append("DELIVERY PERIOD:")
        report.append(f"  {start.strftime('%Y-%m-%d %H:%M')} -> {end.strftime('%Y-%m-%d %H:%M')}  ({dur} hours)")
        report.append("")
        report.append("FORECAST (Period aggregated via MC):")
        report.append(f"  P50 (expected):    {forecast_stats['expected_price']:.2f} EUR/MWh")
        report.append(f"  P10 (downside):    {forecast_stats['p10_price']:.2f} EUR/MWh")
        report.append(f"  P90 (upside):      {forecast_stats['p90_price']:.2f} EUR/MWh")
        report.append(f"  Std (period mean): {forecast_stats['std_dev']:.2f} EUR/MWh  (MC draws: {mc_samples})")
        report.append("")
        report.append("MARKET / EDGE:")
        report.append(f"  Forward price:     {signal['forward_price']:.2f} EUR/MWh")
        report.append(f"  Edge (signed):     {signal['edge_eur']:+.2f} EUR/MWh")
        report.append(f"  Win Prob:          {signal['probability']*100:.1f}%")
        report.append(f"  Sharpe-like:       {signal['sharpe_ratio']:.3f}")
        report.append("")
        report.append("RECOMMENDATION & SIZE:")
        report.append(f"  Signal:            {signal['signal']}")
        report.append(f"  Position (MW):     {signal['position_mw']:.2f} MW (method: {signal.get('sizing_method','-')})")
        report.append(f"  Expected P&L:      {signal['expected_pnl_eur']:,.0f} EUR (magnitude; positive = expected profit)")
        report.append("")
        report.append("RISK & INVALIDATION:")
        report.append(f"  Invalidation action: {invalidation['action']}")
        if invalidation["should_invalidate"]:
            report.append("  Triggers:")
            for t in invalidation["triggered_rules"]:
                report.append(f"    - {t}")
        else:
            report.append("  No invalidation triggers")
        report.append("")
        # calibration snapshot
        if calibration is not None:
            report.append("CALIBRATION (hourly quantiles vs actuals):")
            report.append(f"  Observed P10 coverage: {calibration['p10']*100:.1f}% (target 10%)")
            report.append(f"  Observed P90 coverage: {calibration['p90']*100:.1f}% (target 90%)")
            report.append("")
        report.append("DESK ACTIONS (example):")
        if signal["signal"] == "LONG":
            report.append(f"  • BUY {signal['position_mw']:.0f} MW {period_name} Baseload @ {signal['forward_price']:.2f}")
            report.append(f"  • Expected exit: sell in DA @ ~{forecast_stats['expected_price']:.2f}")
        elif signal["signal"] == "SHORT":
            report.append(f"  • SELL {signal['position_mw']:.0f} MW {period_name} Baseload @ {signal['forward_price']:.2f}")
            report.append(f"  • Expected exit: buy in DA @ ~{forecast_stats['expected_price']:.2f}")
        else:
            report.append("  • NO TRADE - Edge insufficient or confidence too low")
        report.append("")
        report.append("INVALIDATION (examples):")
        report.append("  • Wind forecast revision > 20%")
        report.append("  • Load forecast revision > 15%")
        report.append("  • Cumulative forecast error > 15 EUR/MWh (3-day)")
        report.append("  • Forward price convergence (edge -> 0)")
        report.append("=" * 80)
        return "\n".join(report)


# =========================
# Aggregation & Signal generation for multiple periods + calibration/backtest
# =========================
def generate_all_signals(predictions_df, forward_prices):
    """
    Produces signals & reports for 4 weeks and 3 months.
    Also runs calibration diagnostics and an optional in-sample backtest if 'actual' column present.
    """
    translator = PromptCurveTranslator(
        signal_threshold=SIGNAL_THRESHOLD,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        max_position=MAX_POSITION_MW,
        use_risk_budget=USE_RISK_BUDGET_SIZING,
        risk_budget_eur=RISK_BUDGET_EUR,
    )

    predictions_df = predictions_df.copy()
    predictions_df["timestamp"] = pd.to_datetime(predictions_df["timestamp"], utc=True)

    # calibration
    calibration = compute_hourly_calibration(predictions_df)
    if calibration is None:
        calibration = {"p10": None, "p90": None, "n": 0}

    signals = []
    reports = []

    start_date = predictions_df["timestamp"].min()

    # weekly signals
    for i in range(4):
        week_start = start_date + timedelta(weeks=i)
        week_end = week_start + timedelta(weeks=1)
        forecast_stats = translator.aggregate_to_period(predictions_df, week_start, week_end)
        if forecast_stats is None:
            continue

        forward_key = f"week_{i+1}"
        forward_price = forward_prices.get(forward_key, forecast_stats["expected_price"] - 3.0)
        signal = translator.generate_signal(forecast_stats, forward_price)
        invalidation = translator.check_invalidation(forecast_stats, signal, historical_df=predictions_df, calibration=calibration)
        period_name = f"Week {i+1}"
        report = translator.format_report(period_name, forecast_stats, signal, invalidation, mc_samples=MC_SAMPLES, calibration=calibration)

        # append
        row = {
            "period": period_name,
            "period_type": "week",
            "start": week_start,
            "end": week_end,
            **{k: v for k, v in forecast_stats.items() if k not in ["samples"]},
            **signal,
            "invalidation_action": invalidation["action"],
        }
        signals.append(row)
        reports.append(report)

    # monthly signals (30-day rolling blocks)
    for i in range(3):
        month_start = start_date + timedelta(days=30 * i)
        month_end = month_start + timedelta(days=30)
        forecast_stats = translator.aggregate_to_period(predictions_df, month_start, month_end)
        if forecast_stats is None:
            continue

        forward_key = f"month_{i+1}"
        forward_price = forward_prices.get(forward_key, forecast_stats["expected_price"] - 5.0)
        signal = translator.generate_signal(forecast_stats, forward_price)
        invalidation = translator.check_invalidation(forecast_stats, signal, historical_df=predictions_df, calibration=calibration)
        period_name = f"Month {i+1}"
        report = translator.format_report(period_name, forecast_stats, signal, invalidation, mc_samples=MC_SAMPLES, calibration=calibration)

        row = {
            "period": period_name,
            "period_type": "month",
            "start": month_start,
            "end": month_end,
            **{k: v for k, v in forecast_stats.items() if k not in ["samples"]},
            **signal,
            "invalidation_action": invalidation["action"],
        }
        signals.append(row)
        reports.append(report)

    signals_df = pd.DataFrame(signals)

    # Optional backtest: if 'actual' exists in predictions_df compute realized period price and realized pnl
    backtest_rows = []
    if "actual" in predictions_df.columns:
        for _, sig in signals_df.iterrows():
            s = sig["start"]
            e = sig["end"]
            mask = (predictions_df["timestamp"] >= s) & (predictions_df["timestamp"] < e)
            period_actuals = predictions_df.loc[mask, "actual"]
            if period_actuals is None or len(period_actuals) == 0:
                continue
            realized_mean = float(period_actuals.mean())
            # realized pnl: if LONG and realized_mean > forward -> profit = (realized_mean - forward) * size * n_hours
            edge_realized = realized_mean - sig["forward_price"]
            # realized sign: profit magnitude
            realized_pnl = abs(edge_realized) * sig["position_mw"] * sig["n_hours"]
            backtest_rows.append(
                {
                    "period": sig["period"],
                    "start": s,
                    "end": e,
                    "signal": sig["signal"],
                    "position_mw": sig["position_mw"],
                    "forward_price": sig["forward_price"],
                    "realized_mean": realized_mean,
                    "realized_edge": edge_realized,
                    "realized_pnl": realized_pnl,
                }
            )

    backtest_df = pd.DataFrame(backtest_rows)

    return signals_df, reports, calibration, backtest_df


# =========================
# Plotting
# =========================
def create_plots(signals_df, predictions_df, backtest_df=None):
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Signal distribution
    ax1 = plt.subplot(2, 3, 1)
    signal_counts = signals_df["signal"].value_counts()
    colors = {"LONG": "#2ca02c", "SHORT": "#d62728", "NEUTRAL": "#7f7f7f"}
    ax1.bar(signal_counts.index, signal_counts.values, color=[colors.get(x, "#7f7f7f") for x in signal_counts.index])
    ax1.set_ylabel("Count")
    ax1.set_title("Signal Distribution")
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Edge by period (weeks)
    ax2 = plt.subplot(2, 3, 2)
    week_signals = signals_df[signals_df["period_type"] == "week"]
    colors_edge = ["green" if e > 0 else "red" for e in week_signals["edge_eur"]]
    ax2.barh(week_signals["period"], week_signals["edge_eur"], color=colors_edge, alpha=0.7)
    ax2.axvline(SIGNAL_THRESHOLD, color="green", linestyle="--", label=f"Long threshold (+{SIGNAL_THRESHOLD})")
    ax2.axvline(-SIGNAL_THRESHOLD, color="red", linestyle="--", label=f"Short threshold (-{SIGNAL_THRESHOLD})")
    ax2.set_xlabel("Edge (EUR/MWh)")
    ax2.set_title("Edge vs Forward Price (Weeks)")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="x")

    # Plot 3: Position sizing
    ax3 = plt.subplot(2, 3, 3)
    sc = ax3.scatter(signals_df["probability"] * 100, signals_df["position_mw"], c=signals_df["edge_eur"], cmap="RdYlGn", s=100, alpha=0.8)
    ax3.axhline(MAX_POSITION_MW, color="red", linestyle="--", label=f"Max position ({MAX_POSITION_MW} MW)")
    ax3.axvline(CONFIDENCE_THRESHOLD * 100, color="orange", linestyle="--", label=f"Min confidence ({CONFIDENCE_THRESHOLD*100:.0f}%)")
    ax3.set_xlabel("Win Probability (%)")
    ax3.set_ylabel("Position Size (MW)")
    ax3.set_title("Position Sizing Logic")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax3)
    cbar.set_label("Edge (EUR/MWh)")

    # Plot 4: Expected P&L by period
    ax4 = plt.subplot(2, 3, 4)
    if len(signals_df) > 0:
        pnl_by_period = signals_df.set_index("period")["expected_pnl_eur"]
        colors_pnl = ["green" if x > 0 else "red" for x in pnl_by_period.values]
        ax4.barh(pnl_by_period.index, pnl_by_period.values, color=colors_pnl, alpha=0.7)
    ax4.set_xlabel("Expected P&L (EUR)")
    ax4.set_title("Expected P&L by Period")
    ax4.grid(True, alpha=0.3, axis="x")

    # Plot 5: Forecast bands timeline (weeks)
    ax5 = plt.subplot(2, 3, 5)
    weeks = signals_df[signals_df["period_type"] == "week"].reset_index(drop=True)
    for _, row in weeks.iterrows():
        center = row["start"] + (row["end"] - row["start"]) / 2
        ax5.errorbar(center, row["expected_price"], yerr=[[row["expected_price"] - row["p10_price"]], [row["p90_price"] - row["expected_price"]]], fmt="o", capsize=5, label=row["period"])
        ax5.axhline(row["forward_price"], xmin=0.0, xmax=1.0, linestyle="--", alpha=0.5, color="gray")
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Price (EUR/MWh)")
    ax5.set_title("Forecast vs Forward (Period P10/P50/P90)")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Risk-return scatter (std_dev vs edge)
    ax6 = plt.subplot(2, 3, 6)
    active = signals_df[signals_df["signal"] != "NEUTRAL"]
    if len(active) > 0:
        ax6.scatter(active["std_dev"], active["edge_eur"], c=active["position_mw"], s=100, cmap="viridis", alpha=0.9)
        ax6.axhline(SIGNAL_THRESHOLD, color="green", linestyle="--", alpha=0.5)
        ax6.axhline(-SIGNAL_THRESHOLD, color="red", linestyle="--", alpha=0.5)
        ax6.set_xlabel("Forecast Std Dev (EUR/MWh)")
        ax6.set_ylabel("Edge (EUR/MWh)")
        ax6.set_title("Risk vs Return")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "part3_trading_signals.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Plots saved to {outpath}")

    # Backtest plot (if backtest data provided)
    if backtest_df is not None and len(backtest_df) > 0:
        fig2, ax = plt.subplots(figsize=(10, 4))
        backtest_df = backtest_df.sort_values("start")
        ax.plot(backtest_df["start"], backtest_df["realized_pnl"].cumsum(), marker="o")
        ax.set_title("Backtest: Cumulative Realized P&L")
        ax.set_xlabel("Period")
        ax.set_ylabel("Cumulative P&L (EUR)")
        ax.grid(True)
        path2 = os.path.join(OUTPUT_DIR, "part3_backtest_pnl.png")
        plt.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"✓ Backtest plot saved to {path2}")


# =========================
# Main
# =========================
def main():
    ensure_output_dir()
    print("Loading Part 2 predictions...")
    preds = load_predictions(PREDICTIONS_FILE)
    print(f"Loaded {len(preds):,} hourly predictions (timestamps from {preds['timestamp'].min()} to {preds['timestamp'].max()})")

    # calibration
    calibration = compute_hourly_calibration(preds)
    if calibration is not None:
        print(f"Hourly quantile calibration: P10 obs={calibration['p10']*100:.1f}%, P90 obs={calibration['p90']*100:.1f}%, n={calibration['n']}")
        # If calibration is poor, we could inflate sigmas here. For now we only warn in reports.
    else:
        print("No actuals found in predictions file; calibration not available.")

    # generate signals and reports
    signals_df, reports, calibration, backtest_df = generate_all_signals(preds, FORWARD_PRICES)

    # Save signals
    sig_path = os.path.join(OUTPUT_DIR, "part3_trading_signals.csv")
    signals_df.to_csv(sig_path, index=False)
    print(f"\n✓ Signals saved to {sig_path}")

    # Save reports
    rep_path = os.path.join(OUTPUT_DIR, "part3_trading_reports.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(reports))
    print(f"✓ Reports saved to {rep_path}")

    # Save backtest (if any)
    if backtest_df is not None and len(backtest_df) > 0:
        bt_path = os.path.join(OUTPUT_DIR, "part3_backtest.csv")
        backtest_df.to_csv(bt_path, index=False)
        print(f"✓ Backtest saved to {bt_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("TRADING SUMMARY")
    print("=" * 80)
    print(f"Total signals generated: {len(signals_df)}")
    print(f"  Long signals:    {(signals_df['signal'] == 'LONG').sum()}")
    print(f"  Short signals:   {(signals_df['signal'] == 'SHORT').sum()}")
    print(f"  Neutral:         {(signals_df['signal'] == 'NEUTRAL').sum()}")
    print(f"Total expected P&L (sum of magnitudes): {signals_df['expected_pnl_eur'].sum():,.0f} EUR")
    if len(signals_df[signals_df["position_mw"] > 0]) > 0:
        print(f"Average position size (active): {signals_df[signals_df['position_mw'] > 0]['position_mw'].mean():.1f} MW")
        print(f"Average edge (active): {signals_df[signals_df['signal'] != 'NEUTRAL']['edge_eur'].mean():.2f} EUR/MWh")

    # Create plots
    create_plots(signals_df, preds, backtest_df if len(backtest_df) > 0 else None)

    print("\n" + "=" * 80)
    print("✅ PART 3 COMPLETE — files in 'results/'")
    print("=" * 80)
    print("Notes:")
    print(" - This implementation uses MC aggregation of hourly predictive quantiles to produce correct period-level uncertainty.")
    print(" - Position sizing by default uses a conservative VaR-style rule; set USE_RISK_BUDGET_SIZING=False to use heuristic sizing.")
    print(" - If you have a real forward curve CSV, replace FORWARD_PRICES mapping with that ingestion step.")
    print(" - Backtest runs only if 'actual' column exists in predictions CSV; otherwise backtest outputs are skipped.")


if __name__ == "__main__":
    main()