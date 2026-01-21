import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import truncnorm
import os, math, logging
from datetime import timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ---------------------------
# Configuration (tweakable)
# ---------------------------
PREDICTIONS_FILE = "results/part2_final_predictions.csv"
OUTPUT_DIR = "results"
PNG_OUT = os.path.join(OUTPUT_DIR, "part3_trading_signals.png")
CSV_OUT = os.path.join(OUTPUT_DIR, "part3_trading_signals.csv")
REPORT_OUT = os.path.join(OUTPUT_DIR, "part3_trading_reports.txt")

SIGNAL_THRESHOLD = 5.0          # EUR/MWh minimum edge to consider trading
CONFIDENCE_THRESHOLD = 0.60     # minimum adjusted win probability to trade
MAX_POSITION_MW = 20.0          # absolute cap on MW position
MC_SAMPLES = 5000
RANDOM_SEED = 2026
MIN_STD_FLOOR = 0.5             # floor on hourly std to avoid division by zero
PROB_CLIP = (0.02, 0.98)        # avoid certainty
UNCERTAINTY_DECAY_SCALE = 30.0  # scale for uncertainty penalty (higher -> less penalty)
MAX_EXPECTED_PNL_CAP = 5e6      # safety cap on expected P&L (EUR)
FORWARD_PRICE_FALLBACK_DELTA_W = -3.0   # if forward not provided, fallback p50-3 for weeks
FORWARD_PRICE_FALLBACK_DELTA_M = -5.0   # fallback for months

# SIMULATION: Example forward prices placeholder (replace by real API-fetch if available)
FORWARD_PRICES = {
    'week_1': 75.0, 'week_2': 78.5, 'week_3': 76.0, 'week_4': 80.0, 'month_1': 79.5, 'month_2': 88.0
}

# ---------------------------
# Utilities
# ---------------------------
def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)

def safe_parse_predictions(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Predictions file not found: {path}")
    df = pd.read_csv(path)
    if 'timestamp' not in df.columns:
        raise ValueError("Predictions file must contain 'timestamp' column")
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    if df['timestamp'].isna().any():
        raise ValueError("Some timestamps could not be parsed; check input CSV")
    # ensure quantile columns exist
    for c in ['p10', 'p50', 'p90']:
        if c not in df.columns:
            raise ValueError(f"Predictions file missing required column: {c}")
    # enforce p10 <= p50 <= p90 per hour
    df['p10'] = np.minimum(df['p10'], df['p50'])
    df['p90'] = np.maximum(df['p90'], df['p50'])
    return df.sort_values('timestamp').reset_index(drop=True)

# ---------------------------
# MC aggregation (hourly -> period)
# ---------------------------
def quantiles_to_sigma(p10, p90):
    # For normal approx: p90 - p10 = 2 * z(0.9) * sigma = 2.563102...
    denom = 2.563102  # 2*z(0.9)
    sigma = (p90 - p10) / denom
    return np.maximum(sigma, MIN_STD_FLOOR)

def aggregate_period_mc(period_df, n_samples=MC_SAMPLES, rng=None):
    """
    period_df: DataFrame with columns p10,p50,p90
    returns: dict with p10,p50,p90,std,samples (period means)
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    # drop any missing hours
    period_df = period_df.dropna(subset=['p10','p50','p90'])
    n_hours = len(period_df)
    if n_hours == 0:
        return None

    p10 = period_df['p10'].values
    p50 = period_df['p50'].values
    p90 = period_df['p90'].values

    sigma = quantiles_to_sigma(p10, p90)

    # Use truncated normal between p10-4*sigma and p90+4*sigma to limit extreme draws
    # For vectorized sampling we'll sample hour-by-hour
    samples = np.empty((n_samples, n_hours), dtype=float)
    # precompute trunc parameters per hour
    for i in range(n_hours):
        loc = p50[i]
        scale = sigma[i]
        a, b = (p10[i] - loc) / scale, (p90[i] - loc) / scale
        # expand a/b a bit to allow draws slightly outside the p10/p90 band but bounded
        a = a - 0.5
        b = b + 0.5
        # handle ill-conditioned scale
        if scale <= 0:
            samples[:, i] = loc
            continue
        tn = truncnorm(a, b, loc=loc, scale=scale)
        samples[:, i] = tn.rvs(size=n_samples, random_state=rng)

    # ensure no negative prices
    samples = np.clip(samples, 0.0, None)

    # period mean per MC sample
    period_means = samples.mean(axis=1)

    p10_period = np.percentile(period_means, 10)
    p50_period = np.percentile(period_means, 50)
    p90_period = np.percentile(period_means, 90)
    std_period = period_means.std(ddof=1)

    return {
        'p10': float(p10_period),
        'p50': float(p50_period),
        'p90': float(p90_period),
        'std': float(std_period),
        'samples': period_means,
        'n_hours': n_hours
    }

# ---------------------------
# Calibration check
# ---------------------------
def check_calibration(df):
    # df should have hourly p10/p90 and y_true where available
    if 'y_true' not in df.columns:
        return None
    eval_df = df.dropna(subset=['y_true','p10','p90'])
    if len(eval_df) == 0:
        return None
    total = len(eval_df)
    p10_cov = (eval_df['y_true'] <= eval_df['p10']).sum() / total
    p90_cov = (eval_df['y_true'] <= eval_df['p90']).sum() / total
    return {
        'p10_coverage': float(p10_cov),
        'p90_coverage': float(p90_cov),
        'p10_calibrated': abs(p10_cov - 0.10) < 0.08,
        'p90_calibrated': abs(p90_cov - 0.90) < 0.08,
        'n': int(total)
    }

# ---------------------------
# Signal generator (safe)
# ---------------------------
class TradingSignalGenerator:
    def __init__(self, signal_thresh=SIGNAL_THRESHOLD, conf_thresh=CONFIDENCE_THRESHOLD,
                 max_pos=MAX_POSITION_MW):
        self.signal_thresh = signal_thresh
        self.conf_thresh = conf_thresh
        self.max_pos = max_pos

    def generate(self, period_stats, forward_price):
        """
        period_stats: output of aggregate_period_mc
        forward_price: float
        """
        expected = period_stats['p50']
        std = max(period_stats['std'], MIN_STD_FLOOR)
        samples = period_stats['samples']
        n_hours = period_stats['n_hours']

        edge = expected - forward_price

        # MC probability that period_mean > forward (for long) or < forward (for short)
        if edge > 0:
            prob_raw = float((samples > forward_price).mean())
        else:
            prob_raw = float((samples < forward_price).mean())

        # Uncertainty penalty based on band width (wider -> reduce effective prob)
        band_width = period_stats['p90'] - period_stats['p10']
        uncertainty_pen = math.exp(-band_width / UNCERTAINTY_DECAY_SCALE)
        prob_adj = prob_raw * (0.6 + 0.4 * uncertainty_pen)  # keep base conservatism

        # clip probability
        prob_adj = float(np.clip(prob_adj, PROB_CLIP[0], PROB_CLIP[1]))

        # Signal decision
        signal = 'NEUTRAL'
        position_mw = 0.0

        if abs(edge) >= self.signal_thresh and prob_adj >= self.conf_thresh:
            signal = 'LONG' if edge > 0 else 'SHORT'

            # Confidence multiplier in (0,1)
            conf_mult = max(0.0, (prob_adj - 0.5) / 0.5)

            # Signal-to-noise: use edge relative to std (normalized by sqrt(hours) since averaging)
            snr = abs(edge) / (std / math.sqrt(max(1, n_hours)))
            # normalize snr (soft cap)
            snr_factor = min(snr / 2.0, 1.0)  # SNR ~2 considered strong

            # combine into a size fraction
            size_frac = conf_mult * snr_factor
            position_mw = float(min(self.max_pos, max(0.0, self.max_pos * size_frac)))
        else:
            prob_adj = float(np.clip(prob_adj, 0.0, 1.0))

        # Conservative expected P&L: edge * position * hours, but apply execution slippage factor (0.7)
        expected_pnl = float(np.clip(abs(edge) * position_mw * n_hours * 0.7, 0.0, MAX_EXPECTED_PNL_CAP))

        # Sharpe-like metric (edge / std)
        sharpe = float(edge / std)

        return {
            'signal': signal,
            'edge': float(edge),
            'sharpe': sharpe,
            'probability_raw': float(prob_raw),
            'probability': prob_adj,
            'position_mw': position_mw,
            'expected_pnl': expected_pnl,
            'forward_price': float(forward_price),
            'band_width': float(band_width)
        }

    def invalidation_checks(self, period_stats, signal_dict, calibration):
        triggers = []
        # Edge erosion
        if abs(signal_dict['edge']) < self.signal_thresh * 0.5:
            triggers.append(f"Edge eroded ({signal_dict['edge']:+.1f} EUR/MWh)")

        # Low confidence
        if signal_dict['probability'] < 0.6:
            triggers.append(f"Low confidence: {signal_dict['probability']*100:.0f}%")

        # Wide uncertainty band
        if period_stats['p90'] - period_stats['p10'] > 40:
            triggers.append(f"Wide band: {period_stats['p90'] - period_stats['p10']:.1f} EUR/MWh")

        # Calibration warnings
        if calibration is not None:
            if not calibration.get('p10_calibrated', True):
                triggers.append(f"P10 miscalibrated ({calibration['p10_coverage']*100:.0f}%)")
            if not calibration.get('p90_calibrated', True):
                triggers.append(f"P90 miscalibrated ({calibration['p90_coverage']*100:.0f}%)")

        # Verdict
        if len(triggers) == 0:
            action = 'HOLD'
        elif len(triggers) == 1:
            action = 'REDUCE_50%'
        else:
            action = 'CLOSE'

        return {'triggers': triggers, 'action': action}

    def format_report(self, period_name, period_range, period_stats, signal, invalidation, calibration):
        s = []
        s.append("="*70)
        s.append(f"TRADING SIGNAL: {period_name}")
        s.append("="*70)
        s.append(f"DELIVERY: {period_range[0].isoformat()} -> {period_range[1].isoformat()} ({period_stats['n_hours']} hours)")
        s.append("")
        s.append("FORECAST (MC aggregated):")
        s.append(f"  P50: {period_stats['p50']:.2f} EUR/MWh")
        s.append(f"  P10: {period_stats['p10']:.2f} EUR/MWh")
        s.append(f"  P90: {period_stats['p90']:.2f} EUR/MWh")
        s.append(f"  Std: {period_stats['std']:.2f} EUR/MWh")
        s.append("")
        s.append("MARKET:")
        s.append(f"  Forward price: {signal['forward_price']:.2f} EUR/MWh")
        s.append(f"  Edge: {signal['edge']:+.2f} EUR/MWh")
        s.append(f"  Sharpe: {signal['sharpe']:.2f}")
        s.append(f"  Win Prob (adj): {signal['probability']*100:.1f}%")
        s.append("")
        s.append("RECOMMENDATION:")
        s.append(f"  Signal: {signal['signal']}")
        s.append(f"  Position: {signal['position_mw']:.2f} MW (baseload)")
        s.append(f"  Expected P&L (conservative): {signal['expected_pnl']:,.0f} EUR")
        s.append("")
        s.append("RISK MANAGEMENT:")
        s.append(f"  Action: {invalidation['action']}")
        if invalidation['triggers']:
            s.append("  Alerts:")
            for t in invalidation['triggers']:
                s.append(f"    - {t}")
        else:
            s.append("  No alerts.")
        s.append("")
        s.append("INVALIDATION TRIGGERS (examples):")
        s.append("  • Wind forecast revision >20%")
        s.append("  • 3-day forecast error >15 EUR/MWh")
        s.append("  • Forward price converges to forecast")
        if calibration is not None:
            s.append("")
            s.append("CALIBRATION (hourly Evidence):")
            s.append(f"  P10 observed: {calibration['p10_coverage']*100:.0f}% (target 10%)  N={calibration['n']}")
            s.append(f"  P90 observed: {calibration['p90_coverage']*100:.0f}% (target 90%)")
        s.append("")
        if signal['signal'] == 'LONG':
            s.append(f"DESK ACTION: BUY {signal['position_mw']:.0f} MW {period_name} @ {signal['forward_price']:.2f}")
        elif signal['signal'] == 'SHORT':
            s.append(f"DESK ACTION: SELL {signal['position_mw']:.0f} MW {period_name} @ {signal['forward_price']:.2f}")
        else:
            s.append("DESK ACTION: NO TRADE (edge/confidence insufficient)")
        s.append("\n")
        return "\n".join(s)

# ---------------------------
# Generate all period signals
# ---------------------------
def generate_period_signals(preds_df, forward_prices):
    preds_df = preds_df.copy()
    preds_df = preds_df.sort_values('timestamp').reset_index(drop=True)
    start_date = preds_df['timestamp'].dt.floor('D').min()

    calibration = check_calibration(preds_df)
    gen = TradingSignalGenerator()
    rng = np.random.default_rng(RANDOM_SEED)

    signals = []
    reports = []

    # Weekly signals (next 4 calendar weeks from first available day)
    for i in range(4):
        week_start = start_date + timedelta(weeks=i)
        week_end = week_start + timedelta(weeks=1)
        mask = (preds_df['timestamp'] >= week_start) & (preds_df['timestamp'] < week_end)
        period_df = preds_df.loc[mask]
        if len(period_df) < 6:  # require at least some coverage; skip sparse periods
            continue
        period_stats = aggregate_period_mc(period_df, n_samples=MC_SAMPLES, rng=rng)
        if period_stats is None:
            continue
        period_stats['start'] = week_start
        period_stats['end'] = week_end

        forward_price = forward_prices.get(f'week_{i+1}', period_stats['p50'] + FORWARD_PRICE_FALLBACK_DELTA_W)
        signal = gen.generate(period_stats, forward_price)
        invalidation = gen.invalidation_checks(period_stats, signal, calibration)
        period_name = f"Week {i+1}"

        report = gen.format_report(period_name, (week_start, week_end), period_stats, signal, invalidation, calibration)
        signals.append({
            'period': period_name,
            'type': 'week',
            'start': week_start.isoformat(),
            'end': week_end.isoformat(),
            'p10': period_stats['p10'],
            'p50': period_stats['p50'],
            'p90': period_stats['p90'],
            'std': period_stats['std'],
            'n_hours': period_stats['n_hours'],
            **signal,
            'invalidation_action': invalidation['action']
        })
        reports.append(report)

    # Monthly signals (30-day rolling windows)
    for i in range(2):
        month_start = start_date + timedelta(days=30 * i)
        month_end = month_start + timedelta(days=30)
        mask = (preds_df['timestamp'] >= month_start) & (preds_df['timestamp'] < month_end)
        period_df = preds_df.loc[mask]
        if len(period_df) < 24:  # require at least a day's coverage
            continue
        period_stats = aggregate_period_mc(period_df, n_samples=MC_SAMPLES, rng=rng)
        if period_stats is None:
            continue
        period_stats['start'] = month_start
        period_stats['end'] = month_end

        forward_price = forward_prices.get(f'month_{i+1}', period_stats['p50'] + FORWARD_PRICE_FALLBACK_DELTA_M)
        signal = gen.generate(period_stats, forward_price)
        invalidation = gen.invalidation_checks(period_stats, signal, calibration)
        period_name = f"Month {i+1}"

        report = gen.format_report(period_name, (month_start, month_end), period_stats, signal, invalidation, calibration)
        signals.append({
            'period': period_name,
            'type': 'month',
            'start': month_start.isoformat(),
            'end': month_end.isoformat(),
            'p10': period_stats['p10'],
            'p50': period_stats['p50'],
            'p90': period_stats['p90'],
            'std': period_stats['std'],
            'n_hours': period_stats['n_hours'],
            **signal,
            'invalidation_action': invalidation['action']
        })
        reports.append(report)

    signals_df = pd.DataFrame(signals)
    return signals_df, reports, calibration

# ---------------------------
# Visualization
# ---------------------------
def create_plots(signals_df):
    if signals_df is None or len(signals_df) == 0:
        logging.warning("No signals to plot.")
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    ax = axes.flatten()

    # 1. Signal counts
    counts = signals_df['signal'].value_counts()
    ax[0].bar(counts.index.astype(str), counts.values)
    ax[0].set_title("Signal distribution")
    ax[0].grid(True, alpha=0.3)

    # 2. Edge by period (weeks)
    weeks = signals_df[signals_df['type'] == 'week']
    if not weeks.empty:
        ax[1].barh(weeks['period'], weeks['edge'], color=['green' if x > 0 else 'red' for x in weeks['edge']])
        ax[1].axvline(SIGNAL_THRESHOLD, color='green', linestyle='--')
        ax[1].axvline(-SIGNAL_THRESHOLD, color='red', linestyle='--')
        ax[1].set_title("Weekly edges")
        ax[1].grid(True, alpha=0.3)

    # 3. Position sizing vs probability
    ax[2].scatter(signals_df['probability'] * 100, signals_df['position_mw'], s=80, alpha=0.9)
    ax[2].axvline(CONFIDENCE_THRESHOLD * 100, color='orange', linestyle='--')
    ax[2].axhline(MAX_POSITION_MW, color='red', linestyle='--')
    ax[2].set_xlabel("Win probability (%)")
    ax[2].set_ylabel("Position (MW)")
    ax[2].set_title("Position sizing")

    # 4. Expected P&L
    ax[3].barh(signals_df['period'], signals_df['expected_pnl'], color=['green' if x > 0 else 'red' for x in signals_df['expected_pnl']])
    ax[3].set_title("Expected P&L by period")
    ax[3].grid(True, alpha=0.3)

    # 5. Forecast bands vs forward (weeks)
    if not weeks.empty:
        for _, r in weeks.iterrows():
            center = pd.to_datetime(r['start']) + (pd.to_datetime(r['end']) - pd.to_datetime(r['start'])) / 2
            ax[4].errorbar(center, r['p50'],
                           yerr=[[r['p50'] - r['p10']], [r['p90'] - r['p50']]],
                           fmt='o', capsize=5)
            ax[4].axhline(r['forward_price'], color='gray', linestyle='--', alpha=0.6)
        ax[4].set_title("Weekly forecast bands vs forward")
        ax[4].set_ylabel("EUR/MWh")
        ax[4].grid(True, alpha=0.3)

    # 6. Risk-return scatter (std vs edge)
    active = signals_df[signals_df['signal'] != 'NEUTRAL']
    if not active.empty:
        for t, df_t in active.groupby('signal'):
            ax[5].scatter(df_t['std'], df_t['edge'], label=t, s=80, alpha=0.8)
        ax[5].axhline(SIGNAL_THRESHOLD, color='green', linestyle='--')
        ax[5].axhline(-SIGNAL_THRESHOLD, color='red', linestyle='--')
        ax[5].set_xlabel("Forecast std (EUR/MWh)")
        ax[5].set_ylabel("Edge (EUR/MWh)")
        ax[5].set_title("Risk vs Return")
        ax[5].legend()
        ax[5].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PNG_OUT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved plot to {PNG_OUT}")

# ---------------------------
# Main
# ---------------------------
def main():
    ensure_output_dir(OUTPUT_DIR)
    logging.info(f"Loading predictions: {PREDICTIONS_FILE}")
    preds = safe_parse_predictions(PREDICTIONS_FILE)
    logging.info(f"Loaded {len(preds):,} hourly predictions from {preds['timestamp'].min().isoformat()} to {preds['timestamp'].max().isoformat()}")

    signals_df, reports, calibration = generate_period_signals(preds, FORWARD_PRICES)

    # Save CSV & reports
    if signals_df is None or signals_df.empty:
        logging.warning("No signals generated. Exiting.")
        return

    signals_df.to_csv(CSV_OUT, index=False)
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write("\n\n".join(reports))

    # Summary to stdout
    total_long = int((signals_df['signal'] == 'LONG').sum())
    total_short = int((signals_df['signal'] == 'SHORT').sum())
    total_neutral = int((signals_df['signal'] == 'NEUTRAL').sum())
    total_expected_pnl = float(signals_df['expected_pnl'].sum())

    logging.info("SUMMARY (Part 3)")
    logging.info(f"Long: {total_long} | Short: {total_short} | Neutral: {total_neutral}")
    logging.info(f"Total expected P&L (conservative cap applied): {total_expected_pnl:,.0f} EUR")
    if calibration is not None:
        logging.info(f"Calibration: P10 obs {calibration['p10_coverage']*100:.0f}% (target 10%), P90 obs {calibration['p90_coverage']*100:.0f}% (target 90%), N={calibration['n']}")

    # plots
    create_plots(signals_df)
    logging.info("Part 3 completed.")

if __name__ == "__main__":
    main()
