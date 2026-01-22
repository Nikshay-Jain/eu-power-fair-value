import os, re, json, math, logging
from datetime import datetime, timedelta
from collections import OrderedDict

import pandas as pd
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
    from langchain_google_genai import ChatGoogleGenerativeAI
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

# -----------------------
# Configuration
# -----------------------
OUTPUT_DIR = "results"
LOG_FILE = os.path.join(OUTPUT_DIR, "part4_ai_log.txt")
TXT_OUT = os.path.join(OUTPUT_DIR, "part4_trader_commentary.txt")
JSON_OUT = os.path.join(OUTPUT_DIR, "part4_trader_commentary.json")
EVIDENCE_OUT = os.path.join(OUTPUT_DIR, "part4_trader_commentary_evidence.txt")

IMPORTANCE_FILE = os.path.join(OUTPUT_DIR, "part2_feature_importance.csv")
PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, "part2_final_predictions.csv")
SIGNALS_FILE = os.path.join(OUTPUT_DIR, "part3_trading_signals.csv")
FEATURE_DATA_FILE = os.path.join("data", "featured_energy_data.csv")

# LLM config
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = "gemini-2.5-flash"

# Numeric tolerances for fact-checking
REL_TOL = 0.12   # 12% relative tolerance for floats extracted from LLM text
ABS_TOL = 0.5    # or 0.5 EUR/MWh absolute tolerance

# Logging
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -----------------------
# Helpers
# -----------------------
def read_csv_safe(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            logging.error(f"Failed to read {path}: {e}")
            return None
    else:
        logging.info(f"File not found: {path}")
        return None

def choose_primary_signal(signals_df):
    """
    Choose the most relevant row from part3 signals:
     - prefer WEEK signals (Week 1) if present, else top expected_pnl, else first row
    """
    if signals_df is None or signals_df.empty:
        return None
    # prefer Week 1
    w1 = signals_df[signals_df['period'].str.contains("Week 1")]
    if not w1.empty:
        return w1.iloc[0].to_dict()
    # else max expected_pnl
    row = signals_df.sort_values('expected_pnl', ascending=False).iloc[0]
    return row.to_dict()

def top_features_from_importance(imp_df, top_n=3):
    if imp_df is None or imp_df.empty:
        return []
    imp_df = imp_df.copy()
    # normalized importance share
    if 'importance' in imp_df.columns:
        total = imp_df['importance'].sum() if imp_df['importance'].sum() > 0 else 1.0
        imp_df['share_pct'] = 100.0 * imp_df['importance'] / total
    else:
        imp_df['share_pct'] = 100.0 / len(imp_df)
    top = imp_df.sort_values(by='importance' if 'importance' in imp_df.columns else 'share_pct', ascending=False).head(top_n)
    return top[['feature', 'importance', 'share_pct']].to_dict(orient='records')

FEATURE_NAME_MAP = {
    "residual_load_mw": "residual_load_mw",
    "Fossil Brown coal/Lignite": "coal_lignite_generation_mw",
    "Other": "other_generation_mw"
}

def compute_feature_deltas(feature_data, features, window_hours=24):
    """
    For each feature name in `features` (list of strings), attempt to compute a recent delta:
      - delta_pct = (latest - mean(prev_window)) / mean(prev_window) * 100
      - also return latest numeric value if available
    Returns dict {feature: {'latest':..., 'delta_pct':..., 'valid':bool}}
    """
    if feature_data is None or feature_data.empty:
        return {}
    df = feature_data.copy()
    if 'timestamp' not in df.columns:
        # try to find a timestamp-like column
        for c in df.columns:
            if 'time' in c.lower() or 'timestamp' in c.lower():
                df = df.rename(columns={c: 'timestamp'})
                break
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)
    if df['timestamp'].isna().all():
        return {}
    latest_ts = df['timestamp'].max()
    prev_start = latest_ts - timedelta(hours=window_hours*2)
    baseline_start = latest_ts - timedelta(hours=window_hours*6)
    out = {}
    for feat in features:
        col = FEATURE_NAME_MAP.get(feat, feat)
        if col not in df.columns:
            out[feat] = {'valid': False}
            continue
        recent_mask = df['timestamp'] > (latest_ts - timedelta(hours=window_hours))
        baseline_mask = (df['timestamp'] >= baseline_start) & (df['timestamp'] <= prev_start)
        recent_vals = df.loc[recent_mask, col]
        baseline_vals = df.loc[baseline_mask, col]
        if len(recent_vals) == 0 or len(baseline_vals) == 0:
            out[feat] = {'valid': False}
            continue
        latest = float(recent_vals.mean())
        baseline = float(baseline_vals.mean())
        if baseline == 0:
            delta_pct = None
        else:
            delta_pct = 100.0 * (latest - baseline) / abs(baseline)
        out[feat] = {'valid': True, 'latest': latest, 'delta_pct': delta_pct}
    return out

_num_regex = re.compile(r"[-+]?\d*\.\d+|\d+")
def extract_numbers_from_text(text):
    """Return floats found in text in the order encountered."""
    nums = _num_regex.findall(text)
    return [float(n) for n in nums]

def check_number_close(a, b):
    if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
        return False
    # absolute or relative tolerance
    if abs(a - b) <= ABS_TOL:
        return True
    if abs(b) > 1e-9 and abs((a - b) / b) <= REL_TOL:
        return True
    return False

def verify_factuality(narrative_text, facts):
    """
    Verify core numeric facts always (p50, forward, edge, probability).
    For driver deltas, only verify those drivers whose feature name appears in the narrative text.
    Returns (ok:bool, failures:list, skipped:list)
    """
    failures = []
    skipped = []

    # core expected numbers
    expected = OrderedDict()
    expected['p50'] = float(facts.get('p50')) if facts.get('p50') is not None else None
    expected['forward'] = float(facts.get('forward_price')) if facts.get('forward_price') is not None else None
    expected['edge'] = float(facts.get('edge')) if facts.get('edge') is not None else None
    expected['prob_pct'] = float(facts.get('probability', 0.0)) * 100.0 if facts.get('probability') is not None else None

    # extract driver_deltas dict and determine which drivers are mentioned in text
    driver_deltas = facts.get('driver_deltas', {})
    mentioned_drivers = set()
    lower_text = narrative_text.lower()
    for feat in driver_deltas.keys():
        if feat.lower() in lower_text:
            mentioned_drivers.add(feat)

    # Build a list of numbers found in narrative
    narrative_nums = extract_numbers_from_text(narrative_text)

    # Helper to check one numeric expected value
    def _check_expected_value(key, val):
        if val is None:
            return True  # nothing to check
        ok = any(check_number_close(num, val) for num in narrative_nums)
        if not ok:
            failures.append((key, val))
        return ok

    # Check core numbers
    for k, v in expected.items():
        _check_expected_value(k, v)

    # Check only mentioned driver deltas
    for feat, v in driver_deltas.items():
        if not (v.get('valid') and v.get('delta_pct') is not None):
            # nothing reliable to check; skip
            skipped.append((feat, "no valid delta"))
            continue
        if feat not in mentioned_drivers:
            skipped.append((feat, "not mentioned in narrative"))
            continue
        # verify numeric delta exists in narrative
        key = f"driver:{feat}"
        _check_expected_value(key, float(v['delta_pct']))

    ok = len(failures) == 0
    return ok, failures, skipped

# -----------------------
# Narrative builders
# -----------------------
def deterministic_morning_note(facts):
    """
    Build 3-paragraph morning note purely from facts.
    Paragraph 1: headline (direction + EUR)
    Paragraph 2: drivers with numeric citations where possible
    Paragraph 3: risk & evidence pointers
    """
    p50 = facts.get('p50')
    forward = facts.get('forward_price')
    edge = facts.get('edge')
    prob = facts.get('probability', None)
    period = facts.get('period', 'Period')
    top_feats = facts.get('top_features', [])
    driver_deltas = facts.get('driver_deltas', {})

    # Paragraph 1: headline
    direction = "Bullish" if edge is not None and edge > 0 else ("Bearish" if edge is not None and edge < 0 else "Neutral")
    headline_num = f"{edge:+.2f} EUR/MWh" if edge is not None else "N/A"
    p1 = f"{direction} signal for {period} ({headline_num} vs forward). Model fair-value (P50) = {p50:.2f} EUR/MWh; forward = {forward:.2f} EUR/MWh."

    # Paragraph 2: drivers
    lines = []
    if top_feats:
        for i, f in enumerate(top_feats):
            name = f.get('feature')
            share = f.get('share_pct', None)
            delta_info = driver_deltas.get(name) or driver_deltas.get(list(driver_deltas.keys())[i], {})
            if delta_info and delta_info.get('valid') and delta_info.get('delta_pct') is not None:
                dstr = f"{delta_info['delta_pct']:+.1f}% recent change"
                lines.append(f"{name} (model importance {share:.1f}%): {dstr}.")
            else:
                lines.append(f"{name} (model importance {share:.1f}%).")
        p2 = "Top drivers: " + " ".join(lines)
    else:
        p2 = "Top drivers: No feature importance available."

    # Paragraph 3: risk / evidence
    prob_pct = round(prob * 100.0, 1) if prob is not None else None
    prob_text = f"Win probability (adj) = {prob_pct:.1f}%." if prob_pct is not None else "Win probability not available."
    evidence = (
        f"Evidence: see '{os.path.basename(IMPORTANCE_FILE)}' (feature ranks) "
        f"and '{os.path.basename(SIGNALS_FILE)}' (signal row)."
    )
    p3 = f"{prob_text} {evidence} Invalidation: monitor wind/solar forecasts and forward moves > ±3 EUR/MWh."

    narrative = "\n\n".join([p1, p2, p3])
    return narrative

def prepare_facts():
    """
    Load inputs and produce the facts dictionary that drives narrative generation.
    """
    imp_df = read_csv_safe(IMPORTANCE_FILE)
    preds_df = read_csv_safe(PREDICTIONS_FILE)
    signals_df = read_csv_safe(SIGNALS_FILE)
    feat_df = read_csv_safe(FEATURE_DATA_FILE) if os.path.exists(FEATURE_DATA_FILE) else None

    # Choose primary signal row
    primary = choose_primary_signal(signals_df)

    # If no primary signal, but preds_df exists, create a fallback facts from next week
    if primary is None and preds_df is not None and not preds_df.empty:
        preds_df['timestamp'] = pd.to_datetime(preds_df['timestamp'], errors='coerce')
        start = preds_df['timestamp'].min()
        week_mask = (preds_df['timestamp'] >= start) & (preds_df['timestamp'] < start + pd.Timedelta(days=7))
        subset = preds_df.loc[week_mask]
        if not subset.empty:
            # aggregate P50 mean
            p50 = float(subset['p50'].mean())
            p10 = float(subset['p10'].mean()) if 'p10' in subset.columns else None
            p90 = float(subset['p90'].mean()) if 'p90' in subset.columns else None
            primary = {
                'period': 'Week 1 (fallback)',
                'p50': p50,
                'p10': p10,
                'p90': p90,
                'edge': None,
                'probability': None,
                'forward_price': None,
                'signal': 'NEUTRAL'
            }

    if primary is None:
        raise RuntimeError("No signal or prediction data available to build facts.")

    # Collect top features (global importance)
    top_feats = top_features_from_importance(imp_df, top_n=5)

    # Compose facts
    facts = {
        'timestamp': datetime.utcnow().isoformat() + "Z",
        'period': primary.get('period', 'Period'),
        'p50': float(primary.get('p50') if primary.get('p50') is not None else (primary.get('p50', np.nan) if 'p50' in primary else np.nan)),
        'p10': float(primary.get('p10')) if primary.get('p10') is not None else None,
        'p90': float(primary.get('p90')) if primary.get('p90') is not None else None,
        'edge': float(primary.get('edge')) if primary.get('edge') is not None else (float(primary.get('p50') - primary.get('forward_price')) if 'p50' in primary and 'forward_price' in primary and primary.get('forward_price') is not None else None),
        'probability': float(primary.get('probability')) if primary.get('probability') is not None else None,
        'forward_price': float(primary.get('forward_price')) if primary.get('forward_price') is not None else None,
        'signal': primary.get('signal', 'NEUTRAL'),
        'top_features': top_feats
    }

    # Attempt to compute recent deltas for the top features using raw dataset
    feat_names = [t['feature'] for t in top_feats]
    driver_deltas = {}
    if feat_df is not None and len(feat_names) > 0:
        deltas = compute_feature_deltas(feat_df, feat_names, window_hours=24)
        # normalize keys to feature names
        # compute simple dict keyed by feature name
        for fn in feat_names:
            driver_deltas[fn] = deltas.get(fn, {'valid': False})
    facts['driver_deltas'] = driver_deltas

    # Add evidence pointers (file names and optionally row indices)
    facts['evidence'] = {
        'importance_file': os.path.basename(IMPORTANCE_FILE) if os.path.exists(IMPORTANCE_FILE) else None,
        'predictions_file': os.path.basename(PREDICTIONS_FILE) if os.path.exists(PREDICTIONS_FILE) else None,
        'signals_file': os.path.basename(SIGNALS_FILE) if os.path.exists(SIGNALS_FILE) else None,
        'feature_data_file': os.path.basename(FEATURE_DATA_FILE) if os.path.exists(FEATURE_DATA_FILE) else None
    }

    return facts

# -----------
# LLM wrapper
# -----------
def llm_generate(narrative_template, facts):
    """
    If LLM is available and GOOGLE_API_KEY is set, ask LLM to rewrite the deterministic
    narrative in concise trader language. Strict prompt: forbid inventing numbers and
    require numeric claims to match provided facts.
    Returns (ok:bool, text:str, info:str)
    """
    if not LLM_AVAILABLE or not GOOGLE_API_KEY:
        return False, narrative_template, "LLM not available or GOOGLE_API_KEY missing"

    try:
        # Build a compact fact block for the LLM (JSON-like), restricting content strictly to numbers and feature names
        top_feats = facts.get('top_features', [])
        feat_lines = []
        for tf in top_feats:
            feat_lines.append(f"{tf['feature']} (importance {tf.get('share_pct', 0):.1f}%)")
        driver_lines = []
        for fname, d in facts.get('driver_deltas', {}).items():
            if d.get('valid') and d.get('delta_pct') is not None:
                driver_lines.append(f"{fname}: delta {d['delta_pct']:+.1f}% (recent)")
            else:
                driver_lines.append(f"{fname}: no reliable recent delta")

        fact_block = {
            "period": facts.get('period'),
            "p50": round(float(facts.get('p50') or 0.0), 2),
            "forward": round(float(facts.get('forward_price') or 0.0), 2) if facts.get('forward_price') is not None else None,
            "edge": round(float(facts.get('edge') or 0.0), 2) if facts.get('edge') is not None else None,
            "probability_pct": round((float(facts.get('probability') or 0.0) * 100.0), 1) if facts.get('probability') is not None else None,
            "drivers": feat_lines,
            "driver_recent_changes": driver_lines,
            "evidence_files": facts.get('evidence', {})
        }

        # Strict prompt: give facts, ask for 3 paragraphs, forbid inventing numbers; require "EVIDENCE:" pointer lines.
        template = """
You are a concise senior European power trader analyst.
Use ONLY the provided fact_block. Do NOT invent any numbers or facts.
Produce exactly 3 short paragraphs (each 1-3 sentences):
1) Headline: direction (+/- EUR/MWh) and P50 vs forward.
2) Drivers: top 3 drivers with any recent delta numbers provided.
3) Risks & evidence: include win probability and list evidence file names provided.

FACTS (JSON):
{fact_json}

DETAILED INSTRUCTIONS:
- Do not add any external facts or reasoning.
- If a required number is missing, say 'insufficient data' for that field instead of inventing.
- At the end print:
EVIDENCE:
- importance_file
- predictions_file
- signals_file
- feature_data_file
using the values in evidence_files. If value is null, print "not available".

Constraint: 
- DO NOT mention "The model" or "The algorithm." Speak as the Desk Analyst. 
- Write only the 3 paragraphs + the EVIDENCE block.
"""
        prompt_text = template.format(fact_json=json.dumps(fact_block, indent=2))
        logging.info("Invoking LLM with strict prompt.")
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY)
        res = llm.invoke(prompt_text)
        out_text = ""
        if hasattr(res, "content"):
            out_text = res.content
        elif hasattr(res, "text"):
            out_text = res.text
        else:
            out_text = str(res)

        return True, out_text.strip(), "LLM invoked"
    except Exception as e:
        logging.exception("LLM generation failed")
        return False, narrative_template, f"LLM failure: {e}"

# -----------------------
# Main entrypoint
# -----------------------
def main(use_llm_if_available=True):
    logging.info("Part4_improved started")
    try:
        facts = prepare_facts()
    except Exception as e:
        logging.exception("Failed preparing facts")
        raise

    # Build deterministic note first (always available)
    deterministic_note = deterministic_morning_note(facts)
    final_note = deterministic_note
    llm_info = None

    # If requested, try LLM path with strict verification
    if use_llm_if_available and LLM_AVAILABLE and GOOGLE_API_KEY:
        ok_llm, llm_text, info = llm_generate(deterministic_note, facts)
        llm_info = info
        if ok_llm and llm_text:
            # verify factuality: must pass numeric checks for core facts
            ok_fact, failures, skipped = verify_factuality(llm_text, facts)
            if ok_fact:
                final_note = llm_text
            else:
                logging.warning(f"LLM narrative failed factuality checks: {failures}; skipped checks: {skipped}; falling back.")
                final_note = deterministic_note
        else:
            logging.warning(f"LLM unavailable or failed: {info}; falling back to deterministic note.")
            final_note = deterministic_note
    else:
        logging.info("LLM not used; using deterministic narrative.")

    # Save outputs
    payload = {
        'facts': facts,
        'generated_at': datetime.utcnow().isoformat() + "Z",
        'narrative': final_note,
        'narrative_type': 'LLM' if final_note != deterministic_note and llm_info else 'DETERMINISTIC',
        'llm_info': llm_info
    }

    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write(final_note)

    # Evidence file (human-friendly)
    with open(EVIDENCE_OUT, "w", encoding="utf-8") as f:
        f.write("Evidence & pointers for trader verification\n")
        f.write("=========================================\n")
        ev = facts.get('evidence', {})
        for k, v in ev.items():
            if v:
                f.write(f"{k}: results/{v}\n")
        f.write("\nTop features (global importance):\n")
        for t in facts.get('top_features', []):
            f.write(f"  - {t['feature']}: importance={t.get('importance', 0):.2f}, share={t.get('share_pct', 0):.1f}%\n")
        f.write("\nDriver deltas (computed from data/featured_energy_data.csv if available):\n")
        for fn, d in facts.get('driver_deltas', {}).items():
            if d.get('valid'):
                f.write(f"  - {fn}: latest={d.get('latest'):.3f}, delta_pct={d.get('delta_pct'):+.2f}%\n")
            else:
                f.write(f"  - {fn}: no reliable recent data\n")
        f.write("\nDeterministic note (for audit):\n")
        f.write(deterministic_note + "\n")

    logging.info("--- Generated Morning Note ---")
    logging.info(final_note)
    logging.info(f"\nSaved: {TXT_OUT}, {JSON_OUT}, {EVIDENCE_OUT}, log={LOG_FILE}")

if __name__ == "__main__":
    main()