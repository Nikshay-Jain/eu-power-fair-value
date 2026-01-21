import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Dict, Any
from functools import reduce

import os, shutil, zipfile, math, warnings, logging
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

os.makedirs('data', exist_ok=True)
os.makedirs('qa_report', exist_ok=True)

# =========================
# DATA SOURCE DOCUMENTATION
# =========================
SOURCE_METADATA = {
    "url": "https://transparency.entsoe.eu/",
    "description": "ENTSO-E Transparency Platform - European electricity market data",
    "DA Prices": "ENTSO-E / Transmission / Day Ahead Prices [12.1.D]",
    "Actual Load": "ENTSO-E / Load / Actual Total Load [6.1.A]",
    "Load Forecast": "ENTSO-E / Load / Day Ahead Total Load Forecast [6.1.B]",
    "Generation": "ENTSO-E / Generation / Actual Generation per Production Type [16.1.B&C]",
    "BZN Code": "DE-LU (10Y1001A1001A83F)"
}

def load_and_stack_data(zip_path):
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join('data', 'extracted'))

    # Get a list of all CSV files in the extracted folder and concatenate into one DataFrame
    csv_files = []
    for root, _, files in os.walk(os.path.join('data','extracted')):
        for f in files:
            if f.endswith('.csv'):
                csv_files.append(os.path.join(root, f))

    dataframes = [pd.read_csv(csv, low_memory=False) for csv in sorted(csv_files)]     # dealing with large dataframes
    shutil.rmtree(os.path.join('data', 'extracted'))
    
    return pd.concat(dataframes, ignore_index=True)

def enforce_hourly_granularity(df, value_columns):
    """
    Enforces hourly granularity on dataframe with 'MTU (UTC)' time intervals.
    
    - Leaves hourly data unchanged (1 hour intervals)
    - Averages sub-hourly data (15-min, 30-min intervals) into 1-hour blocks
    - Handles both input formats (with/without seconds)
    - Output format: "DD/MM/YYYY HH:MM - DD/MM/YYYY HH:MM" (no seconds)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'MTU (UTC)' column
    value_columns : list
        Column names to average when aggregating (e.g., capacity, flow values)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with strict hourly granularity
    
    Example:
    --------
    >>> value_cols = ['Offered Capacity from BZN|DE-LU (MW)', 
    ...               'Offered Capacity to BZN|DE-LU (MW)']
    >>> df_hourly = enforce_hourly_granularity(df, value_cols)
    """
    
    df = df.copy()
    
    # Convert value columns to numeric (in case they're still strings)
    for col in value_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Parse start time - handle both formats flexibly
    time_str = df['MTU (UTC)'].str.split(' - ').str[0]
    df['_start_time'] = pd.to_datetime(time_str, format='mixed', dayfirst=True)
    
    # Create hourly group (floor to nearest hour)
    df['_hour_group'] = df['_start_time'].dt.floor('h')
    
    # Build aggregation dictionary
    agg_dict = {}
    for col in df.columns:
        if col in value_columns:
            agg_dict[col] = 'mean'  # Average numeric values
        elif col not in ['MTU (UTC)', '_start_time', '_hour_group']:
            agg_dict[col] = 'first'  # Keep first value for metadata
    
    # Group by hour and aggregate
    result = df.groupby('_hour_group', as_index=False).agg(agg_dict)
    
    # Recreate MTU (UTC) in proper hourly format (without seconds)
    result['MTU (UTC)'] = (
        result['_hour_group'].dt.strftime('%d/%m/%Y %H:%M') + ' - ' +
        (result['_hour_group'] + pd.Timedelta(hours=1)).dt.strftime('%d/%m/%Y %H:%M')
    )
    
    # Drop helper column
    result = result.drop(columns=['_hour_group'])
    
    # Restore original column order
    original_cols = [c for c in df.columns if c not in ['_start_time', '_hour_group']]
    result = result[original_cols]
    
    return result.reset_index(drop=True)

def pivot_generation_by_type(df):
    """
    Pivots generation data from long to wide format.
    Each Production Type becomes a separate column with Generation (MW) values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: 'MTU (UTC)', 'Area', 'Production Type', 'Generation (MW)'
    
    Returns:
    --------
    pd.DataFrame
        Wide-format DataFrame with separate columns for each Production Type
    
    Example:
    --------
    Input (long format):
        MTU (UTC)                        Area      Production Type    Generation (MW)
        01/01/2023 00:00 - 00:15:00     BZN|DE-LU  Biomass           4020.74
        01/01/2023 00:00 - 00:15:00     BZN|DE-LU  Solar             1500.00
        01/01/2023 00:00 - 00:15:00     BZN|DE-LU  Wind Onshore      8000.00
        
    Output (wide format):
        MTU (UTC)                        Area      Biomass  Solar   Wind Onshore
        01/01/2023 00:00 - 00:15:00     BZN|DE-LU  4020.74  1500.0  8000.0
    """
    
    # Pivot: Production Type values become columns, Generation (MW) fills the cells
    df_wide = df.pivot_table(
        index=['MTU (UTC)', 'Area'],   # Keep these as index
        columns='Production Type',     # Spread this into columns
        values='Generation (MW)',      # Use these values to fill
        aggfunc='sum'                # If duplicates exist, take first (shouldn't happen)
    ).reset_index()
    
    dups = df.groupby(["MTU (UTC)", "Area", "Production Type"]).size()
    if (dups > 1).any():
        logging.info("Multiple rows per MTU/Area/Production Type found — summing as intended.")


    # Flatten column names (remove multi-index if present)
    df_wide.columns.name = None
    
    return df_wide

def merge_on_mtu_union(dfs, mtu_col="MTU (UTC)"):
    """
    Full outer merge of multiple dataframes on MTU (UTC).
    Keeps union of all MTU rows.
    Missing values become NaN.
    """

    return reduce(
        lambda left, right: pd.merge(
            left, right,
            on=mtu_col,
            how="outer"
        ),
        dfs
    )

def generate_qa_report(df: pd.DataFrame,
                       mtu_col: str = "MTU (UTC)",
                       parse_dayfirst: bool = True,
                       output_prefix: str = None) -> Dict[str, Any]:
    """
    Run dataset QA: missingness, duplicates, outliers, coverage by field/time.
    Returns a dict of DataFrames and summary info. Optionally writes CSVs with output_prefix.
    """
    report = {}
    working = df.copy()

    # --- Parse MTU start time and set index (safe parsing) ---
    # MTU format assumed "DD/MM/YYYY HH:MM - DD/MM/YYYY HH:MM" (dayfirst common)
    split = working[mtu_col].astype(str).str.split(" - ", expand=True)
    try:
        start = pd.to_datetime(split[0], dayfirst=parse_dayfirst, utc=True, errors="raise")
    except Exception as e:
        # fallback: mixed parsing, coerce to find bad rows
        start = pd.to_datetime(split[0], dayfirst=parse_dayfirst, utc=True, errors="coerce")
        bad = working[split[0].isna() | start.isna()].head(10)
        raise ValueError("MTU parsing failure. Example bad rows:\n" + bad.to_string()) from e

    working["_start"] = start
    working = working.set_index("_start", drop=False).sort_index()

    total_rows = len(working)
    report["meta"] = pd.DataFrame({
        "total_rows": [total_rows],
        "time_index_min": [working.index.min()],
        "time_index_max": [working.index.max()]
    })

    # --- Missingness summary (columns) ---
    missing_count = working.isna().sum()
    missing_pct = (missing_count / total_rows) * 100
    missing_df = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct
    }).sort_values("missing_count", ascending=False)
    report["missing_by_column"] = missing_df
    if output_prefix:
        missing_df.to_csv(os.path.join(output_prefix, f"{output_prefix}_missing_by_column.csv"))

    # --- Rows with at least one NaN (indices & contiguous blocks) ---
    rows_with_nan = working[working.isna().any(axis=1)].index
    report["rows_with_nan_index"] = rows_with_nan
    # contiguous blocks (by hourly freq)
    # Ensure hourly freq for range detection
    rows_with_nan_sorted = rows_with_nan.sort_values()
    # For contiguous detection use pd.date_range diff trick
    # Convert to list and build ranges
    if len(rows_with_nan_sorted) > 0:
        # Ensure freq-awareness by reindexing expected hourly freq
        # Build contiguous ranges manually using 1-hour delta
        ranges = []
        start = prev = rows_with_nan_sorted[0]
        length = 1
        for ts in rows_with_nan_sorted[1:]:
            if (ts - prev) == pd.Timedelta(hours=1):
                prev = ts
                length += 1
            else:
                ranges.append((start, prev, length))
                start = prev = ts
                length = 1
        ranges.append((start, prev, length))
    else:
        ranges = []
    ranges_df = pd.DataFrame(ranges, columns=["start", "end", "length_hours"])
    report["missing_blocks"] = ranges_df
    if output_prefix:
        ranges_df.to_csv(os.path.join(output_prefix, f"{output_prefix}_missing_blocks.csv"), index=False)

    # --- Duplicates ---
    # 1) duplicate MTU timestamps
    dup_mtu_mask = working.index.duplicated(keep=False)
    dup_mtu = working[dup_mtu_mask].sort_index()
    report["duplicate_mtu_count"] = pd.DataFrame({"duplicate_mtu_rows": [dup_mtu.shape[0]]})
    report["duplicate_mtu_sample"] = dup_mtu.head(20)
    # 2) full-row duplicates
    full_dups = working[working.duplicated(keep=False)]
    report["full_duplicate_rows_count"] = pd.DataFrame({"full_duplicate_rows": [full_dups.shape[0]]})
    report["full_duplicates_sample"] = full_dups.head(20)
    if output_prefix:
        dup_mtu.head(200).to_csv(os.path.join(output_prefix, f"{output_prefix}_dup_mtu_sample.csv"))
        full_dups.head(200).to_csv(os.path.join(output_prefix, f"{output_prefix}_full_dups_sample.csv"))

    # --- Time continuity check ---
    expected_idx = pd.date_range(start=working.index.min(), end=working.index.max(), freq="h", tz="UTC")
    missing_hours = expected_idx.difference(working.index)
    extra_hours = working.index.difference(expected_idx)
    report["time_continuity"] = pd.DataFrame({
        "expected_hours": [len(expected_idx)],
        "present_rows": [len(working.index.unique())],
        "missing_hours_count": [len(missing_hours)],
        "extra_hours_count": [len(extra_hours)]
    })
    report["missing_hours_list"] = missing_hours
    if output_prefix:
        pd.Series(missing_hours).to_csv(os.path.join(output_prefix, f"{output_prefix}_missing_hours.csv"), index=False)

    # --- Coverage by field/time ---
    # Monthly coverage % per column
    monthly = working.copy()
    # Remove timezone before converting to PeriodIndex to avoid warning
    if hasattr(monthly.index, 'tz') and monthly.index.tz is not None:
        monthly.index = monthly.index.tz_convert(None)
    monthly["__month"] = monthly.index.to_period("M")
    monthly_coverage = monthly.groupby("__month").apply(lambda g: g.isna().mean()).T
    monthly_coverage.columns = monthly_coverage.columns.astype(str)
    report["monthly_missing_pct_by_column"] = monthly_coverage
    if output_prefix:
        monthly_coverage.to_csv(os.path.join(output_prefix, f"{output_prefix}_monthly_missing_pct_by_column.csv"))

    # Hour-of-day coverage (0-23) per column
    hod_cov = working.groupby(working.index.hour).apply(lambda g: g.isna().mean()).T
    hod_cov.columns = [f"hour_{h}" for h in hod_cov.columns]
    report["hour_of_day_missing_pct_by_column"] = hod_cov
    if output_prefix:
        hod_cov.to_csv(os.path.join(output_prefix, f"{output_prefix}_hod_missing_pct_by_column.csv"))

    # --- Outlier detection (robust) ---
    numeric = working.select_dtypes(include=[np.number]).copy()
    outlier_summary = []
    outlier_samples = {}
    for col in numeric.columns:
        s = numeric[col].dropna()
        if s.empty:
            outlier_summary.append((col, 0, 0, np.nan, np.nan))
            outlier_samples[col] = pd.Series([], dtype=float)
            continue

        # IQR method
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # MAD-based robust zscore flag (redundant)
        med = s.median()
        mad = float(np.median(np.abs(s - med))) if len(s) > 0 else 0.0
        if mad == 0:
            mad = np.std(s) if np.std(s) > 0 else 1.0
        robust_z = (s - med) / mad

        outlier_mask = (numeric[col] < lower) | (numeric[col] > upper)
        outlier_count = int(outlier_mask.sum())
        outlier_pct = (outlier_count / total_rows) * 100

        outlier_summary.append((col, outlier_count, outlier_pct, lower, upper))
        outlier_samples[col] = working.loc[outlier_mask, [mtu_col, col]].head(10)

    outlier_df = pd.DataFrame(outlier_summary, columns=["column", "outlier_count", "outlier_pct", "lower_bound", "upper_bound"])\
                  .sort_values("outlier_count", ascending=False)
    report["outlier_summary"] = outlier_df
    report["outlier_samples"] = outlier_samples
    if output_prefix:
        outlier_df.to_csv(os.path.join(output_prefix, f"{output_prefix}_outlier_summary.csv"))
        # write sample outliers to separate files per column (limited)
        for c, sample in outlier_samples.items():
            if sample.shape[0] > 0:
                safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in c)
                sample.to_csv(os.path.join(output_prefix, f"{output_prefix}_outliers_{safe_name}.csv"), index=False)

    # --- Negative-value check (Context-aware) ---
    neg_summary = []
    for col in numeric.columns:
        neg_count = int((working[col] < 0).sum())

        # Highlight: DA Prices can be negative, Load/Gen cannot.
        is_error = neg_count > 0 and "Price" not in col 
        neg_summary.append((col, neg_count, "DATA_ERROR" if is_error else "MARKET_SIGNAL"))
    neg_df = pd.DataFrame(neg_summary, columns=["column", "negative_count", "interpretation"])
    report["negative_values_summary"] = neg_df
    if output_prefix:
        neg_df.to_csv(os.path.join(output_prefix, f"{output_prefix}_negative_values_summary.csv"))

    # --- Quick textual summary (few lines) ---
    summary_lines = []
    summary_lines.append(f"Total rows: {total_rows}")
    summary_lines.append(f"Columns with missing values (top 10):\n{missing_df.head(10).to_string()}")
    summary_lines.append(f"Duplicate MTU rows: {report['duplicate_mtu_count'].iloc[0,0]}")
    summary_lines.append(f"Full duplicate rows: {report['full_duplicate_rows_count'].iloc[0,0]}")
    summary_lines.append(f"Missing hourly timestamps: {len(missing_hours)}")
    summary_text = "\n\n".join(summary_lines)
    report["summary_text"] = summary_text
    if output_prefix:
        neg_df.to_csv(os.path.join(output_prefix, f"{output_prefix}_negative_values_summary.csv"))
        with open(os.path.join(output_prefix, f"{output_prefix}_qa_summary.txt"), "w") as f:
            f.write(summary_text)

    return report

def impute_energy_ts(df, mtu_col):
    out = df.copy()

    # datetime index from MTU start
    time = pd.to_datetime(out[mtu_col].str.split(" - ").str[0], dayfirst=True, utc=True)
    out.index = time

    num_cols = out.select_dtypes(include="number").columns

    for col in num_cols:
        s = out[col]

        # 1. same hour previous da
        s = s.fillna(s.shift(24))

        # 2. same hour previous week
        s = s.fillna(s.shift(168))

        # 3. hour-of-day median fallback
        hod_median = s.groupby(s.index.hour).transform("median")
        s = s.fillna(hod_median)

        out[col] = s

    return out.reset_index(drop=True)

class PowerFeatureEngineer:
    def __init__(self, target_col: str = "Day Ahead Price (EUR/MWh)"):
        self.target_col = target_col
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create complete feature set with no NaN gaps."""
        df = df.copy()
        
        # Parse timestamp
        df["timestamp"] = pd.to_datetime(
            df["MTU (UTC)"].str.split(" - ").str[0],
            dayfirst=True, utc=True
        )
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Core features
        df = self._create_residual_load(df)
        df = self._create_temporal_features(df)
        df = self._create_lag_features(df)
        df = self._create_rolling_features(df)
        df = self._create_interaction_features(df)
        
        return df
    
    def _create_residual_load(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Residual Load = Load - Renewables
        Most important feature (0.9 correlation with price)
        """
        # Total renewables (actual)
        wind_on = df["Wind Onshore"].fillna(0)
        wind_off = df["Wind Offshore"].fillna(0)
        solar = df["Solar"].fillna(0)
        
        df["total_renewable_mw"] = wind_on + wind_off + solar
        
        # Residual load (actual)
        load_actual = df["Actual Total Load (MW)"].fillna(df["Day-ahead Total Load Forecast (MW)"])
        df["residual_load_mw"] = load_actual - df["total_renewable_mw"]
        
        # Renewable penetration
        df["renewable_penetration"] = df["total_renewable_mw"] / (load_actual + 1e-6)
        
        # Individual renewable components
        df["wind_total_mw"] = wind_on + wind_off
        df["solar_mw"] = solar
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based patterns"""
        ts = df["timestamp"]
        
        df["hour"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["month"] = ts.dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        
        # Cyclical encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lagged values (avoid leakage)"""
        # Price lags (if target exists)
        if self.target_col in df.columns:
            df["price_lag_24h"] = df[self.target_col].shift(24)
            df["price_lag_168h"] = df[self.target_col].shift(168)
        
        # Residual load lags
        df["resload_lag_24h"] = df["residual_load_mw"].shift(24)
        df["resload_lag_168h"] = df["residual_load_mw"].shift(168)
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling statistics"""
        # 24-hour rolling mean (shifted to avoid leakage)
        df["resload_rolling_mean_24h"] = (
            df["residual_load_mw"].shift(1)
            .rolling(24, min_periods=1)
            .mean()
        )
        
        # Price volatility (if target exists)
        if self.target_col in df.columns:
            df["price_rolling_std_24h"] = (
                df[self.target_col].shift(1)
                .rolling(24, min_periods=1)
                .std()
                .fillna(0)
            )
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross features"""
        df["hour_x_weekend"] = df["hour"] * df["is_weekend"]
        df["resload_x_hour"] = df["residual_load_mw"] * df["hour"]
        df["renew_pen_x_hour"] = df["renewable_penetration"] * df["hour"]
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame):
        """Return list of feature columns (exclude target and metadata)"""
        exclude = ["MTU (UTC)", "timestamp", self.target_col]
        # Also exclude raw generation columns
        raw_gen = ["Biomass", "Fossil Brown coal/Lignite", "Fossil Coal-derived gas",
                   "Fossil Gas", "Fossil Hard coal", "Fossil Oil", "Geothermal",
                   "Hydro Pumped Storage", "Hydro Run-of-river and pondage",
                   "Other", "Other renewable"]
        exclude.extend(raw_gen)
        
        return [c for c in df.columns if c not in exclude]

def corr_analysis(df: pd.DataFrame,
                 target_col: str,
                 timestamp_col: str = "timestamp",
                 output_prefix: str = "corr"):

    # ---- Prepare numeric-only dataframe ----
    dfc = df.copy()
    if timestamp_col in dfc.columns:
        dfc = dfc.drop(columns=[timestamp_col])

    dfc = dfc.select_dtypes(include=[np.number])

    if target_col not in dfc.columns:
        raise KeyError("Target column not found in dataframe")

    # ---- Correlation matrix ----
    corr_matrix = dfc.corr(method="pearson")

    # ---- Save full heatmap using seaborn clustermap ----
    # Mask small correlations for clarity
    mask = np.abs(corr_matrix) < 0.2
    # Set diagonal to False so main diagonal is always shown
    np.fill_diagonal(mask.values, False)
    sns.set(font_scale=0.7)
    g = sns.clustermap(
        corr_matrix,
        cmap="coolwarm",
        vmin=-1, vmax=1,
        linewidths=0.1,
        figsize=(14, 12),
        mask=mask,
        annot=False,
        cbar_kws={"label": "Correlation"}
    )
    plt.title("Feature Correlation Matrix (Clustered)", pad=80)
    plt.savefig(os.path.join("qa_report", f"{output_prefix}_heatmap.png"), dpi=300)
    plt.close()

    # ---- Correlation vs target ----
    target_corr = corr_matrix[target_col].drop(target_col).sort_values()

    plt.figure(figsize=(8,10))
    target_corr.plot(kind="barh")
    plt.title(f"Feature Correlation vs Target: {target_col}")
    plt.xlabel("Pearson Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join("qa_report", f"{output_prefix}_target_bar.png"), dpi=300)
    plt.close()

def plotting(df: pd.DataFrame, 
             target_col: str,
             resload_col: str = "residual_load_mw", 
             output_path: str = os.path.join("qa_report", "price_vs_residual_load.png"),
             timestamp_col: str = "timestamp",
             max_points: int = 20000):
    
    # Sort by time
    dfp = df.copy()
    dfp[timestamp_col] = pd.to_datetime(dfp[timestamp_col], utc=True)
    dfp = dfp.sort_values(timestamp_col)

    # Downsample if too large
    if len(dfp) > max_points:
        dfp = dfp.iloc[np.linspace(0, len(dfp)-1, max_points).astype(int)]

    time = dfp[timestamp_col].values

    # Numeric features only (excluding target)
    numeric_cols = dfp.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]

    n_feats = len(numeric_cols)
    n_cols = 2
    n_rows = math.ceil(n_feats / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows), sharex=True)
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]

        # Plot target
        ax.scatter(time, dfp[target_col].values, s=8, alpha=0.25, label=target_col)

        # Plot feature (scaled for visibility)
        feature_vals = dfp[col].values
        # Normalize feature for overlay
        f_norm = (feature_vals - np.nanmean(feature_vals)) / (np.nanstd(feature_vals) + 1e-6)
        f_norm = f_norm * np.nanstd(dfp[target_col].values) + np.nanmean(dfp[target_col].values)

        ax.scatter(time, f_norm, s=8, alpha=0.25, label=col)

        ax.set_title(col, fontsize=9)
        ax.grid(True, alpha=0.3)

        if i % 2 == 0:
            ax.set_ylabel("Scaled Value")

        ax.legend(fontsize=8)

    # Remove unused axes
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join("qa_report", f"feature_vs_target_timeseries_grid.png"), dpi=300)
    plt.close()
    logging.info("Saved feature vs target time-series grid plot to qa_report/feature_vs_target_timeseries_grid.png")

    plt.figure(figsize=(8, 6))
    plt.scatter(df[resload_col], df[target_col], alpha=0.3, s=10, color='royalblue')
    plt.xlabel("Residual Load (MW)")
    plt.ylabel("Day-ahead Price (EUR/MWh)")
    plt.title("Price vs. Residual Load (Merit Order Curve)")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info("Saved Price vs. Residual Load plot to %s", output_path)

def main():
    gen_path = os.path.join('data', 'generation.zip')
    load_path = os.path.join('data', 'load.zip')
    market_path = os.path.join('data', 'market.zip')
    prices_path = os.path.join('data', 'prices.zip')

    logging.info("Loading and processing data...")
    gen_data = load_and_stack_data(gen_path)
    load_data = load_and_stack_data(load_path)
    market_data = load_and_stack_data(market_path)
    prices_data = load_and_stack_data(prices_path)
    
    logging.info("Data loaded: Generation (%d rows), Load (%d rows), Market (%d rows), Prices (%d rows)",
                 len(gen_data), len(load_data), len(market_data), len(prices_data))
    
    value_cols = [
        'Offered Capacity from BZN|DE-LU (MW)',
        'Offered Capacity to BZN|DE-LU (MW)'
    ]

    logging.info("Enforcing hourly granularity...")
    market_hourly = enforce_hourly_granularity(market_data, value_cols)
    market_hourly.drop(columns=['Time Interval (UTC)', 'In Area', 'Out Area', 'Classification Sequence', 'Instance Code'], errors='ignore', inplace=True)
    value_cols = [
        'Actual Total Load (MW)',
        'Day-ahead Total Load Forecast (MW)'
    ]
    load_hourly = enforce_hourly_granularity(load_data, value_cols)
    load_hourly.drop(columns=['Area'], inplace=True)

    # Pivot generation data
    df_wide = pivot_generation_by_type(gen_data)
    
    # Enforce hourly granularity
    generation_cols = [col for col in df_wide.columns 
                       if col not in ['MTU (UTC)', 'Area']]
    gen_hourly = enforce_hourly_granularity(df_wide, generation_cols)

    # drop columns like 'Area' and less relevant production types and Nuclear as Germany droped the usage to 0 MW popst 2023
    gen_hourly.drop(columns=['Area', 'Energy storage', 'Fossil Oil shale', 'Fossil Peat', 'Hydro Water Reservoir', 'Marine', 'Nuclear', 'Waste'], 
                    errors='ignore',
                    inplace=True)

    prices_data = prices_data[prices_data['Sequence'] == 'Sequence Sequence 1']
    prices_data.drop(columns=['Area', 'Sequence', 'Intraday Period (UTC)', 'Intraday Price (EUR/MWh)'], errors='ignore', inplace=True)
    prices_data.reset_index(drop=True, inplace=True)
    
    value_cols = [
        'Day-ahead Price (EUR/MWh)'
    ]
    prices_hourly = enforce_hourly_granularity(prices_data, value_cols)
    
    logging.info("Merging datasets...")
    df_merged = merge_on_mtu_union(
        [prices_hourly, market_hourly, load_hourly, gen_hourly],
        mtu_col="MTU (UTC)"
    )
    df_merged = df_merged.sort_values("MTU (UTC)").reset_index(drop=True)

    report = generate_qa_report(df_merged, mtu_col="MTU (UTC)", output_prefix="qa_report")
    logging.info("\nQA Report Summary:\n%s\n", report["summary_text"])

    # Inspect detailed tables:
    _ = report["missing_by_column"].head(50)
    _ = report["missing_blocks"]
    _ = report["duplicate_mtu_sample"]
    _ = report["outlier_summary"].head(50)
    _ = df_merged[df_merged.isnull().any(axis=1)].index.tolist()

    df_imputed = impute_energy_ts(df_merged, "MTU (UTC)")
    df_imputed.to_csv(os.path.join("data", "cleaned_energy_data.csv"), index=False)

    df = pd.read_csv(os.path.join("data", "cleaned_energy_data.csv"))

    # Create features
    engineer = PowerFeatureEngineer(target_col="Day-ahead Price (EUR/MWh)")
    df_featured = engineer.create_all_features(df)

    # Get feature list
    features = engineer.get_feature_columns(df_featured)
    logging.info("Created %d features", len(features))
    logging.info("Sample features: %s", features[:15])

    # Check for NaNs
    nan_counts = df_featured[features].isna().sum()
    if nan_counts.sum() > 0:
        logging.warning("%d total NaN values found", nan_counts.sum())
        logging.warning("NaN counts by feature: %s", nan_counts[nan_counts > 0].to_dict())
    else:
        logging.info("No NaN values in features")

    df_featured.to_csv(os.path.join("data", "featured_energy_data.csv"), index=False)
    logging.info("Saved to data/featured_energy_data.csv (%d rows)", len(df_featured))

    corr_analysis(
        df=df_featured,
        target_col="Day-ahead Price (EUR/MWh)"
    )
    logging.info("Saved correlation analysis plots to qa_report/")

    plotting(
        df=df_featured,
        target_col="Day-ahead Price (EUR/MWh)",
        resload_col="residual_load_mw",
        output_path="qa_report/price_vs_residual_load.png"
    )
    logging.info("Saved plotting outputs to qa_report/")

if __name__ == "__main__":
    main()