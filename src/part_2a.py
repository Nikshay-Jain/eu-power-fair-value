import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os, math, warnings, logging

os.makedirs('results', exist_ok=True)
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class PowerFeatureEngineer:
    def __init__(self, target_col: str = "Day-ahead Price (EUR/MWh)"):
        self.target_col = target_col
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create complete feature set with no NaN gaps."""
        
        # Parse timestamp
        df["timestamp"] = pd.to_datetime(
            df["MTU (UTC)"].str.split(" - ").str[0],
            dayfirst=True, utc=True
        )
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        df = self._create_time_features(df)

        # Core features
        df = self._create_residual_load(df)
        df = self._create_temporal_features(df)
        df = self._create_interaction_features(df)

        # Rolling Features with SAFE LAG (48h minimum for Day-Ahead - safe). Shift(24) is risky (requires perfect data availability).
        col = "residual_load_mw"
        df[f"{col}_roll24_mean"] = df[col].shift(48).rolling(24).mean()
        df[f"{col}_roll24_std"] = df[col].shift(48).rolling(24).std()

        df = self._create_lag_features(df)

        return df.dropna().reset_index(drop=True)
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical time features (critical for price shape)."""
        df = df.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.weekday
        df['month'] = df['timestamp'].dt.month

        # Cyclical encoding (best practice for ML)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        return df
    
    def _create_residual_load(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Residual Load = Load - Renewables
        Most important feature (0.9 correlation with price)
        """
        # Total renewables (actual)
        df["wind_total_mw"] = df["Wind Onshore"].fillna(0) + df["Wind Offshore"].fillna(0)
        df["Solar"] = df["Solar"].fillna(0)
        df["total_renewable_mw"] = df["wind_total_mw"] + df["Solar"]
        
        # Residual load (actual) and penetration
        df["residual_load_mw"] = df["Day-ahead Total Load Forecast (MW)"] - df["total_renewable_mw"]
        df["renewable_penetration"] = df["total_renewable_mw"] / (df["Day-ahead Total Load Forecast (MW)"] + 1e-6)
                    
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
        # Price lags (if target exists). Add Price Lags (The "Fuel Proxy")
        # We know D-1 prices when predicting D.
        if self.target_col in df.columns:
            # Lag 24: Yesterday's price at this hour (captures recent price level)
            df['price_lag_24h'] = df[self.target_col].shift(24)
            # Lag 168: Last week's price (captures weekly seasonality)
            df['price_lag_168h'] = df[self.target_col].shift(168)
            
            # Rolling Price Average (The "Trend" / "Gas Proxy")
            # Average price of the last 7 days. If gas is expensive, this will be high.
            df['price_rolling_7d'] = df[self.target_col].shift(24).rolling(24*7).mean()
            
        # Residual load lags
        df["resload_lag_48h"] = df["residual_load_mw"].shift(48) 
        df["resload_lag_168h"] = df["residual_load_mw"].shift(168)
        
        return df
     
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Capture non-linear dependencies."""
        df = df.copy()
        
        # Interaction: Residual Load x Hour Harmonic (Peak vs Offpeak sensitivity)
        # Note: Ensure _create_time_features is run FIRST so 'hour_cos' exists
        if 'residual_load_mw' in df.columns and 'hour_cos' in df.columns:
            df['resload_x_hour_cos'] = df['residual_load_mw'] * df['hour_cos']
            
        # Interaction: Solar x Time (Solar only matters during the day)
        if 'solar_mw' in df.columns and 'hour' in df.columns:
            # Create a simple day/night flag proxy or interacting with hour curve
            df['solar_x_hour_sin'] = df['solar_mw'] * df['hour_sin']
            
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
                 output_prefix: str = "corr",
                 threshold: float = 0.1):

    # Prepare numeric-only dataframe
    dfc = df.copy()
    if timestamp_col in dfc.columns:
        dfc = dfc.drop(columns=[timestamp_col])

    dfc = dfc.select_dtypes(include=[np.number])

    if target_col not in dfc.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe")

    # Correlation matrix
    corr_matrix = dfc.corr(method="pearson")

    # Full heatmap plotting (existing logic)
    mask = np.abs(corr_matrix) < 0.2
    np.fill_diagonal(mask.values, False)
    sns.set_theme(font_scale=0.7)
    g = sns.clustermap(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, 
                       linewidths=0.1, figsize=(14, 12), mask=mask)
    plt.title("Feature Correlation Matrix (Clustered)", pad=80)
    plt.savefig(os.path.join("results", f"part2_{output_prefix}_heatmap.png"), dpi=300)
    plt.close()

    # Dynamic Feature Selection Logic
    target_corr = corr_matrix[target_col].drop(target_col)
    
    # Filter features where |correlation| >= threshold
    selected_features = target_corr[target_corr.abs() >= threshold].index.tolist()
    
    # Correlation vs target bar plot (existing logic)
    plt.figure(figsize=(8,10))
    target_corr.sort_values().plot(kind="barh")
    plt.title(f"Feature Correlation vs Target: {target_col}")
    plt.axvline(threshold, color='r', linestyle='--', alpha=0.5)
    plt.axvline(-threshold, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join("results", f"part2_{output_prefix}_target_bar.png"), dpi=300)
    plt.close()
    return selected_features

def feat_plotting(df: pd.DataFrame, 
             target_col: str,
             resload_col: str = "residual_load_mw", 
             output_path: str = os.path.join("results", "part2_price_vs_residual_load.png"),
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
    plt.savefig(os.path.join("results", f"part2_feature_vs_target_timeseries_grid.png"), dpi=300)
    plt.close()
    logging.info("Saved feature vs target time-series grid plot to results/feature_vs_target_timeseries_grid.png")

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

# ==============
# Main Execution
# ==============
def main():
    # Load data
    logging.info("Loading data...")
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
    logging.info("Saved correlation analysis plots to results/")

    feat_plotting(
        df=df_featured,
        target_col="Day-ahead Price (EUR/MWh)",
        resload_col="residual_load_mw",
        output_path=os.path.join("results", "part2_price_vs_residual_load.png")
    )
    logging.info("Saved plotting outputs to results/")

if __name__ == "__main__":
    main()