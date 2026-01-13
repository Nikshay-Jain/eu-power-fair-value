"""
Feature Engineering for Power Price Forecasting
Clean, minimal implementation focused on core predictors
"""

import pandas as pd
import numpy as np

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
            df["price_lag_24h"] = df[self.target_col].shift(24).fillna(method='bfill')
            df["price_lag_168h"] = df[self.target_col].shift(168).fillna(method='bfill')
        
        # Residual load lags
        df["resload_lag_24h"] = df["residual_load_mw"].shift(24).fillna(method='bfill')
        df["resload_lag_168h"] = df["residual_load_mw"].shift(168).fillna(method='bfill')
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling statistics"""
        # 24-hour rolling mean (shifted to avoid leakage)
        df["resload_rolling_mean_24h"] = (
            df["residual_load_mw"].shift(1)
            .rolling(24, min_periods=1)
            .mean()
            .fillna(method='bfill')
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


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/cleaned_energy_data.csv")
    
    # Create features
    engineer = PowerFeatureEngineer(target_col="Day Ahead Price (EUR/MWh)")
    df_featured = engineer.create_all_features(df)
    
    # Get feature list
    features = engineer.get_feature_columns(df_featured)
    print(f"Created {len(features)} features")
    print(f"Sample features: {features[:15]}")
    
    # Check for NaNs
    nan_counts = df_featured[features].isna().sum()
    if nan_counts.sum() > 0:
        print(f"\nWarning: {nan_counts.sum()} total NaN values found")
        print(nan_counts[nan_counts > 0])
    else:
        print("\n✓ No NaN values in features")
    
    # Save
    df_featured.to_csv("data/featured_energy_data.csv", index=False)
    print(f"\n✓ Saved to data/featured_energy_data.csv ({len(df_featured)} rows)")