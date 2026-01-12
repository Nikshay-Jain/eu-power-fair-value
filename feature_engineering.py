"""
Feature Engineering for Power Price Forecasting
Leakage-safe option + pragmatic fallback so no long leading-NaNs.

Creates:
- Residual Load features (forecast-first, actual-fallback)
- Temporal & cyclical features
- Congestion features
- Lag & rolling statistics
- Holiday & ramp features

Usage:
    engineer = PowerFeatureEngineer(target_col="Day Ahead Price (EUR/MWh)")
    df_features = engineer.create_all_features(df, leakage_safe=False)
"""

import pandas as pd
import numpy as np
import holidays

class PowerFeatureEngineer:
    def __init__(self, target_col: str, country: str = "DE"):
        """
        Parameters
        ----------
        target_col : str
            Column name of Day-Ahead Price (€/MWh) or whatever target you use.
        country : str
            Country code for holiday calendar (default "DE").
        """
        self.target_col = target_col
        # use CountryHoliday object for membership testing
        self.holiday_calendar = holidays.CountryHoliday(country)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def create_all_features(self, df: pd.DataFrame, leakage_safe: bool = False) -> pd.DataFrame:
        """
        Master function building the feature set.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing 'MTU (UTC)' and fundamental columns.
        leakage_safe : bool
            If True, fallback to lagged actuals (shifted) when forecasts are missing.
            This avoids leakage but will produce leading NaNs for the first 24 hours.
            If False (default), fallback uses actuals (no shift) to keep rows populated.
        """
        df = df.copy()

        # parse timestamp (start of MTU)
        df["timestamp"] = pd.to_datetime(df["MTU (UTC)"].str.split(" - ").str[0],
                                         dayfirst=True, utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # core features
        df = self._create_residual_load(df, leakage_safe=leakage_safe)
        df = self._create_temporal_features(df)
        df = self._create_congestion_features(df)
        df = self._create_lag_features(df)
        df = self._create_rolling_features(df)

        return df

    # ---------------------------------------------------------------------
    # Residual load: forecast preferred, actual fallback
    # ---------------------------------------------------------------------
    def _create_residual_load(self, df: pd.DataFrame, leakage_safe: bool = False) -> pd.DataFrame:
        # Primary load source for day-ahead modeling: Day-ahead forecast
        if "Day-ahead Total Load Forecast (MW)" not in df.columns:
            raise KeyError("Day-ahead Total Load Forecast (MW) must be present for residual calculation.")
        load_forecast = df["Day-ahead Total Load Forecast (MW)"]

        # Renewable forecasts: use forecast cols if present
        # If missing, fallback to actuals (or shifted actuals when leakage_safe=True)
        def _fallback(col_forecast: str, col_actual: str) -> pd.Series:
            if col_forecast in df.columns:
                return df[col_forecast].astype(float)
            if col_actual in df.columns:
                if leakage_safe:
                    return df[col_actual].astype(float).shift(24)  # avoids leakage, creates leading NaNs
                return df[col_actual].astype(float)  # pragmatic fallback -> no leading NaNs
            # if neither present return zeros (safe)
            return pd.Series(0.0, index=df.index)

        wind_on = _fallback("Wind Onshore Forecast", "Wind Onshore")
        wind_off = _fallback("Wind Offshore Forecast", "Wind Offshore")
        solar = _fallback("Solar Forecast", "Solar")

        df["wind_on_used_mw"] = wind_on
        df["wind_off_used_mw"] = wind_off
        df["solar_used_mw"] = solar

        df["total_renewable_mw_forecast_or_fallback"] = wind_on + wind_off + solar

        # Also provide pure-actual residuals for diagnostics (no shift)
        wind_on_act = df["Wind Onshore"] if "Wind Onshore" in df.columns else pd.Series(0.0, index=df.index)
        wind_off_act = df["Wind Offshore"] if "Wind Offshore" in df.columns else pd.Series(0.0, index=df.index)
        solar_act = df["Solar"] if "Solar" in df.columns else pd.Series(0.0, index=df.index)

        df["total_renewable_mw_actual"] = wind_on_act + wind_off_act + solar_act

        # Choose model-side residual: prefer forecast-or-fallback
        df["total_renewable_mw"] = df["total_renewable_mw_forecast_or_fallback"]

        # Residual load (forecast-based when possible)
        df["residual_load_mw"] = load_forecast - df["total_renewable_mw"]

        # Also store the actual residual for diagnostics
        if "Actual Total Load (MW)" in df.columns:
            df["residual_load_mw_actual"] = df["Actual Total Load (MW)"] - df["total_renewable_mw_actual"]

        # penetration
        df["renewable_penetration"] = df["total_renewable_mw"] / (load_forecast + 1e-6)

        # ramp signal (hour-to-hour)
        df["residual_load_ramp"] = df["residual_load_mw"].diff()

        return df

    # ---------------------------------------------------------------------
    # Temporal features and holidays
    # ---------------------------------------------------------------------
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = df["timestamp"]

        df["hour"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["month"] = ts.dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # boolean holiday flag
        df["is_holiday"] = ts.dt.date.apply(lambda d: d in self.holiday_calendar).astype(int)

        # cyclical encodings
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        return df

    # ---------------------------------------------------------------------
    # Congestion / cross-border features
    # ---------------------------------------------------------------------
    def _create_congestion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Offered Capacity from BZN|DE-LU (MW)" in df.columns and "Offered Capacity to BZN|DE-LU (MW)" in df.columns:
            df["ntc_spread"] = (
                df["Offered Capacity from BZN|DE-LU (MW)"].astype(float)
                - df["Offered Capacity to BZN|DE-LU (MW)"].astype(float)
            )
        else:
            df["ntc_spread"] = 0.0
        return df

    # ---------------------------------------------------------------------
    # Lag features (safe design)
    # ---------------------------------------------------------------------
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # target lags and residual lags; shifts always use past values
        for lag in [24, 168]:
            # guard if target column not present
            if self.target_col in df.columns:
                df[f"{self.target_col}_lag_{lag}h"] = df[self.target_col].shift(lag)
            else:
                df[f"{self.target_col}_lag_{lag}h"] = np.nan

            df[f"residual_load_lag_{lag}h"] = df["residual_load_mw"].shift(lag)
            df[f"renew_pen_lag_{lag}h"] = df["renewable_penetration"].shift(lag)

        return df

    # ---------------------------------------------------------------------
    # Rolling trend & volatility
    # ---------------------------------------------------------------------
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["residual_load_rollmean_24h"] = df["residual_load_mw"].shift(1).rolling(24, min_periods=1).mean()

        if self.target_col in df.columns:
            df["price_rollstd_24h"] = df[self.target_col].shift(1).rolling(24, min_periods=1).std()
        else:
            df["price_rollstd_24h"] = np.nan

        return df

    # ---------------------------------------------------------------------
    # Utility: feature list
    # ---------------------------------------------------------------------
    def get_feature_columns(self, df: pd.DataFrame):
        exclude = ["MTU (UTC)", "timestamp", self.target_col]
        return [c for c in df.columns if c not in exclude and not c.startswith("_")]


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # load
    df = pd.read_csv("data/cleaned_energy_data.csv")

    # set your true price column here (if you have price)
    TARGET = "Day Ahead Price (EUR/MWh)"  # replace with actual price column

    fe = PowerFeatureEngineer(target_col=TARGET)
    # leakage_safe=False will keep early rows populated; set to True when preparing final training frames that must avoid any forecast-absence fallbacks
    df_feat = fe.create_all_features(df, leakage_safe=False)

    fe_cols = fe.get_feature_columns(df_feat)
    print("Features:", fe_cols[:20])
    df_feat.to_csv("data/featured_energy_data.csv", index=False)
    print("Saved.")
