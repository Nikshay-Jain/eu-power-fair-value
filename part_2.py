import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')

# ============================================================================
# Configuration
# ============================================================================
TARGET = "Day-ahead Price (EUR/MWh)"
DATA_FILE = "data/featured_energy_data.csv"

# High correlation features (|corr| > 0.28 from analysis)
SELECTED_FEATURES = ['renewable_penetration', 'total_renewable_mw', 'renew_pen_x_hour',
       'Solar', 'solar_mw', 'Wind Onshore', 'wind_total_mw',
       'Day-ahead Total Load Forecast (MW)', 'Actual Total Load (MW)',
       'Other', 'resload_lag_168h', 'resload_rolling_mean_24h',
       'Hydro Pumped Storage', 'Fossil Hard coal', 'resload_x_hour',
       'resload_lag_24h', 'Biomass', 'Fossil Gas',
       'Fossil Brown coal/Lignite', 'residual_load_mw']

# ============================================================================
# Walk-Forward Validator
# ============================================================================
class WalkForwardValidator:
    def __init__(self, train_years=2, test_days=30, step_days=30):
        self.train_size = train_years * 365 * 24
        self.test_size = test_days * 24
        self.step = step_days * 24
    
    def split(self, df):
        n = len(df)
        start = self.train_size
        while start + self.test_size <= n:
            yield np.arange(start), np.arange(start, start + self.test_size)
            start += self.step

# ============================================================================
# Metrics
# ============================================================================
def calculate_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    
    if len(y_true) == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'tail_mae_p90': np.nan}
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Tail MAE (P90 hours)
    p90_threshold = np.percentile(y_true, 90)
    tail_mask = y_true >= p90_threshold
    tail_mae = mean_absolute_error(y_true[tail_mask], y_pred[tail_mask]) if tail_mask.sum() > 0 else np.nan
    
    return {'mae': mae, 'rmse': rmse, 'tail_mae_p90': tail_mae, 'n': len(y_true)}

# ============================================================================
# Models
# ============================================================================
def baseline_naive_24h(df, target):
    return df[target].shift(24)

def baseline_naive_168h(df, target):
    return df[target].shift(168)

def train_ridge(X_train, y_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.fillna(0))
    X_test_scaled = scaler.transform(X_test.fillna(0))
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    return model.predict(X_test_scaled), model

def train_lightgbm(X_train, y_train, X_test, quantile=None):
    params = {
        'objective': 'quantile' if quantile else 'regression',
        'metric': 'quantile' if quantile else 'rmse',
        'num_leaves': 64,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 8,
        'min_data_in_leaf': 50,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1
    }
    
    if quantile:
        params['alpha'] = quantile
    
    train_data = lgb.Dataset(X_train.fillna(0), label=y_train)
    model = lgb.train(params, train_data, num_boost_round=500)
    return model.predict(X_test.fillna(0)), model

# ============================================================================
# Walk-Forward Evaluation
# ============================================================================
def run_walk_forward_cv(df, features, target):
    print("="*70)
    print("WALK-FORWARD CROSS-VALIDATION")
    print("="*70)
    
    validator = WalkForwardValidator(train_years=2, test_days=30, step_days=30)
    
    results = {
        'naive_24h': [],
        'naive_168h': [],
        'ridge': [],
        'lightgbm': []
    }
    
    all_predictions = []
    
    for fold, (train_idx, test_idx) in enumerate(validator.split(df), 1):
        train = df.iloc[:train_idx.max()].copy()
        test = df.iloc[test_idx].copy()
        
        y_test = test[target].values
        
        # Baseline: 24h naive
        pred_24h = baseline_naive_24h(df, target).iloc[test_idx].fillna(train[target].median()).values
        results['naive_24h'].append(calculate_metrics(y_test, pred_24h))
        
        # Baseline: 168h naive
        pred_168h = baseline_naive_168h(df, target).iloc[test_idx].fillna(train[target].median()).values
        results['naive_168h'].append(calculate_metrics(y_test, pred_168h))
        
        # Available features
        feat_cols = [f for f in features if f in df.columns]
        X_train = train[feat_cols]
        y_train = train[target]
        X_test = test[feat_cols]
        
        # Ridge
        pred_ridge, _ = train_ridge(X_train, y_train, X_test)
        results['ridge'].append(calculate_metrics(y_test, pred_ridge))
        
        # LightGBM
        pred_lgbm, _ = train_lightgbm(X_train, y_train, X_test)
        results['lightgbm'].append(calculate_metrics(y_test, pred_lgbm))
        
        # Store predictions
        fold_preds = pd.DataFrame({
            'timestamp': test['timestamp'],
            'y_true': y_test,
            'pred_24h': pred_24h,
            'pred_168h': pred_168h,
            'pred_ridge': pred_ridge,
            'pred_lgbm': pred_lgbm,
            'fold': fold
        })
        all_predictions.append(fold_preds)
        
        print(f"Fold {fold}: test period {test['timestamp'].min()} to {test['timestamp'].max()}")
    
    # Aggregate results
    print("\n" + "="*70)
    print("AGGREGATED CV RESULTS")
    print("="*70)
    
    summary = {}
    for model_name, fold_results in results.items():
        mae_vals = [r['mae'] for r in fold_results if not np.isnan(r['mae'])]
        rmse_vals = [r['rmse'] for r in fold_results if not np.isnan(r['rmse'])]
        tail_vals = [r['tail_mae_p90'] for r in fold_results if not np.isnan(r['tail_mae_p90'])]
        
        summary[model_name] = {
            'mae': np.mean(mae_vals),
            'rmse': np.mean(rmse_vals),
            'tail_mae_p90': np.mean(tail_vals)
        }
        
        print(f"\n{model_name.upper()}")
        print(f"  MAE:          {summary[model_name]['mae']:.2f} EUR/MWh")
        print(f"  RMSE:         {summary[model_name]['rmse']:.2f} EUR/MWh")
        print(f"  Tail MAE P90: {summary[model_name]['tail_mae_p90']:.2f} EUR/MWh")
    
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    return summary, predictions_df

# ============================================================================
# Train Final Model & Generate Submission
# ============================================================================
def train_final_model(df, features, target, test_start='2025-11-01'):
    print("\n" + "="*70)
    print("TRAINING FINAL MODEL")
    print("="*70)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    train_mask = df['timestamp'] < test_start
    test_mask = df['timestamp'] >= test_start
    
    train = df[train_mask].copy()
    test = df[test_mask].copy()
    
    feat_cols = [f for f in features if f in df.columns]
    X_train = train[feat_cols]
    y_train = train[target]
    X_test = test[feat_cols]
    
    print(f"Training samples: {len(train):,}")
    print(f"Test samples: {len(test):,}")
    
    # Point forecast (P50)
    pred_p50, model_p50 = train_lightgbm(X_train, y_train, X_test, quantile=0.5)
    
    # Quantile forecasts
    pred_p10, _ = train_lightgbm(X_train, y_train, X_test, quantile=0.1)
    pred_p90, _ = train_lightgbm(X_train, y_train, X_test, quantile=0.9)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feat_cols,
        'importance': model_p50.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(importance.head(15).to_string(index=False))
    
    # Create submission
    submission = pd.DataFrame({
        'id': test['MTU (UTC)'],
        'y_pred': pred_p50
    })
    
    # Save detailed predictions
    detailed_preds = pd.DataFrame({
        'timestamp': test['timestamp'],
        'y_true': test[target] if target in test.columns else np.nan,
        'p10': pred_p10,
        'p50': pred_p50,
        'p90': pred_p90
    })
    
    # Calculate week/month averages
    next_week_mean = pred_p50[:24*7].mean() if len(pred_p50) >= 24*7 else pred_p50.mean()
    next_month_mean = pred_p50[:24*30].mean() if len(pred_p50) >= 24*30 else pred_p50.mean()
    
    print(f"\nNext Week Expected Mean:  {next_week_mean:.2f} EUR/MWh")
    print(f"Next Month Expected Mean: {next_month_mean:.2f} EUR/MWh")
    
    return submission, detailed_preds, importance

# ============================================================================
# Visualization
# ============================================================================
def create_plots(cv_predictions, final_predictions, importance):
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: CV predictions vs actuals (LightGBM)
    ax1 = plt.subplot(3, 2, 1)
    sample = cv_predictions.sample(min(1000, len(cv_predictions)))
    ax1.scatter(sample['y_true'], sample['pred_lgbm'], alpha=0.3, s=10)
    ax1.plot([sample['y_true'].min(), sample['y_true'].max()],
             [sample['y_true'].min(), sample['y_true'].max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Price (EUR/MWh)')
    ax1.set_ylabel('Predicted Price (EUR/MWh)')
    ax1.set_title('LightGBM: Predicted vs Actual (CV)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    ax2 = plt.subplot(3, 2, 2)
    residuals = cv_predictions['y_true'] - cv_predictions['pred_lgbm']
    ax2.hist(residuals.dropna(), bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residual (EUR/MWh)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residuals Distribution (LightGBM)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Time series (last fold)
    ax3 = plt.subplot(3, 2, 3)
    last_fold = cv_predictions[cv_predictions['fold'] == cv_predictions['fold'].max()].iloc[:240]
    ax3.plot(last_fold.index, last_fold['y_true'], label='Actual', linewidth=2)
    ax3.plot(last_fold.index, last_fold['pred_lgbm'], label='LightGBM', linewidth=2, alpha=0.7)
    ax3.legend()
    ax3.set_xlabel('Hour')
    ax3.set_ylabel('Price (EUR/MWh)')
    ax3.set_title('Forecast vs Actual (Last 10 Days of CV)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Model comparison
    ax4 = plt.subplot(3, 2, 4)
    models = {
        'naive_24h': 'pred_24h',
        'naive_168h': 'pred_168h',
        'ridge': 'pred_ridge',
        'lightgbm': 'pred_lgbm'
    }

    mae_vals = []
    for name, col in models.items():
        mae = mean_absolute_error(
            cv_predictions['y_true'],
            cv_predictions[col]
        )
        mae_vals.append(mae)

    ax4.barh(list(models.keys()), mae_vals,
            color=['#d62728','#ff7f0e','#2ca02c','#1f77b4'])
    ax4.set_xlabel('MAE (EUR/MWh)')
    ax4.set_title('Model Comparison (CV)')
    ax4.grid(True, alpha=0.3, axis='x')

    # ax4.barh(models, mae_vals, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
    # ax4.set_xlabel('MAE (EUR/MWh)')
    # ax4.set_title('Model Comparison (CV)')
    # ax4.grid(True, alpha=0.3, axis='x')
    
    # Plot 5: Feature importance
    ax5 = plt.subplot(3, 2, 5)
    top15 = importance.head(15)
    ax5.barh(range(len(top15)), top15['importance'], color='steelblue')
    ax5.set_yticks(range(len(top15)))
    ax5.set_yticklabels(top15['feature'], fontsize=9)
    ax5.set_xlabel('Importance (Gain)')
    ax5.set_title('Top 15 Feature Importance')
    ax5.grid(True, alpha=0.3, axis='x')
    
    # Plot 6: Quantile bands (test set)
    ax6 = plt.subplot(3, 2, 6)
    sample_final = final_predictions.iloc[:240]  # First 10 days
    x = range(len(sample_final))
    ax6.fill_between(x, sample_final['p10'], sample_final['p90'], alpha=0.3, label='P10-P90 Band')
    ax6.plot(x, sample_final['p50'], label='P50 (Median)', linewidth=2, color='darkblue')
    if not sample_final['y_true'].isna().all():
        ax6.plot(x, sample_final['y_true'], label='Actual', linewidth=2, color='red', alpha=0.7)
    ax6.legend()
    ax6.set_xlabel('Hour')
    ax6.set_ylabel('Price (EUR/MWh)')
    ax6.set_title('Quantile Forecast Bands (Test Set)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/part2_forecasting_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plots saved to results/part2_forecasting_results.png")

# ============================================================================
# Main Execution
# ============================================================================
def main():
    import os
    os.makedirs('results', exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in data")
    
    print(f"Loaded {len(df):,} rows")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Walk-forward CV
    summary, cv_predictions = run_walk_forward_cv(df, SELECTED_FEATURES, TARGET)
    
    # Save CV results
    pd.DataFrame(summary).T.to_csv('results/part2_cv_summary.csv')
    cv_predictions.to_csv('results/part2_cv_predictions.csv', index=False)
    
    # Train final model
    submission, final_predictions, importance = train_final_model(df, SELECTED_FEATURES, TARGET)
    
    # Save outputs
    submission.to_csv('results/submission.csv', index=False)
    final_predictions.to_csv('results/part2_final_predictions.csv', index=False)
    importance.to_csv('results/part2_feature_importance.csv', index=False)
    
    # Create plots
    create_plots(cv_predictions, final_predictions, importance)
    
    print("\n" + "="*70)
    print("✅ PART 2 COMPLETE")
    print("="*70)
    print("Outputs:")
    print("  • results/submission.csv - Out-of-sample predictions")
    print("  • results/part2_cv_summary.csv - CV metrics")
    print("  • results/part2_cv_predictions.csv - All CV predictions")
    print("  • results/part2_final_predictions.csv - Test set with P10/P50/P90")
    print("  • results/part2_feature_importance.csv - Feature rankings")
    print("  • results/part2_forecasting_results.png - Visualizations")

if __name__ == "__main__":
    main()