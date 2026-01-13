"""
Part 3: Prompt Curve Translation
Convert DA forecasts → Tradable week/month-ahead signals
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from datetime import timedelta

sns.set_style('whitegrid')

# ============================================================================
# Configuration
# ============================================================================
PREDICTIONS_FILE = "results/part2_final_predictions.csv"

# Mock forward prices (in production: fetch from EEX/ICE API)
# Assume forward trades at slight discount to expected DA
FORWARD_PRICES = {
    'week_1': 75.0,   # Week 1 baseload
    'week_2': 73.5,
    'week_3': 72.0,
    'week_4': 71.0,
    'month_1': 70.0,  # Month 1 baseload
    'month_2': 69.0,
    'month_3': 68.5
}

SIGNAL_THRESHOLD = 5.0  # Minimum edge (EUR/MWh) to trade
CONFIDENCE_THRESHOLD = 0.65  # Min probability for signal
MAX_POSITION_MW = 20  # Position size cap

# ============================================================================
# Prompt Curve Translator
# ============================================================================
class PromptCurveTranslator:
    def __init__(self, signal_threshold=5.0, confidence_threshold=0.65, max_position=20):
        self.signal_threshold = signal_threshold
        self.confidence_threshold = confidence_threshold
        self.max_position = max_position
    
    def aggregate_to_period(self, df, start, end):
        """Aggregate hourly forecasts to delivery period."""
        mask = (df['timestamp'] >= start) & (df['timestamp'] < end)
        period_df = df[mask]
        
        if len(period_df) == 0:
            return None
        
        expected = period_df['p50'].mean()
        p10 = period_df['p10'].mean()
        p90 = period_df['p90'].mean()
        std_dev = (p90 - p10) / 2.56  # Assume normal distribution
        
        return {
            'expected_price': expected,
            'p10_price': p10,
            'p90_price': p90,
            'std_dev': std_dev,
            'n_hours': len(period_df),
            'start': start,
            'end': end
        }
    
    def generate_signal(self, forecast_stats, forward_price):
        """Generate trading signal by comparing forecast to forward."""
        expected = forecast_stats['expected_price']
        std_dev = forecast_stats['std_dev']
        
        # Edge calculation
        edge = expected - forward_price
        
        # Risk-adjusted signal (Sharpe-style)
        sharpe = edge / (std_dev + 1e-6)
        
        # Probability of profitability
        if edge > 0:
            prob = 1 - norm.cdf(forward_price, loc=expected, scale=std_dev)
        else:
            prob = norm.cdf(forward_price, loc=expected, scale=std_dev)
        
        # Decision logic
        if abs(edge) >= self.signal_threshold and prob >= self.confidence_threshold:
            signal = 'LONG' if edge > 0 else 'SHORT'
            # Position sizing: scale with confidence, cap at max
            confidence_mult = (prob - 0.5) / 0.5
            size = 10 * min(abs(sharpe), 2) * confidence_mult
            size = min(size, self.max_position)
        else:
            signal = 'NEUTRAL'
            size = 0
        
        # Expected P&L
        expected_pnl = edge * size * forecast_stats['n_hours']
        
        return {
            'signal': signal,
            'edge_eur': edge,
            'sharpe_ratio': sharpe,
            'probability': prob,
            'position_mw': size,
            'expected_pnl_eur': expected_pnl,
            'forward_price': forward_price
        }
    
    def check_invalidation(self, forecast_stats, signal_dict):
        """Check invalidation triggers."""
        triggers = []
        
        # Trigger 1: Edge eroded
        if abs(signal_dict['edge_eur']) < self.signal_threshold * 0.5:
            triggers.append(f"Edge eroded to {signal_dict['edge_eur']:.1f} EUR/MWh")
        
        # Trigger 2: Low confidence
        if signal_dict['probability'] < 0.6:
            triggers.append(f"Confidence dropped to {signal_dict['probability']*100:.0f}%")
        
        # Trigger 3: Wide uncertainty band
        band_width = forecast_stats['p90_price'] - forecast_stats['p10_price']
        if band_width > 50:  # Very wide uncertainty
            triggers.append(f"High uncertainty: P10-P90 band = {band_width:.1f} EUR/MWh")
        
        if len(triggers) == 0:
            action = 'HOLD'
        elif len(triggers) == 1:
            action = 'REDUCE_50%'
        else:
            action = 'CLOSE'
        
        return {
            'should_invalidate': len(triggers) > 0,
            'triggered_rules': triggers,
            'action': action
        }
    
    def format_report(self, period_name, forecast_stats, signal, invalidation):
        """Generate trader-ready report."""
        report = f"""
{'='*70}
TRADING SIGNAL: {period_name}
{'='*70}

DELIVERY PERIOD:
  {forecast_stats['start'].strftime('%Y-%m-%d %H:%M')} to {forecast_stats['end'].strftime('%Y-%m-%d %H:%M')}
  Duration: {forecast_stats['n_hours']} hours

FORECAST:
  Expected DA (P50):     {forecast_stats['expected_price']:.2f} EUR/MWh
  P10 (Downside):        {forecast_stats['p10_price']:.2f} EUR/MWh
  P90 (Upside):          {forecast_stats['p90_price']:.2f} EUR/MWh
  Std Deviation:         {forecast_stats['std_dev']:.2f} EUR/MWh

MARKET:
  Forward Price:         {signal['forward_price']:.2f} EUR/MWh
  Edge:                  {signal['edge_eur']:+.2f} EUR/MWh
  Sharpe Ratio:          {signal['sharpe_ratio']:.2f}
  Win Probability:       {signal['probability']*100:.1f}%

RECOMMENDATION:
  Signal:                {signal['signal']}
  Position Size:         {signal['position_mw']:.1f} MW Baseload
  Expected P&L:          {signal['expected_pnl_eur']:,.0f} EUR

RISK MANAGEMENT:
  Action:                {invalidation['action']}
"""
        
        if invalidation['should_invalidate']:
            report += "  ⚠️  ALERTS:\n"
            for trigger in invalidation['triggered_rules']:
                report += f"    - {trigger}\n"
        else:
            report += "  ✓ No invalidation triggers\n"
        
        report += "\nDESK ACTIONS:\n"
        if signal['signal'] == 'LONG':
            report += f"  • BUY {signal['position_mw']:.0f} MW {period_name} Baseload @ {signal['forward_price']:.2f} EUR/MWh\n"
            report += f"  • Expect to sell in DA @ ~{forecast_stats['expected_price']:.2f} EUR/MWh\n"
            report += f"  • Profit: {signal['expected_pnl_eur']:,.0f} EUR if forecast realized\n"
        elif signal['signal'] == 'SHORT':
            report += f"  • SELL {signal['position_mw']:.0f} MW {period_name} Baseload @ {signal['forward_price']:.2f} EUR/MWh\n"
            report += f"  • Expect to buy back in DA @ ~{forecast_stats['expected_price']:.2f} EUR/MWh\n"
            report += f"  • Profit: {signal['expected_pnl_eur']:,.0f} EUR if forecast realized\n"
        else:
            report += "  • NO TRADE - Edge insufficient or confidence too low\n"
        
        report += "\nINVALIDATION TRIGGERS:\n"
        report += "  • Wind forecast revision >20%\n"
        report += "  • Load forecast revision >15%\n"
        report += "  • 3-day cumulative forecast error >15 EUR/MWh\n"
        report += "  • Forward price converges to forecast (edge →0)\n"
        
        report += f"\n{'='*70}\n"
        return report

# ============================================================================
# Generate Signals for Multiple Periods
# ============================================================================
def generate_all_signals(predictions_df, forward_prices):
    """Generate signals for weeks and months."""
    translator = PromptCurveTranslator(
        signal_threshold=SIGNAL_THRESHOLD,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        max_position=MAX_POSITION_MW
    )
    
    predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
    start_date = predictions_df['timestamp'].min()
    
    signals = []
    reports = []
    
    # Weekly signals (4 weeks)
    for i in range(4):
        week_start = start_date + timedelta(weeks=i)
        week_end = week_start + timedelta(weeks=1)
        
        forecast_stats = translator.aggregate_to_period(predictions_df, week_start, week_end)
        if forecast_stats is None:
            continue
        
        forward_key = f'week_{i+1}'
        forward_price = forward_prices.get(forward_key, forecast_stats['expected_price'] - 3)
        
        signal = translator.generate_signal(forecast_stats, forward_price)
        invalidation = translator.check_invalidation(forecast_stats, signal)
        
        period_name = f"Week {i+1}"
        report = translator.format_report(period_name, forecast_stats, signal, invalidation)
        
        signals.append({
            'period': period_name,
            'period_type': 'week',
            'start': week_start,
            'end': week_end,
            **forecast_stats,
            **signal,
            'invalidation_action': invalidation['action']
        })
        
        reports.append(report)
        print(report)
    
    # Monthly signals (3 months)
    for i in range(3):
        month_start = start_date + timedelta(days=30*i)
        month_end = month_start + timedelta(days=30)
        
        forecast_stats = translator.aggregate_to_period(predictions_df, month_start, month_end)
        if forecast_stats is None:
            continue
        
        forward_key = f'month_{i+1}'
        forward_price = forward_prices.get(forward_key, forecast_stats['expected_price'] - 5)
        
        signal = translator.generate_signal(forecast_stats, forward_price)
        invalidation = translator.check_invalidation(forecast_stats, signal)
        
        period_name = f"Month {i+1}"
        report = translator.format_report(period_name, forecast_stats, signal, invalidation)
        
        signals.append({
            'period': period_name,
            'period_type': 'month',
            'start': month_start,
            'end': month_end,
            **forecast_stats,
            **signal,
            'invalidation_action': invalidation['action']
        })
        
        reports.append(report)
        print(report)
    
    return pd.DataFrame(signals), reports

# ============================================================================
# Visualization
# ============================================================================
def create_plots(signals_df, predictions_df):
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Signal distribution
    ax1 = plt.subplot(2, 3, 1)
    signal_counts = signals_df['signal'].value_counts()
    colors = {'LONG': '#2ca02c', 'SHORT': '#d62728', 'NEUTRAL': '#7f7f7f'}
    ax1.bar(signal_counts.index, signal_counts.values, 
            color=[colors.get(x, '#7f7f7f') for x in signal_counts.index])
    ax1.set_ylabel('Count')
    ax1.set_title('Signal Distribution')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Edge by period
    ax2 = plt.subplot(2, 3, 2)
    week_signals = signals_df[signals_df['period_type'] == 'week']
    colors_edge = ['green' if e > 0 else 'red' for e in week_signals['edge_eur']]
    ax2.barh(week_signals['period'], week_signals['edge_eur'], color=colors_edge, alpha=0.7)
    ax2.axvline(SIGNAL_THRESHOLD, color='green', linestyle='--', label=f'Long threshold (+{SIGNAL_THRESHOLD})')
    ax2.axvline(-SIGNAL_THRESHOLD, color='red', linestyle='--', label=f'Short threshold (-{SIGNAL_THRESHOLD})')
    ax2.set_xlabel('Edge (EUR/MWh)')
    ax2.set_title('Edge vs Forward Price (Weeks)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Position sizing
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(signals_df['probability']*100, signals_df['position_mw'], 
                c=signals_df['edge_eur'], cmap='RdYlGn', s=100, alpha=0.7)
    ax3.axhline(MAX_POSITION_MW, color='red', linestyle='--', label=f'Max position ({MAX_POSITION_MW} MW)')
    ax3.axvline(CONFIDENCE_THRESHOLD*100, color='orange', linestyle='--', 
                label=f'Min confidence ({CONFIDENCE_THRESHOLD*100:.0f}%)')
    ax3.set_xlabel('Win Probability (%)')
    ax3.set_ylabel('Position Size (MW)')
    ax3.set_title('Position Sizing Logic')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax3.collections[0], ax=ax3)
    cbar.set_label('Edge (EUR/MWh)')
    
    # Plot 4: Expected P&L
    ax4 = plt.subplot(2, 3, 4)
    pnl_by_period = signals_df.groupby('period')['expected_pnl_eur'].sum()
    colors_pnl = ['green' if x > 0 else 'red' for x in pnl_by_period.values]
    ax4.barh(pnl_by_period.index, pnl_by_period.values, color=colors_pnl, alpha=0.7)
    ax4.set_xlabel('Expected P&L (EUR)')
    ax4.set_title('Expected P&L by Period')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Plot 5: Forecast bands timeline
    ax5 = plt.subplot(2, 3, 5)
    for _, row in week_signals.iterrows():
        period_center = row['start'] + (row['end'] - row['start']) / 2
        ax5.errorbar(period_center, row['expected_price'], 
                     yerr=[[row['expected_price']-row['p10_price']], 
                           [row['p90_price']-row['expected_price']]], 
                     fmt='o', capsize=5, label=row['period'])
        ax5.axhline(row['forward_price'], xmin=0.1, xmax=0.9, 
                    linestyle='--', alpha=0.5, color='gray')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Price (EUR/MWh)')
    ax5.set_title('Forecast vs Forward (P10/P50/P90)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Risk-return scatter
    ax6 = plt.subplot(2, 3, 6)
    active_signals = signals_df[signals_df['signal'] != 'NEUTRAL']
    if len(active_signals) > 0:
        for signal_type in ['LONG', 'SHORT']:
            subset = active_signals[active_signals['signal'] == signal_type]
            ax6.scatter(subset['std_dev'], subset['edge_eur'], 
                       label=signal_type, s=100, alpha=0.7)
        ax6.axhline(SIGNAL_THRESHOLD, color='green', linestyle='--', alpha=0.5)
        ax6.axhline(-SIGNAL_THRESHOLD, color='red', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Forecast Std Dev (EUR/MWh)')
        ax6.set_ylabel('Edge (EUR/MWh)')
        ax6.set_title('Risk vs Return')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/part3_trading_signals.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plots saved to results/part3_trading_signals.png")

# ============================================================================
# Main Execution
# ============================================================================
def main():
    os.makedirs('results', exist_ok=True)
    
    # Load predictions
    print("Loading Part 2 predictions...")
    predictions = pd.read_csv(PREDICTIONS_FILE)
    print(f"Loaded {len(predictions):,} hourly predictions")
    
    # Generate signals
    signals_df, reports = generate_all_signals(predictions, FORWARD_PRICES)
    
    # Save outputs
    signals_df.to_csv('results/part3_trading_signals.csv', index=False)
    
    with open('results/part3_trading_reports.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(reports))
    
    # Summary statistics
    print("\n" + "="*70)
    print("TRADING SUMMARY")
    print("="*70)
    print(f"Total signals generated: {len(signals_df)}")
    print(f"  Long signals:    {(signals_df['signal'] == 'LONG').sum()}")
    print(f"  Short signals:   {(signals_df['signal'] == 'SHORT').sum()}")
    print(f"  Neutral:         {(signals_df['signal'] == 'NEUTRAL').sum()}")
    print(f"\nTotal expected P&L: {signals_df['expected_pnl_eur'].sum():,.0f} EUR")
    print(f"Average position size: {signals_df[signals_df['position_mw'] > 0]['position_mw'].mean():.1f} MW")
    print(f"Average edge (active): {signals_df[signals_df['signal'] != 'NEUTRAL']['edge_eur'].mean():.2f} EUR/MWh")
    
    # Create plots
    create_plots(signals_df, predictions)
    
    print("\n" + "="*70)
    print("✅ PART 3 COMPLETE")
    print("="*70)
    print("Outputs:")
    print("  • results/part3_trading_signals.csv - Signal details")
    print("  • results/part3_trading_reports.txt - Trader reports")
    print("  • results/part3_trading_signals.png - Visualizations")
    print("\nNext: Review signals and adjust FORWARD_PRICES for real market conditions")

if __name__ == "__main__":
    main()