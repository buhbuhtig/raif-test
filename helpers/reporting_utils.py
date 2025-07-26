# reporting_utils.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import json

# In reporting_utils.py


def calculate_performance_metrics(strategy_returns, returns_df):
    """
    Calculates all required performance metrics for a given strategy returns series.

    Args:
        strategy_returns (pd.Series): The daily returns of the strategy.
        returns_df (pd.DataFrame): DataFrame with daily returns for benchmarks.

    Returns:
        dict: A dictionary with all calculated performance metrics.
    """
    # 1. Calculate annualized metrics from the provided strategy_returns
    TRADING_DAYS_PER_YEAR = 252

    # Annualized Return (Geometric)
    cumulative_return = (1 + strategy_returns).cumprod()
    if cumulative_return.empty:
        return {
            "Annualized Return": 0, "Annualized Volatility": 0, "Annualized Sharpe Ratio": 0,
            "Annualized Alpha (vs 6 factors)": 0, "Information Ratio": 0
        }
    total_return = cumulative_return.iloc[-1]
    n_years = len(strategy_returns) / TRADING_DAYS_PER_YEAR
    annualized_return = total_return**(1/n_years) - 1 if n_years > 0 else 0

    # Annualized Volatility
    annualized_volatility = strategy_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Annualized Sharpe Ratio
    annual_risk_free_rate = returns_df['acc_rate'].mean(
    ) * TRADING_DAYS_PER_YEAR
    sharpe_ratio = (annualized_return - annual_risk_free_rate) / \
        annualized_volatility if annualized_volatility > 0 else 0

    # 2. Calculate Alpha and Information Ratio against benchmarks
    y = strategy_returns
    benchmark_cols = ['low_risk', 'momentum',
                      'quality', 'size', 'value', 'spx']
    X = returns_df[benchmark_cols].fillna(0)
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    alpha = model.params['const'] * TRADING_DAYS_PER_YEAR

    tracking_error = model.resid.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    information_ratio = alpha / tracking_error if tracking_error > 1e-6 else 0.0

    # 3. Compile metrics into a dictionary
    metrics = {
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Annualized Sharpe Ratio": sharpe_ratio,
        "Annualized Alpha (vs 6 factors)": alpha,
        "Information Ratio": information_ratio
    }

    return metrics


def plot_performance_summary(strategy_returns, returns_df, metrics, save_png_path=None, save_json_path=None, figsize = None):
    """
    Plots the cumulative performance graph, displays a formatted metrics table,
    and optionally saves the plot to PNG and metrics to JSON.

    Args:
        strategy_returns (pd.Series): The daily returns of the strategy.
        returns_df (pd.DataFrame): DataFrame with benchmark returns.
        metrics (dict): A dictionary with pre-calculated performance metrics.
        save_png_path (str, optional): Path to save the output plot PNG.
        save_json_path (str, optional): Path to save the metrics dictionary as a JSON file.
    """
    if figsize is None:
        figsize = (12,8)
    # 1. Prepare data for plotting
    plot_df = pd.DataFrame({
        'Strategy': (1 + strategy_returns).cumprod(),
        'Benchmark Momentum': (1 + returns_df['momentum']).cumprod(),
        'S&P 500 (SPX)': (1 + returns_df['spx']).cumprod()
    })

    # 2. Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    plot_df.plot(ax=ax, logy=True)
    
    ax.set_title('Cumulative Strategy Performance vs. Benchmarks', fontsize=16)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Cumulative Value (Log Scale)')
    ax.grid(True, which="both", linestyle='--', linewidth=0.5)
    ax.legend(title='Series', loc='upper left', fontsize=10)
    
    if save_png_path:
        try:
            fig.savefig(save_png_path, dpi=150, bbox_inches='tight')
        except Exception as e:
            print(f"   Error saving PNG file: {e}")

    plt.show()

    # 3. Save metrics to JSON if a path is provided
    if save_json_path:
        try:
            with open(save_json_path, 'w') as f:
                json.dump(metrics, f, indent=4)
        except Exception as e:
            print(f"   Error saving JSON file: {e}")

    # 4. Create and print a nicely formatted summary table
    print("\n" + "="*40)
    print("      Strategy Performance Summary")
    print("="*40)
    
    for name, value in metrics.items():
        # Determine the format based on the metric name
        if "Ratio" in name:
            formatted_value = f"{value:.2f}"
        else:
            formatted_value = f"{value:.2%}"
        
        # Print with aligned columns
        print(f"   {name:<30} {formatted_value:>8}")
        
    print("="*40)