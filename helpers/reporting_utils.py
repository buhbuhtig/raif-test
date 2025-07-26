# reporting_utils.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import json
from typing import Dict, Optional, Tuple, List, Any
from helpers.logger_setup import setup_logger # Import the logger setup

# Create a logger specific to this module
log = setup_logger(__name__)

def calculate_performance_metrics(strategy_returns: pd.Series, returns_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates all required performance metrics for a given strategy returns series.
    """
    TRADING_DAYS_PER_YEAR: int = 252

    # Handle the case of an empty returns series
    if strategy_returns.empty or not strategy_returns.notna().any():
        log.warning("Strategy returns series is empty or contains only NaNs. Returning zero metrics.")
        return {
            "Annualized Return": 0.0, "Annualized Volatility": 0.0, "Annualized Sharpe Ratio": 0.0,
            "Annualized Alpha (vs 6 factors)": 0.0, "Information Ratio": 0.0
        }

    # Annualized Return (Geometric)
    cumulative_return: pd.Series = (1 + strategy_returns).cumprod()
    total_return: float = cumulative_return.iloc[-1]
    n_years: float = len(strategy_returns) / TRADING_DAYS_PER_YEAR
    annualized_return: float = total_return**(1/n_years) - 1 if n_years > 0 else 0.0

    # Annualized Volatility
    annualized_volatility: float = strategy_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Annualized Sharpe Ratio
    annual_risk_free_rate: float = returns_df['acc_rate'].mean() * TRADING_DAYS_PER_YEAR
    sharpe_ratio: float = (annualized_return - annual_risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0.0

    # Calculate Alpha and Information Ratio against benchmarks
    y: pd.Series = strategy_returns
    benchmark_cols: List[str] = ['low_risk', 'momentum', 'quality', 'size', 'value', 'spx']
    X: pd.DataFrame = returns_df[benchmark_cols].fillna(0)
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    alpha: float = model.params['const'] * TRADING_DAYS_PER_YEAR
    tracking_error: float = model.resid.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    information_ratio: float = alpha / tracking_error if tracking_error > 1e-6 else 0.0

    # Compile metrics into a dictionary
    metrics: Dict[str, float] = {
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Annualized Sharpe Ratio": sharpe_ratio,
        "Annualized Alpha (vs 6 factors)": alpha,
        "Information Ratio": information_ratio
    }

    return metrics


def plot_performance_summary(
    strategy_returns: pd.Series, 
    returns_df: pd.DataFrame, 
    metrics: Dict[str, float], 
    save_png_path: Optional[str] = None, 
    save_json_path: Optional[str] = None, 
    figsize: Optional[Tuple[int, int]] = None
) -> None:
    """
    Plots the cumulative performance graph, displays a formatted metrics table,
    and optionally saves the plot to PNG and metrics to JSON.
    """
    if figsize is None:
        figsize = (12, 8)
        
    # Prepare data for plotting
    plot_df: pd.DataFrame = pd.DataFrame({
        'Strategy': (1 + strategy_returns).cumprod(),
        'Benchmark Momentum': (1 + returns_df['momentum']).cumprod(),
        'S&P 500 (SPX)': (1 + returns_df['spx']).cumprod()
    })

    # Create the plot
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
            log.info(f"Performance summary plot saved to {save_png_path}")
        except Exception as e:
            log.error(f"Error saving PNG file to {save_png_path}: {e}")

    plt.show()

    # Save metrics to JSON if a path is provided
    if save_json_path:
        try:
            with open(save_json_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            log.info(f"Performance metrics saved to {save_json_path}")
        except Exception as e:
            log.error(f"Error saving JSON file to {save_json_path}: {e}")

    # Create the summary table as a multi-line string for logging
    summary_lines = ["\n" + "="*40, "      Strategy Performance Summary", "="*40]
    for name, value in metrics.items():
        if "Ratio" in name:
            formatted_value: str = f"{value:.2f}"
        else:
            formatted_value: str = f"{value:.2%}"
        
        summary_lines.append(f"   {name:<30} {formatted_value:>8}")
    summary_lines.append("="*40)
    
    # Join all lines into a single string and log it at INFO level
    log.info("\n".join(summary_lines))