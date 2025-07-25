# plotting_utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.dates as mdates

def plot_dashboard(returns_df, presence_matrix, non_stock_columns, stock_columns, figsize=None):
    """
    Generates a 2x2 dashboard. The colorbar is now smaller and positioned
    only under the bottom-left plot.
    """
    if figsize is None:
        figsize = (20, 12)
        
    master_index = returns_df.index
    
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    
    # --- TOP-LEFT & TOP-RIGHT plots (no changes here) ---
    ax_tl = axes[0, 0]
    ax_tl.plot(master_index, (1 + returns_df[non_stock_columns]).cumprod())
    ax_tl.set_title('Cumulative Factor & Benchmark Returns')
    ax_tl.set_ylabel("Cumulative Value")
    ax_tl.grid(True, linestyle='--', alpha=0.5)
    ax_tl.legend(non_stock_columns, fontsize='small', loc='upper left')

    ax_tr = axes[0, 1]
    ax_tr.plot(master_index, (1 + returns_df[stock_columns]).cumprod(), color='gray', alpha=0.2)
    ax_tr.set_title('Cumulative Stock Value')
    ax_tr.set_yscale('log')
    ax_tr.grid(True, linestyle='--', alpha=0.5)

    # --- Data Prep for Bottom Row (no changes here) ---
    common_stocks = pd.Index(stock_columns).intersection(presence_matrix.columns)
    returns_sync = returns_df.reindex(index=master_index, columns=common_stocks)
    presence_sync = presence_matrix.reindex(index=master_index, columns=common_stocks)
    sorted_stocks = presence_sync.apply(lambda s: s.first_valid_index()).fillna(pd.Timestamp.max).sort_values().index
    returns_sync = returns_sync[sorted_stocks]
    presence_sync = presence_sync[sorted_stocks]

    start_num = mdates.date2num(master_index[0])
    end_num = mdates.date2num(master_index[-1])
    
    # --- BOTTOM-LEFT & BOTTOM-RIGHT plots (no changes here) ---
    ax_bl = axes[1, 0]
    map_data = pd.DataFrame(0, index=returns_sync.index, columns=returns_sync.columns)
    map_data[returns_sync > 0] = 1; map_data[returns_sync < 0] = 2
    map_data[returns_sync == 0] = 3; map_data[returns_sync.isnull()] = 4
    map_data[presence_sync.isnull()] = 0
    colors = ['white', 'green', 'red', 'yellow', 'black']
    cmap_detailed = ListedColormap(colors)
    im1 = ax_bl.imshow(map_data.T.values, cmap=cmap_detailed, aspect='auto', interpolation='none', 
                       extent=[start_num, end_num, len(sorted_stocks) - 0.5, -0.5])
    ax_bl.set_title('Detailed Stock Returns Activity')
    ax_bl.set_ylabel('Stocks (sorted)')

    ax_br = axes[1, 1]
    cmap_simple = ListedColormap(['white', 'dimgray'])
    ax_br.imshow(presence_sync[sorted_stocks].T.fillna(0).values, cmap=cmap_simple, aspect='auto', interpolation='none',
                 extent=[start_num, end_num, len(sorted_stocks) - 0.5, -0.5])
    ax_br.set_title('Expected Presence (`presence_matrix`)')

    # --- Final Formatting (no changes here) ---
    for ax in axes.flatten():
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_xlim(master_index.min(), master_index.max())

    plt.setp(ax_tr.get_yticklabels(), visible=False)
    plt.setp(ax_bl.get_yticklabels(), visible=False)
    plt.setp(ax_br.get_yticklabels(), visible=False)

    # --- MODIFIED COLORBAR BLOCK ---
    # We remove the manually created axis and instead attach the colorbar
    # directly to the bottom-left plot (ax_bl).
    
    labels = ['None', 'Pos', 'Neg', 'Zero', 'NaN']
    
    cbar = fig.colorbar(im1, 
                        ax=axes[1, 0],            # Attach to bottom-left plot
                        location='bottom',        # Place it underneath
                        orientation='horizontal',
                        pad=0.2,                  # Distance from the plot
                        fraction=0.05,            # Controls the thickness
                        ticks=np.arange(len(colors)))
                        
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(labelsize=8) # Set the font size for the labels
    
    # Use tight_layout to adjust everything else automatically
    fig.tight_layout()
    plt.show()
    
    
# In plotting_utils.py

def plot_leverage_and_volatility(strategy_leverage, vol_proxy_df, figsize=None):
    """
    Generates a side-by-side plot of portfolio leverage and underlying volatility proxies.
    
    Args:
        strategy_leverage (pd.Series): Series of the strategy's daily leverage.
        vol_proxy_df (pd.DataFrame): DataFrame of daily volatility proxies for SPX and Momentum.
        figsize (tuple, optional): The figure size for the plot.
    """
    if figsize is None:
        figsize = (15, 4)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Plot 1: Portfolio Leverage (on the left, axes[0]) ---
    # Now plots the pre-calculated strategy_leverage Series
    strategy_leverage.plot(
        ax=axes[0],
        title='Portfolio Leverage Over Time',
        grid=True,
        color='navy'
    )
    axes[0].set_ylabel('Gross Exposure (Leverage)')
    # Max leverage is 2 * w_scale_max = 2 * 2.0 = 4.0
    axes[0].axhline(4.0, color='r', linestyle='--', label='Max Leverage (4.0)')
    axes[0].legend()

    # --- Plot 2: Volatility Proxies (on the right, axes[1]) ---
    vol_proxy_df.plot(
        ax=axes[1],
        title='Underlying Volatility Proxies',
        grid=True,
        logy=True
    )
    axes[1].set_ylabel('Volatility Proxy (Log Scale)')
    axes[1].legend(['SPX Vol', 'Momentum Vol'])

    plt.tight_layout()
    plt.show()