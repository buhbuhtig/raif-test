# plotting_utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.dates as mdates
from typing import List, Tuple, Optional
from helpers.logger_setup import setup_logger # Import the logger setup

# Create a logger specific to this module
log = setup_logger(__name__)

def plot_dashboard(
    returns_df: pd.DataFrame, 
    presence_matrix: pd.DataFrame, 
    non_stock_columns: List[str], 
    stock_columns: List[str], 
    figsize: Optional[Tuple[int, int]] = None, 
    save_png_path: Optional[str] = None
) -> None:
    """
    Generates a 2x2 dashboard and optionally saves it as a static PNG file.
    """
    if figsize is None:
        figsize = (12, 6)

    master_index: pd.DatetimeIndex = returns_df.index
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)

    # --- TOP ROW PLOTS ---
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

    # --- DATA PREP FOR BOTTOM ROW ---
    common_stocks: pd.Index = pd.Index(stock_columns).intersection(presence_matrix.columns)
    presence_sync: pd.DataFrame = presence_matrix.reindex(index=master_index, columns=common_stocks)
    sorted_stocks: pd.Index = presence_sync.apply(lambda s: s.first_valid_index()).fillna(pd.Timestamp.max).sort_values().index
    returns_sorted: pd.DataFrame = returns_df[sorted_stocks]
    start_num: float = mdates.date2num(master_index[0])
    end_num: float = mdates.date2num(master_index[-1])

    # --- BOTTOM-LEFT: Detailed Returns Activity Map ---
    ax_bl = axes[1, 0]
    is_nan: pd.DataFrame = returns_sorted.isnull()
    internal_nan_mask: pd.DataFrame = is_nan & returns_sorted.shift(1).notnull() & returns_sorted.shift(-1).notnull()
    map_data: pd.DataFrame = pd.DataFrame(0, index=returns_sorted.index, columns=returns_sorted.columns)
    map_data[returns_sorted > 0] = 1; map_data[returns_sorted < 0] = 2
    map_data[returns_sorted == 0] = 3; map_data[internal_nan_mask] = 4
    colors: List[str] = ['white', 'green', 'red', 'yellow', 'black']
    cmap_detailed = ListedColormap(colors)
    im1 = ax_bl.imshow(map_data.T.values, cmap=cmap_detailed, aspect='auto', interpolation='none', extent=[start_num, end_num, len(sorted_stocks) - 0.5, -0.5])
    ax_bl.set_title('Detailed Stock Returns Activity')
    ax_bl.set_ylabel('Stocks (sorted by presence)')

    # --- BOTTOM-RIGHT: Presence Matrix Activity Map ---
    ax_br = axes[1, 1]
    presence_sorted: pd.DataFrame = presence_sync[sorted_stocks]
    cmap_simple = ListedColormap(['white', 'dimgray'])
    norm_simple = plt.Normalize(vmin=0, vmax=1)
    ax_br.imshow(presence_sorted.T.fillna(0).values, cmap=cmap_simple, norm=norm_simple, aspect='auto', interpolation='none', extent=[start_num, end_num, len(sorted_stocks) - 0.5, -0.5])
    ax_br.set_title('Expected Presence (`presence_matrix`)')

    # --- FINAL FORMATTING ---
    for ax in axes.flatten():
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_xlim(master_index.min(), master_index.max())
    plt.setp(ax_tr.get_yticklabels(), visible=False)
    plt.setp(ax_bl.get_yticklabels(), visible=False)
    plt.setp(ax_br.get_yticklabels(), visible=False)

    labels: List[str] = ['None', 'Pos', 'Neg', 'Zero', 'NaN']
    cbar = fig.colorbar(im1, ax=axes[1, 0], location='bottom', orientation='horizontal', pad=0.1, fraction=0.03, ticks=np.arange(len(colors)))
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()

    if save_png_path:
        try:
            fig.savefig(save_png_path, dpi=150, bbox_inches='tight')
            log.info(f"Dashboard saved to {save_png_path}")
        except Exception as e:
            log.error(f"Error saving PNG file to {save_png_path}: {e}")
    plt.show()


def plot_leverage_and_volatility(
    strategy_leverage: pd.Series, 
    vol_proxy_df: pd.DataFrame, 
    figsize: Optional[Tuple[int, int]] = None, 
    save_png_path: Optional[str] = None
) -> None:
    """
    Generates a side-by-side plot and optionally saves it as a static PNG file.
    """
    if figsize is None:
        figsize = (12, 3)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Portfolio Leverage
    strategy_leverage.plot(ax=axes[0], title='Portfolio Leverage Over Time', grid=True, color='navy')
    axes[0].set_ylabel('Gross Exposure (Leverage)')
    axes[0].axhline(4.0, color='r', linestyle='--', label='Max Leverage (4.0)')
    axes[0].legend()

    # Plot 2: Volatility Proxies
    vol_proxy_df.plot(ax=axes[1], title='Underlying Volatility Proxies', grid=True, logy=True)
    axes[1].set_ylabel('Volatility Proxy (Log Scale)')
    axes[1].legend(['SPX Vol', 'Momentum Vol'])

    plt.tight_layout()

    if save_png_path:
        try:
            fig.savefig(save_png_path, dpi=150, bbox_inches='tight')
            log.info(f"Leverage plot saved to {save_png_path}")
        except Exception as e:
            log.error(f"Error saving PNG file to {save_png_path}: {e}")
    plt.show()