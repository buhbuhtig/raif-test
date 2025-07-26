# data_utils.py
import pandas as pd
import numpy as np
import os
from typing import List, Dict
# NEW: Import the logger setup function
from helpers.logger_setup import setup_logger

# NEW: Create a logger specific to this module
log = setup_logger(__name__)

def data_loader(returns_file_path: str, presence_file_path: str, non_stock_columns: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Loads returns and presence data. If presence_matrix fails to load,
    it generates a fallback matrix assuming all stocks are always present.
    """
    try:
        returns_df: pd.DataFrame = pd.read_parquet(returns_file_path)
        if 'date' in returns_df.columns:
            returns_df.set_index('date', inplace=True)
        log.info("'returns_df' loaded successfully.")
    
    except Exception as e:
        log.critical(f"Failed to load returns_df from {returns_file_path}. Halting. Details: {e}")
        return {}

    try:
        presence_matrix: pd.DataFrame = pd.read_parquet(presence_file_path)
        if 'date' in presence_matrix.columns:
            presence_matrix.set_index('date', inplace=True)
        log.info("'presence_matrix' loaded successfully.")

    except Exception as e:
        log.warning(f"'presence_matrix' not found or failed to parse. Details: {e}")
        log.info("Creating a fallback presence matrix assuming all stocks are always present (filled with 1s).")
        
        stock_columns: pd.Index = returns_df.columns.difference(non_stock_columns)
        presence_matrix = pd.DataFrame(1.0, index=returns_df.index, columns=stock_columns)

    stock_columns = returns_df.columns.difference(non_stock_columns)
    presence_matrix_aligned: pd.DataFrame = presence_matrix.reindex(index=returns_df.index, fill_value=0)
    presence_matrix_final: pd.DataFrame = presence_matrix_aligned.reindex(columns=stock_columns, fill_value=0)
    
    return {'returns_df': returns_df, 'presence_matrix': presence_matrix_final}


def create_spx_patch_file(returns_df: pd.DataFrame, patch_filename: str = 'spx_patch.parquet', force_creation: bool = False) -> bool:
    """Checks for missing SPX data and creates a patch file with the missing returns."""
    log.debug("Running SPX patch file creator...")
    import yfinance as yf

    if os.path.exists(patch_filename) and not force_creation:
        log.info(f"Patch file '{patch_filename}' already exists. Skipping creation.")
        return True

    if 'spx' not in returns_df.columns:
        log.error("'spx' column not found in DataFrame.")
        return False

    spx_series: pd.Series = returns_df['spx']
    missing_dates_mask: pd.Series = spx_series.isnull()

    if not missing_dates_mask.any():
        log.info("No missing SPX data found. Patch file not needed.")
        return True

    first_missing_date: pd.Timestamp = spx_series[missing_dates_mask].index.min()
    last_missing_date: pd.Timestamp = spx_series[missing_dates_mask].index.max()

    log.info(f"Missing SPX data found from {first_missing_date.date()} to {last_missing_date.date()}.")
    log.info("Attempting to download data from Yahoo Finance...")

    start_download: pd.Timestamp = first_missing_date - pd.Timedelta(days=5)
    end_download: pd.Timestamp = last_missing_date + pd.Timedelta(days=2)

    try:
        spx_yahoo_data: pd.DataFrame = yf.download('^GSPC', start=start_download, end=end_download, progress=False, auto_adjust=True)

        if spx_yahoo_data.empty:
            log.warning("Download failed: No data returned from yfinance.")
            return False
        log.info("Download successful.")
        
        # ... (rest of the logic is the same) ...
        returns_series: pd.Series = spx_yahoo_data['Close'].pct_change()
        missing_dates_index: pd.DatetimeIndex = spx_series[missing_dates_mask].index
        clean_patch_series: pd.Series = returns_series.loc[returns_series.index.isin(missing_dates_index)]
        spx_patch_df: pd.DataFrame = pd.DataFrame(clean_patch_series)
        spx_patch_df.columns = ['spx']
        spx_patch_df.to_parquet(patch_filename)
        log.info(f"Successfully created patch file '{patch_filename}' with {len(spx_patch_df)} rows.")
        return True

    except Exception as e:
        log.error(f"An error occurred during patch creation: {e}")
        return False


def apply_spx_hole_patch(patch_filename: str, returns_df: pd.DataFrame) -> bool:
    """Applies the SPX patch file to the main returns DataFrame."""
    try:
        spx_patch_df: pd.DataFrame = pd.read_parquet(patch_filename)
        returns_df.update(spx_patch_df)
        log.info(f"Successfully applied patch from '{patch_filename}'.")
    except FileNotFoundError:
        log.warning(f"Patch file '{patch_filename}' not found. Continuing without patching.")
        return False
    except Exception as e:
        log.error(f"An error occurred while applying the patch: {e}")
        return False
    return True

def generate_custom_presence_matrix(
    returns_df: pd.DataFrame, 
    stock_columns: List[str], 
    window_long: int = 30, 
    threshold_long: float = 0.90, 
    window_short: int = 2
) -> pd.DataFrame:
    """
    Generates a custom presence matrix based on trading activity.
    """
    was_traded: pd.DataFrame = returns_df[stock_columns].notna()
    
    long_term_active: pd.DataFrame = was_traded.rolling(window=window_long).mean() >= threshold_long
    short_term_active: pd.DataFrame = was_traded.rolling(window=window_short).sum() == window_short

    custom_presence_matrix: pd.DataFrame = (long_term_active & short_term_active).astype(float).replace(0, np.nan)
    
    return custom_presence_matrix