# data_utils.py
import pandas as pd
import numpy as np
import math
import os


# In data_utils.py
import pandas as pd
import numpy as np

def data_loader(returns_file_path, presence_file_path, non_stock_columns):
    """
    Loads returns and presence data. If presence_matrix fails to load,
    it generates a fallback matrix assuming all stocks are always present.
    """
    try:
        # --- Load primary data (returns_df) ---
        returns_df = pd.read_parquet(returns_file_path)
        if 'date' in returns_df.columns:
            returns_df.set_index('date', inplace=True)
        print("   'returns_df' loaded successfully.")
    
    except Exception as e:
        print(f"   Critical Error: Failed to load returns_df from {returns_file_path}. Halting.")
        print(f"   Details: {e}")
        return {}

    # --- Load or Create presence_matrix ---
    try:
        # Attempt to load the presence matrix file
        presence_matrix = pd.read_parquet(presence_file_path)
        if 'date' in presence_matrix.columns:
            presence_matrix.set_index('date', inplace=True)
        print("   'presence_matrix' loaded successfully.")

    except Exception as e:
        # --- FALLBACK LOGIC ---
        # If loading fails for any reason (file not found, parsing error, etc.)
        print("   Warning: 'presence_matrix' not found or failed to parse.")
        print("   Creating a fallback presence matrix assuming all stocks are always present (filled with 1).")
        
        # Get the shape from returns_df
        stock_columns = returns_df.columns.difference(non_stock_columns)
        
        # Create a new DataFrame with the same shape, filled with 1.0
        presence_matrix = pd.DataFrame(1.0, 
                                     index=returns_df.index, 
                                     columns=stock_columns)

    # --- Final Alignment ---
    # This ensures consistency even if the loaded presence_matrix is misaligned.
    stock_columns = returns_df.columns.difference(non_stock_columns)
    presence_matrix_aligned = presence_matrix.reindex(index=returns_df.index, fill_value=0)
    presence_matrix_final = presence_matrix_aligned.reindex(columns=stock_columns, fill_value=0)
    
    return {'returns_df': returns_df, 'presence_matrix': presence_matrix_final}


def create_spx_patch_file(returns_df, patch_filename='spx_patch.parquet', force_creation=True):
    """
    Checks for missing SPX data and creates a patch file with the missing returns.
    This function ONLY creates the file, it does not apply it.

    Args:
        returns_df (pd.DataFrame): The DataFrame with returns to diagnose.
        patch_filename (str): The name for the output patch file.
        force_creation (bool): If True, creates the file even if it already exists.

    Returns:
        bool: True if the file was created or already existed, False on error.
    """
    print("     ... running SPX patch file creator ")
    import yfinance as yf # local import! we dont need yfinance if we had patch file already

    # 1. Check if file exists and we don't need to force its creation
    if os.path.exists(patch_filename) and not force_creation:
        print(
            f"     ... patch file '{patch_filename}' already exists. Skipping creation.")
        return True

    # 2. Diagnose missing data in the 'spx' column
    if 'spx' not in returns_df.columns:
        print("Error: 'spx' column not found in DataFrame.")
        return False

    spx_series = returns_df['spx']
    missing_dates_mask = spx_series.isnull()

    if not missing_dates_mask.any():
        print("No missing SPX data found. Patch file not needed.")
        return True

    first_missing_date = spx_series[missing_dates_mask].index.min()
    last_missing_date = spx_series[missing_dates_mask].index.max()

    print(
        f"Missing SPX data found from {first_missing_date.date()} to {last_missing_date.date()}.")
    print("Attempting to download data from Yahoo Finance...")

    # 3. Download, process, and save the patch
    start_download = first_missing_date - pd.Timedelta(days=5)
    end_download = last_missing_date + pd.Timedelta(days=2)

    try:
        spx_yahoo_data = yf.download(
            '^GSPC', start=start_download, end=end_download, progress=False, auto_adjust=True)

        if spx_yahoo_data.empty:
            print("Download failed: No data returned from yfinance.")
            return False

        print("Download successful.")

        if isinstance(spx_yahoo_data.index, pd.MultiIndex):
            spx_yahoo_data = spx_yahoo_data.reset_index().set_index('Date')

        returns_series = spx_yahoo_data['Close'].pct_change()

        missing_dates_index = spx_series[missing_dates_mask].index
        clean_patch_series = returns_series.loc[returns_series.index.isin(
            missing_dates_index)]

        spx_patch_df = pd.DataFrame(clean_patch_series)
        spx_patch_df.columns = ['spx']

        spx_patch_df.to_parquet(patch_filename)
        print(
            f"     ... successfully created patch file '{patch_filename}' with {len(spx_patch_df)} rows.")

        return True

    except Exception as e:
        print(f"An error occurred during patch creation: {e}")
        return False


def apply_spx_hole_patch(patch_filename, returns_df):
    try:
        # Read the patch file
        spx_patch_df = pd.read_parquet(patch_filename)

        # Update returns_df in-place with data from the patch
        returns_df.update(spx_patch_df)

        print(f"     ... successfully applied patch from '{patch_filename}'.")

    except FileNotFoundError:
        print(
            f"Patch file '{patch_filename}' not found. Continuing without patching.")
        return False
    except Exception as e:
        print(f"An error occurred while applying the patch: {e}")
        return False
    return True

def generate_custom_presence_matrix(returns_df, stock_columns, window_long=30, threshold_long=0.90, window_short=2):
    """
    Generates a custom presence matrix based on trading activity.

    A stock is considered "present" on a given day if two conditions are met:
    1. Long-term: It traded on at least `threshold_long` fraction of days over the `window_long` period.
    2. Short-term: It traded on all days over the `window_short` period.

    Args:
        returns_df (pd.DataFrame): DataFrame with daily returns.
        stock_columns (list): A list of column names for stocks.
        window_long (int): The number of days for the long-term activity check.
        threshold_long (float): The required trading activity threshold (e.g., 0.90 for 90%).
        window_short (int): The number of days for the short-term activity check.

    Returns:
        pd.DataFrame: A presence matrix with the same index as returns_df, containing 1.0 or NaN.
    """
    # Create a boolean matrix: True where a return exists (i.e., trading occurred)
    was_traded = returns_df[stock_columns].notna()

    # 1. Long-term activity check
    # The rolling mean of a boolean series gives the percentage of True values.
    long_term_active = was_traded.rolling(window=window_long).mean() >= threshold_long

    # 2. Short-term activity check
    # The rolling sum equals the window size only if all values in the window are True.
    short_term_active = was_traded.rolling(window=window_short).sum() == window_short

    # 3. Combine conditions and format the output
    # `&` performs a logical AND operation.
    # Convert boolean (True/False) to float (1.0/0.0) and then replace 0 with NaN.
    custom_presence_matrix = (long_term_active & short_term_active).astype(float).replace(0, np.nan)
    
    return custom_presence_matrix