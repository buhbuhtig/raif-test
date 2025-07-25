# data_utils.py
import pandas as pd

import os


def data_loader(returns_file_path, presence_file_path, non_stock_columns):
    try:
        # Load the Parquet files into pandas DataFrames.
        returns_df = pd.read_parquet(returns_file_path)
        presence_matrix = pd.read_parquet(presence_file_path)
        
        if 'date' in returns_df.columns:
            returns_df.set_index('date', inplace=True)
        
        if 'date' in presence_matrix.columns:
            presence_matrix.set_index('date', inplace=True)

        print("   Files loaded successfully!")
        

    except FileNotFoundError:
        # Handle the case where files are not found.
        print(f"Error: One of the files was not found. Please check the paths:\n- {returns_file_path}\n- {presence_file_path}")
    except Exception as e:
        # Handle other potential errors during file loading.
        print(f"An error occurred while loading the files: {e}")
        
    stock_columns = returns_df.columns.difference(non_stock_columns)
    
    
    presence_matrix = presence_matrix.reindex(columns=stock_columns, fill_value=0)
    presence_matrix = presence_matrix.reindex(index=returns_df.index, fill_value=0)
    
    return {'returns_df': returns_df, 'presence_matrix': presence_matrix}


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
    import yfinance as yf

    # 1. Check if file exists and we don't need to force its creation
    if os.path.exists(patch_filename) and not force_creation:
        print(f"     ... patch file '{patch_filename}' already exists. Skipping creation.")
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

    print(f"Missing SPX data found from {first_missing_date.date()} to {last_missing_date.date()}.")
    print("Attempting to download data from Yahoo Finance...")

    # 3. Download, process, and save the patch
    start_download = first_missing_date - pd.Timedelta(days=5)
    end_download = last_missing_date + pd.Timedelta(days=2)
    
    try:
        spx_yahoo_data = yf.download('^GSPC', start=start_download, end=end_download, progress=False, auto_adjust=True)

        if spx_yahoo_data.empty:
            print("Download failed: No data returned from yfinance.")
            return False
        
        print("Download successful.")
        
        if isinstance(spx_yahoo_data.index, pd.MultiIndex):
            spx_yahoo_data = spx_yahoo_data.reset_index().set_index('Date')
        
        returns_series = spx_yahoo_data['Close'].pct_change()
        
        missing_dates_index = spx_series[missing_dates_mask].index
        clean_patch_series = returns_series.loc[returns_series.index.isin(missing_dates_index)]
        
        spx_patch_df = pd.DataFrame(clean_patch_series)
        spx_patch_df.columns = ['spx']

        spx_patch_df.to_parquet(patch_filename)
        print(f"     ... successfully created patch file '{patch_filename}' with {len(spx_patch_df)} rows.")

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
        print(f"Patch file '{patch_filename}' not found. Continuing without patching.")
        return False
    except Exception as e:
        print(f"An error occurred while applying the patch: {e}")
        return False
    return True