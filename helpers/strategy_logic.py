# strategy_logic.py

import pandas as pd
import numpy as np

# In strategy_logic.py
import pandas as pd
import numpy as np
import math # Import math module for ceiling function

def get_base_weights(rebal_date, presence_matrix, prices_df, stock_columns, percentile_leg=0.20):
    """
    Calculates base portfolio weights based on the momentum strategy.
    The number of stocks in each leg is dynamically calculated based on the
    number of available stocks on the rebalancing date.

    Args:
        rebal_date (pd.Timestamp): The date of rebalancing.
        presence_matrix (pd.DataFrame): DataFrame indicating stock presence.
        prices_df (pd.DataFrame): DataFrame with historical stock prices.
        stock_columns (pd.Index): List of stock column names.
        percentile_leg (float): The percentile (e.g., 0.20 for 20%) to select for each leg.

    Returns:
        pd.Series: A Series with equal weights for long/short legs, or zeros.
    """
    
    # 1. Determine the universe of available stocks
    available_mask = presence_matrix.loc[rebal_date] == 1
    if not available_mask.any():
        return pd.Series(0.0, index=stock_columns)
    available_stocks = available_mask[available_mask].index

    # 2. Calculate momentum scores for available stocks
    date_1m_ago = rebal_date - pd.DateOffset(months=1)
    date_12m_ago = rebal_date - pd.DateOffset(months=12)
    
    price_t_1m = prices_df.asof(date_1m_ago)
    price_t_12m = prices_df.asof(date_12m_ago)
    
    # Avoid division by zero and handle potential NaNs from .asof()
    if price_t_1m is None or price_t_12m is None:
        return pd.Series(0.0, index=stock_columns)
        
    price_t_12m[price_t_12m == 0] = np.nan
    momentum_scores = (price_t_1m / price_t_12m - 1).loc[available_stocks].dropna()

    # --- DYNAMIC CALCULATION OF LEG SIZE ---
    num_available_for_scoring = len(momentum_scores)
    if num_available_for_scoring == 0:
        return pd.Series(0.0, index=stock_columns)

    # Calculate number of stocks per leg based on the percentile
    # Use math.ceil to ensure we get at least 1 stock if possible, and handle rounding.
    n_stocks_leg = math.ceil(num_available_for_scoring * percentile_leg)
    
    # 3. Select long and short portfolios
    # Ensure we have enough stocks to form both a long and a short leg
    if num_available_for_scoring < n_stocks_leg * 2:
        return pd.Series(0.0, index=stock_columns)
        
    long_stocks = momentum_scores.nlargest(n_stocks_leg).index
    short_stocks = momentum_scores.nsmallest(n_stocks_leg).index

    # 4. Form the base weights Series
    base_weights = pd.Series(0.0, index=stock_columns)
    if n_stocks_leg > 0:
        base_weights.loc[long_stocks] = 1 / n_stocks_leg
        base_weights.loc[short_stocks] = -1 / n_stocks_leg
    
    return base_weights

def get_risk_scale(rebal_date, vol_proxy_df):
    """
    Calculates the risk control scaling factor (w_scale).

    Args:
        rebal_date (pd.Timestamp): The date of rebalancing.
        vol_proxy_df (pd.DataFrame): DataFrame with pre-calculated volatility proxies.

    Returns:
        float: The scaling factor w_scale.
    """
    vol_spx = vol_proxy_df.loc[rebal_date, 'spx']
    vol_mom = vol_proxy_df.loc[rebal_date, 'momentum']
    
    if pd.isna(vol_mom) or pd.isna(vol_spx) or vol_mom == 0:
        w_scale = 1.0
    else:
        # Note: using sqrt as vol_proxy_df contains variance (sigma^2)
        w_scale = min(np.sqrt(vol_spx / vol_mom), 2.0)
        
    return w_scale