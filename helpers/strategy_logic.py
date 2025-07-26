# strategy_logic.py

import pandas as pd
import numpy as np
import math
from typing import List

def get_base_weights(
    rebal_date: pd.Timestamp, 
    presence_matrix: pd.DataFrame, 
    prices_df: pd.DataFrame, 
    stock_columns: List[str], 
    percentile_leg: float = 0.20
) -> pd.Series:
    """
    Calculates base portfolio weights based on the momentum strategy.
    The number of stocks in each leg is dynamically calculated based on the
    number of available stocks on the rebalancing date.
    """
    # 1. Determine the universe of available stocks
    available_mask: pd.Series = presence_matrix.loc[rebal_date] == 1
    if not available_mask.any():
        return pd.Series(0.0, index=stock_columns)
    available_stocks: pd.Index = available_mask[available_mask].index

    # 2. Calculate momentum scores for available stocks
    date_1m_ago: pd.Timestamp = rebal_date - pd.DateOffset(months=1)
    date_12m_ago: pd.Timestamp = rebal_date - pd.DateOffset(months=12)

    price_t_1m: pd.Series = prices_df.asof(date_1m_ago)
    price_t_12m: pd.Series = prices_df.asof(date_12m_ago)

    if price_t_1m is None or price_t_12m is None:
        return pd.Series(0.0, index=stock_columns)

    price_t_12m[price_t_12m == 0] = np.nan
    momentum_scores: pd.Series = (price_t_1m / price_t_12m - 1).loc[available_stocks].dropna()

    # Dynamic calculation of leg size
    num_available_for_scoring: int = len(momentum_scores)
    if num_available_for_scoring == 0:
        return pd.Series(0.0, index=stock_columns)

    n_stocks_leg: int = math.ceil(num_available_for_scoring * percentile_leg)

    # 3. Select long and short portfolios
    if num_available_for_scoring < n_stocks_leg * 2:
        return pd.Series(0.0, index=stock_columns)

    long_stocks: pd.Index = momentum_scores.nlargest(n_stocks_leg).index
    short_stocks: pd.Index = momentum_scores.nsmallest(n_stocks_leg).index

    # 4. Form the base weights Series
    base_weights: pd.Series = pd.Series(0.0, index=stock_columns)
    if n_stocks_leg > 0:
        base_weights.loc[long_stocks] = 1 / n_stocks_leg
        base_weights.loc[short_stocks] = -1 / n_stocks_leg

    return base_weights


def get_risk_scale(rebal_date: pd.Timestamp, vol_proxy_df: pd.DataFrame) -> float:
    """
    Calculates the risk control scaling factor (w_scale).
    """
    vol_spx: float = vol_proxy_df.loc[rebal_date, 'spx']
    vol_mom: float = vol_proxy_df.loc[rebal_date, 'momentum']
    
    w_scale: float
    if pd.isna(vol_mom) or pd.isna(vol_spx) or vol_mom == 0:
        w_scale = 1.0
    else:
        # Note: using sqrt as vol_proxy_df contains variance (sigma^2)
        w_scale = min(np.sqrt(vol_spx / vol_mom), 2.0)

    return w_scale