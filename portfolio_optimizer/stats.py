"""Estimate annualized expected returns and covariance from daily returns.

Annualization (ARCHITECTURE.md §5): mu = mean_daily * 252, Sigma = cov_daily * 252.
Pure functions over a returns DataFrame; no network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_mean(returns) -> np.ndarray:
    """Annualized expected return per asset, ordered by column."""
    r = pd.DataFrame(returns)
    return (r.mean(axis=0) * TRADING_DAYS).to_numpy(dtype=float)


def annualized_cov(returns) -> np.ndarray:
    """Annualized covariance matrix, ordered by column."""
    r = pd.DataFrame(returns)
    return (r.cov() * TRADING_DAYS).to_numpy(dtype=float)
