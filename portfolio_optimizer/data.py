"""Load and cache historical close prices, and convert them to returns.

Prices are fetched once via yfinance and cached to disk as CSV, so repeat runs
are reproducible and work offline on the cached data (ARCHITECTURE.md §4, G6).
Inputs are validated (G3) before they reach the optimizer.
"""
from __future__ import annotations

import os

import pandas as pd

from .guardrails import GuardrailError, validate_prices

CACHE_DIR_DEFAULT = "data_cache"


def _cache_path(cache_dir: str, tickers: list[str], start: str, end: str) -> str:
    key = f"{'-'.join(tickers)}__{start}__{end}.csv"
    return os.path.join(cache_dir, key)


def load_prices(
    tickers,
    start,
    end,
    cache_dir: str = CACHE_DIR_DEFAULT,
    validate: bool = True,
    min_rows: int = 60,
) -> pd.DataFrame:
    """Return a DataFrame of adjusted close prices (one column per ticker, in the
    given order), reading from cache when available and otherwise fetching from
    yfinance and caching the result.
    """
    tickers = list(tickers)
    path = _cache_path(cache_dir, tickers, str(start), str(end))

    if os.path.exists(path):
        prices = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        import yfinance as yf

        raw = yf.download(
            tickers, start=start, end=end, progress=False, auto_adjust=True
        )
        if raw is None or len(raw) == 0:
            raise GuardrailError(
                f"No data returned for {tickers} in {start}..{end} (failed tickers?)."
            )
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        prices = pd.DataFrame(close)
        if not isinstance(close.columns, pd.MultiIndex) and len(tickers) == 1:
            prices.columns = tickers
        prices = prices.reindex(columns=tickers)
        os.makedirs(cache_dir, exist_ok=True)
        prices.to_csv(path)

    prices.columns = [str(c) for c in prices.columns]
    if validate:
        validate_prices(prices, min_rows=min_rows)
    return prices


def to_returns(prices) -> pd.DataFrame:
    """Daily simple returns with leading NaNs dropped."""
    return pd.DataFrame(prices).pct_change().dropna(how="any")
