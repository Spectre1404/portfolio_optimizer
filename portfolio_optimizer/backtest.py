"""Out-of-sample backtest (ARCHITECTURE.md §7 stretch; makes G2/G7 real).

Weights are solved on a training window ONLY, then applied unchanged to a
strictly-later test window. The realized test-window stats are honest
out-of-sample numbers — no re-optimization on test data. The no-look-ahead
split is enforced with :func:`guardrails.no_lookahead`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import to_returns
from .stats import annualized_mean, annualized_cov
from .guardrails import (
    no_lookahead,
    validate_prices,
    ensure_psd,
    equal_weight,
)
from .optimize import PortfolioResult, max_sharpe, min_variance


@dataclass
class BacktestEntry:
    """One portfolio: fixed training weights, and its in/out-of-sample stats."""

    name: str
    weights: np.ndarray
    in_sample: PortfolioResult
    out_of_sample: PortfolioResult


@dataclass
class BacktestResult:
    split_date: str
    train_span: tuple
    train_days: int
    test_span: tuple
    test_days: int
    entries: dict  # name -> BacktestEntry


def split_by_date(prices, split_date):
    """Split into (train, test): train is strictly before ``split_date``, test is
    on/after it. No overlap by construction."""
    df = pd.DataFrame(prices)
    cutoff = pd.Timestamp(split_date)
    train = df[df.index < cutoff]
    test = df[df.index >= cutoff]
    return train, test


def evaluate_weights(weights, returns, rf: float = 0.0) -> PortfolioResult:
    """Apply FIXED weights to a returns window and report realized annualized stats.

    Formula (kept consistent with the in-sample PortfolioResult):
      mu = annualized_mean(returns); cov = annualized_cov(returns)
      ret = w@mu; vol = sqrt(w@cov@w); sharpe = (ret - rf) / vol
    """
    w = np.asarray(weights, dtype=float)
    mu = annualized_mean(returns)
    cov = annualized_cov(returns)
    ret = float(w @ mu)
    vol = float(np.sqrt(max(float(w @ cov @ w), 0.0)))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return PortfolioResult(weights=w, expected_return=ret, volatility=vol, sharpe=sharpe)


def backtest_windows(
    train_prices,
    test_prices,
    rf: float = 0.0,
    bounds=(0.0, 1.0),
    min_rows: int = 60,
) -> BacktestResult:
    """Solve on ``train_prices`` only, evaluate the fixed weights on ``test_prices``.

    Enforces no look-ahead (G7) and validates both windows (G3) before computing.
    """
    train = pd.DataFrame(train_prices)
    test = pd.DataFrame(test_prices)

    # G7: training window must strictly precede the test window.
    no_lookahead(train.index, test.index)
    # G3: both windows need enough history to estimate / evaluate on.
    validate_prices(train, min_rows=min_rows)
    validate_prices(test, min_rows=min_rows)

    # --- Train only: estimate and optimize ---
    train_returns = to_returns(train)
    mu_tr = annualized_mean(train_returns)
    cov_tr = ensure_psd(annualized_cov(train_returns))  # G4

    ms = max_sharpe(mu_tr, cov_tr, rf=rf, bounds=bounds)
    mv = min_variance(mu_tr, cov_tr, bounds=bounds)
    ew = equal_weight(mu_tr, cov_tr, rf=rf)  # G2 benchmark

    # --- Test only: apply the FIXED training weights (no re-optimization) ---
    test_returns = to_returns(test)
    entries = {}
    for name, in_sample in (
        ("max_sharpe", ms),
        ("min_variance", mv),
        ("equal_weight", ew),
    ):
        oos = evaluate_weights(in_sample.weights, test_returns, rf=rf)
        entries[name] = BacktestEntry(
            name=name,
            weights=np.asarray(in_sample.weights, dtype=float),
            in_sample=in_sample,
            out_of_sample=oos,
        )

    return BacktestResult(
        split_date=str(pd.Timestamp(test.index.min()).date()),
        train_span=(train.index.min().date(), train.index.max().date()),
        train_days=len(train_returns),
        test_span=(test.index.min().date(), test.index.max().date()),
        test_days=len(test_returns),
        entries=entries,
    )


def run_backtest(prices, split_date, rf: float = 0.0, bounds=(0.0, 1.0)) -> BacktestResult:
    """Split ``prices`` at ``split_date`` and run the out-of-sample backtest."""
    train, test = split_by_date(prices, split_date)
    return backtest_windows(train, test, rf=rf, bounds=bounds)
