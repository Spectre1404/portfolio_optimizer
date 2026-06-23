"""Tests for the out-of-sample backtest (no look-ahead, fixed train weights).

Synthetic, deterministic, offline (no yfinance). A regime change at the split
makes in-sample and out-of-sample stats genuinely diverge.
"""
import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.guardrails import GuardrailError, equal_weight
from portfolio_optimizer.stats import annualized_mean, annualized_cov
from portfolio_optimizer.optimize import max_sharpe, min_variance
from portfolio_optimizer.data import to_returns
from portfolio_optimizer.backtest import (
    split_by_date,
    backtest_windows,
    run_backtest,
    evaluate_weights,
)

SPLIT = "2022-01-01"
TOL = 1e-9


def _regime_prices(seed=0):
    """Prices with a different drift/vol regime before vs. after SPLIT, so
    in-sample and out-of-sample stats clearly differ."""
    rng = np.random.default_rng(seed)
    cols = ["A", "B", "C"]
    train_idx = pd.date_range("2020-01-01", "2021-12-31", freq="B")
    test_idx = pd.date_range("2022-01-03", "2023-06-30", freq="B")

    train_drift = np.array([0.0008, 0.0004, 0.0006])
    test_drift = np.array([-0.0003, 0.0009, 0.0001])  # regime flip
    tr = rng.normal(train_drift, 0.010, size=(len(train_idx), 3))
    te = rng.normal(test_drift, 0.015, size=(len(test_idx), 3))

    rets = np.vstack([tr, te])
    idx = train_idx.append(test_idx)
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=cols)


def test_no_lookahead_is_enforced():
    # Hand an overlapping train/test pair straight to the engine -> must raise.
    prices = _regime_prices()
    train = prices.loc[:"2022-03-31"]
    test = prices.loc["2022-01-01":]  # overlaps the tail of train
    with pytest.raises(GuardrailError):
        backtest_windows(train, test)


def test_training_weights_applied_unchanged_to_test():
    prices = _regime_prices()
    res = run_backtest(prices, SPLIT, rf=0.02)

    # Recompute what training-only optimization should have produced.
    train, _ = split_by_date(prices, SPLIT)
    mu_tr = annualized_mean(to_returns(train))
    cov_tr = annualized_cov(to_returns(train))
    expected = {
        "max_sharpe": max_sharpe(mu_tr, cov_tr, rf=0.02).weights,
        "min_variance": min_variance(mu_tr, cov_tr).weights,
        "equal_weight": equal_weight(mu_tr, cov_tr, rf=0.02).weights,
    }
    for name, exp_w in expected.items():
        entry = res.entries[name]
        # The fixed weights are the training weights, unchanged.
        assert entry.weights == pytest.approx(exp_w, abs=1e-6)
        # And the out-of-sample result carries those very same weights.
        assert entry.out_of_sample.weights == pytest.approx(exp_w, abs=1e-6)


def test_out_of_sample_differs_from_in_sample():
    prices = _regime_prices()
    res = run_backtest(prices, SPLIT, rf=0.02)
    # With a regime change, OOS Sharpe should not equal IS Sharpe for the
    # optimized portfolios (sanity that test data is really re-evaluated).
    for name in ("max_sharpe", "min_variance"):
        e = res.entries[name]
        assert abs(e.out_of_sample.sharpe - e.in_sample.sharpe) > 1e-3


def test_equal_weight_test_window_sharpe_is_correct():
    prices = _regime_prices()
    rf = 0.02
    res = run_backtest(prices, SPLIT, rf=rf)

    # Independently recompute the equal-weight realized OOS Sharpe on the test
    # window using the documented formula.
    _, test = split_by_date(prices, SPLIT)
    test_rets = to_returns(test)
    n = test_rets.shape[1]
    w = np.full(n, 1.0 / n)
    mu_te = annualized_mean(test_rets)
    cov_te = annualized_cov(test_rets)
    ret = float(w @ mu_te)
    vol = float(np.sqrt(w @ cov_te @ w))
    expected_sharpe = (ret - rf) / vol

    got = res.entries["equal_weight"].out_of_sample
    assert got.sharpe == pytest.approx(expected_sharpe, rel=1e-9, abs=1e-9)
    # evaluate_weights should agree with the engine's stored result.
    direct = evaluate_weights(w, test_rets, rf=rf)
    assert direct.sharpe == pytest.approx(got.sharpe, abs=TOL)
