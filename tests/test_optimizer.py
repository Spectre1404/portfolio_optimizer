"""Seed tests for the optimizer core.

These FAIL until `portfolio_optimizer.optimize` (and `guardrails.equal_weight`) are
implemented — that is the point. They use small, hand-built mu/cov so they are
deterministic and offline (no yfinance), and they encode the properties the optimizer
must satisfy.

Run:  pytest -q
"""
import numpy as np
import pytest

# These imports fail until you build the modules (TDD seed).
from portfolio_optimizer.optimize import (
    PortfolioResult,
    min_variance,
    max_sharpe,
    efficient_frontier,
)
from portfolio_optimizer.guardrails import equal_weight

TOL = 1e-4


@pytest.fixture
def two_assets():
    # Asset 1: var 0.04 (vol 20%); Asset 2: var 0.01 (vol 10%); uncorrelated.
    mu = np.array([0.10, 0.15])  # annualized expected returns
    cov = np.array([[0.04, 0.00],
                    [0.00, 0.01]])
    return mu, cov


@pytest.fixture
def four_assets():
    mu = np.array([0.08, 0.12, 0.15, 0.10])
    vols = np.array([0.15, 0.20, 0.25, 0.18])
    corr = np.array([
        [1.0, 0.2, 0.1, 0.0],
        [0.2, 1.0, 0.3, 0.1],
        [0.1, 0.3, 1.0, 0.2],
        [0.0, 0.1, 0.2, 1.0],
    ])
    cov = np.outer(vols, vols) * corr
    return mu, cov


def _feasible(w, lower=0.0, upper=1.0):
    w = np.asarray(w, dtype=float)
    return (
        abs(w.sum() - 1.0) < TOL
        and bool((w >= lower - TOL).all())
        and bool((w <= upper + TOL).all())
    )


def test_min_variance_weights_are_feasible(four_assets):
    mu, cov = four_assets
    res = min_variance(mu, cov)
    assert isinstance(res, PortfolioResult)
    assert _feasible(res.weights)


def test_max_sharpe_weights_are_feasible(four_assets):
    mu, cov = four_assets
    res = max_sharpe(mu, cov, rf=0.02)
    assert _feasible(res.weights)


def test_two_asset_min_variance_matches_closed_form(two_assets):
    # Long-only constraint is inactive here, so the closed form applies:
    #   w1 = (s2^2 - s12) / (s1^2 + s2^2 - 2*s12)
    mu, cov = two_assets
    res = min_variance(mu, cov)
    w1 = (cov[1, 1] - cov[0, 1]) / (cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    assert res.weights[0] == pytest.approx(w1, abs=1e-3)        # 0.20
    assert res.weights[1] == pytest.approx(1 - w1, abs=1e-3)    # 0.80


def test_min_variance_is_actually_minimal(four_assets):
    # Min-variance vol must be <= equal-weight and <= many random feasible portfolios.
    mu, cov = four_assets
    mv = min_variance(mu, cov)
    ew = equal_weight(mu, cov, rf=0.0)
    assert mv.volatility <= ew.volatility + TOL

    rng = np.random.default_rng(0)
    for _ in range(200):
        w = rng.random(len(mu))
        w /= w.sum()
        vol = float(np.sqrt(w @ cov @ w))
        assert mv.volatility <= vol + TOL


def test_max_sharpe_beats_equal_weight_in_sample(four_assets):
    mu, cov = four_assets
    rf = 0.02
    ms = max_sharpe(mu, cov, rf=rf)
    ew = equal_weight(mu, cov, rf=rf)
    assert ms.sharpe >= ew.sharpe - TOL


def test_sharpe_field_is_internally_consistent(four_assets):
    mu, cov = four_assets
    rf = 0.02
    res = max_sharpe(mu, cov, rf=rf)
    expected = (res.expected_return - rf) / res.volatility
    assert res.sharpe == pytest.approx(expected, rel=1e-3)


def test_efficient_frontier_shape_and_feasibility(four_assets):
    mu, cov = four_assets
    ef = efficient_frontier(mu, cov, n_points=25)
    assert len(ef) == 25
    mv_vol = min_variance(mu, cov).volatility
    for point in ef:
        assert _feasible(point.weights)
        # nothing on the frontier can be less risky than the global min-variance portfolio
        assert point.volatility >= mv_vol - TOL


def test_efficient_frontier_returns_are_monotonic(four_assets):
    mu, cov = four_assets
    ef = efficient_frontier(mu, cov, n_points=25)
    rets = [p.expected_return for p in ef]
    assert rets == sorted(rets)  # swept from low to high target return