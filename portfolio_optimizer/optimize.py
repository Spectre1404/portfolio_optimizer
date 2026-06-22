"""Mean-variance optimization core (long-only, fully invested).

Pure and deterministic: functions take ``mu`` (annualized expected returns) and
``cov`` (annualized covariance) directly and return a :class:`PortfolioResult`.
No network, no global state. Weights align to the order of ``mu``/``cov``.

Constraints (CLAUDE.md hard rules): ``w >= 0`` and ``sum(w) == 1``. No shorting,
no leverage. The solver is SLSQP; non-convergence fails loudly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


class OptimizationError(RuntimeError):
    """Raised when the optimizer fails to converge (fail loudly, never silently)."""


@dataclass
class PortfolioResult:
    """A portfolio and its annualized stats.

    ``weights`` align to the order of ``mu``/``cov``. ``sharpe`` is
    ``(expected_return - rf) / volatility`` for the ``rf`` used to build it.
    """

    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe: float


def _as_arrays(mu, cov):
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    return mu, cov


def _stats(weights, mu, cov, rf=0.0):
    w = np.asarray(weights, dtype=float)
    ret = float(w @ mu)
    var = float(w @ cov @ w)
    vol = float(np.sqrt(max(var, 0.0)))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def _sum_to_one(n):
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def _solve_min_var(mu, cov, bounds, extra_constraints=()):
    """Minimize wᵀΣw subject to sum(w)=1, bounds, and any extra constraints."""
    n = len(mu)
    w0 = np.full(n, 1.0 / n)
    cons = (_sum_to_one(n), *extra_constraints)
    res = minimize(
        lambda w: w @ cov @ w,
        w0,
        method="SLSQP",
        bounds=[bounds] * n,
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not res.success:
        raise OptimizationError(f"SLSQP failed to converge: {res.message}")
    return res.x


def min_variance(mu, cov, bounds=(0.0, 1.0)) -> PortfolioResult:
    """Global minimum-variance portfolio: minimize wᵀΣw s.t. sum(w)=1, bounds."""
    mu, cov = _as_arrays(mu, cov)
    w = _solve_min_var(mu, cov, bounds)
    ret, vol, sharpe = _stats(w, mu, cov, rf=0.0)
    return PortfolioResult(weights=w, expected_return=ret, volatility=vol, sharpe=sharpe)


def max_sharpe(mu, cov, rf=0.0, bounds=(0.0, 1.0)) -> PortfolioResult:
    """Maximum-Sharpe (tangency) portfolio: maximize (wᵀμ − rf)/sqrt(wᵀΣw)."""
    mu, cov = _as_arrays(mu, cov)
    n = len(mu)
    w0 = np.full(n, 1.0 / n)

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(max(float(w @ cov @ w), 1e-18))
        return -(ret - rf) / vol

    res = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=[bounds] * n,
        constraints=(_sum_to_one(n),),
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not res.success:
        raise OptimizationError(f"SLSQP failed to converge: {res.message}")
    ret, vol, sharpe = _stats(res.x, mu, cov, rf=rf)
    return PortfolioResult(weights=res.x, expected_return=ret, volatility=vol, sharpe=sharpe)


def efficient_frontier(mu, cov, n_points=50, bounds=(0.0, 1.0)) -> list[PortfolioResult]:
    """Trace the frontier: for target returns swept low→high, minimize variance
    subject to wᵀμ = target, sum(w)=1, bounds. Returned ordered by target return.
    """
    mu, cov = _as_arrays(mu, cov)
    targets = np.linspace(float(mu.min()), float(mu.max()), n_points)
    out: list[PortfolioResult] = []
    for target in targets:
        hit_target = {"type": "eq", "fun": (lambda w, t=target: w @ mu - t)}
        w = _solve_min_var(mu, cov, bounds, extra_constraints=(hit_target,))
        ret, vol, sharpe = _stats(w, mu, cov, rf=0.0)
        out.append(PortfolioResult(weights=w, expected_return=ret, volatility=vol, sharpe=sharpe))
    return out
