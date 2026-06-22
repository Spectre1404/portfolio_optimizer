"""Guardrails — a first-class layer, not a disclaimer footer (see ARCHITECTURE.md §2).

These checks exist so the tool can't quietly mislead: validate inputs (G3), check
constraint feasibility (G5) and numerical integrity (G4), provide the equal-weight
benchmark and concentration warnings (G2), and forbid look-ahead in backtests (G7).
Failures raise :class:`GuardrailError` loudly rather than computing on garbage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .optimize import PortfolioResult


class GuardrailError(Exception):
    """Raised when an input or constraint would make a result misleading or invalid."""


# --- G3: input validation -------------------------------------------------

def validate_prices(prices, min_rows: int = 60) -> None:
    """Reject price data we shouldn't optimize on: too little history, or columns
    that are entirely NaN (a failed ticker). Raises :class:`GuardrailError`.
    """
    df = pd.DataFrame(prices)
    if len(df) < min_rows:
        raise GuardrailError(
            f"Insufficient history: {len(df)} rows < required min_rows={min_rows}."
        )
    all_nan = df.columns[df.isna().all(axis=0)].tolist()
    if all_nan:
        raise GuardrailError(f"Column(s) entirely NaN (failed ticker?): {all_nan}.")


# --- G5: constraint feasibility -------------------------------------------

def check_constraints_feasible(n_assets: int, lower: float, upper: float) -> None:
    """Ensure per-asset bounds can sum to 1 across ``n_assets``. E.g. 3 assets
    capped at 20% each can never sum to 100% — fail before optimizing.
    """
    if lower > upper:
        raise GuardrailError(f"lower bound {lower} exceeds upper bound {upper}.")
    if n_assets * upper < 1.0 - 1e-9:
        raise GuardrailError(
            f"Infeasible caps: {n_assets} assets x upper={upper} = "
            f"{n_assets * upper:.3f} < 1.0; weights cannot sum to 1."
        )
    if n_assets * lower > 1.0 + 1e-9:
        raise GuardrailError(
            f"Infeasible floors: {n_assets} assets x lower={lower} = "
            f"{n_assets * lower:.3f} > 1.0; weights cannot sum to 1."
        )


# --- G4: numerical integrity ----------------------------------------------

def ensure_psd(cov, tol: float = 1e-8):
    """Verify ``cov`` is symmetric positive semi-definite. Returns it unchanged
    if valid; raises :class:`GuardrailError` otherwise.
    """
    c = np.asarray(cov, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise GuardrailError(f"Covariance must be square 2-D; got shape {c.shape}.")
    if not np.allclose(c, c.T, atol=tol):
        raise GuardrailError("Covariance matrix is not symmetric.")
    eigvals = np.linalg.eigvalsh(c)
    if float(eigvals.min()) < -tol:
        raise GuardrailError(
            f"Covariance is not PSD: minimum eigenvalue {eigvals.min():.3e} < 0."
        )
    return c


# --- G2: benchmark + concentration ----------------------------------------

def equal_weight(mu, cov, rf: float = 0.0) -> PortfolioResult:
    """The 1/N benchmark — a famously hard portfolio to beat. Every report shows
    this beside the 'optimal' result so the optimizer can't flatter itself.
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(mu)
    w = np.full(n, 1.0 / n)
    ret = float(w @ mu)
    vol = float(np.sqrt(max(float(w @ cov @ w), 0.0)))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return PortfolioResult(weights=w, expected_return=ret, volatility=vol, sharpe=sharpe)


def assess_portfolio(result: PortfolioResult, max_weight_warn: float = 0.40) -> list[str]:
    """Return human-readable warnings about a result (currently: concentration).
    Empty list means nothing flagged.
    """
    warnings: list[str] = []
    w = np.asarray(result.weights, dtype=float)
    heavy = np.where(w > max_weight_warn)[0]
    if heavy.size:
        worst = float(w.max())
        warnings.append(
            f"Concentration: {heavy.size} asset(s) exceed {max_weight_warn:.0%} "
            f"(max weight {worst:.0%}). Optimal weights may be unstable / overfit."
        )
    return warnings


# --- G7: no look-ahead in the backtest ------------------------------------

def no_lookahead(train_index, test_index) -> None:
    """Enforce that the training window strictly precedes the test window.
    Any overlap means weights were fit on data they're later 'tested' on.
    """
    train = pd.DatetimeIndex(train_index)
    test = pd.DatetimeIndex(test_index)
    if len(train) == 0 or len(test) == 0:
        raise GuardrailError("Both train and test windows must be non-empty.")
    if train.max() >= test.min():
        raise GuardrailError(
            f"Look-ahead: train ends {train.max().date()} but test starts "
            f"{test.min().date()}; train must strictly precede test."
        )
