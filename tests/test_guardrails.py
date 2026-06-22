import numpy as np
import pandas as pd
import pytest
 
from portfolio_optimizer.guardrails import (
    GuardrailError,
    validate_prices,
    check_constraints_feasible,
    ensure_psd,
    equal_weight,
    assess_portfolio,
    no_lookahead,
)
from portfolio_optimizer.optimize import PortfolioResult, min_variance
 
 
def _prices(rows=120, cols=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=rows, freq="B")
    data = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(rows, cols)), axis=0))
    return pd.DataFrame(data, index=idx, columns=[f"A{i}" for i in range(cols)])
 
 
# --- input validation (G3) ---
 
def test_validate_prices_accepts_clean_data():
    validate_prices(_prices())  # should not raise
 
 
def test_validate_prices_rejects_too_few_rows():
    with pytest.raises(GuardrailError):
        validate_prices(_prices(rows=10), min_rows=60)
 
 
def test_validate_prices_rejects_all_nan_column():
    p = _prices()
    p["A1"] = np.nan
    with pytest.raises(GuardrailError):
        validate_prices(p)
 
 
# --- constraint feasibility (G5) ---
 
def test_infeasible_caps_raise():
    # 3 assets, max 20% each -> can't sum to 1. Must fail loudly, not silently.
    with pytest.raises(GuardrailError):
        check_constraints_feasible(n_assets=3, lower=0.0, upper=0.20)
 
 
def test_feasible_caps_pass():
    check_constraints_feasible(n_assets=3, lower=0.0, upper=0.50)  # no raise
 
 
# --- numerical integrity (G4) ---
 
def test_ensure_psd_rejects_non_psd():
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1 -> not PSD
    with pytest.raises(GuardrailError):
        ensure_psd(bad)
 
 
def test_ensure_psd_accepts_valid_covariance():
    good = np.array([[0.04, 0.00], [0.00, 0.01]])
    out = ensure_psd(good)
    assert np.allclose(out, good)
 
 
# --- benchmark + concentration (G2) ---
 
def test_equal_weight_is_uniform():
    mu = np.array([0.10, 0.20, 0.15])
    cov = np.eye(3) * 0.04
    res = equal_weight(mu, cov, rf=0.0)
    assert res.weights == pytest.approx(np.full(3, 1 / 3), abs=1e-9)
 
 
def test_concentration_warning_fires():
    # A portfolio with 95% in one asset must trigger a concentration warning.
    concentrated = PortfolioResult(
        weights=np.array([0.95, 0.03, 0.02]),
        expected_return=0.0, volatility=0.0, sharpe=0.0,
    )
    warnings = assess_portfolio(concentrated, max_weight_warn=0.40)
    assert any("concentr" in w.lower() for w in warnings)
 
 
def test_diversified_result_no_concentration_warning():
    mu = np.array([0.10, 0.20, 0.15])
    cov = np.eye(3) * 0.04
    mv = min_variance(mu, cov)  # roughly balanced for identity-ish cov
    warnings = assess_portfolio(mv, max_weight_warn=0.90)
    assert not any("concentr" in w.lower() for w in warnings)
 
 
# --- no look-ahead in the backtest (G7) ---
 
def test_no_lookahead_rejects_overlap():
    train = pd.date_range("2023-01-01", "2023-06-30", freq="B")
    test_overlap = pd.date_range("2023-06-01", "2023-12-31", freq="B")
    with pytest.raises(GuardrailError):
        no_lookahead(train, test_overlap)
 
 
def test_no_lookahead_accepts_clean_split():
    train = pd.date_range("2023-01-01", "2023-06-30", freq="B")
    test_clean = pd.date_range("2023-07-01", "2023-12-31", freq="B")
    no_lookahead(train, test_clean)  # no raise