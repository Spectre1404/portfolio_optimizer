# Portfolio Optimizer — Architecture
 
Co-architected spec for the 3-day rebuild. Goal: turn the Week-1 learning skeleton
(hardcoded weights, no optimization) into a real, tested mean-variance optimizer with a
self-contained HTML dashboard — with **guardrails as a first-class layer, not a footnote**.
 
## 1. What this is (and the honest one-liner)
 
An **educational / analytical** mean-variance (Markowitz) portfolio optimizer. Given a set of
tickers and a date window, it estimates expected returns and risk from historical prices, finds
the max-Sharpe and min-variance portfolios, traces the efficient frontier, and reports the
result honestly against a plain equal-weight benchmark.
 
Honest one-liner, stated in the product itself:
 
> "On this historical data, under these assumptions, these were the mean-variance-optimal
> weights." NOT "buy these." It is a tool for reasoning about risk/return trade-offs, not
> investment advice.
 
## 2. Guardrails (first-class layer)
 
Mean-variance optimization is fragile: it is very sensitive to the expected-return estimates,
in-sample-optimal routinely underperforms out of sample, and 1/N (equal weight) is a famously
hard benchmark to beat. The guardrails layer exists so the tool can't quietly mislead. It is a
real module (`guardrails.py`) and a real part of the output contract.
 
- **G1 — Not-advice framing, enforced in output.** Every report renders a prominent
  "Educational tool — not investment advice" notice. The tool never emits "buy/sell/should"; it
  describes what was optimal on historical data under stated assumptions.
- **G2 — Honesty about estimation error / overfitting.**
  - Always compute and show the **equal-weight (1/N) benchmark** beside the "optimal" portfolio.
  - **In-sample vs out-of-sample:** if a backtest was run, show out-of-sample stats; if not,
    render an explicit "in-sample only — not a performance prediction" warning.
  - **Concentration check:** if any weight exceeds a threshold (default 40%), surface a warning.
    Optional per-asset cap available to keep the optimizer from dumping everything into one name.
- **G3 — Input validation.** Reject insufficient history, all-/mostly-NaN columns, failed
  tickers, and non-overlapping date ranges with clear errors (`GuardrailError`) — never compute
  on garbage.
- **G4 — Numerical integrity.** Covariance must be symmetric PSD (`ensure_psd`); guard against
  singular covariance; check optimizer convergence and fail loudly if SLSQP doesn't converge.
- **G5 — Constraint feasibility.** Validate that requested bounds are satisfiable (e.g. 3 assets
  capped at 20% each can't sum to 1) before optimizing.
- **G6 — Reproducibility / auditability.** Cache the exact prices used and record run metadata
  (tickers, date window, data source, risk-free rate, annualization factor, timestamp) in the
  report, so any result is reproducible.
- **G7 — No look-ahead (backtest).** Training window must strictly precede the test window;
  weights are estimated on training data only. Enforced by `no_lookahead`.
## 3. Modules & public API
 
```
portfolio_optimizer/
  __init__.py
  data.py        load_prices(tickers, start, end, cache_dir="data_cache") -> DataFrame  # cached
                 to_returns(prices) -> DataFrame   # daily simple returns, NaNs dropped
  stats.py       TRADING_DAYS = 252
                 annualized_mean(returns) -> ndarray
                 annualized_cov(returns) -> ndarray
  optimize.py    @dataclass PortfolioResult: weights, expected_return, volatility, sharpe
                 min_variance(mu, cov, bounds=(0.0, 1.0)) -> PortfolioResult
                 max_sharpe(mu, cov, rf=0.0, bounds=(0.0, 1.0)) -> PortfolioResult
                 efficient_frontier(mu, cov, n_points=50, bounds=(0.0, 1.0))
                     -> list[PortfolioResult]   # ordered by increasing target return
  guardrails.py  class GuardrailError(Exception)
                 validate_prices(prices, min_rows=60) -> None
                 check_constraints_feasible(n_assets, lower, upper) -> None
                 ensure_psd(cov) -> ndarray
                 equal_weight(mu, cov, rf=0.0) -> PortfolioResult
                 assess_portfolio(result, max_weight_warn=0.40) -> list[str]
                 no_lookahead(train_index, test_index) -> None
  backtest.py    (stretch) train/test split + out-of-sample evaluation vs equal weight
  report.py      build_report(...) -> str (self-contained HTML); writes to disk
  __main__.py    CLI: python -m portfolio_optimizer --tickers AAPL MSFT ... --start ...
```
 
`mu` = array-like of annualized expected returns; `cov` = 2-D array-like annualized covariance.
Functions accept numpy or pandas; **weights align to the order of mu/cov**.
 
## 4. Data flow
 
```
tickers + window
  -> data.load_prices (yfinance, cached to disk)            [G3 validate_prices, G6 metadata]
  -> data.to_returns -> stats.annualized_mean / _cov        [G4 ensure_psd]
  -> optimize.{min_variance, max_sharpe, efficient_frontier}[G5 feasibility]
  -> guardrails.{equal_weight, assess_portfolio}            [G2 benchmark + warnings]
  -> report.build_report (self-contained HTML)              [G1 notice, G6 metadata, warnings]
(optional) backtest.train_test -> out-of-sample stats       [G7 no_lookahead] -> into report
```
 
## 5. The optimization (brief)
 
Long-only, fully invested: `w >= 0`, `sum(w) = 1`.
 
- **Min variance:** minimize `wᵀΣw` s.t. `sum(w)=1`, bounds.
- **Max Sharpe:** maximize `(wᵀμ − rf) / sqrt(wᵀΣw)` (minimize the negative) s.t. `sum(w)=1`, bounds.
- **Efficient frontier:** for target returns swept from min to max feasible, minimize `wᵀΣw`
  s.t. `wᵀμ = target`, `sum(w)=1`, bounds.
Solver: `scipy.optimize.minimize(method="SLSQP")`. Annualize: `μ = mean_daily·252`, `Σ = cov_daily·252`.
 
## 6. Output contract (the HTML report must always include)
 
1. The not-advice notice (G1), prominent.
2. Run metadata for reproducibility (G6).
3. Efficient-frontier chart: random-portfolio cloud + frontier curve + max-Sharpe & min-var markers.
4. Recommended weights (default headline objective: **max-Sharpe**) as a bar/table.
5. Key stats: expected return, volatility, Sharpe.
6. Equal-weight benchmark, side-by-side (G2).
7. In-sample-only warning, or out-of-sample stats if the backtest ran (G2).
8. Any guardrail warnings (concentration, short window, instability) (G2).
Self-contained: inline CSS/JS, no server, opens by double-click. (Charting: a small JS lib via
CDN, or pre-rendered SVG / matplotlib embedded as base64 — your call; self-contained is the rule.)
 
## 7. Scope
 
- **Core (must land):** data+cache, stats, optimizer (3 functions), guardrails layer, HTML
  report, repo cleanup. Green test suite.
- **Stretch (only if Core is done):** out-of-sample backtest (highest-value add — it's what makes
  G2 real); CLI flags for per-asset caps; covariance shrinkage (Ledoit–Wolf) or a risk-parity
  objective.
- **Non-goals (scope guardrails — do not drift here):** no ML / return prediction; no buy/sell
  "signals" or ratings; no shorting or leverage (long-only only); no future-performance claims;
  no brokerage / live trading; not a general backtesting framework.
## 8. Verification
 
- `pytest -q` after every change; keep it green. Seed tests live in `tests/`.
- Pure-math tests on synthetic mu/cov (deterministic, offline — no yfinance in tests).
- Property tests: feasibility, min-variance minimality, Sharpe consistency, frontier monotonicity.
- Closed-form check: 2-asset min-variance weight.
- Guardrail tests: validation, feasibility, PSD, concentration, no-look-ahead.
- Manual smoke test: run the CLI on ~5 liquid tickers; open the HTML; sanity-check the frontier.
## 9. Repo hygiene (do these in the session — they read well and the repo needs them)
 
- Add `.gitignore`: `venv/`, `__pycache__/`, `*.pyc`, `data_cache/`, `.pytest_cache/`, output PNGs.
- Remove the committed venv from tracking: `git rm -r --cached venv` (it's ~214 MB / ~9k files).
- Real `README.md` at repo root (currently empty): what it is, the not-advice framing, install, run.
- Rename `portfolio_optimizer/project notes ` (trailing space) -> `notes/project_notes.md`.
- One `requirements.txt`; add the missing **scipy** and **pytest** (current file has neither).