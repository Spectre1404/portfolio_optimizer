# Portfolio Optimizer

An **educational** mean-variance (Markowitz) portfolio optimizer. Given a set of
tickers and a date window, it estimates annualized expected returns and risk from
cached historical prices, finds the **max-Sharpe** and **min-variance** portfolios,
traces the **efficient frontier**, and produces a self-contained HTML report that
compares the result honestly against a plain equal-weight (1/N) benchmark.

> **Not investment advice.** The report describes what *was* mean-variance-optimal
> on historical data under stated assumptions — not what to buy, and not a
> prediction of future performance. See `portfolio_optimizer/CLAUDE.md` and
> `portfolio_optimizer/ARCHITECTURE.md`.

## Design

- **Long-only, fully invested:** weights `>= 0` and sum to `1`. No shorting, no leverage.
- **Guardrails are a first-class layer** (`guardrails.py`), not a footnote: input
  validation, constraint feasibility, PSD covariance, equal-weight benchmark,
  concentration warnings, and no-look-ahead checks.
- **Reproducible / offline:** fetched prices are cached to disk so re-runs need no network.

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python -m portfolio_optimizer \
    --tickers AAPL MSFT GOOGL BLK \
    --start 2021-01-01 --end 2024-01-01 \
    --rf 0.02 --out report.html
```

Open `report.html` by double-clicking it — the file is fully self-contained
(inline CSS, the frontier chart embedded as a base64 PNG, no server).

## Test

```bash
pytest -q
```

## Modules

| Module | Responsibility |
|--------|----------------|
| `data.py` | load + cache prices (yfinance), convert to returns |
| `stats.py` | annualized expected returns and covariance |
| `optimize.py` | `min_variance`, `max_sharpe`, `efficient_frontier` (scipy SLSQP) |
| `guardrails.py` | validation, feasibility, PSD, benchmark, concentration, no-look-ahead |
| `report.py` | self-contained HTML report (the output contract) |
| `__main__.py` | CLI wiring the pipeline together |
