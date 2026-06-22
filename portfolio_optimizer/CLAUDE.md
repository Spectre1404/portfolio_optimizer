# CLAUDE.md — operating rules for this repo

This is an **educational** mean-variance portfolio optimizer. Full design is in `ARCHITECTURE.md`
— read it before building. These are the rules you must hold to.

## What we're building
A real Markowitz optimizer (max-Sharpe, min-variance, efficient frontier) over cached historical
prices, with a guardrails layer and a self-contained HTML report.

## Hard rules
- Long-only, fully invested: weights `>= 0` and sum to `1`. No shorting, no leverage.
- **Guardrails are a real module and part of the output contract — not a disclaimer footer.**
  Every report MUST include: the "educational tool — not investment advice" notice; the
  equal-weight benchmark; an in-sample-only warning (or out-of-sample stats); and run metadata.
- Validate inputs before computing (insufficient data, NaNs, failed tickers, infeasible caps).
  Covariance must be symmetric PSD. Check optimizer convergence; fail loudly, never silently.
- Cache fetched prices to disk; runs must be reproducible and work offline on cached data.

## Non-goals — do not add these
- No ML / return prediction. No buy/sell "signals" or ratings. No future-performance claims.
- No shorting / leverage. No brokerage or live trading. Not a general backtesting framework.

## How we work
- **Test-first.** Run `pytest -q` after each change and keep it green. Seed tests are in `tests/`.
- Functions take `mu` (annualized returns) and `cov` (annualized covariance) directly and return
  `PortfolioResult`. Pure, deterministic, testable without network.
- Small commits with clear messages. Keep modules focused (data / stats / optimize / guardrails /
  report).
- **If you think the scope or the API should change, STOP and ask before doing it.**

## Repo hygiene
- `venv/` is currently committed (~214 MB) — gitignore it and `git rm -r --cached venv`.
- Add `scipy` and `pytest` to `requirements.txt` (missing today).