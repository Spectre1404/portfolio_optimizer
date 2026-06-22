"""CLI: fetch prices, optimize, run guardrails, write the HTML report.

    python -m portfolio_optimizer --tickers AAPL MSFT GOOGL BLK \
        --start 2021-01-01 --end 2024-01-01 --rf 0.02 --out report.html

Prices are cached under --cache-dir so repeat runs are reproducible and offline.
"""
from __future__ import annotations

import argparse
import datetime as _dt

from .data import load_prices, to_returns, CACHE_DIR_DEFAULT
from .stats import annualized_mean, annualized_cov, TRADING_DAYS
from .guardrails import (
    check_constraints_feasible,
    ensure_psd,
    equal_weight,
    assess_portfolio,
)
from .optimize import min_variance, max_sharpe, efficient_frontier
from .report import build_report


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="portfolio_optimizer", description=__doc__)
    p.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols.")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD.")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD.")
    p.add_argument("--rf", type=float, default=0.0, help="Annual risk-free rate.")
    p.add_argument("--lower", type=float, default=0.0, help="Per-asset min weight.")
    p.add_argument("--upper", type=float, default=1.0, help="Per-asset max weight (cap).")
    p.add_argument("--n-points", type=int, default=50, help="Frontier resolution.")
    p.add_argument("--max-weight-warn", type=float, default=0.40,
                   help="Concentration warning threshold.")
    p.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    p.add_argument("--out", default="report.html", help="Output HTML path.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    tickers = list(args.tickers)
    bounds = (args.lower, args.upper)

    # G5: fail before computing if the caps can't sum to 1.
    check_constraints_feasible(len(tickers), args.lower, args.upper)

    # Load (G3 validates), then estimate annualized mu/cov.
    prices = load_prices(tickers, args.start, args.end, cache_dir=args.cache_dir)
    returns = to_returns(prices)
    mu = annualized_mean(returns)
    cov = ensure_psd(annualized_cov(returns))  # G4

    # Optimize (G5 bounds threaded through).
    ms = max_sharpe(mu, cov, rf=args.rf, bounds=bounds)
    mv = min_variance(mu, cov, bounds=bounds)
    frontier = efficient_frontier(mu, cov, n_points=args.n_points, bounds=bounds)
    ew = equal_weight(mu, cov, rf=args.rf)  # G2 benchmark

    warnings = assess_portfolio(ms, max_weight_warn=args.max_weight_warn)  # G2

    metadata = {  # G6
        "Tickers": ", ".join(tickers),
        "Date window": f"{args.start} -> {args.end}",
        "Trading days observed": len(returns),
        "Data source": "yfinance (adjusted close), cached to disk",
        "Risk-free rate": f"{args.rf:.2%}",
        "Annualization factor": TRADING_DAYS,
        "Generated": _dt.datetime.now().isoformat(timespec="seconds"),
    }

    build_report(
        tickers=tickers, mu=mu, cov=cov, rf=args.rf,
        max_sharpe_result=ms, min_variance_result=mv, equal_weight_result=ew,
        frontier=frontier, warnings=warnings, metadata=metadata,
        out_of_sample=None,  # in-sample only; backtest is a stretch goal
        output_path=args.out,
    )
    print(f"Wrote report to {args.out}")
    print(f"Max-Sharpe Sharpe={ms.sharpe:.3f} vs equal-weight Sharpe={ew.sharpe:.3f}")
    for w in warnings:
        print(f"  warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
