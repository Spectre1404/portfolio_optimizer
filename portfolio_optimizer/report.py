"""Render a self-contained HTML report (ARCHITECTURE.md §6, output contract).

Every report MUST carry: the not-advice notice (G1), run metadata (G6), the
efficient-frontier chart, the recommended (max-Sharpe) weights, key stats, the
equal-weight benchmark side-by-side (G2), an in-sample-only warning or
out-of-sample stats (G2), and any guardrail warnings (G2).

The chart is rendered with matplotlib and embedded as a base64 PNG so the file
is fully self-contained: inline CSS, no server, no network, opens by double-click.
"""
from __future__ import annotations

import base64
import html
import io

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: never tries to open a window
import matplotlib.pyplot as plt  # noqa: E402

from .optimize import PortfolioResult  # noqa: E402

NOT_ADVICE = (
    "Educational tool — not investment advice. These are the weights that were "
    "mean-variance-optimal on the historical data below, under the stated "
    "assumptions. This is not a recommendation to buy or sell, and not a "
    "prediction of future performance."
)


def _frontier_png(mu, cov, frontier, max_sharpe_res, min_var_res, seed=0) -> str:
    """Return a base64-encoded PNG: random-portfolio cloud + frontier curve +
    max-Sharpe and min-variance markers."""
    mu = np.asarray(mu, float)
    cov = np.asarray(cov, float)
    rng = np.random.default_rng(seed)
    n = len(mu)

    cloud = rng.random((2000, n))
    cloud /= cloud.sum(axis=1, keepdims=True)
    cloud_ret = cloud @ mu
    cloud_vol = np.sqrt(np.einsum("ij,jk,ik->i", cloud, cov, cloud))

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
    ax.scatter(cloud_vol, cloud_ret, s=6, c="#c9d4e3", alpha=0.5,
               label="Random portfolios")
    f_vol = [p.volatility for p in frontier]
    f_ret = [p.expected_return for p in frontier]
    ax.plot(f_vol, f_ret, color="#2b6cb0", lw=2, label="Efficient frontier")
    ax.scatter([max_sharpe_res.volatility], [max_sharpe_res.expected_return],
               marker="*", s=240, c="#dd6b20", edgecolor="white",
               zorder=5, label="Max Sharpe")
    ax.scatter([min_var_res.volatility], [min_var_res.expected_return],
               marker="o", s=90, c="#2f855a", edgecolor="white",
               zorder=5, label="Min variance")
    ax.set_xlabel("Volatility (annualized)")
    ax.set_ylabel("Expected return (annualized)")
    ax.set_title("Efficient frontier")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _weights_rows(tickers, result: PortfolioResult) -> str:
    cells = []
    for t, w in zip(tickers, np.asarray(result.weights, float)):
        cells.append(f"<tr><td>{html.escape(str(t))}</td><td>{w:.2%}</td></tr>")
    return "\n".join(cells)


def _stats_block(result: PortfolioResult) -> str:
    return (
        f"<ul class='stats'>"
        f"<li>Expected return <b>{result.expected_return:.2%}</b></li>"
        f"<li>Volatility <b>{result.volatility:.2%}</b></li>"
        f"<li>Sharpe <b>{result.sharpe:.2f}</b></li>"
        f"</ul>"
    )


def _backtest_block(bt) -> str:
    """Render an in-sample vs out-of-sample comparison from a BacktestResult.

    Duck-typed on ``bt`` (``.entries``, ``.train_span``/``.train_days``,
    ``.test_span``/``.test_days``) so report.py stays decoupled from backtest.py.
    """
    labels = {
        "max_sharpe": "Max Sharpe",
        "min_variance": "Min variance",
        "equal_weight": "Equal weight (1/N)",
    }
    rows = ""
    for key, label in labels.items():
        e = bt.entries[key]
        i, o = e.in_sample, e.out_of_sample
        rows += (
            f"<tr><td>{label}</td>"
            f"<td>{i.expected_return:.2%}</td><td>{i.volatility:.2%}</td><td>{i.sharpe:.2f}</td>"
            f"<td>{o.expected_return:.2%}</td><td>{o.volatility:.2%}</td><td>{o.sharpe:.2f}</td>"
            f"</tr>"
        )
    note = (
        "<div class='ok'><b>Out-of-sample backtest.</b> Weights were fit on the "
        f"TRAIN window ({bt.train_span[0]} &rarr; {bt.train_span[1]}, {bt.train_days} days) "
        "and evaluated <b>unchanged</b> on the TEST window "
        f"({bt.test_span[0]} &rarr; {bt.test_span[1]}, {bt.test_days} days). The "
        "out-of-sample columns are realized results &mdash; no re-optimization on test data.</div>"
    )
    table = (
        "<table><thead>"
        "<tr><th rowspan='2'>Portfolio</th>"
        "<th colspan='3'>In-sample (train)</th>"
        "<th colspan='3'>Out-of-sample (test)</th></tr>"
        "<tr><th>Return</th><th>Vol</th><th>Sharpe</th>"
        "<th>Return</th><th>Vol</th><th>Sharpe</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )
    return note + table


def build_report(
    tickers,
    mu,
    cov,
    rf,
    max_sharpe_result: PortfolioResult,
    min_variance_result: PortfolioResult,
    equal_weight_result: PortfolioResult,
    frontier,
    warnings,
    metadata: dict,
    out_of_sample: dict | None = None,
    backtest=None,
    output_path: str | None = None,
) -> str:
    """Assemble the report HTML (and write it to ``output_path`` if given)."""
    tickers = [str(t) for t in tickers]
    chart_b64 = _frontier_png(mu, cov, frontier, max_sharpe_result, min_variance_result)

    meta_rows = "\n".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in metadata.items()
    )

    if backtest is not None:
        sample_block = _backtest_block(backtest)
    elif out_of_sample:
        oos_rows = "".join(
            f"<li>{html.escape(str(k))}: <b>{html.escape(str(v))}</b></li>"
            for k, v in out_of_sample.items()
        )
        sample_block = (
            "<div class='ok'><b>Out-of-sample results.</b>"
            f"<ul class='stats'>{oos_rows}</ul></div>"
        )
    else:
        sample_block = (
            "<div class='warn'><b>In-sample only — not a performance prediction.</b> "
            "These stats are measured on the same history used to choose the weights; "
            "real out-of-sample results are typically worse.</div>"
        )

    if warnings:
        warn_block = "<div class='warn'><b>Guardrail warnings</b><ul>" + "".join(
            f"<li>{html.escape(str(w))}</li>" for w in warnings
        ) + "</ul></div>"
    else:
        warn_block = "<div class='ok'>No guardrail warnings.</div>"

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Portfolio Optimizer — Report</title>
<style>
  :root {{ --ink:#1a202c; --line:#e2e8f0; --accent:#2b6cb0; }}
  * {{ box-sizing: border-box; }}
  body {{ font: 15px/1.5 -apple-system, system-ui, sans-serif; color: var(--ink);
         max-width: 880px; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ font-size: 1.6rem; margin-bottom: .2rem; }}
  h2 {{ font-size: 1.15rem; margin-top: 2rem; border-bottom: 2px solid var(--line);
        padding-bottom: .3rem; }}
  .notice {{ background:#fffaf0; border:1px solid #f6ad55; border-left:5px solid #dd6b20;
             padding:.8rem 1rem; border-radius:6px; margin:1rem 0; }}
  .warn {{ background:#fffaf0; border-left:4px solid #dd6b20; padding:.6rem .9rem;
           border-radius:4px; margin:.8rem 0; }}
  .ok {{ background:#f0fff4; border-left:4px solid #2f855a; padding:.6rem .9rem;
         border-radius:4px; margin:.8rem 0; }}
  table {{ border-collapse: collapse; width: 100%; margin:.5rem 0; }}
  th, td {{ text-align: left; padding:.4rem .6rem; border-bottom:1px solid var(--line); }}
  .cols {{ display:flex; gap:2rem; flex-wrap:wrap; }}
  .cols > div {{ flex:1 1 320px; }}
  .stats {{ list-style:none; padding:0; }}
  .stats li {{ padding:.15rem 0; }}
  img {{ max-width:100%; height:auto; border:1px solid var(--line); border-radius:6px; }}
  .muted {{ color:#718096; font-size:.85rem; }}
</style></head><body>

<h1>Portfolio Optimizer — Report</h1>
<p class="muted">Mean-variance analysis of: {html.escape(", ".join(tickers))}</p>

<div class="notice">{html.escape(NOT_ADVICE)}</div>

<h2>Run metadata</h2>
<table><tbody>{meta_rows}</tbody></table>

<h2>Efficient frontier</h2>
<img alt="Efficient frontier chart" src="data:image/png;base64,{chart_b64}">

<h2>Recommended portfolio (max Sharpe)</h2>
<div class="cols">
  <div>
    <table><thead><tr><th>Asset</th><th>Weight</th></tr></thead>
    <tbody>{_weights_rows(tickers, max_sharpe_result)}</tbody></table>
  </div>
  <div>{_stats_block(max_sharpe_result)}</div>
</div>

<h2>Equal-weight benchmark (1/N)</h2>
<div class="cols">
  <div>
    <table><thead><tr><th>Asset</th><th>Weight</th></tr></thead>
    <tbody>{_weights_rows(tickers, equal_weight_result)}</tbody></table>
  </div>
  <div>{_stats_block(equal_weight_result)}</div>
</div>

<h2>Honesty checks</h2>
{sample_block}
{warn_block}

<p class="muted">Generated by portfolio_optimizer. Risk-free rate: {rf:.2%}.
Long-only, fully invested (weights &ge; 0 and sum to 1).</p>

</body></html>"""

    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(doc)
    return doc
