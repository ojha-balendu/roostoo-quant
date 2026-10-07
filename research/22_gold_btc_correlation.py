"""How do tokenized gold (PAXG) returns move with BTC returns? VWAP prices, several frequencies.

    python research/22_gold_btc_correlation.py
    python research/22_gold_btc_correlation.py --start 2025-01-01 --a PAXG/USD --b ETH/USD

VWAP of each bar = USD traded / coins traded (exact, from Binance klines); longer bars aggregate the same
way. Returns are log changes of VWAP. Reported per frequency: Pearson and Spearman correlation, beta of A
to B, whether B's previous bar predicts A's next one, and the close-to-close correlation for comparison;
then behaviour on B's worst and best days, a quarter-by-quarter breakdown, and a chart.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.analysis.vwap import resample_vwap
from quantbot.config import load_config
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe

OUT = Path("reports/research")
FREQS = {"1h": "1 hour", "4h": "4 hours", "1D": "1 day", "7D": "1 week"}


def stats(ra: pd.Series, rb: pd.Series) -> dict[str, float]:
    df = pd.concat([ra, rb], axis=1).dropna()
    if len(df) < 10:
        return {"n": len(df), "pearson": np.nan, "spearman": np.nan, "beta": np.nan, "lead": np.nan}
    a, b = df.iloc[:, 0], df.iloc[:, 1]
    lead = pd.concat([ra, rb.shift(1)], axis=1).dropna()
    return {"n": len(df), "pearson": a.corr(b), "spearman": a.rank().corr(b.rank()),  # Spearman without scipy
            "beta": a.cov(b) / b.var(), "lead": lead.iloc[:, 0].corr(lead.iloc[:, 1])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="PAXG/USD", help="first asset (default tokenized gold)")
    ap.add_argument("--b", default="BTC/USD", help="second asset")
    ap.add_argument("--start", default=None, help="default: data.start in the config")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    universe = load_universe(cfg.data.universe_path)
    missing = [p for p in (args.a, args.b) if p not in universe]
    if missing:
        raise SystemExit(f"{missing} not in {cfg.data.universe_path}")
    pair = {p: universe[p] for p in (args.a, args.b)}
    start = args.start or cfg.data.start
    qv = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", pair, start=start, field="quote_volume")
    vol = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", pair, start=start, field="volume")
    close = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", pair, start=start, field="close")
    if qv.empty or args.a not in qv or args.b not in qv:
        raise SystemExit("no hourly data with volume for both assets: run  quantbot download")
    common = qv[[args.a, args.b]].dropna().index
    qv, vol, close = qv.loc[common], vol.loc[common], close.loc[common]
    a, b = args.a.split("/")[0], args.b.split("/")[0]
    zero = float((vol[args.a] <= 0).mean())
    print(f"{a} vs {b}, hourly klines {common.min():%Y-%m-%d} -> {common.max():%Y-%m-%d}; "
          f"{a} had no trades in {zero:.1%} of hours (VWAP carried forward there)\n")

    rows, daily = {}, None
    for rule, label in FREQS.items():
        vw = resample_vwap(qv, vol, rule).ffill()
        cl = close.resample(rule).last() if rule != "1h" else close
        r = np.log(vw).diff()
        rc = np.log(cl).diff()
        s = stats(r[args.a], r[args.b])
        sc = stats(rc[args.a], rc[args.b])
        s["close_pearson"], s["lead"] = sc["pearson"], sc["lead"]    # lead from closes: VWAP bars overlap
        rows[label] = s
        if rule == "1D":
            daily = r
    t = pd.DataFrame(rows).T
    pd.set_option("display.width", 200)
    print(f"Correlation of {a} and {b} VWAP returns:")
    print(pd.DataFrame({
        "observations": t["n"].astype(int), "correlation": t["pearson"].map("{:+.3f}".format),
        "rank correlation": t["spearman"].map("{:+.3f}".format),
        f"beta of {a} to {b}": t["beta"].map("{:+.3f}".format),
        f"{b} last bar vs {a} next bar*": t["lead"].map("{:+.3f}".format),
        "close-to-close correlation": t["close_pearson"].map("{:+.3f}".format)}).to_string())
    print(f"beta = how much {a} moves per 1% move in {b}. * measured on closing prices: consecutive VWAPs overlap in\n"
          f"time, which alone makes VWAP returns look slightly delayed. A clearly non-zero value would mean {a} reacts\n"
          f"to {b} with a delay.")

    d = daily[[args.a, args.b]].dropna()
    lo, hi = d[args.b].quantile(0.10), d[args.b].quantile(0.90)
    worst, best = d[d[args.b] <= lo], d[d[args.b] >= hi]
    print(f"\nWhen {b} moves a lot (daily VWAP returns):")
    print(f"  {b}'s worst 10% of days ({len(worst)} days, {b} {worst[args.b].mean() * 100:+.2f}% on average): "
          f"{a} {worst[args.a].mean() * 100:+.2f}% on average, up on {(worst[args.a] > 0).mean():.0%} of them, "
          f"correlation {worst[args.a].corr(worst[args.b]):+.2f}")
    print(f"  {b}'s best 10% of days ({len(best)} days, {b} {best[args.b].mean() * 100:+.2f}% on average):  "
          f"{a} {best[args.a].mean() * 100:+.2f}% on average, up on {(best[args.a] > 0).mean():.0%} of them, "
          f"correlation {best[args.a].corr(best[args.b]):+.2f}")
    print(f"  all days: {a} {d[args.a].mean() * 100:+.3f}% on average, annual volatility "
          f"{a} {d[args.a].std() * np.sqrt(365):.0%}, {b} {d[args.b].std() * np.sqrt(365):.0%}")

    q = d.groupby(d.index.tz_localize(None).to_period("Q").rename("quarter"))
    by_q = pd.DataFrame({"days": q.size(), "correlation": q.apply(lambda x: x[args.a].corr(x[args.b])),
                         f"{a} return": q[args.a].sum().map(lambda v: f"{np.expm1(v):+.1%}"),
                         f"{b} return": q[args.b].sum().map(lambda v: f"{np.expm1(v):+.1%}")})
    by_q["correlation"] = by_q["correlation"].map("{:+.2f}".format)
    print("\nQuarter by quarter (daily VWAP returns):")
    print(by_q.to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / f"{a}_{b}_correlation.csv")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    vwd = resample_vwap(qv, vol, "1D").ffill().dropna()
    fig = plt.figure(figsize=(14, 9))
    ax1 = fig.add_subplot(2, 2, (1, 2))
    for col, color in ((args.a, "goldenrod"), (args.b, "tab:orange")):
        ax1.plot(vwd.index, vwd[col] / vwd[col].iloc[0], label=f"{col.split('/')[0]} (daily VWAP)", color=color)
    ax1.set_yscale("log")
    ax1.set_ylabel("price / start (log)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_title(f"{a} vs {b}")
    ax2 = fig.add_subplot(2, 2, 3)
    roll = d[args.a].rolling(30, min_periods=20).corr(d[args.b])
    ax2.plot(roll.index, roll, color="tab:blue")
    ax2.axhline(d[args.a].corr(d[args.b]), color="k", ls="--", lw=0.8, label="full-period")
    ax2.axhline(0, color="grey", lw=0.5)
    ax2.set_ylabel("30-day rolling correlation")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.scatter(d[args.b] * 100, d[args.a] * 100, s=6, alpha=0.4)
    beta = d[args.a].cov(d[args.b]) / d[args.b].var()
    xs = np.linspace(d[args.b].min(), d[args.b].max(), 50)
    ax3.plot(xs * 100, beta * xs * 100, color="red", lw=1, label=f"beta {beta:+.2f}")
    ax3.set_xlabel(f"{b} daily return %")
    ax3.set_ylabel(f"{a} daily return %")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    fig.tight_layout()
    png = OUT / f"{a}_{b}_correlation.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"\nchart: {png}")
    if sys.platform == "darwin" and not args.no_open:
        subprocess.run(["open", str(png)], check=False)


if __name__ == "__main__":
    main()
