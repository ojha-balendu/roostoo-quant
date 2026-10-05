"""The configured portfolio over the most recent N days, as one fresh $1M run, with a chart.

    python research/09_recent.py                         # last 30 days, config/competition.yaml
    python research/09_recent.py --days 14
    python research/09_recent.py --every-days 3 --threshold 0.04     # try lower-turnover settings

Plots equity (portfolio, ensemble members, portfolio with zero fees, holding BTC), drawdown and
exposure, prints the metrics, and saves reports/final/recent_<N>d.png (opened automatically on macOS).
Warm-up history before the start date is used for signals only; the run itself starts flat.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.backtest.engine import run_backtest
from quantbot.config import load_config
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe
from quantbot.risk import clip_weights_frame
from quantbot.strategy.ensemble import Ensemble
from quantbot.strategy.registry import get_strategy

BTC = "BTC/USD"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/competition.yaml")
    ap.add_argument("--days", type=float, default=30)
    ap.add_argument("--every-days", type=int, default=1, help="rebalance every N days (at the configured UTC hour)")
    ap.add_argument("--threshold", type=float, default=None, help="override execution.rebalance_threshold")
    ap.add_argument("--no-open", action="store_true", help="don't open the chart")
    args = ap.parse_args()

    cfg = load_config(args.config)
    strategy = get_strategy(cfg.strategy.name, cfg.strategy.params)
    threshold = cfg.execution.rebalance_threshold if args.threshold is None else args.threshold
    universe = load_universe(cfg.data.universe_path)
    probe = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", {BTC: universe[BTC]})
    if probe.empty:
        raise SystemExit("no hourly data: run  quantbot download")
    end = probe.index.max()
    start = end - pd.Timedelta(days=args.days)
    warm = start - pd.Timedelta(hours=strategy.warmup_bars * 3)
    closes = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", universe,
                        start=str(warm.date())).ffill(limit=3)

    hours = cfg.live.rebalance_utc_hours or [0]
    closes_at = closes.index + closes.index.to_series().diff().median()
    day_no = (closes_at.normalize() - pd.Timestamp("1970-01-01", tz="UTC")).days
    trade_on = np.asarray((closes_at.minute == 0) & closes_at.hour.isin(hours) & (day_no % args.every_days == 0))
    test = closes.index >= start
    trade_on[np.flatnonzero(test)[0]] = True  # the competition starts invested: trade on the first bar
    c = closes.loc[test]
    bt = dict(initial_capital=cfg.backtest.initial_capital, long_fee=cfg.backtest.long_fee,
              spread_bps=cfg.backtest.spread_bps, delay_bars=0, min_trade_usd=cfg.execution.min_trade_usd,
              min_close_usd=cfg.execution.min_close_usd, threshold=threshold, cash_buffer=cfg.execution.cash_buffer,
              trade_on=trade_on[test])

    weights = {"portfolio": clip_weights_frame(strategy.generate_weights(closes), cfg.risk)}
    if isinstance(strategy, Ensemble):
        for i, (m, _) in enumerate(strategy.members, 1):
            tag = "" if not hasattr(m, "long_in_bear") else (" (regime)" if m.long_in_bear == 0 else " (no regime)")
            weights[f"member {i}{tag}"] = clip_weights_frame(m.generate_weights(closes), cfg.risk)
    weights["hold BTC"] = pd.DataFrame(0.0, index=closes.index, columns=closes.columns).assign(**{BTC: 0.95})

    runs = {name: run_backtest(c, w.loc[test], fees=cfg.fees, **bt) for name, w in weights.items()}
    zero = replace(cfg.fees, taker=0.0, maker=0.0, short_open=0.0, short_close=0.0)
    runs["portfolio, zero fees"] = run_backtest(c, weights["portfolio"].loc[test], fees=zero,
                                                **{**bt, "spread_bps": 0.0})

    rows = {}
    for name, r in runs.items():
        m = r.metrics
        rows[name] = {"return": f"{m['total_return']:+.2%}", "max DD": f"{m['max_drawdown']:.2%}",
                      "Sharpe": f"{m['sharpe']:.2f}", "Sortino": f"{m['sortino']:.2f}",
                      "Calmar": f"{m['calmar']:.1f}", "fees $": f"{m['fees_paid']:,.0f}", "trades": m["n_trades"]}
    pd.set_option("display.width", 200)
    sched = "daily" if args.every_days == 1 else f"every {args.every_days} days"
    print(f"{c.index.min():%Y-%m-%d %H:%M} -> {c.index.max():%Y-%m-%d %H:%M} UTC, ${cfg.backtest.initial_capital:,.0f}, "
          f"rebalance {sched} at {hours} UTC, threshold {threshold:.1%}")
    print(pd.DataFrame(rows).T.to_string())
    print("(Sharpe/Sortino/Calmar are annualised from a short period: compare them, don't read them literally.)")

    out = Path(cfg.backtest.report_dir) / "final"
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"recent_{args.days:g}d.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib to get the chart")
        return
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    styles = {"portfolio": dict(lw=2.4, color="tab:blue"), "portfolio, zero fees": dict(lw=1.4, ls="--", color="tab:blue"),
              "hold BTC": dict(lw=1.6, color="tab:orange")}
    for name, r in runs.items():
        eq = r.equity / r.equity.iloc[0] - 1
        st = styles.get(name, dict(lw=1.0, alpha=0.75))
        axes[0].plot(eq.index, eq * 100, label=f"{name} ({eq.iloc[-1]:+.1%})", **st)
        if name in ("portfolio", "hold BTC"):
            dd = r.equity / r.equity.cummax() - 1
            axes[1].plot(dd.index, dd * 100, label=name, **{k: v for k, v in st.items() if k != "lw"}, lw=1.4)
    w = weights["portfolio"].loc[test]
    axes[2].plot(w.index, w.abs().sum(axis=1), label="target gross exposure", color="tab:blue")
    axes[2].plot(w.index, w.sum(axis=1), label="target net exposure (long - short)", color="tab:green")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[0].set_title(f"Competition portfolio, last {args.days:g} days (fresh ${cfg.backtest.initial_capital:,.0f}, "
                      f"rebalance {sched}, threshold {threshold:.0%})")
    axes[0].set_ylabel("return %")
    axes[1].set_ylabel("drawdown %")
    axes[2].set_ylabel("exposure")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].axhline(0, color="k", lw=0.5)
    locator = mdates.AutoDateLocator(maxticks=10)
    axes[2].xaxis.set_major_locator(locator)
    axes[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    print(f"\nchart saved to {png}")
    if sys.platform == "darwin" and not args.no_open:
        subprocess.run(["open", str(png)], check=False)


if __name__ == "__main__":
    main()
