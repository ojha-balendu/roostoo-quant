"""Every strategy on the most recent N days (default 30), as one fresh $1M competition, with charts.

    quantbot download                      # first: bring the hourly data up to yesterday
    python research/19_last_30_days.py
    python research/19_last_30_days.py --days 14

Strategies: momentum sleeve variants A-G (research/18), the trend portfolio in config/trend_ensemble.yaml,
composite indicator signal v1 and v2 (research/14), and benchmarks (hold BTC, equal-weight basket,
never trade). All start flat with the same capital, pay the same fees, and use only data available at
each bar; history before the window is used only to warm up the signals.
Score = 0.4 x Sortino + 0.3 x Sharpe + 0.3 x Calmar (annualised, so short windows give large numbers).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.analysis.composite import COEFFS, affine_unit, composite
from quantbot.analysis.indicators import compute_named
from quantbot.analysis.xsection import eligibility, residual_returns
from quantbot.backtest.composite_bt import (
    CompositeConfig,
    calibrate,
    run_metrics,
    simulate,
    v1_weights,
    v2_weights,
)
from quantbot.backtest.momentum import momentum_inputs, sleeve_weights
from quantbot.config import load_config
from quantbot.data.loader import load_ohlcv
from quantbot.data.universe import load_universe
from quantbot.risk import clip_weights_frame
from quantbot.rl.sim import simple_returns
from quantbot.strategy.registry import get_strategy

BTC = "BTC/USD"
OUT = Path("reports/research")


def load_variants():
    import importlib.util

    path = Path(__file__).with_name("18_momentum_v2.py")
    spec = importlib.util.spec_from_file_location("momentum_v2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.variants()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=30)
    ap.add_argument("--warmup-days", type=float, default=150, help="history used only to warm up signals")
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--config", default="config/trend_ensemble.yaml", help="trend portfolio and fee settings")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    universe = load_universe(cfg.data.universe_path)
    probe = load_ohlcv(cfg.data.processed_dir, cfg.data.market, {BTC: universe[BTC]}, fields=("close",))["close"]
    if probe.empty:
        raise SystemExit("no hourly data: run  quantbot download")
    end_ts = probe.index.max()
    start_ts = end_ts - pd.Timedelta(days=args.days)
    load_from = str((start_ts - pd.Timedelta(days=args.warmup_days)).date())
    f = load_ohlcv(cfg.data.processed_dir, cfg.data.market, universe, start=load_from)
    close = f["close"].ffill(limit=3)
    resid, beta = residual_returns(close, BTC)
    coins = list(resid.columns)
    eligible = eligibility(close, f["quote_volume"])[coins]
    idx = close.index
    P = close[coins + [BTC]].to_numpy(dtype=float)
    rets = simple_returns(P)
    E, B = eligible.to_numpy(dtype=bool), beta[coins].to_numpy()
    a, end = int(idx.searchsorted(start_ts)), len(idx) - 1
    closes_at = idx + pd.Timedelta(hours=1)
    midnight = np.asarray((closes_at.hour == 0) & (closes_at.minute == 0))
    bc = CompositeConfig(capital=args.capital, fee=cfg.fees.taker, spread_bps=cfg.backtest.spread_bps)
    days = (idx[end] - idx[a]).total_seconds() / 86400
    print(f"{len(coins)} coins; window {idx[a]:%Y-%m-%d %H:%M} -> {idx[end]:%Y-%m-%d %H:%M} UTC ({days:.1f} days), "
          f"${args.capital:,.0f} each, fees {cfg.fees.taker:.2%} + spread {cfg.backtest.spread_bps:g} bps\n")

    strategies: dict[str, tuple[str, callable]] = {}

    # momentum sleeve variants (daily at 00:00 UTC; enter on the first bar)
    for key, (label, mc) in load_variants().items():
        x = {k: v.reindex(columns=coins).to_numpy(dtype=float) for k, v in momentum_inputs(close, resid, eligible, mc).items()}

        def mom(t, cur, x=x, mc=mc):
            return sleeve_weights(t, cur, x, B, E, rets, mc) if (midnight[t] or t == a) else cur
        strategies[f"momentum {key}"] = (f"momentum {key}: {label}", mom)

    # trend portfolio exactly as configured
    trend = clip_weights_frame(get_strategy(cfg.strategy.name, cfg.strategy.params).generate_weights(close), cfg.risk)
    TW = trend.reindex(columns=coins + [BTC]).fillna(0.0).to_numpy()
    hours = cfg.live.rebalance_utc_hours or [0]
    rebal = np.asarray((closes_at.minute == 0) & closes_at.hour.isin(hours))
    strategies["trend portfolio"] = ("trend portfolio (trend_ensemble.yaml)",
                                     lambda t, cur: TW[t] if (rebal[t] or t == a) else cur)

    # composite indicator signal v1 / v2
    named = {k: v[coins] for k, v in compute_named(f["open"], f["high"], f["low"], close, f["volume"],
                                                     f["quote_volume"]).items()}
    sp = affine_unit(composite(named, eligible, COEFFS))
    cal = calibrate(sp, resid, bc)
    S, A = sp.to_numpy(), cal["A"].to_numpy()
    strategies["composite v1"] = ("composite v1 (top/bottom 10% hourly)", lambda t, cur: v1_weights(S[t], E[t], B[t], bc))
    strategies["composite v2"] = ("composite v2 (profit > fee)", lambda t, cur: v2_weights(A[t], cur, E[t], B[t], bc))

    # benchmarks
    btc_t = np.zeros(P.shape[1])
    btc_t[-1] = bc.gross
    ew = np.zeros(P.shape[1])
    ok = E[a] & np.isfinite(P[a, :-1])
    ew[:-1][ok] = bc.gross / max(ok.sum(), 1)
    strategies["hold BTC"] = ("hold BTC", lambda t, cur: btc_t if t == a else cur)
    strategies["equal-weight"] = (f"equal-weight {int(ok.sum())} coins", lambda t, cur: ew if t == a else cur)
    strategies["never trade"] = ("never trade", lambda t, cur: np.zeros(P.shape[1]))

    rows, curves = {}, {}
    for label, decide in strategies.values():
        res = simulate(P, idx, decide, a, end, bc.sim_config())
        m = run_metrics(res, args.capital)
        rows[label] = {**m, "orders_day": res["trades"] / days}
        curves[label] = res["equity"]
    t = pd.DataFrame(rows).T.sort_values("score", ascending=False, na_position="last")
    def ratio(v: float) -> str:
        return f"{v:+.2f}" if np.isfinite(v) else "n/a"

    show = pd.DataFrame({
        "SCORE": t["score"].map(ratio), "return": t["return"].map("{:+.2%}".format), "Sortino": t["sortino"].map(ratio),
        "Sharpe": t["sharpe"].map(ratio), "Calmar": t["calmar"].map(ratio),
        "max DD": t["max_dd"].map("{:.2%}".format), "P&L before fees": t["gross_pnl"].map("{:+.2%}".format),
        "fees": t["fees_pct"].map("{:.2%}".format), "orders/day": t["orders_day"].map("{:.1f}".format),
        "avg exposure": t["avg_gross"].map("{:.2f}".format)})
    pd.set_option("display.width", 220)
    print(f"Last {days:.0f} days, sorted by SCORE = 0.4 Sortino + 0.3 Sharpe + 0.3 Calmar:")
    print(show.to_string())
    print("\nOne 30-day window is a single draw: a strategy can top this table by luck. Compare with the 14-day "
          "window statistics from research/17-18 before deciding anything.")
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / f"last_{args.days:g}_days.csv")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib for the charts")
        return
    order = list(t.index)
    cmap = plt.get_cmap("tab20")
    colors = {name: cmap(i % 20) for i, name in enumerate(order)}
    colors["hold BTC"], colors["never trade"] = "tab:orange", "black"
    fig, ax = plt.subplots(figsize=(14, 7))
    for name in order:
        eq = curves[name]
        ax.plot(eq.index, (eq / eq.iloc[0] - 1) * 100, color=colors[name], lw=2.2 if name == order[0] else 1.2,
                ls="--" if name in ("hold BTC", "never trade") else "-",
                label=f"{name}  (score {ratio(t.loc[name, 'score'])}, {t.loc[name, 'return']:+.1%})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("return %")
    ax.set_title(f"All strategies, last {days:.0f} days, ${args.capital:,.0f} each, fees included (legend sorted by score)")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.tight_layout()
    overview = OUT / f"last_{args.days:g}_days_all.png"
    fig.savefig(overview, dpi=130)
    plt.close(fig)

    names = [n for n in order if n != "never trade"]
    cols = 3
    rows_n = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows_n * 2, cols, figsize=(15, 3.6 * rows_n), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1] * rows_n})
    btc_eq = curves["hold BTC"]
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        top, bot = axes[2 * r, c], axes[2 * r + 1, c]
        eq = curves[name]
        top.plot(btc_eq.index, (btc_eq / btc_eq.iloc[0] - 1) * 100, color="tab:orange", lw=0.8, alpha=0.5, label="BTC")
        top.plot(eq.index, (eq / eq.iloc[0] - 1) * 100, color="tab:blue", lw=1.6)
        top.axhline(0, color="k", lw=0.4)
        m = t.loc[name]
        top.set_title(f"{name}\nscore {ratio(m['score'])} | ret {m['return']:+.1%} | Sortino {ratio(m['sortino'])} | "
                      f"max DD {m['max_dd']:.1%} | fees {m['fees_pct']:.1%}", fontsize=8)
        top.grid(alpha=0.3)
        top.tick_params(labelsize=7)
        dd = (eq / eq.cummax() - 1) * 100
        bot.fill_between(dd.index, dd, 0, color="tab:red", alpha=0.4)
        bot.set_ylabel("DD %", fontsize=7)
        bot.grid(alpha=0.3)
        bot.tick_params(labelsize=7)
        bot.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for j in range(len(names), rows_n * cols):
        r, c = divmod(j, cols)
        axes[2 * r, c].axis("off")
        axes[2 * r + 1, c].axis("off")
    fig.suptitle(f"Each strategy, last {days:.0f} days (blue: strategy, orange: BTC; red: drawdown)", fontsize=11)
    fig.tight_layout()
    grid = OUT / f"last_{args.days:g}_days_each.png"
    fig.savefig(grid, dpi=120)
    plt.close(fig)
    print(f"\ncharts: {overview}\n        {grid}")
    if sys.platform == "darwin" and not args.no_open:
        subprocess.run(["open", str(overview), str(grid)], check=False)


if __name__ == "__main__":
    main()
