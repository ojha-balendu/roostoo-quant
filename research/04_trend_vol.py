"""Trend following + volatility targeting + regime filter, scored over every 14-day window.

    python research/04_trend_vol.py                         # majors, 1h bars, last 365 days
    python research/04_trend_vol.py --universe all          # all coins with data
    python research/04_trend_vol.py --interval 1m           # slower, more realistic fills
    python research/04_trend_vol.py --target-vol 0.5 --rebalance-hours 4
    python research/04_trend_vol.py --step-days 7           # overlapping windows (more samples)

Each window is scored as a fresh competition: flat start, $100k, fees and spread included.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.backtest.windows import evaluate_windows, summarize
from quantbot.config import load_config
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe
from quantbot.strategy.trend_vol import TrendVolTarget

BTC = "BTC/USD"
MAJORS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "BNB/USD", "DOGE/USD", "ADA/USD", "LINK/USD",
          "AVAX/USD", "LTC/USD", "DOT/USD", "SUI/USD", "NEAR/USD", "UNI/USD", "AAVE/USD"]
OUT = Path("reports/research")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h", choices=["1h", "1m"])
    ap.add_argument("--universe", default="majors", choices=["majors", "all"])
    ap.add_argument("--history-days", type=float, default=365, help="span covered by the windows")
    ap.add_argument("--window-days", type=float, default=14)
    ap.add_argument("--step-days", type=float, default=None, help="default = window length (no overlap)")
    ap.add_argument("--target-vol", type=float, default=0.30, help="annualised portfolio volatility target")
    ap.add_argument("--lookbacks", default="24,72,168", help="trend lookbacks in hours")
    ap.add_argument("--regime-ma-hours", type=int, default=168)
    ap.add_argument("--regime-band", type=float, default=0.01)
    ap.add_argument("--signal", default="tanh", choices=["tanh", "sign"])
    ap.add_argument("--long-in-bear", type=float, default=0.0)
    ap.add_argument("--short-in-bull", type=float, default=0.0)
    ap.add_argument("--rebalance-hours", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.005, help="skip weight changes below this")
    ap.add_argument("--spread-bps", type=float, default=2.0)
    args = ap.parse_args()

    cfg = load_config()
    universe = load_universe(cfg.data.universe_path)
    if args.universe == "majors":
        universe = {p: s for p, s in universe.items() if p in MAJORS}
    probe = load_panel(cfg.data.processed_dir, cfg.data.market, args.interval, {BTC: universe[BTC]})
    if probe.empty:
        raise SystemExit(f"no {args.interval} BTC data; run quantbot download --interval {args.interval}")
    end = probe.index.max()
    first_window = end - pd.Timedelta(days=args.history_days)
    closes = load_panel(cfg.data.processed_dir, cfg.data.market, args.interval, universe,
                        start=str((first_window - pd.Timedelta(days=12)).date())).ffill()
    print(f"{closes.shape[1]} coins, {args.interval} bars, windows from {first_window:%Y-%m-%d} to {end:%Y-%m-%d}")

    lookbacks = [int(x) for x in args.lookbacks.split(",")]
    common = dict(lookbacks_hours=lookbacks, target_vol=args.target_vol, regime_ma_hours=args.regime_ma_hours,
                  regime_band=args.regime_band, signal=args.signal)
    variants = {
        "trend+vol+regime": TrendVolTarget(long_in_bear=args.long_in_bear, short_in_bull=args.short_in_bull,
                                           **common).generate_weights(closes),
        "trend+vol (no regime)": TrendVolTarget(long_in_bear=1.0, short_in_bull=1.0, **common)
        .generate_weights(closes),
    }
    avail = closes.notna().astype(float)
    variants["hold BTC"] = pd.DataFrame(0.0, index=closes.index, columns=closes.columns).assign(**{BTC: 0.95})
    variants["equal-weight long"] = avail.div(avail.sum(axis=1), axis=0) * 0.95

    # Trade on the first bar of each new rebalance period (works for any bar alignment).
    bar = closes.index.to_series().diff().median()
    decided = pd.Series(closes.index + bar if args.interval == "1h" else closes.index)
    bucket = decided.dt.floor(f"{args.rebalance_hours}h")
    on_hour = (bucket != bucket.shift(1)).to_numpy()
    bt = dict(
        initial_capital=cfg.backtest.initial_capital, fees=cfg.fees, long_fee=cfg.backtest.long_fee,
        spread_bps=args.spread_bps, delay_bars=0 if args.interval == "1h" else 1,
        min_trade_usd=cfg.execution.min_trade_usd, min_close_usd=cfg.execution.min_close_usd,
        threshold=args.threshold, cash_buffer=cfg.execution.cash_buffer,
    )
    results = {}
    for name, w in variants.items():
        results[name] = evaluate_windows(closes, w, window_days=args.window_days, step_days=args.step_days,
                                         start=first_window, trade_on=on_hour, **bt)
        print(f"  evaluated {name}: {len(results[name])} windows")

    summary = pd.DataFrame({n: summarize(r, versus=results["hold BTC"]) for n, r in results.items()})
    pct_rows = ["median_return", "mean_return", "worst_return", "best_return", "pct_positive",
                "median_max_dd", "worst_max_dd", "pct_beats_benchmark"]
    shown = summary.copy().astype(object)
    for row in summary.index:
        for col in summary.columns:
            v = summary.loc[row, col]
            shown.loc[row, col] = (f"{v:.1%}" if row in pct_rows else
                                   f"{v:,.0f}" if row in ("windows", "avg_fees") else f"{v:.2f}")
    shown = shown.rename(index={"pct_beats_benchmark": "pct_beats_btc"})
    pd.set_option("display.width", 180)
    print(f"\nAcross all {args.window_days:g}-day windows (each starts flat with $100k, fees included):")
    print(shown.to_string())

    main_r, btc_r = results["trend+vol+regime"], results["hold BTC"]
    per = pd.DataFrame({
        "start": main_r["start"].dt.strftime("%Y-%m-%d"),
        "strategy": main_r["return"].map("{:+.1%}".format),
        "max_dd": main_r["max_dd"].map("{:.1%}".format),
        "avg_gross": main_r["avg_gross"].map("{:.2f}".format),
        "btc": btc_r["return"].map("{:+.1%}".format),
        "fees_$": main_r["fees"].map("{:,.0f}".format),
    })
    print("\ntrend+vol+regime, window by window (vs holding BTC):")
    print(per.to_string(index=False))

    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(results, names=["variant"]).to_csv(OUT / "trend_vol_windows.csv")
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(13, 5))
        x = np.arange(len(main_r))
        ax.bar(x - 0.2, main_r["return"] * 100, 0.4, label="trend+vol+regime")
        ax.bar(x + 0.2, btc_r["return"] * 100, 0.4, label="hold BTC")
        ax.set_xticks(x, main_r["start"].dt.strftime("%m-%d"), rotation=90, fontsize=7)
        ax.set_ylabel("window return %")
        ax.axhline(0, color="k", lw=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "trend_vol_windows.png", dpi=130)
        print(f"\nsaved {OUT}/trend_vol_windows.csv and .png")
    except ImportError:
        print(f"\nsaved {OUT}/trend_vol_windows.csv")


if __name__ == "__main__":
    main()
