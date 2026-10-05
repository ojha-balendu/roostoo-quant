"""Backtest the beta-neutral alt basket on 1-minute candles.

Long every alt (equal weight), short BTC by the basket's beta, re-estimated and
rebalanced once an hour. Compared against the same alt basket without the hedge
and against simply holding BTC.

Needs 1m data covering the test window plus the beta lookback, e.g.:
    quantbot download --interval 1m --start 2026-08-01
Run:
    python research/02_beta_neutral.py                 # last 15 days
    python research/02_beta_neutral.py --days 30 --sizing inverse_vol
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.backtest.engine import run_backtest
from quantbot.config import load_config
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe
from quantbot.strategy.beta_neutral import BetaNeutralBasket

BTC = "BTC/USD"
OUT = Path("reports/research")


def realised_beta(equity: pd.Series, btc_price: pd.Series) -> tuple[float, float]:
    """Beta and correlation of the strategy's hourly returns to BTC's."""
    e = equity.resample("1h").last().pct_change()
    b = btc_price.resample("1h").last().pct_change()
    df = pd.concat([e, b], axis=1).dropna()
    if len(df) < 10 or df.iloc[:, 1].var() == 0:
        return float("nan"), float("nan")
    beta = df.iloc[:, 0].cov(df.iloc[:, 1]) / df.iloc[:, 1].var()
    return float(beta), float(df.iloc[:, 0].corr(df.iloc[:, 1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=15)
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--beta-hours", type=int, default=720, help="beta lookback (default 30 days)")
    ap.add_argument("--sizing", default="equal", choices=["equal", "inverse_vol"])
    ap.add_argument("--threshold", type=float, default=0.002,
                    help="skip trades smaller than this fraction of equity per coin")
    ap.add_argument("--spread-bps", type=float, default=2.0)
    args = ap.parse_args()

    cfg = load_config()
    universe = load_universe(cfg.data.universe_path)
    probe = load_panel(cfg.data.processed_dir, cfg.data.market, args.interval, {BTC: universe[BTC]})
    if probe.empty:
        raise SystemExit(f"no {args.interval} data: run  quantbot download --interval {args.interval} --start <date>")
    end = probe.index.max()
    test_start = end - pd.Timedelta(days=args.days)
    load_start = test_start - pd.Timedelta(hours=args.beta_hours + 48)
    closes = load_panel(cfg.data.processed_dir, cfg.data.market, args.interval, universe,
                        start=str(load_start.date())).ffill()
    print(f"loaded {closes.shape[1]} coins, {closes.index.min():%Y-%m-%d} -> {end:%Y-%m-%d %H:%M} ({args.interval})")
    if closes.index.min() > test_start - pd.Timedelta(hours=args.beta_hours * 0.5):
        print("WARNING: less beta history than requested before the test window; download an earlier --start")

    strat = BetaNeutralBasket(beta_lookback_hours=args.beta_hours, min_hours=min(240, args.beta_hours // 2),
                              sizing=args.sizing)
    weights = strat.generate_weights(closes)

    test = closes.index >= test_start
    c, w = closes.loc[test], weights.loc[test]
    hourly = c.index.minute == 0  # decisions made at the top of the hour execute on the next bar
    alts = [p for p in c.columns if p != BTC]
    variants = {
        "beta-neutral": w,
        "alt basket, no hedge": w.assign(**{BTC: 0.0}),
        "hold BTC": pd.DataFrame(0.0, index=c.index, columns=c.columns).assign(**{BTC: 0.95}),
    }

    held = (w[alts] > 0).sum(axis=1)
    print(f"test window {c.index.min():%Y-%m-%d %H:%M} -> {c.index.max():%Y-%m-%d %H:%M}, "
          f"{int(held.median())} alts held (median)")
    print(f"average long book {w[alts].sum(axis=1).mean():.1%} of equity, "
          f"BTC short {(-w[BTC]).mean():.1%}, implied basket beta "
          f"{((-w[BTC]) / w[alts].sum(axis=1)).replace([np.inf, -np.inf], np.nan).mean():.2f}\n")

    rows, curves = {}, {}
    for name, wv in variants.items():
        res = run_backtest(
            c, wv, initial_capital=cfg.backtest.initial_capital, fees=cfg.fees, long_fee=cfg.backtest.long_fee,
            spread_bps=args.spread_bps, delay_bars=1, min_trade_usd=cfg.execution.min_trade_usd,
            min_close_usd=cfg.execution.min_close_usd, threshold=args.threshold,
            cash_buffer=cfg.execution.cash_buffer, trade_on=hourly,
        )
        beta, corr = realised_beta(res.equity, c[BTC])
        m = res.metrics
        rows[name] = {
            "return_%": m.get("total_return", np.nan) * 100,
            "max_dd_%": m.get("max_drawdown", np.nan) * 100,
            "sharpe(ann)": m.get("sharpe", np.nan),
            "sortino(ann)": m.get("sortino", np.nan),
            "beta_to_btc": beta,
            "corr_to_btc": corr,
            "fees_$": m.get("fees_paid", 0.0),
            "trades": m.get("n_trades", 0),
        }
        curves[name] = res.equity

    pd.set_option("display.width", 160)
    print(pd.DataFrame(rows).T.round(3).to_string())
    print("\nNote: Sharpe/Sortino over ~2 weeks are annualised from very few days; treat them as rough.")

    OUT.mkdir(parents=True, exist_ok=True)
    eq = pd.DataFrame(curves)
    eq.to_csv(OUT / "beta_neutral_equity.csv")
    try:
        import matplotlib.pyplot as plt

        ax = (eq / eq.iloc[0] - 1).mul(100).plot(figsize=(12, 5), title="Beta-neutral alt basket vs alternatives")
        ax.set_ylabel("return %")
        plt.tight_layout()
        plt.savefig(OUT / "beta_neutral_equity.png", dpi=130)
        print(f"saved {OUT}/beta_neutral_equity.csv and .png")
    except ImportError:
        print(f"saved {OUT}/beta_neutral_equity.csv")


if __name__ == "__main__":
    main()
