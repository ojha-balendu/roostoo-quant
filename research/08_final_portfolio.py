"""Evaluate the portfolio exactly as configured for the competition.

    python research/08_final_portfolio.py                          # uses config/competition.yaml
    python research/08_final_portfolio.py --config config/config.yaml

Builds the strategy, risk limits, rebalancing schedule, thresholds and capital from the
config file (the same objects the live bot uses), then reports:
  * every 14-day window since the data starts, as a fresh competition, vs benchmarks
  * the ensemble's members on their own (does averaging help?)
  * one continuous backtest over the whole history
Not simulated: the drawdown kill switch and exchange quantity rounding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.backtest.engine import run_backtest
from quantbot.backtest.windows import evaluate_windows, summarize
from quantbot.config import Config, load_config
from quantbot.data.loader import load_panel
from quantbot.data.universe import load_universe
from quantbot.risk import clip_weights_frame
from quantbot.strategy.ensemble import Ensemble
from quantbot.strategy.registry import get_strategy

BTC = "BTC/USD"


def schedule(index: pd.DatetimeIndex, cfg: Config) -> np.ndarray:
    """Bars on which the live bot would rebalance (same rule as the runner)."""
    bar = index.to_series().diff().median()
    closes_at = index + bar
    if cfg.live.rebalance_utc_hours:
        return np.asarray((closes_at.minute == 0) & closes_at.hour.isin(cfg.live.rebalance_utc_hours))
    every = max(cfg.live.rebalance_every_bars, 1)
    return np.arange(len(index)) % every == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/competition.yaml")
    ap.add_argument("--window-days", type=float, default=14)
    args = ap.parse_args()

    cfg = load_config(args.config)
    strategy = get_strategy(cfg.strategy.name, cfg.strategy.params)
    universe = load_universe(cfg.data.universe_path)
    closes = load_panel(cfg.data.processed_dir, cfg.data.market, "1h", universe, start=cfg.data.start).ffill(limit=3)
    if closes.empty or BTC not in closes:
        raise SystemExit("no hourly data: run  quantbot download")
    first_window = closes.index.min() + pd.Timedelta(days=45)
    trade_on = schedule(closes.index, cfg)
    bt = dict(initial_capital=cfg.backtest.initial_capital, fees=cfg.fees, long_fee=cfg.backtest.long_fee,
              spread_bps=cfg.backtest.spread_bps, delay_bars=0, min_trade_usd=cfg.execution.min_trade_usd,
              min_close_usd=cfg.execution.min_close_usd, threshold=cfg.execution.rebalance_threshold,
              cash_buffer=cfg.execution.cash_buffer)
    print(f"config {args.config}: strategy '{cfg.strategy.name}', {closes.shape[1]} coins loaded, "
          f"capital ${cfg.backtest.initial_capital:,.0f}, rebalancing on {int(trade_on.sum())} of {len(trade_on)} bars")

    variants: dict[str, pd.DataFrame] = {
        "competition portfolio": clip_weights_frame(strategy.generate_weights(closes), cfg.risk)}
    if isinstance(strategy, Ensemble):
        for i, (member, _) in enumerate(strategy.members, 1):
            label = f"member {i}: {member.name}"
            if hasattr(member, "long_in_bear"):
                label += " (regime filter)" if member.long_in_bear == 0 else " (no regime filter)"
            variants[label] = clip_weights_frame(member.generate_weights(closes), cfg.risk)
    majors = [c for c in closes.columns if (variants["competition portfolio"][c] != 0).any()] or list(closes.columns)
    variants["hold BTC"] = pd.DataFrame(0.0, index=closes.index, columns=closes.columns).assign(**{BTC: 0.95})
    avail = closes[majors].notna().astype(float)
    variants["equal-weight long (same coins)"] = (avail.div(avail.sum(axis=1), axis=0) * 0.95).reindex(
        columns=closes.columns, fill_value=0.0)

    results = {name: evaluate_windows(closes, w, window_days=args.window_days, start=first_window,
                                      trade_on=trade_on, **bt) for name, w in variants.items()}
    summary = pd.DataFrame({n: summarize(r, versus=results["hold BTC"]) for n, r in results.items()})
    pct = {"median_return", "mean_return", "worst_return", "best_return", "pct_positive", "median_max_dd",
           "worst_max_dd", "pct_beats_benchmark"}
    show = summary.astype(object).copy()
    for row in summary.index:
        for col in summary.columns:
            v = summary.loc[row, col]
            show.loc[row, col] = (f"{v:.1%}" if row in pct else f"{v:,.0f}" if row in ("windows", "avg_fees")
                                  else f"{v:.2f}")
    show = show.rename(index={"pct_beats_benchmark": "pct_beats_btc"}).drop(index=["avg_gross"], errors="ignore")
    pd.set_option("display.width", 220)
    print(f"\nEvery {args.window_days:g}-day window since {first_window:%Y-%m-%d} "
          f"(each starts flat with ${cfg.backtest.initial_capital:,.0f}, fees included):")
    print(show.to_string())

    print("\nOne continuous backtest over the whole history:")
    rows = {}
    for name in ("competition portfolio", "hold BTC"):
        res = run_backtest(closes.loc[closes.index >= first_window], variants[name].loc[closes.index >= first_window],
                           trade_on=trade_on[closes.index >= first_window], **bt)
        m = res.metrics
        rows[name] = {"total return": f"{m['total_return']:+.1%}", "CAGR": f"{m['cagr']:+.1%}",
                      "volatility": f"{m['ann_vol']:.1%}", "Sharpe": f"{m['sharpe']:.2f}",
                      "Sortino": f"{m['sortino']:.2f}", "max drawdown": f"{m['max_drawdown']:.1%}",
                      "Calmar": f"{m['calmar']:.2f}", "fees": f"${m['fees_paid']:,.0f}", "trades": m["n_trades"]}
        if name == "competition portfolio":
            curve = res.equity
    print(pd.DataFrame(rows).to_string())

    out = Path(cfg.backtest.report_dir) / "final"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "window_summary.csv")
    pd.concat(results, names=["variant"]).to_csv(out / "windows.csv")
    curve.to_csv(out / "equity_full_history.csv")
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        (curve / curve.iloc[0] - 1).mul(100).plot(ax=ax, label="competition portfolio")
        btc = closes.loc[curve.index, BTC]
        ((btc / btc.iloc[0] - 1) * 100).plot(ax=ax, label="BTC", alpha=0.6)
        ax.set_ylabel("return %")
        ax.legend()
        ax.set_title("Competition portfolio, continuous backtest")
        fig.tight_layout()
        fig.savefig(out / "equity_full_history.png", dpi=130)
    except ImportError:
        pass
    print(f"\nsaved results to {out}/")


if __name__ == "__main__":
    main()
