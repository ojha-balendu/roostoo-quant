"""Bar-based backtester that mirrors Roostoo mechanics.

* Longs: spot BUY/SELL, fee = taker (or maker if you assume passive fills).
* Shorts: sized by USD collateral (qty = collateral / entry price), 0.1% open and close
  fees, close at the ask, loss capped at the collateral.
* Longs and shorts both consume cash, so gross exposure is bounded by equity.
* Uses the same `decide()` as live trading (thresholds, min trade size, flip logic).

Simplifications: trades fill at bar close +/- half the spread, no quantity rounding,
no partial fills, no market impact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantbot.backtest.metrics import compute_metrics
from quantbot.config import FeeConfig
from quantbot.execution.decision import Action, decide


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float]

    def summary(self) -> str:
        return "\n".join(f"{k:>18}: {v:,.4f}" if isinstance(v, float) else f"{k:>18}: {v}"
                         for k, v in self.metrics.items())


def run_backtest(
    closes: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    initial_capital: float = 100_000.0,
    fees: FeeConfig | None = None,
    long_fee: str = "taker",
    spread_bps: float = 2.0,
    delay_bars: int = 1,
    min_trade_usd: float = 50.0,
    min_close_usd: float = 2.0,
    threshold: float = 0.02,
    cash_buffer: float = 0.005,
    trade_on: pd.Series | np.ndarray | None = None,
) -> BacktestResult:
    """trade_on: optional boolean per bar; when given, orders are only placed on True bars
    (e.g. once an hour) while equity is still marked every bar."""
    fees = fees or FeeConfig()
    closes = closes.sort_index().ffill()
    w = target_weights.reindex(index=closes.index, columns=closes.columns).fillna(0.0)
    if delay_bars:
        w = w.shift(delay_bars).fillna(0.0)
    P = closes.to_numpy(dtype=float)
    W = w.to_numpy(dtype=float)
    pairs = list(closes.columns)
    index = closes.index
    n_t, n_a = P.shape
    if trade_on is None:
        can_trade = np.ones(n_t, dtype=bool)
    else:
        can_trade = np.asarray(trade_on, dtype=bool)
        if len(can_trade) != n_t:
            raise ValueError("trade_on must have one entry per bar")

    long_rate = fees.taker if long_fee == "taker" else fees.maker
    hs = spread_bps / 2e4
    cash = float(initial_capital)
    lq = np.zeros(n_a)   # long qty
    sq = np.zeros(n_a)   # short qty
    se = np.zeros(n_a)   # short entry price
    sc = np.zeros(n_a)   # short collateral
    eq = np.full(n_t, np.nan)
    trades: list[tuple] = []

    for t in range(n_t):
        p = P[t]
        valid = np.isfinite(p) & (p > 0)
        pv = np.where(valid, p, 0.0)
        short_val = np.where(valid, sc + sq * (se - pv), sc)
        # Loss on a short is capped at its collateral: liquidate exhausted shorts.
        for i in np.flatnonzero((sq > 0) & (short_val <= 0)):
            trades.append((index[t], pairs[i], "CLOSE", sq[i], p[i], 0.0, 0.0))
            sq[i] = sc[i] = se[i] = 0.0
        short_val = np.where(sq > 0, np.where(valid, sc + sq * (se - pv), sc), 0.0)
        equity = cash + float(np.sum(lq * pv)) + float(np.sum(short_val))
        eq[t] = equity
        if equity <= 0:
            eq[t:] = max(equity, 0.0)
            break
        if not can_trade[t]:
            continue

        active = np.flatnonzero(valid & ((W[t] != 0) | (lq > 0) | (sq > 0)))
        reducing, increasing = [], []
        for i in active:
            for it in decide(pairs[i], lq[i], sq[i], p[i], equity, W[t, i], min_trade_usd=min_trade_usd,
                             min_close_usd=min_close_usd, threshold=threshold):
                (reducing if it.is_reducing else increasing).append((i, it))

        for i, it in reducing:
            if it.action is Action.SELL:
                q = lq[i] if it.full else min(it.quantity or 0.0, lq[i])
                px = p[i] * (1 - hs)
                fee = q * px * long_rate
                cash += q * px - fee
                lq[i] = 0.0 if it.full else lq[i] - q
            else:  # CLOSE
                q = sq[i] if it.full else min(it.quantity or 0.0, sq[i])
                if q <= 0:
                    continue
                px = p[i] * (1 + hs)
                coll = sc[i] * q / sq[i]
                pnl = max(q * (se[i] - px), -coll)
                fee = q * px * fees.short_close
                cash += coll + pnl - fee
                if it.full or sq[i] - q <= 1e-12:
                    sq[i] = sc[i] = se[i] = 0.0
                else:
                    sq[i] -= q
                    sc[i] -= coll
            trades.append((index[t], pairs[i], it.action.value, q, px, q * px, fee))

        if increasing:
            def rate(a: Action) -> float:
                return long_rate if a is Action.LONG else fees.short_open

            budget = cash - cash_buffer * equity
            need = sum(it.notional * (1 + rate(it.action)) for _, it in increasing)
            scale = 0.0 if budget <= 0 else min(1.0, budget / need)
            for i, it in increasing:
                n = it.notional * scale
                if n < min_trade_usd:
                    continue
                fee = n * rate(it.action)
                cash -= n + fee
                if it.action is Action.LONG:
                    px = p[i] * (1 + hs)
                    q = n / px
                    lq[i] += q
                else:  # SHORT: fills at the bid, merged at weighted-average entry
                    px = p[i] * (1 - hs)
                    q = n / px
                    se[i] = (sq[i] * se[i] + q * px) / (sq[i] + q)
                    sq[i] += q
                    sc[i] += n
                trades.append((index[t], pairs[i], it.action.value, q, px, n, fee))

    equity_s = pd.Series(eq, index=index, name="equity").ffill()
    trades_df = pd.DataFrame(trades, columns=["ts", "pair", "action", "qty", "price", "notional", "fee"])
    metrics = compute_metrics(equity_s)
    span_years = (index[-1] - index[0]).total_seconds() / (365 * 86400) if len(index) > 1 else float("nan")
    metrics.update(
        n_trades=int(len(trades_df)),
        fees_paid=float(trades_df["fee"].sum()) if len(trades_df) else 0.0,
        annual_turnover=float(trades_df["notional"].sum() / equity_s.mean() / span_years)
        if len(trades_df) and span_years > 0 else 0.0,
    )
    return BacktestResult(equity=equity_s, trades=trades_df, metrics=metrics)
