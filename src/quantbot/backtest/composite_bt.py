"""Backtest of the composite-signal strategies (v1: decile cutoff, v2: expected profit vs fee).

Timing (no lookahead): every input used at bar t is known at the close of bar t (indicators from
OHLCV up to t, betas from returns up to t, the v2 calibration from outcomes realised by t). Trades
execute at the close of bar t (+ half the spread), and positions earn bar t+1's price change.

Accounting: positions are dollar weights per coin plus a BTC hedge; P&L uses real price changes;
every traded dollar pays fee + half spread. Each run checks equity == start + P&L - fees.

v1  rank coins by s' (the composite mapped to [-1, 1]); long the top `decile`, short the bottom
    `decile`, equal weight, BTC hedge to zero beta; gross = `gross`.
v2  expected cumulative relative return of each coin, A = b * s' * persistence, where
      b           = average cross-sectional slope of next-bar relative return on s', over the last
                    `calib_bars` (only realised outcomes); negative b -> 0, i.e. stay out
      persistence = (1 - rho^H) / (1 - rho), rho = how much the ranking persists bar to bar, H = max hold
    For each coin choose -1, 0 or +1 position units to maximise   T*A - c*|T - current| - c*|T|
    (c = fee + half spread; the last term is the future exit fee). Consequences: open only if
    |A| > 2c (entry + exit), hold while A keeps its sign, flip only if the gain covers both trades.
    Capacity: at most as many coins as v1 holds (2 x decile), chosen by expected profit |A|, with a
    2c bonus for coins already held (their entry is paid; swapping must beat that). A position kept on
    the same side is left exactly as it is (no resizing: that would pay fees for no expected gain).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantbot.backtest.metrics import compute_metrics
from quantbot.rl.sim import PortfolioSim, SimConfig, simple_returns


@dataclass
class CompositeConfig:
    capital: float = 1_000_000.0
    fee: float = 0.001
    spread_bps: float = 2.0
    gross: float = 0.95              # longs + shorts + BTC hedge, as a share of equity
    decile: float = 0.10             # v1 cutoff; also sets v2's position size
    band: float = 0.005              # skip per-coin weight changes smaller than this (dust from price drift)
    calib_bars: int = 90 * 24        # v2 calibration window
    max_hold_bars: int = 24          # v2 persistence horizon cap
    allow_flip: bool = False         # v2: if the calibration says the signal is inverted, trade it inverted
    min_coins: int = 10

    @property
    def cost(self) -> float:
        return self.fee + self.spread_bps / 2e4

    def sim_config(self, zero_cost: bool = False) -> SimConfig:
        return SimConfig(initial_capital=self.capital, fee=0.0 if zero_cost else self.fee,
                         spread_bps=0.0 if zero_cost else self.spread_bps, max_gross=1.0, max_weight=1.0,
                         trade_band=self.band)


# ------------------------------------------------------------------------------------- calibration
def _row_slope_and_corr(S: np.ndarray, Y: np.ndarray, min_coins: int) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(S) & np.isfinite(Y)
    n = m.sum(axis=1)
    x, y = np.where(m, S, np.nan), np.where(m, Y, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # rows without data (e.g. the last bar) -> NaN, handled below
        xc = x - np.nanmean(x, axis=1, keepdims=True)
        yc = y - np.nanmean(y, axis=1, keepdims=True)
        sxx, sxy = np.nansum(xc ** 2, axis=1), np.nansum(xc * yc, axis=1)
        slope = np.where((n >= min_coins) & (sxx > 0), sxy / sxx, np.nan)
        syy = np.nansum(yc ** 2, axis=1)
        corr = np.where((n >= min_coins) & (sxx > 0) & (syy > 0), sxy / np.sqrt(sxx * syy), np.nan)
    return slope, corr


def calibrate(sp: pd.DataFrame, resid: pd.DataFrame, cfg: CompositeConfig) -> dict[str, pd.Series | pd.DataFrame]:
    """b (bps of next-bar relative return per unit of s'), rho and A, each known at bar t."""
    S = sp.to_numpy(dtype=float)
    nxt = resid.reindex(index=sp.index, columns=sp.columns).shift(-1).to_numpy(dtype=float)
    slope, _ = _row_slope_and_corr(S, nxt, cfg.min_coins)
    prev = np.vstack([np.full((1, S.shape[1]), np.nan), S[:-1]])
    _, persist = _row_slope_and_corr(S, prev, cfg.min_coins)
    w, mp = cfg.calib_bars, max(cfg.calib_bars // 4, 2)
    b = pd.Series(slope, index=sp.index).rolling(w, min_periods=mp).mean().shift(1)  # slope at u is known at u+1
    rho = pd.Series(persist, index=sp.index).rolling(w, min_periods=mp).mean().clip(0.0, 0.999)
    mult = (1 - rho ** cfg.max_hold_bars) / (1 - rho)
    b_used = b if cfg.allow_flip else b.clip(lower=0.0)
    A = sp.mul(b_used * mult, axis=0)
    return {"b": b, "rho": rho, "persistence": mult, "A": A, "slope": pd.Series(slope, index=sp.index)}


# --------------------------------------------------------------------------------------- strategies
def v1_weights(sp_row: np.ndarray, elig_row: np.ndarray, beta_row: np.ndarray, cfg: CompositeConfig) -> np.ndarray:
    n = len(sp_row)
    w = np.zeros(n + 1)
    idx = np.flatnonzero(elig_row & np.isfinite(sp_row) & np.isfinite(beta_row))
    if len(idx) < cfg.min_coins:
        return w
    k = max(1, int(round(cfg.decile * len(idx))))
    order = idx[np.argsort(sp_row[idx], kind="stable")]
    w[order[-k:]], w[order[:k]] = 1.0 / k, -1.0 / k
    w[n] = -float(np.dot(w[:n][idx], beta_row[idx]))
    return w * (cfg.gross / np.abs(w).sum())


def v2_weights(A_row: np.ndarray, cur: np.ndarray, elig_row: np.ndarray, beta_row: np.ndarray,
               cfg: CompositeConfig) -> np.ndarray:
    n = len(A_row)
    ok = elig_row & np.isfinite(A_row) & np.isfinite(beta_row)
    w = np.zeros(n + 1)
    n_ok = int(ok.sum())
    if n_ok < cfg.min_coins:
        return w
    k = max(1, int(round(cfg.decile * n_ok)))
    unit = cfg.gross * 0.8 / (2 * k)                       # same position size as v1; 20% of gross kept for the hedge
    held = np.sign(cur[:n])
    cur_units = held                                       # current position in units (-1, 0, +1)
    c, A = cfg.cost, np.where(ok, A_row, 0.0)
    options = np.array([-1.0, 0.0, 1.0])
    value = options[None, :] * A[:, None] - c * np.abs(options[None, :] - cur_units[:, None]) - c * np.abs(options)[None, :]
    choice = options[np.argmax(value, axis=1)]
    choice[~ok] = 0.0
    picked = np.flatnonzero(choice != 0)
    if len(picked) > 2 * k:                                # capacity: best expected profit first, held coins favoured
        priority = np.abs(A[picked]) + 2 * c * (held[picked] == choice[picked])
        choice[picked[np.argsort(-priority, kind="stable")[2 * k:]]] = 0.0
    w[:n] = choice * unit
    keep = (choice != 0) & (held == choice)                # same side as now: leave the position exactly as it is
    w[:n][keep] = cur[:n][keep]
    w[n] = -float(np.dot(w[:n][ok], beta_row[ok]))
    total = np.abs(w).sum()
    return w * (cfg.gross / total) if total > cfg.gross else w


# --------------------------------------------------------------------------------------- simulation
def simulate(prices: np.ndarray, index: pd.DatetimeIndex, decide: Callable[[int, np.ndarray], np.ndarray],
             start: int, end: int, sim_cfg: SimConfig) -> dict:
    """Run bars start..end-1 from a flat book. decide(t, current_weights) -> target weights."""
    rets = simple_returns(prices)
    tradable = np.isfinite(prices)
    end = min(end, len(prices) - 1)
    sim = PortfolioSim(prices.shape[1], sim_cfg)
    curve, gross, npos, pnl_total = [sim.equity], [], [], 0.0
    for t in range(start, end):
        sim.rebalance(np.nan_to_num(decide(t, sim.weights)), tradable[t])
        gross.append(float(np.abs(sim.weights).sum()))
        npos.append(int(np.count_nonzero(np.abs(sim.weights[:-1]) > 1e-9)))
        pnl_total += sim.advance(rets[t])
        curve.append(sim.equity)
    scale = max(sim_cfg.initial_capital, abs(sim.equity), sim.traded)
    if abs(sim.equity - (sim_cfg.initial_capital + pnl_total - sim.fees)) > 1e-9 * scale:  # beyond float rounding
        raise AssertionError("accounting identity violated: equity != start + P&L - fees")
    eq = pd.Series(curve, index=index[start:end + 1])
    return {"equity": eq, "fees": sim.fees, "traded": sim.traded, "trades": sim.trades, "pnl": pnl_total,
            "avg_gross": float(np.mean(gross)) if gross else 0.0, "avg_positions": float(np.mean(npos)) if npos else 0.0}


def run_metrics(res: dict, capital: float) -> dict[str, float]:
    eq = res["equity"]
    m = compute_metrics(eq) if len(eq) > 2 else {}
    years = (eq.index[-1] - eq.index[0]).total_seconds() / (365 * 86400) if len(eq) > 1 else np.nan
    return {
        "return": m.get("total_return", 0.0), "cagr": m.get("cagr", np.nan), "vol": m.get("ann_vol", np.nan),
        "sharpe": m.get("sharpe", np.nan), "sortino": m.get("sortino", np.nan), "calmar": m.get("calmar", np.nan),
        "score": m.get("score", np.nan),
        "max_dd": m.get("max_drawdown", 0.0), "gross_pnl": res["pnl"] / capital, "fees_pct": res["fees"] / capital,
        "turnover_yr": res["traded"] / eq.mean() / years if years and years > 0 else np.nan,
        "trades": res["trades"], "avg_gross": res["avg_gross"], "avg_positions": res["avg_positions"],
    }


def topk_weights(sig_row: np.ndarray, elig_row: np.ndarray, beta_row: np.ndarray, k: int, gross: float) -> np.ndarray:
    """Long the k highest-signal coins, short the k lowest, equal weight, BTC hedge to zero beta (last slot)."""
    n = len(sig_row)
    w = np.zeros(n + 1)
    idx = np.flatnonzero(elig_row & np.isfinite(sig_row) & np.isfinite(beta_row))
    if len(idx) < 2 * k:
        return w
    order = idx[np.argsort(sig_row[idx], kind="stable")]
    w[order[-k:]], w[order[:k]] = 1.0 / k, -1.0 / k
    w[n] = -float(np.dot(w[:n][idx], beta_row[idx]))
    return w * (gross / np.abs(w).sum())
