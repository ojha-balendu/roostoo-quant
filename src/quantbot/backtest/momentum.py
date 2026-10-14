"""Relative-momentum sleeve with the standard improvements, each switchable so they can be tested one by one.

Every rebalance (daily at 00:00 UTC), using only data up to that bar's close:
  score        raw trailing return (baseline) or, with `residual_riskadj`, the trailing return left after
               removing each coin's BTC beta, divided by its own volatility over the same window;
               with several `lookbacks_days` the cross-coin z-scores are averaged; `skip_days` leaves out
               the most recent days (short-term reversal)
  consistency  optional filter: the share of days that moved in the trend's direction must be at least
               `min_consistency` (gradual trends persist more than jumps: "frog in the pan")
  proximity    optional: z-score of price / 30-day high added to the score (anchoring to recent highs)
  selection    long the k best, short the k worst (or `k_short` worst); with `buffer_k` (`buffer_short`)
               a held coin stays until it falls outside the best (worst) buffer, which cuts turnover
  weights      within each leg, by `weighting`, using each coin's residual volatility sigma:
                 equal        1 / n
                 inverse_vol  (1/sigma_i) / sum(1/sigma)
                 mul_vol      sigma_i / sum(sigma)          (the most volatile coin gets the most)
                 exp_vol      exp(-sigma_i / sum(sigma)) / sum_k exp(-sigma_k / sum(sigma))
               k = 0 or k_short = 0 leaves that leg empty (the BTC hedge offsets the other leg alone)
               each leg totals the same dollars; BTC hedge to zero beta; gross <= `gross`; per-coin
               cap; optional portfolio volatility target from the last 30 days
Between rebalances positions are held unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 24 * 365
HELD_MIN_WEIGHT = 0.001   # smaller positions (float residue, exchange dust) do not count as held


@dataclass
class MomentumConfig:
    k: int = 5
    buffer_k: int | None = None
    k_short: int | None = None        # shorts, if different from k
    buffer_short: int | None = None   # short-side buffer, if different from buffer_k
    lookbacks_days: list[int] = field(default_factory=lambda: [16])
    skip_days: int = 0
    residual_riskadj: bool = False
    min_consistency: float | None = None
    proximity: bool = False
    weighting: str = "equal"          # "equal", "inverse_vol", "mul_vol" or "exp_vol"
    vol_target: float | None = None   # annualised, e.g. 0.20
    max_weight: float = 1.0
    gross: float = 0.95


def _xs_z(df: pd.DataFrame) -> pd.DataFrame:
    sd = df.std(axis=1)
    return df.sub(df.mean(axis=1), axis=0).div(sd.where(sd > 0), axis=0)


def momentum_inputs(close: pd.DataFrame, resid: pd.DataFrame, eligible: pd.DataFrame,
                    cfg: MomentumConfig) -> dict[str, pd.DataFrame]:
    """Hourly panels (coins only) of the score, trend direction, consistency and risk."""
    coins = list(resid.columns)
    logp = np.log(close[coins])
    skip = cfg.skip_days * 24
    parts, direction = [], None
    for L in cfg.lookbacks_days:
        n = L * 24
        if cfg.residual_riskadj:
            ret = resid.shift(skip).rolling(n - skip, min_periods=(n - skip) // 2).sum()
            vol = resid.rolling(n, min_periods=n // 2).std() * np.sqrt(n - skip)
            s = ret / vol.where(vol > 0)
        else:
            s = logp.shift(skip) - logp.shift(n)
        parts.append(_xs_z(s.where(eligible)) if len(cfg.lookbacks_days) > 1 else s)
        if direction is None:
            direction = np.sign(s)
    score = sum(parts) / len(parts)
    if cfg.proximity:
        prox = close[coins] / close[coins].rolling(720, min_periods=360).max()
        score = _xs_z(score.where(eligible)) + _xs_z(prox.where(eligible))
    daily = resid.rolling(24, min_periods=24).sum()
    L0 = cfg.lookbacks_days[0]
    ups = sum((daily.shift(24 * d) > 0).astype(float) for d in range(L0)) / L0
    consistency = ups.where(direction > 0, 1 - ups)          # share of days moving with the trend
    risk = resid.rolling(16 * 24, min_periods=8 * 24).std()
    return {"score": score.where(eligible), "direction": direction, "consistency": consistency, "risk": risk}


def _pick(pool: np.ndarray, score: np.ndarray, held: np.ndarray, k: int, buffer_k: int | None,
          descending: bool) -> np.ndarray:
    idx = np.flatnonzero(pool)
    if len(idx) == 0:
        return idx
    order = idx[np.argsort(-score[idx] if descending else score[idx], kind="stable")]
    if not buffer_k:
        return order[:k]
    keep = [i for i in order[:buffer_k] if held[i]][:k]
    fill = [i for i in order if i not in keep][: k - len(keep)]
    return np.array(keep + fill, dtype=int)


def leg_weights(sigma: np.ndarray, weighting: str) -> np.ndarray:
    """Relative weights of the coins in one leg (they are normalised to the leg's total afterwards).
    Missing or non-positive volatilities are replaced by the leg's average."""
    if weighting == "equal" or len(sigma) == 0:
        return np.ones(len(sigma))
    sig = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
    sig = np.where(np.isfinite(sig), sig, np.nanmean(sig) if np.isfinite(sig).any() else 1.0)
    if weighting == "inverse_vol":
        return 1.0 / sig
    if weighting == "mul_vol":
        return sig
    if weighting == "exp_vol":
        return np.exp(-sig / sig.sum())
    raise ValueError(f"unknown weighting {weighting!r}: use equal, inverse_vol, mul_vol or exp_vol")


def sleeve_weights(t: int, cur: np.ndarray, x: dict[str, np.ndarray], beta: np.ndarray, eligible: np.ndarray,
                   rets: np.ndarray, cfg: MomentumConfig) -> np.ndarray:
    """Target weights at rebalance bar t: coins then BTC (last slot)."""
    n = beta.shape[1]
    w = np.zeros(n + 1)
    s, b = x["score"][t], beta[t]
    ok = eligible[t] & np.isfinite(s) & np.isfinite(b)
    k_short = cfg.k_short if cfg.k_short is not None else cfg.k
    buffer_short = cfg.buffer_short if cfg.buffer_short is not None else cfg.buffer_k
    if ok.sum() < max(cfg.k + k_short, 1) or cfg.k + k_short == 0:
        return w
    long_pool, short_pool = ok.copy(), ok.copy()
    if cfg.min_consistency is not None:
        cons, d = x["consistency"][t], x["direction"][t]
        good = np.isfinite(cons) & (cons >= cfg.min_consistency)
        long_pool &= good & (d > 0)
        short_pool &= good & (d < 0)
    longs = _pick(long_pool, s, cur[:n] > HELD_MIN_WEIGHT, cfg.k, cfg.buffer_k, True)
    shorts = _pick(short_pool & ~np.isin(np.arange(n), longs), s, cur[:n] < -HELD_MIN_WEIGHT, k_short, buffer_short,
                   False)
    for leg, sign in ((longs, 1.0), (shorts, -1.0)):
        if len(leg) == 0:
            continue
        raw = leg_weights(x["risk"][t][leg], cfg.weighting)
        w[leg] = sign * raw / raw.sum()
    w[n] = -float(np.dot(w[:n][ok], b[ok]))
    total = np.abs(w).sum()
    if total == 0:
        return w
    w *= cfg.gross / total
    if cfg.max_weight < 1.0:
        w[:n] = np.clip(w[:n], -cfg.max_weight, cfg.max_weight)
        w[n] = -float(np.dot(w[:n][ok], b[ok]))
    if cfg.vol_target:
        past = rets[max(0, t - 720):t]
        if len(past) > 48:
            sigma = float(np.nanstd(np.nan_to_num(past) @ w) * np.sqrt(HOURS_PER_YEAR))
            if sigma > cfg.vol_target:
                w *= cfg.vol_target / sigma
    return w
