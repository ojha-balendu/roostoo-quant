"""Threshold momentum: hold every coin whose momentum score is strong enough, with hysteresis.

At each check bar (every hour, or once a day), with score = 16-day BTC-removed return / its volatility:
  exit   a long when its score < exit_frac x T, a short when its score > -exit_frac x T (or the coin
         is no longer eligible)
  enter  a long when score > T, a short when score < -T, at most `max_per_side` per side; free slots
         go to the strongest candidates first (longs and shorts compete by |score|)
  size   fixed: `unit` of equity; inverse_vol: unit x median sigma / sigma_i (capped at `cap`); a position
         keeps its size while held (no resizing, so it only trades on entry and exit)
  hedge  BTC = -sum(position x beta), reset when positions change or it is `hedge_band` off; capped at `cap`
  gross  an entry is skipped if longs + shorts + |hedge| would exceed `max_gross`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantbot.backtest.momentum import HELD_MIN_WEIGHT


@dataclass
class ThresholdConfig:
    threshold: float = 1.0
    exit_frac: float = 0.5
    max_per_side: int = 8
    unit: float = 0.95 / 16
    sizing: str = "fixed"            # "fixed" or "inverse_vol"
    cap: float = 0.25
    max_gross: float = 0.95
    hedge_band: float = 0.02


def threshold_decider(tc: ThresholdConfig, score: np.ndarray, risk: np.ndarray, beta: np.ndarray,
                      eligible: np.ndarray, check: np.ndarray, first: int, log: list | None = None):
    """decide(t, cur) for the simulator; columns are the coins, then BTC (hedge) last.
    `log`, if given, receives (n_longs, n_shorts) after every check."""
    def decide(t: int, cur: np.ndarray) -> np.ndarray:
        if not (check[t] or t == first):
            return cur
        n = score.shape[1]
        w = cur.copy()
        s, b = score[t], np.nan_to_num(beta[t], nan=1.0)
        ok = eligible[t] & np.isfinite(score[t]) & np.isfinite(beta[t])
        lng, sht = w[:n] > HELD_MIN_WEIGHT, w[:n] < -HELD_MIN_WEIGHT
        out = (lng & (~ok | ~(s >= tc.exit_frac * tc.threshold))) | (sht & (~ok | ~(s <= -tc.exit_frac * tc.threshold)))
        w[:n][out] = 0.0
        w[:n][~(lng | sht)] = 0.0                                            # clear dust
        changed = bool(out.any())
        lng, sht = w[:n] > 0, w[:n] < 0
        free = {1: tc.max_per_side - int(lng.sum()), -1: tc.max_per_side - int(sht.sum())}
        cand = np.flatnonzero(ok & ~(lng | sht) & ((s > tc.threshold) | (s < -tc.threshold)))
        if len(cand):
            sig = risk[t]
            med = np.nanmedian(np.where(ok & np.isfinite(sig) & (sig > 0), sig, np.nan))
            for c in cand[np.argsort(-np.abs(s[cand]), kind="stable")]:
                side = 1 if s[c] > 0 else -1
                if free[side] <= 0:
                    continue
                size = tc.unit
                if tc.sizing == "inverse_vol" and np.isfinite(sig[c]) and sig[c] > 0 and np.isfinite(med):
                    size = min(tc.unit * med / sig[c], tc.cap)
                trial = w[:n].copy()
                trial[c] = side * size
                hedge = float(np.clip(-np.dot(trial, b), -tc.cap, tc.cap))
                if np.abs(trial).sum() + abs(hedge) > tc.max_gross + 1e-12:
                    continue
                w[:n] = trial
                free[side] -= 1
                changed = True
        need = float(np.clip(-np.dot(w[:n], b), -tc.cap, tc.cap))
        if changed or t == first or abs(need - w[n]) > tc.hedge_band:
            w[n] = need
        if log is not None:
            log.append((int((w[:n] > 0).sum()), int((w[:n] < 0).sum())))
        return w

    return decide
