"""Portfolio-level risk: position caps, gross cap and an automatic drawdown kill switch."""

from __future__ import annotations

import logging
import math
import time

import pandas as pd

from quantbot.config import RiskConfig

log = logging.getLogger(__name__)


def clip_weights_frame(weights: pd.DataFrame, cfg: RiskConfig) -> pd.DataFrame:
    """Vectorised version of RiskManager.clip_targets (used by the backtester)."""
    cap = cfg.max_weight_per_asset
    w = weights.fillna(0.0).clip(-cap, cap)
    gross = w.abs().sum(axis=1)
    scale = (cfg.max_gross_leverage / gross).where(gross > cfg.max_gross_leverage, 1.0)
    return w.mul(scale, axis=0)


class RiskManager:
    def __init__(self, cfg: RiskConfig, store=None):
        self.cfg = cfg
        self.store = store
        self._mem: dict[str, float] = {}

    def _get(self, key: str, default: float) -> float:
        if self.store is not None:
            return self.store.get_float(key, default)
        return self._mem.get(key, default)

    def _set(self, key: str, value: float) -> None:
        if self.store is not None:
            self.store.set(key, str(value))
        else:
            self._mem[key] = value

    def clip_targets(self, targets: dict[str, float], tradable: set[str] | None = None) -> dict[str, float]:
        clean = {
            p: float(w)
            for p, w in targets.items()
            if (tradable is None or p in tradable) and w is not None and math.isfinite(w) and abs(w) > 1e-12
        }
        if not clean:
            return {}
        frame = clip_weights_frame(pd.DataFrame([clean]), self.cfg)
        return {p: float(v) for p, v in frame.iloc[0].items() if abs(v) > 1e-12}

    def update_drawdown(self, equity: float, now: float | None = None) -> bool:
        """Track peak equity. Returns False while the kill switch is active (caller should go flat)."""
        now = time.time() if now is None else now
        halted_until = self._get("halted_until", 0.0)
        if halted_until:
            if now < halted_until:
                return False
            log.warning("drawdown cooldown over; resuming trading with a fresh peak of %.2f", equity)
            self._set("halted_until", 0.0)
            self._set("peak_equity", equity)
        peak = max(self._get("peak_equity", equity), equity)
        self._set("peak_equity", peak)
        drawdown = 1.0 - equity / peak if peak > 0 else 0.0
        if drawdown >= self.cfg.max_drawdown_kill:
            until = now + self.cfg.kill_cooldown_hours * 3600
            self._set("halted_until", until)
            log.critical("KILL SWITCH: drawdown %.1f%% >= %.1f%%; flattening until %s",
                         drawdown * 100, self.cfg.max_drawdown_kill * 100,
                         time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(until)))
            return False
        return True
