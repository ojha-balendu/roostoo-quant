"""Episode metrics and the composite objective.

composite = 0.5 * return_% + 0.2 * Sortino + 0.15 * Sharpe + 0.15 * Calmar

Scale choices (so no single term can dominate or be gamed):
  * return is in PERCENT (a 2% gain counts as 2.0), comparable in size to the ratios;
  * Sharpe, Sortino and Calmar are annualised and capped at +/- `cap` (default 10);
  * Calmar uses a drawdown floor (default 1%) so a near-zero drawdown can't explode it.
A portfolio that never trades scores exactly 0.
"""

from __future__ import annotations

import numpy as np

BARS_PER_YEAR_15M = 365 * 24 * 4
DEFAULT_WEIGHTS = (0.5, 0.2, 0.15, 0.15)  # return, Sortino, Sharpe, Calmar


def episode_metrics(equity: np.ndarray, periods_per_year: float = BARS_PER_YEAR_15M, cap: float = 10.0,
                    dd_floor: float = 0.01) -> dict[str, float]:
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2 or eq[0] <= 0:
        return {"return_pct": 0.0, "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "max_dd": 0.0}
    rets = eq[1:] / eq[:-1] - 1.0
    total = eq[-1] / eq[0] - 1.0
    mean, std = rets.mean(), rets.std()
    downside = np.sqrt(np.mean(np.minimum(rets, 0.0) ** 2))
    scale = np.sqrt(periods_per_year)
    sharpe = mean / std * scale if std > 1e-12 else 0.0
    if downside > 1e-12:
        sortino = mean / downside * scale
    else:
        sortino = cap if mean > 1e-12 else 0.0
    max_dd = float(np.max(1.0 - eq / np.maximum.accumulate(eq)))
    years = len(rets) / periods_per_year
    annual = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1.0 else -1.0
    calmar = annual / max(max_dd, dd_floor)
    clip = lambda x: float(np.clip(x, -cap, cap))  # noqa: E731
    return {"return_pct": float(total * 100), "sharpe": clip(sharpe), "sortino": clip(sortino),
            "calmar": clip(calmar), "max_dd": max_dd}


def composite_score(m: dict[str, float], weights: tuple[float, float, float, float] = DEFAULT_WEIGHTS) -> float:
    w_ret, w_sortino, w_sharpe, w_calmar = weights
    return (w_ret * m["return_pct"] + w_sortino * m["sortino"] + w_sharpe * m["sharpe"]
            + w_calmar * m["calmar"])
