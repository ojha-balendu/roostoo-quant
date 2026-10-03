"""Performance metrics (crypto trades 24/7, so years are 365 days)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

SECONDS_PER_YEAR = 365 * 24 * 3600


def periods_per_year(index: pd.Index) -> float:
    if len(index) < 2:
        return float("nan")
    step = pd.Series(index).diff().dropna().median().total_seconds()
    return SECONDS_PER_YEAR / step if step > 0 else float("nan")


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def compute_metrics(equity: pd.Series, ppy: float | None = None) -> dict[str, float]:
    equity = equity.dropna()
    if len(equity) < 3 or equity.iloc[0] <= 0:
        return {}
    ppy = ppy or periods_per_year(equity.index)
    rets = equity.pct_change().dropna()
    total = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = len(rets) / ppy
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float("nan")
    std = rets.std()
    downside = math.sqrt(float(np.mean(np.minimum(rets, 0.0) ** 2)))
    mdd = max_drawdown(equity)
    return {
        "total_return": float(total),
        "cagr": float(cagr),
        "ann_vol": float(std * math.sqrt(ppy)),
        "sharpe": float(rets.mean() / std * math.sqrt(ppy)) if std > 0 else float("nan"),
        "sortino": float(rets.mean() / downside * math.sqrt(ppy)) if downside > 0 else float("nan"),
        "max_drawdown": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else float("nan"),
        "final_equity": float(equity.iloc[-1]),
    }
