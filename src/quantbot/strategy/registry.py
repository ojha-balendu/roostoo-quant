from __future__ import annotations

from typing import Any

from quantbot.strategy.base import Strategy
from quantbot.strategy.beta_neutral import BetaNeutralBasket
from quantbot.strategy.ensemble import Ensemble
from quantbot.strategy.flat import Flat
from quantbot.strategy.pairs_spread import PairsSpread
from quantbot.strategy.trend_vol import TrendVolTarget
from quantbot.strategy.vol_managed import VolManaged
from quantbot.strategy.xs_momentum import CrossSectionalMomentum

# Register new strategies here.
STRATEGIES: dict[str, type[Strategy]] = {
    Flat.name: Flat,
    CrossSectionalMomentum.name: CrossSectionalMomentum,
    BetaNeutralBasket.name: BetaNeutralBasket,
    PairsSpread.name: PairsSpread,
    TrendVolTarget.name: TrendVolTarget,
    VolManaged.name: VolManaged,
    Ensemble.name: Ensemble,
}


def get_strategy(name: str, params: dict[str, Any] | None = None) -> Strategy:
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; available: {sorted(STRATEGIES)}") from None
    return cls(**(params or {}))
