from __future__ import annotations

from typing import Any

from quantbot.strategy.base import Strategy
from quantbot.strategy.flat import Flat
from quantbot.strategy.xs_momentum import CrossSectionalMomentum

# Register new strategies here.
STRATEGIES: dict[str, type[Strategy]] = {
    Flat.name: Flat,
    CrossSectionalMomentum.name: CrossSectionalMomentum,
}


def get_strategy(name: str, params: dict[str, Any] | None = None) -> Strategy:
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; available: {sorted(STRATEGIES)}") from None
    return cls(**(params or {}))
