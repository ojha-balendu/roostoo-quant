"""Pure translation of target weights into the four competition actions.

    LONG  -> POST /v3/place_order side=BUY      (open / increase a long)
    SELL  -> POST /v3/place_order side=SELL     (reduce / close a long)
    SHORT -> POST /v6/short_open                (open / increase a short)
    CLOSE -> POST /v6/short_close               (reduce / close a short)

Shared by the live executor and the backtester so both trade identically.
A pair never holds a long and a short at the same time: flipping sides first
fully exits the old side, then builds the new one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    LONG = "LONG"
    SELL = "SELL"
    SHORT = "SHORT"
    CLOSE = "CLOSE"


@dataclass(frozen=True)
class Intent:
    action: Action
    pair: str
    notional: float                # USD estimate at decision price
    quantity: float | None = None  # coin quantity for SELL / CLOSE
    full: bool = False             # exit the entire position on that side
    reason: str = ""

    @property
    def is_reducing(self) -> bool:
        return self.action in (Action.SELL, Action.CLOSE)


def decide(
    pair: str,
    long_qty: float,
    short_qty: float,
    price: float,
    equity: float,
    target_weight: float,
    *,
    min_trade_usd: float,
    min_close_usd: float,
    threshold: float,
) -> list[Intent]:
    if price <= 0 or equity <= 0:
        return []
    long_n = long_qty * price
    short_n = short_qty * price
    current_w = (long_n - short_n) / equity
    flat_target = abs(target_weight) < 1e-9
    if not flat_target and abs(target_weight - current_w) < threshold:
        return []

    intents: list[Intent] = []
    long_after, short_after = long_n, short_n
    if target_weight <= 0 and long_n >= min_close_usd:
        intents.append(Intent(Action.SELL, pair, long_n, long_qty, full=True, reason="exit long"))
        long_after = 0.0
    if target_weight >= 0 and short_n >= min_close_usd:
        intents.append(Intent(Action.CLOSE, pair, short_n, short_qty, full=True, reason="exit short"))
        short_after = 0.0

    target_n = abs(target_weight) * equity
    if target_weight > 0:
        delta = target_n - long_after
        if delta >= min_trade_usd:
            intents.append(Intent(Action.LONG, pair, delta, reason="increase long"))
        elif delta <= -min_trade_usd:
            qty = min(-delta / price, long_qty)
            intents.append(Intent(Action.SELL, pair, qty * price, qty, reason="reduce long"))
    elif target_weight < 0:
        delta = target_n - short_after
        if delta >= min_trade_usd:
            intents.append(Intent(Action.SHORT, pair, delta, reason="increase short"))
        elif delta <= -min_trade_usd:
            qty = min(-delta / price, short_qty)
            intents.append(Intent(Action.CLOSE, pair, qty * price, qty, reason="reduce short"))
    return intents


def plan(
    targets: dict[str, float],
    snapshot,  # PortfolioSnapshot (duck-typed to keep this module dependency-free)
    *,
    min_trade_usd: float,
    min_close_usd: float,
    threshold: float,
) -> list[Intent]:
    """Intents for every pair that has a target or a position. Pairs without a target are flattened.
    Reducing intents come first so they free cash for the increasing ones."""
    equity = snapshot.equity
    intents: list[Intent] = []
    for pair in sorted(set(targets) | snapshot.pairs_with_positions()):
        price = snapshot.price(pair)
        if price <= 0:
            continue
        intents += decide(
            pair,
            snapshot.long_qty(pair),
            snapshot.short_qty(pair),
            price,
            equity,
            float(targets.get(pair, 0.0)),
            min_trade_usd=min_trade_usd,
            min_close_usd=min_close_usd,
            threshold=threshold,
        )
    intents.sort(key=lambda i: 0 if i.is_reducing else 1)
    return intents
