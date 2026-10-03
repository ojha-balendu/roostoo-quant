"""Typed views over Roostoo responses.

Roostoo omits fields whose value is zero (e.g. OpenFee, RemainingQty), so every
numeric field is read with a default of 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def _num(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = data.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class PairInfo:
    pair: str
    coin: str
    unit: str
    can_trade: bool
    price_precision: int
    amount_precision: int
    min_notional: float  # "MiniOrder": price * quantity must exceed this
    full_name: str = ""

    @classmethod
    def from_api(cls, pair: str, data: dict[str, Any]) -> PairInfo:
        coin, _, unit = pair.partition("/")
        return cls(
            pair=pair,
            coin=str(data.get("Coin") or coin),
            unit=str(data.get("Unit") or unit or "USD"),
            can_trade=bool(data.get("CanTrade", True)),
            price_precision=int(_num(data, "PricePrecision")),
            amount_precision=int(_num(data, "AmountPrecision")),
            min_notional=_num(data, "MiniOrder"),
            full_name=str(data.get("CoinFullName") or ""),
        )


@dataclass(frozen=True)
class Ticker:
    pair: str
    bid: float
    ask: float
    last: float
    change_24h: float = 0.0
    quote_volume: float = 0.0
    server_ts: int = 0

    @classmethod
    def from_api(cls, pair: str, data: dict[str, Any], server_ts: int = 0) -> Ticker:
        return cls(
            pair=pair,
            bid=_num(data, "MaxBid"),
            ask=_num(data, "MinAsk"),
            last=_num(data, "LastPrice"),
            change_24h=_num(data, "Change"),
            quote_volume=_num(data, "UnitTradeValue"),
            server_ts=server_ts,
        )

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread_bps(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.ask - self.bid) / self.mid * 1e4
        return math.nan


@dataclass(frozen=True)
class ShortPosition:
    id: int
    pair: str
    entry_price: float
    qty: float
    collateral: float
    current_price: float
    unrealized_pnl: float
    create_ts: int = 0

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> ShortPosition:
        return cls(
            id=int(_num(data, "ID")),
            pair=str(data.get("Pair", "")),
            entry_price=_num(data, "EntryPrice"),
            qty=_num(data, "ShortQty"),
            collateral=_num(data, "Collateral"),
            current_price=_num(data, "CurrentPrice"),
            unrealized_pnl=_num(data, "UnrealizedPNL"),
            create_ts=int(_num(data, "CreateTimestamp")),
        )

    @property
    def value(self) -> float:
        """What a full close returns before the close fee (the API's PositionValue)."""
        return self.collateral + self.unrealized_pnl


@dataclass
class Balance:
    free: dict[str, float] = field(default_factory=dict)
    lock: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Balance:
        # Documented key is "Wallet"; "SpotWallet" is accepted defensively.
        wallet = data.get("Wallet") or data.get("SpotWallet") or {}
        free = {coin: _num(v, "Free") for coin, v in wallet.items()}
        lock = {coin: _num(v, "Lock") for coin, v in wallet.items()}
        return cls(free=free, lock=lock)

    def total(self, coin: str) -> float:
        return self.free.get(coin, 0.0) + self.lock.get(coin, 0.0)

    @property
    def coins(self) -> set[str]:
        return set(self.free) | set(self.lock)
