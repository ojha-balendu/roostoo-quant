"""Portfolio snapshot built from exchange state (the exchange is the source of truth)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from quantbot.exchange.client import RoostooClient
from quantbot.exchange.models import ShortPosition, Ticker

log = logging.getLogger(__name__)
QUOTE = "USD"


@dataclass
class PortfolioSnapshot:
    ts_ms: int
    usd_free: float
    usd_lock: float
    longs: dict[str, float] = field(default_factory=dict)       # pair -> coin qty (free + locked)
    longs_free: dict[str, float] = field(default_factory=dict)  # pair -> sellable coin qty
    shorts: dict[str, ShortPosition] = field(default_factory=dict)
    tickers: dict[str, Ticker] = field(default_factory=dict)
    collateral_in_usd_balance: bool = False

    def price(self, pair: str) -> float:
        t = self.tickers.get(pair)
        return t.mid if t else 0.0

    def _bid(self, pair: str) -> float:
        t = self.tickers.get(pair)
        if t is None:
            return 0.0
        return t.bid if t.bid > 0 else t.mid

    def long_qty(self, pair: str) -> float:
        return self.longs.get(pair, 0.0)

    def short_qty(self, pair: str) -> float:
        pos = self.shorts.get(pair)
        return pos.qty if pos else 0.0

    @property
    def long_value(self) -> float:
        """Longs marked at the bid (what selling would realise, before fees)."""
        return sum(q * self._bid(p) for p, q in self.longs.items())

    @property
    def short_value(self) -> float:
        """Contribution of shorts to equity, avoiding double-counting locked collateral."""
        if self.collateral_in_usd_balance:
            return sum(pos.unrealized_pnl for pos in self.shorts.values())
        return sum(pos.value for pos in self.shorts.values())

    @property
    def usd_total(self) -> float:
        return self.usd_free + self.usd_lock

    @property
    def equity(self) -> float:
        return self.usd_total + self.long_value + self.short_value

    def net_notional(self, pair: str) -> float:
        return (self.long_qty(pair) - self.short_qty(pair)) * self.price(pair)

    def weights(self) -> dict[str, float]:
        eq = self.equity
        if eq <= 0:
            return {}
        return {p: self.net_notional(p) / eq for p in self.pairs_with_positions()}

    def gross_exposure(self) -> float:
        eq = self.equity
        if eq <= 0:
            return 0.0
        gross = sum(abs(self.long_qty(p)) * self.price(p) for p in self.longs)
        gross += sum(pos.qty * self.price(p) for p, pos in self.shorts.items())
        return gross / eq

    def pairs_with_positions(self) -> set[str]:
        return {p for p, q in self.longs.items() if q > 0} | {p for p, s in self.shorts.items() if s.qty > 0}

    def summary(self) -> dict[str, float]:
        return {
            "equity": round(self.equity, 2),
            "usd_free": round(self.usd_free, 2),
            "usd_lock": round(self.usd_lock, 2),
            "long_value": round(self.long_value, 2),
            "short_value": round(self.short_value, 2),
            "gross": round(self.gross_exposure(), 4),
            "n_long": sum(1 for q in self.longs.values() if q > 0),
            "n_short": len(self.shorts),
        }


def fetch_snapshot(
    client: RoostooClient,
    tickers: dict[str, Ticker] | None = None,
    collateral_in_usd_balance: bool = False,
) -> PortfolioSnapshot:
    balance = client.balance()
    shorts = {s.pair: s for s in client.short_positions() if s.qty > 0}
    if tickers is None:
        tickers = client.ticker()
    longs: dict[str, float] = {}
    longs_free: dict[str, float] = {}
    for coin in balance.coins:
        if coin == QUOTE:
            continue
        qty = balance.total(coin)
        if qty <= 0:
            continue
        pair = f"{coin}/{QUOTE}"
        if pair not in tickers:
            log.warning("holding %s %s but no ticker for %s; valued at 0", qty, coin, pair)
        longs[pair] = qty
        longs_free[pair] = balance.free.get(coin, 0.0)
    return PortfolioSnapshot(
        ts_ms=int(time.time() * 1000),
        usd_free=balance.free.get(QUOTE, 0.0),
        usd_lock=balance.lock.get(QUOTE, 0.0),
        longs=longs,
        longs_free=longs_free,
        shorts=shorts,
        tickers=tickers,
        collateral_in_usd_balance=collateral_in_usd_balance,
    )
