from __future__ import annotations

import pytest

from quantbot.exchange.models import Balance, PairInfo, ShortPosition, Ticker
from quantbot.portfolio import PortfolioSnapshot


def make_pair(pair="BTC/USD", amount_precision=6, price_precision=2, min_notional=1.0):
    return PairInfo(pair=pair, coin=pair.split("/")[0], unit="USD", can_trade=True,
                    price_precision=price_precision, amount_precision=amount_precision,
                    min_notional=min_notional, full_name=pair)


def make_ticker(pair="BTC/USD", price=50_000.0, spread=0.0):
    return Ticker(pair=pair, bid=price - spread / 2, ask=price + spread / 2, last=price)


class FakeClient:
    """In-memory stand-in for RoostooClient that records calls and simulates fills."""

    def __init__(self, usd=100_000.0, prices=None):
        self.prices = prices or {"BTC/USD": 50_000.0, "ETH/USD": 2_500.0}
        self.usd = usd
        self.coins: dict[str, float] = {}
        self.shorts: dict[str, ShortPosition] = {}
        self.calls: list[tuple] = []

    def balance(self):
        free = {"USD": self.usd, **self.coins}
        return Balance(free=free, lock={k: 0.0 for k in free})

    def short_positions(self):
        return list(self.shorts.values())

    def ticker(self, pair=None):
        return {p: make_ticker(p, px) for p, px in self.prices.items() if pair in (None, p)}

    def place_order(self, pair, side, quantity, order_type="MARKET", price=None):
        self.calls.append(("place_order", pair, side, quantity, order_type))
        q, px, coin = float(quantity), self.prices[pair], pair.split("/")[0]
        if side == "BUY":
            self.usd -= q * px * 1.001
            self.coins[coin] = self.coins.get(coin, 0.0) + q
        else:
            self.usd += q * px * 0.999
            self.coins[coin] = self.coins.get(coin, 0.0) - q
        return {"Status": "FILLED", "FilledQuantity": q, "FilledAverPrice": px}

    def short_open(self, pair, collateral, price=None):
        self.calls.append(("short_open", pair, collateral))
        c, px = float(collateral), self.prices[pair]
        self.usd -= c * 1.001
        prev = self.shorts.get(pair)
        qty = c / px + (prev.qty if prev else 0.0)
        coll = c + (prev.collateral if prev else 0.0)
        self.shorts[pair] = ShortPosition(1, pair, px, qty, coll, px, 0.0)
        return {"Success": True, "Status": "OPEN", "ShortQty": qty, "EntryPrice": px}

    def short_close(self, pair, close_qty=None, close_pct=None):
        self.calls.append(("short_close", pair, close_qty))
        pos = self.shorts.pop(pair)
        self.usd += pos.collateral
        return {"Success": True, "ClosedQty": pos.qty, "FullyClosed": True}

    def query_order(self, **kw):
        return []

    def cancel_order(self, **kw):
        return []


@pytest.fixture
def fake_client():
    return FakeClient()


@pytest.fixture
def pair_infos():
    return {"BTC/USD": make_pair("BTC/USD"), "ETH/USD": make_pair("ETH/USD", amount_precision=4)}


def snapshot_from(client: FakeClient, collateral_in_usd_balance=False) -> PortfolioSnapshot:
    from quantbot.portfolio import fetch_snapshot

    return fetch_snapshot(client, None, collateral_in_usd_balance)
