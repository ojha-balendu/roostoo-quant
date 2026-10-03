"""Precision / minimum-order helpers based on /v3/exchangeInfo."""

from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal

from quantbot.exchange.models import PairInfo


def floor_to(value: float, decimals: int) -> float:
    """Round DOWN to `decimals` places (never sell/close more than we hold)."""
    if value <= 0:
        return 0.0
    quantum = Decimal(1).scaleb(-decimals)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_DOWN))


def round_price(value: float, decimals: int) -> float:
    quantum = Decimal(1).scaleb(-decimals)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def fmt_decimal(value: float, decimals: int) -> str:
    """Fixed-point string. str(1e-05) == '1e-05' would be rejected by the exchange."""
    return f"{value:.{max(decimals, 0)}f}"


def is_valid_order(quantity: float, price: float, info: PairInfo) -> bool:
    return quantity > 0 and price > 0 and quantity * price > info.min_notional
