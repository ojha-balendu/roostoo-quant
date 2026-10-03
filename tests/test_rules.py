from conftest import make_pair

from quantbot.exchange.rules import floor_to, fmt_decimal, is_valid_order, round_price


def test_floor_never_rounds_up():
    assert floor_to(0.123456789, 6) == 0.123456
    assert floor_to(1.9999, 0) == 1.0
    assert floor_to(-1, 2) == 0.0


def test_fmt_avoids_scientific_notation():
    assert fmt_decimal(1e-05, 6) == "0.000010"
    assert fmt_decimal(3, 0) == "3"


def test_round_price_and_min_notional():
    assert round_price(123.456, 2) == 123.46
    info = make_pair(min_notional=1.0)
    assert is_valid_order(0.0001, 50_000, info)
    assert not is_valid_order(0.00001, 50_000, info)  # 0.5 USD < 1 USD minimum
