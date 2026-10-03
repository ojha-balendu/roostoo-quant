from quantbot.exchange.signing import build_total_params, sign


def test_signature_matches_roostoo_docs_example():
    params = {"type": "MARKET", "side": "BUY", "timestamp": "1580774512000", "pair": "BNB/USD", "quantity": "2000"}
    total = build_total_params(params)
    assert total == "pair=BNB/USD&quantity=2000&side=BUY&timestamp=1580774512000&type=MARKET"
    secret = "S1XP1e3UZj6A7H5fATj0jNhqPxxdSJYdInClVN65XAbvqqMKjVHjA7PZj4W12oep"
    assert sign(secret, total) == "20b7fd5550b67b3bf0c1684ed0f04885261db8fdabd38611e9e6af23c19b7fff"
