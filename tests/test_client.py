import pytest
import requests

from quantbot.exchange.client import RoostooAPIError, RoostooClient, RoostooError
from quantbot.exchange.signing import sign


class Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code, self.text = payload, status, str(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def _next(self, method, url, **kw):
        self.requests.append((method, url, kw))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def get(self, url, **kw):
        return self._next("GET", url, **kw)

    def post(self, url, **kw):
        return self._next("POST", url, **kw)


def client(responses):
    session = FakeSession(responses)
    c = RoostooClient("https://x", "KEY", "SECRET", session=session, min_interval=0, sleep=lambda s: None)
    return c, session


def test_post_is_signed_form_body():
    c, s = client([Resp({"Success": True, "OrderDetail": {"Status": "FILLED"}})])
    c.place_order("BTC/USD", "buy", "0.010000")
    method, url, kw = s.requests[0]
    assert url == "https://x/v3/place_order"
    body = kw["data"]
    assert body.startswith("pair=BTC/USD&quantity=0.010000&side=BUY&timestamp=")
    assert kw["headers"]["MSG-SIGNATURE"] == sign("SECRET", body)
    assert kw["headers"]["RST-API-KEY"] == "KEY"
    assert kw["headers"]["Content-Type"] == "application/x-www-form-urlencoded"


def test_success_false_raises():
    c, _ = client([Resp({"Success": False, "ErrMsg": "insufficient balance"})])
    with pytest.raises(RoostooAPIError, match="insufficient balance"):
        c.short_open("BTC/USD", "100.00")


def test_benign_failures_are_not_errors():
    c, _ = client([
        Resp({"Success": False, "ErrMsg": "no pending order under this account", "TotalPending": 0}),
        Resp({"Success": False, "ErrMsg": "no order matched"}),
    ])
    assert c.pending_count() == 0
    assert c.query_order(pair="BTC/USD") == []


def test_order_not_retried_after_read_timeout():
    c, s = client([requests.ReadTimeout("slow"), Resp({"Success": True})])
    with pytest.raises(RoostooError):
        c.place_order("BTC/USD", "BUY", "1")
    assert len(s.requests) == 1


def test_reads_are_retried():
    c, s = client([requests.ConnectionError("boom"), Resp({"Success": True, "Positions": []})])
    assert c.short_positions() == []
    assert len(s.requests) == 2


def test_timestamp_rejection_resyncs_and_retries():
    c, s = client([
        Resp({"Success": False, "ErrMsg": "timestamp out of range"}),
        Resp({"ServerTime": 1_700_000_000_000}),
        Resp({"Success": True, "Wallet": {"USD": {"Free": 10, "Lock": 0}}}),
    ])
    assert c.balance().free["USD"] == 10
    assert len(s.requests) == 3


def test_missing_fields_read_as_zero():
    c, _ = client([Resp({"Success": True, "Positions": [{"ID": 1, "Pair": "BTC/USD", "ShortQty": 0.2,
                                                          "Collateral": 10000, "EntryPrice": 50000}]})])
    (pos,) = c.short_positions()
    assert pos.unrealized_pnl == 0 and pos.value == 10000


def test_signed_call_without_keys_fails_fast():
    c = RoostooClient("https://x", session=FakeSession([]), min_interval=0)
    with pytest.raises(RoostooError, match="not set"):
        c.balance()
