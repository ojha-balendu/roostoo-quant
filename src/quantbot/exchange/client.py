"""Defensive client for the Roostoo mock-exchange REST API.

Docs: https://github.com/roostoo/Roostoo-API-Documents

Design points:
* Failed requests come back as HTTP 200 with Success=false, so every response is checked.
* Order-placing endpoints are NOT retried after the request may have reached the server
  (a lost response could otherwise double an order). The executor reconciles from
  exchange state on the next cycle instead, so a lost response self-heals.
* Signed requests carry a timestamp that must be within 60s of server time; the client
  keeps an offset to server time and resyncs if a timestamp is rejected.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import requests

from quantbot.exchange.models import Balance, PairInfo, ShortPosition, Ticker
from quantbot.exchange.signing import build_total_params, sign

log = logging.getLogger(__name__)


class RoostooError(Exception):
    """Transport-level or unexpected failure talking to Roostoo."""


class RoostooAPIError(RoostooError):
    """Roostoo answered with Success=false."""

    def __init__(self, path: str, message: str, payload: dict | None = None):
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message
        self.payload = payload or {}


class RoostooClient:
    def __init__(
        self,
        base_url: str = "https://mock-api.roostoo.com",
        api_key: str = "",
        secret_key: str = "",
        *,
        timeout: float = 10.0,
        max_retries: int = 3,
        min_interval: float = 0.2,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.secret_key = secret_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval = min_interval
        self.session = session or requests.Session()
        self._sleep = sleep
        self._offset_ms = 0
        self._last_call = 0.0
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: Any) -> RoostooClient:
        return cls(
            cfg.base_url,
            cfg.api_key,
            cfg.secret_key,
            timeout=cfg.timeout_seconds,
            max_retries=cfg.max_retries,
            min_interval=cfg.min_request_interval_seconds,
        )

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.secret_key)

    # ------------------------------------------------------------------ time
    def timestamp(self) -> str:
        return str(int(time.time() * 1000) + self._offset_ms)

    def now_ms(self) -> int:
        return int(self.timestamp())

    def sync_time(self) -> int:
        """Measure local->server clock offset (ms). Call at startup and periodically."""
        t0 = time.time()
        server = self.server_time()
        t1 = time.time()
        self._offset_ms = int(server - (t0 + t1) / 2 * 1000)
        if abs(self._offset_ms) > 2000:
            log.warning("local clock is %+d ms off Roostoo server time (check chrony/NTP)", self._offset_ms)
        return self._offset_ms

    # --------------------------------------------------------------- plumbing
    def _throttle(self) -> None:
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                self._sleep(wait)
            self._last_call = time.monotonic()

    def _backoff(self, attempt: int) -> None:
        self._sleep(min(10.0, 0.5 * 2 ** (attempt - 1)))

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any],
        *,
        signed: bool,
        timestamp: bool,
        idempotent: bool,
    ) -> dict[str, Any]:
        url = self.base_url + path
        attempt = 0
        while True:
            attempt += 1
            payload = {k: v for k, v in params.items() if v is not None}
            if timestamp:
                payload["timestamp"] = self.timestamp()
            total = build_total_params(payload)
            headers: dict[str, str] = {}
            if signed:
                if not self.has_credentials:
                    raise RoostooError("ROOSTOO_API_KEY / ROOSTOO_SECRET_KEY are not set")
                headers["RST-API-KEY"] = self.api_key
                headers["MSG-SIGNATURE"] = sign(self.secret_key, total)
            self._throttle()
            try:
                if method == "GET":
                    resp = self.session.get(url, params=payload, headers=headers, timeout=self.timeout)
                else:
                    headers["Content-Type"] = "application/x-www-form-urlencoded"
                    resp = self.session.post(url, data=total, headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:
                # A connect timeout means the request never reached the server -> always safe to retry.
                safe = idempotent or isinstance(exc, requests.ConnectTimeout)
                if safe and attempt <= self.max_retries:
                    log.warning("%s %s failed (%s); retry %d/%d", method, path, exc, attempt, self.max_retries)
                    self._backoff(attempt)
                    continue
                raise RoostooError(f"{method} {path} failed: {exc}") from exc

            retryable_status = resp.status_code == 429 or (resp.status_code >= 500 and idempotent)
            if retryable_status and attempt <= self.max_retries:
                log.warning("%s %s -> HTTP %d; retry %d/%d", method, path, resp.status_code, attempt, self.max_retries)
                self._backoff(attempt)
                continue
            if resp.status_code >= 400:
                raise RoostooError(f"{method} {path} -> HTTP {resp.status_code}: {resp.text[:300]}")
            try:
                data = resp.json()
            except ValueError as exc:
                raise RoostooError(f"{method} {path}: non-JSON response: {resp.text[:300]}") from exc
            if not isinstance(data, dict):
                raise RoostooError(f"{method} {path}: unexpected response type {type(data).__name__}")
            return data

    def _call(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        signed: bool = True,
        timestamp: bool = True,
        idempotent: bool = True,
        ok_if: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, Any]:
        params = params or {}
        for attempt in (1, 2):
            data = self._request(method, path, params, signed=signed, timestamp=timestamp, idempotent=idempotent)
            if data.get("Success", True) is not False:
                return data
            if ok_if is not None and ok_if(data):
                return data
            msg = str(data.get("ErrMsg") or "unknown error")
            # A rejected timestamp means the request was not processed, so a retry is safe.
            if attempt == 1 and "timestamp" in msg.lower():
                log.warning("%s: timestamp rejected (%s); resyncing clock", path, msg)
                self.sync_time()
                continue
            raise RoostooAPIError(path, msg, data)
        raise AssertionError("unreachable")

    # ------------------------------------------------------- public endpoints
    def server_time(self) -> int:
        data = self._request("GET", "/v3/serverTime", {}, signed=False, timestamp=False, idempotent=True)
        return int(data["ServerTime"])

    def exchange_info(self) -> dict[str, Any]:
        return self._request("GET", "/v3/exchangeInfo", {}, signed=False, timestamp=False, idempotent=True)

    def pairs(self) -> dict[str, PairInfo]:
        info = self.exchange_info()
        return {p: PairInfo.from_api(p, d) for p, d in (info.get("TradePairs") or {}).items()}

    def ticker(self, pair: str | None = None) -> dict[str, Ticker]:
        data = self._call("GET", "/v3/ticker", {"pair": pair}, signed=False)
        server_ts = int(data.get("ServerTime") or 0)
        return {p: Ticker.from_api(p, d, server_ts) for p, d in (data.get("Data") or {}).items()}

    # ------------------------------------------------------- signed endpoints
    def balance(self) -> Balance:
        return Balance.from_api(self._call("GET", "/v3/balance"))

    def pending_count(self) -> int:
        # "no pending order" is reported as Success=false with TotalPending=0.
        data = self._call("GET", "/v3/pending_count", ok_if=lambda d: "TotalPending" in d)
        return int(data.get("TotalPending") or 0)

    def place_order(
        self,
        pair: str,
        side: str,
        quantity: str,
        order_type: str = "MARKET",
        price: str | None = None,
    ) -> dict[str, Any]:
        """Spot order. BUY opens/increases a long, SELL reduces/closes it."""
        side, order_type = side.upper(), order_type.upper()
        if side not in ("BUY", "SELL"):
            raise ValueError(f"side must be BUY or SELL, got {side!r}")
        if order_type not in ("MARKET", "LIMIT"):
            raise ValueError(f"order_type must be MARKET or LIMIT, got {order_type!r}")
        if order_type == "LIMIT" and price is None:
            raise ValueError("LIMIT orders require a price")
        params = {"pair": pair, "side": side, "type": order_type, "quantity": quantity}
        if order_type == "LIMIT":
            params["price"] = price
        data = self._call("POST", "/v3/place_order", params, idempotent=False)
        return data.get("OrderDetail") or {}

    def query_order(
        self,
        order_id: int | str | None = None,
        pair: str | None = None,
        pending_only: bool | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if order_id is not None:
            params: dict[str, Any] = {"order_id": str(order_id)}  # no other params allowed with order_id
        else:
            params = {
                "pair": pair,
                "pending_only": None if pending_only is None else ("TRUE" if pending_only else "FALSE"),
                "offset": None if offset is None else str(offset),
                "limit": None if limit is None else str(limit),
            }
        data = self._call(
            "POST",
            "/v3/query_order",
            params,
            ok_if=lambda d: "no order" in str(d.get("ErrMsg", "")).lower(),
        )
        return list(data.get("OrderMatched") or [])

    def cancel_order(self, order_id: int | str | None = None, pair: str | None = None) -> list[int]:
        """Cancel by id, by pair, or (both None) ALL pending orders."""
        if order_id is not None and pair is not None:
            raise ValueError("send order_id or pair, not both")
        params = {"order_id": None if order_id is None else str(order_id), "pair": pair}
        data = self._call("POST", "/v3/cancel_order", params)
        return [int(x) for x in data.get("CanceledList") or []]

    def short_open(self, pair: str, collateral: str, price: str | None = None) -> dict[str, Any]:
        """Open/increase a short sized by USD collateral (qty = collateral / EntryPrice)."""
        params: dict[str, Any] = {"pair": pair, "collateral": collateral}
        if price is not None:
            params["order_type"] = "LIMIT"
            params["price"] = price
        return self._call("POST", "/v6/short_open", params, idempotent=False)

    def short_close(self, pair: str, close_qty: str | None = None, close_pct: str | None = None) -> dict[str, Any]:
        """Reduce-only close at MinAsk. No size -> closes the whole position."""
        if close_qty is not None and close_pct is not None:
            raise ValueError("send close_qty or close_pct, not both")
        params = {"pair": pair, "close_qty": close_qty, "close_pct": close_pct}
        return self._call("POST", "/v6/short_close", params, idempotent=False)

    def short_positions(self) -> list[ShortPosition]:
        data = self._call("GET", "/v6/short_positions")
        return [ShortPosition.from_api(p) for p in data.get("Positions") or []]
