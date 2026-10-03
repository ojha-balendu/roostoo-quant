"""Turns intents into Roostoo orders, with exchange rules, cash checks and a dry-run mode."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from quantbot.config import ExecutionConfig, FeeConfig
from quantbot.exchange.client import RoostooAPIError, RoostooClient, RoostooError
from quantbot.exchange.models import PairInfo, Ticker
from quantbot.exchange.rules import floor_to, fmt_decimal, is_valid_order, round_price
from quantbot.execution.decision import Action, Intent, plan
from quantbot.portfolio import PortfolioSnapshot

log = logging.getLogger(__name__)

USD_DECIMALS = 2


@dataclass
class ExecResult:
    intent: Intent
    status: str  # filled | pending | sent | dry_run | skipped | failed
    detail: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class Executor:
    def __init__(
        self,
        client: RoostooClient,
        pair_infos: dict[str, PairInfo],
        cfg: ExecutionConfig,
        fees: FeeConfig,
        *,
        dry_run: bool = True,
        store=None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.client = client
        self.pairs = pair_infos
        self.cfg = cfg
        self.fees = fees
        self.dry_run = dry_run
        self.store = store
        self._sleep = sleep

    def update_pairs(self, pair_infos: dict[str, PairInfo]) -> None:
        self.pairs = pair_infos

    # ---------------------------------------------------------------- public
    def rebalance(
        self,
        targets: dict[str, float],
        snapshot: PortfolioSnapshot,
        refresh: Callable[[], PortfolioSnapshot] | None = None,
    ) -> list[ExecResult]:
        intents = plan(
            targets,
            snapshot,
            min_trade_usd=self.cfg.min_trade_usd,
            min_close_usd=self.cfg.min_close_usd,
            threshold=self.cfg.rebalance_threshold,
        )
        if not intents:
            log.info("rebalance: portfolio already at target")
            return []
        reducing = [i for i in intents if i.is_reducing]
        increasing = [i for i in intents if not i.is_reducing]
        results = [self._execute(i, snapshot) for i in reducing]
        if increasing:
            if reducing and refresh is not None and not self.dry_run:
                try:
                    snapshot = refresh()
                except RoostooError:
                    log.exception("could not refresh balances after reductions; skipping increases this cycle")
                    return results
            for intent in self._fit_to_cash(increasing, snapshot, reducing):
                results.append(self._execute(intent, snapshot))
        return results

    # ------------------------------------------------------------- internals
    def _fee_rate(self, intent: Intent) -> float:
        if intent.action is Action.SHORT:
            return self.fees.short_open
        if self.cfg.long_order_mode == "passive_limit":
            return self.fees.maker
        return self.fees.taker

    def _fit_to_cash(self, intents: list[Intent], snap: PortfolioSnapshot, reducing: list[Intent]) -> list[Intent]:
        budget = snap.usd_free - self.cfg.cash_buffer * snap.equity
        if self.dry_run:
            budget += sum(i.notional for i in reducing)  # reductions were not actually executed
        need = sum(i.notional * (1 + self._fee_rate(i)) for i in intents)
        if need <= 0:
            return intents
        if budget <= 0:
            log.warning("no free USD for %d increasing orders (budget %.2f)", len(intents), budget)
            return []
        scale = min(1.0, budget / need)
        if scale < 1.0:
            log.warning("scaling increasing orders by %.3f to fit free USD %.2f", scale, budget)
        out = []
        for intent in intents:
            notional = intent.notional * scale
            if notional >= self.cfg.min_trade_usd:
                out.append(replace(intent, notional=notional))
        return out

    def _execute(self, intent: Intent, snap: PortfolioSnapshot) -> ExecResult:
        info = self.pairs.get(intent.pair)
        tick = snap.tickers.get(intent.pair)
        if info is None or not info.can_trade:
            result = ExecResult(intent, "skipped", error="pair not tradable")
        elif tick is None:
            result = ExecResult(intent, "skipped", error="no ticker")
        else:
            try:
                if intent.action is Action.LONG:
                    result = self._buy_long(intent, info, tick)
                elif intent.action is Action.SELL:
                    result = self._sell_long(intent, info, tick, snap)
                elif intent.action is Action.SHORT:
                    result = self._open_short(intent, info, tick)
                else:
                    result = self._close_short(intent, info, snap)
            except RoostooError as exc:
                log.error("%s %s failed: %s", intent.action.value, intent.pair, exc)
                result = ExecResult(intent, "failed", error=str(exc))
        if result.status == "skipped":
            log.info("skip %s %s: %s", intent.action.value, intent.pair, result.error)
        self._record(result)
        if result.status not in ("skipped", "dry_run"):
            self._sleep(self.cfg.pause_between_orders_seconds)
        return result

    def _dry(self, intent: Intent, **detail: Any) -> ExecResult:
        log.info("[DRY-RUN] %s %s %s (%s)", intent.action.value, intent.pair, detail, intent.reason)
        return ExecResult(intent, "dry_run", detail=detail)

    # LONG -----------------------------------------------------------------
    def _buy_long(self, intent: Intent, info: PairInfo, tick: Ticker) -> ExecResult:
        price = tick.ask if tick.ask > 0 else tick.mid
        qty = floor_to(intent.notional / price, info.amount_precision)
        if not is_valid_order(qty, price, info):
            return ExecResult(intent, "skipped", error=f"qty {qty} below exchange minimum")
        return self._spot(intent, info, tick, "BUY", qty)

    # SELL -----------------------------------------------------------------
    def _sell_long(self, intent: Intent, info: PairInfo, tick: Ticker, snap: PortfolioSnapshot) -> ExecResult:
        # Use the actual free balance: buy commissions may be charged in the coin,
        # so we can hold slightly less than we bought.
        free = snap.longs_free.get(intent.pair, 0.0)
        qty = free if intent.full else min(intent.quantity or 0.0, free)
        qty = floor_to(qty, info.amount_precision)
        price = tick.bid if tick.bid > 0 else tick.mid
        if not is_valid_order(qty, price, info):
            return ExecResult(intent, "skipped", error=f"qty {qty} below exchange minimum")
        return self._spot(intent, info, tick, "SELL", qty)

    def _spot(self, intent: Intent, info: PairInfo, tick: Ticker, side: str, qty: float) -> ExecResult:
        qty_s = fmt_decimal(qty, info.amount_precision)
        if self.dry_run:
            return self._dry(intent, side=side, quantity=qty_s, mode=self.cfg.long_order_mode)
        if self.cfg.long_order_mode == "passive_limit":
            return self._passive_then_market(intent, info, tick, side, qty)
        detail = self.client.place_order(intent.pair, side, qty_s, "MARKET")
        log.info("%s %s %s qty=%s -> %s @ %s", intent.action.value, intent.pair, side, qty_s,
                 detail.get("Status"), detail.get("FilledAverPrice"))
        return ExecResult(intent, str(detail.get("Status") or "sent").lower(), detail)

    def _passive_then_market(self, intent: Intent, info: PairInfo, tick: Ticker, side: str, qty: float) -> ExecResult:
        """Rest a LIMIT at the touch (maker fee); after the timeout cancel and send the rest at market."""
        price = tick.bid if side == "BUY" else tick.ask
        price_s = fmt_decimal(round_price(price, info.price_precision), info.price_precision)
        detail = self.client.place_order(intent.pair, side, fmt_decimal(qty, info.amount_precision), "LIMIT", price_s)
        order_id = detail.get("OrderID")
        status = str(detail.get("Status") or "")
        deadline = time.monotonic() + self.cfg.limit_timeout_seconds
        while status != "FILLED" and order_id is not None and time.monotonic() < deadline:
            self._sleep(2.0)
            orders = self.client.query_order(order_id=order_id)
            if orders:
                detail, status = orders[0], str(orders[0].get("Status") or "")
        if status == "FILLED":
            return ExecResult(intent, "filled", detail)
        try:
            self.client.cancel_order(order_id=order_id)
        except RoostooAPIError as exc:  # it may have filled in the meantime
            log.info("cancel %s: %s", order_id, exc)
        orders = self.client.query_order(order_id=order_id)
        if orders:
            detail, status = orders[0], str(orders[0].get("Status") or "")
        if status == "FILLED":
            return ExecResult(intent, "filled", detail)
        # Only trust FilledQuantity when an average fill price is reported.
        filled = float(detail.get("FilledQuantity") or 0) if float(detail.get("FilledAverPrice") or 0) > 0 else 0.0
        remaining = floor_to(qty - filled, info.amount_precision)
        if not is_valid_order(remaining, price, info):
            return ExecResult(intent, "filled", detail)
        log.info("limit %s %s not filled in %ss; sending %s at market", side, intent.pair,
                 self.cfg.limit_timeout_seconds, remaining)
        detail = self.client.place_order(intent.pair, side, fmt_decimal(remaining, info.amount_precision), "MARKET")
        return ExecResult(intent, str(detail.get("Status") or "sent").lower(), detail)

    # SHORT ----------------------------------------------------------------
    def _open_short(self, intent: Intent, info: PairInfo, tick: Ticker) -> ExecResult:
        collateral = floor_to(intent.notional, USD_DECIMALS)
        price = tick.bid if tick.bid > 0 else tick.mid  # market shorts fill at MaxBid
        if collateral < max(1.0, info.min_notional):
            return ExecResult(intent, "skipped", error=f"collateral {collateral} below minimum")
        if floor_to(collateral / price, info.amount_precision) <= 0:
            return ExecResult(intent, "skipped", error="collateral too small for amount precision")
        collateral_s = fmt_decimal(collateral, USD_DECIMALS)
        if self.dry_run:
            return self._dry(intent, collateral=collateral_s)
        detail = self.client.short_open(intent.pair, collateral_s)
        log.info("SHORT %s collateral=%s -> %s qty=%s @ %s", intent.pair, collateral_s,
                 detail.get("Status"), detail.get("ShortQty"), detail.get("EntryPrice"))
        return ExecResult(intent, str(detail.get("Status") or "sent").lower(), detail)

    # CLOSE ----------------------------------------------------------------
    def _close_short(self, intent: Intent, info: PairInfo, snap: PortfolioSnapshot) -> ExecResult:
        pos = snap.shorts.get(intent.pair)
        if pos is None:
            return ExecResult(intent, "skipped", error="no open short")
        close_qty: str | None = None
        if not intent.full:
            qty = floor_to(min(intent.quantity or 0.0, pos.qty), info.amount_precision)
            if qty <= 0:
                return ExecResult(intent, "skipped", error="close qty rounds to zero")
            close_qty = fmt_decimal(qty, info.amount_precision)
        if self.dry_run:
            return self._dry(intent, close_qty=close_qty or "ALL")
        detail = self.client.short_close(intent.pair, close_qty=close_qty)
        log.info("CLOSE %s qty=%s -> pnl=%s fee=%s fully_closed=%s", intent.pair, detail.get("ClosedQty"),
                 detail.get("RealizedPNL"), detail.get("CloseFee"), detail.get("FullyClosed"))
        return ExecResult(intent, "filled", detail)

    def _record(self, result: ExecResult) -> None:
        if self.store is None:
            return
        try:
            self.store.record_order(
                ts_ms=int(time.time() * 1000),
                action=result.intent.action.value,
                pair=result.intent.pair,
                notional=result.intent.notional,
                quantity=result.intent.quantity,
                status=result.status,
                dry_run=self.dry_run,
                detail=json.dumps(result.detail, default=str),
                error=result.error,
            )
        except Exception:  # never let bookkeeping break trading
            log.exception("failed to record order")
