"""The autonomous trading loop.

Every `poll_seconds`: one /v3/ticker call (all pairs) -> record ticks -> build bars.
On each closed bar (every `rebalance_every_bars`): snapshot account -> drawdown check ->
strategy targets -> risk clipping -> executor reconciles exchange state toward targets.

Robustness:
* No local position state: every rebalance starts from what the exchange reports, so a
  crash, restart or lost order response heals itself on the next cycle.
* Exceptions are logged and retried with backoff; the loop never exits on its own.
  systemd (Restart=always) covers anything worse.
* SIGTERM/SIGINT stop the loop cleanly between cycles; positions are left as they are.
"""

from __future__ import annotations

import logging
import signal
import time

import pandas as pd

from quantbot.config import Config
from quantbot.exchange.client import RoostooClient
from quantbot.exchange.models import Ticker
from quantbot.execution.executor import Executor
from quantbot.live.bars import BarBuilder, parse_interval
from quantbot.live.store import StateStore
from quantbot.live.warmup import warmup_close_panel
from quantbot.portfolio import PortfolioSnapshot, fetch_snapshot
from quantbot.risk import RiskManager
from quantbot.strategy.base import Strategy

log = logging.getLogger(__name__)

TIME_SYNC_EVERY_S = 30 * 60
EXCHANGE_INFO_EVERY_S = 6 * 3600


class LiveRunner:
    def __init__(self, cfg: Config, client: RoostooClient, strategy: Strategy,
                 universe: dict[str, str], store: StateStore):
        self.cfg = cfg
        self.client = client
        self.strategy = strategy
        self.universe = universe
        self.store = store
        self.interval = parse_interval(cfg.live.bar_interval)
        self.bars = BarBuilder(cfg.live.bar_interval)
        self.risk = RiskManager(cfg.risk, store)
        self.closes = pd.DataFrame()
        self.pair_infos: dict = {}
        self.executor: Executor | None = None
        self._stop = False
        self._bars_since_rebalance = 0
        self._next_time_sync = 0.0
        self._next_info_refresh = 0.0
        self._next_heartbeat = 0.0

    # ------------------------------------------------------------ lifecycle
    def stop(self, *_args) -> None:
        log.info("stop requested; finishing current cycle")
        self._stop = True

    def _sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while not self._stop and time.monotonic() < end:
            time.sleep(min(1.0, end - time.monotonic()))

    @property
    def tradable(self) -> set[str]:
        return {p for p, i in self.pair_infos.items() if i.can_trade}

    def _refresh_exchange_info(self) -> None:
        self.pair_infos = self.client.pairs()
        if self.executor is None:
            self.executor = Executor(self.client, self.pair_infos, self.cfg.execution, self.cfg.fees,
                                     dry_run=self.cfg.live.dry_run, store=self.store)
        else:
            self.executor.update_pairs(self.pair_infos)
        log.info("exchange info: %d pairs, %d tradable", len(self.pair_infos), len(self.tradable))

    def _maintenance(self, now: float) -> None:
        if now >= self._next_time_sync:
            offset = self.client.sync_time()
            log.debug("clock offset to server: %+d ms", offset)
            self._next_time_sync = now + TIME_SYNC_EVERY_S
        if now >= self._next_info_refresh:
            self._refresh_exchange_info()
            self._next_info_refresh = now + EXCHANGE_INFO_EVERY_S

    def _snapshot(self, tickers: dict[str, Ticker] | None = None) -> PortfolioSnapshot:
        return fetch_snapshot(self.client, tickers, self.cfg.accounting.short_collateral_in_usd_balance)

    def _warmup(self) -> None:
        n = max(self.cfg.live.warmup_bars, self.strategy.warmup_bars + 5)
        since_ms = int((pd.Timestamp.now(tz="UTC") - n * self.interval).timestamp() * 1000)
        local = self.store.load_close_panel(since_ms)
        remote = pd.DataFrame()
        if self.cfg.live.warmup_source == "binance":
            mapping = {p: s for p, s in self.universe.items() if p in self.tradable}
            if mapping:
                log.info("warm-up: fetching %d bars x %d pairs from Binance", n, len(mapping))
                try:
                    remote = warmup_close_panel(mapping, self.cfg.live.bar_interval, n)
                except Exception:
                    log.exception("warm-up from Binance failed; continuing with local bars only")
            else:
                log.warning("warm-up: universe is empty; run `quantbot universe`")
        if local.empty:
            closes = remote
        elif remote.empty:
            closes = local
        else:
            closes = local.combine_first(remote)  # Roostoo's own bars win where both exist
        self.closes = closes.sort_index().tail(n * 2)
        log.info("warm-up: %d bars of history (strategy needs %d)", len(self.closes), self.strategy.warmup_bars)

    def _startup(self) -> None:
        mode = "DRY-RUN (no orders sent)" if self.cfg.live.dry_run else "LIVE TRADING"
        log.info("starting quantbot: strategy=%s interval=%s mode=%s",
                 self.strategy.name, self.cfg.live.bar_interval, mode)
        self._maintenance(time.time())
        self._warmup()
        snap = self._snapshot()
        self.store.record_equity(snap)
        log.info("account: %s", snap.summary())
        self._rebalance(snap.tickers)  # reconcile immediately after (re)start

    # --------------------------------------------------------------- loop
    def run(self) -> None:
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        while not self._stop:
            try:
                self._startup()
                break
            except Exception:
                log.exception("startup failed; retrying in 30s")
                self._sleep(30)

        errors = 0
        while not self._stop:
            started = time.time()
            try:
                self._maintenance(started)
                self._tick()
                errors = 0
            except Exception:
                errors += 1
                backoff = min(300.0, self.cfg.live.poll_seconds * 2 ** min(errors, 5))
                log.exception("cycle failed (%d in a row); backing off %.0fs", errors, backoff)
                self._sleep(backoff)
                continue
            self._sleep(max(0.0, self.cfg.live.poll_seconds - (time.time() - started)))
        log.info("quantbot stopped")

    def _tick(self) -> None:
        tickers = self.client.ticker()
        if not tickers:
            log.warning("empty ticker response")
            return
        ts_ms = max(t.server_ts for t in tickers.values()) or self.client.now_ms()
        if self.cfg.live.record_ticks:
            self.store.insert_ticks(ts_ms, tickers)
        closed = self.bars.update(ts_ms, {p: (t.last if t.last > 0 else t.mid) for p, t in tickers.items()})
        if closed is not None:
            bar_ts, bars = closed
            self.store.upsert_bars(int(bar_ts.timestamp() * 1000), bars)
            row = bars["close"].rename(bar_ts).to_frame().T
            closes = pd.concat([self.closes, row]) if not self.closes.empty else row
            self.closes = closes[~closes.index.duplicated(keep="last")].sort_index().tail(
                max(self.cfg.live.warmup_bars, self.strategy.warmup_bars) * 2)
            self._bars_since_rebalance += 1
            if self._bars_since_rebalance >= self.cfg.live.rebalance_every_bars:
                self._rebalance(tickers)
        if time.time() >= self._next_heartbeat:
            snap = self._snapshot(tickers)
            self.store.record_equity(snap)
            log.info("heartbeat: %s", snap.summary())
            self._next_heartbeat = time.time() + self.cfg.live.heartbeat_minutes * 60

    def _rebalance(self, tickers: dict[str, Ticker]) -> None:
        assert self.executor is not None
        self._bars_since_rebalance = 0
        snap = self._snapshot(tickers)
        self.store.record_equity(snap)
        if not self.risk.update_drawdown(snap.equity):
            targets: dict[str, float] = {}
            log.warning("risk halt active: targeting flat")
        else:
            closes = self.closes.ffill()
            if len(closes) < self.strategy.warmup_bars:
                log.info("warming up (%d/%d bars); holding positions", len(closes), self.strategy.warmup_bars)
                return
            raw = self.strategy.target_weights(closes)
            targets = self.risk.clip_targets(raw, self.tradable)
        log.info("targets (%d): %s", len(targets), {p: round(w, 4) for p, w in sorted(targets.items())})
        results = self.executor.rebalance(targets, snap, refresh=self._snapshot)
        if results:
            counts: dict[str, int] = {}
            for r in results:
                counts[r.status] = counts.get(r.status, 0) + 1
            log.info("rebalance done: %s", counts)
