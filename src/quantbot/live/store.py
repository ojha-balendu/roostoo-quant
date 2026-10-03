"""SQLite persistence: bot state (peak equity, kill switch), bars, ticks, equity curve, orders.

Survives restarts so the bot can resume autonomously; also a research dataset of
Roostoo's own prices and spreads.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS ticks (ts INTEGER, pair TEXT, bid REAL, ask REAL, last REAL);
CREATE INDEX IF NOT EXISTS ticks_ts ON ticks(ts);
CREATE TABLE IF NOT EXISTS bars (
    ts INTEGER, pair TEXT, open REAL, high REAL, low REAL, close REAL, PRIMARY KEY (ts, pair)
);
CREATE TABLE IF NOT EXISTS equity (
    ts INTEGER PRIMARY KEY, equity REAL, usd REAL, long_value REAL, short_value REAL, gross REAL
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, action TEXT, pair TEXT, notional REAL,
    quantity REAL, status TEXT, dry_run INTEGER, detail TEXT, error TEXT
);
"""


class StateStore:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        if path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    def _write(self, sql: str, rows: list[tuple] | tuple) -> None:
        with self._lock, self.conn:
            if isinstance(rows, list):
                self.conn.executemany(sql, rows)
            else:
                self.conn.execute(sql, rows)

    # key/value -------------------------------------------------------------
    def get(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set(self, key: str, value: str) -> None:
        self._write("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, value))

    def get_float(self, key: str, default: float) -> float:
        value = self.get(key)
        try:
            return float(value) if value is not None else default
        except ValueError:
            return default

    # market data -------------------------------------------------------------
    def insert_ticks(self, ts_ms: int, tickers: dict) -> None:
        rows = [(ts_ms, p, t.bid, t.ask, t.last) for p, t in tickers.items()]
        if rows:
            self._write("INSERT INTO ticks VALUES (?, ?, ?, ?, ?)", rows)

    def upsert_bars(self, bar_ts_ms: int, bars: pd.DataFrame) -> None:
        rows = [(bar_ts_ms, pair, float(r.open), float(r.high), float(r.low), float(r.close))
                for pair, r in bars.iterrows()]
        if rows:
            self._write("INSERT OR REPLACE INTO bars VALUES (?, ?, ?, ?, ?, ?)", rows)

    def load_close_panel(self, since_ms: int = 0) -> pd.DataFrame:
        df = pd.read_sql_query("SELECT ts, pair, close FROM bars WHERE ts >= ?", self.conn, params=(since_ms,))
        if df.empty:
            return pd.DataFrame()
        panel = df.pivot(index="ts", columns="pair", values="close")
        panel.index = pd.to_datetime(panel.index, unit="ms", utc=True)
        panel.columns.name = None
        return panel.sort_index()

    # account -----------------------------------------------------------------
    def record_equity(self, snap) -> None:
        s = snap.summary()
        self._write(
            "INSERT OR REPLACE INTO equity VALUES (?, ?, ?, ?, ?, ?)",
            (snap.ts_ms, s["equity"], snap.usd_total, s["long_value"], s["short_value"], s["gross"]),
        )

    def record_order(self, *, ts_ms, action, pair, notional, quantity, status, dry_run, detail, error) -> None:
        self._write(
            "INSERT INTO orders (ts, action, pair, notional, quantity, status, dry_run, detail, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts_ms, action, pair, notional, quantity, status, int(dry_run), detail, error),
        )

    def close(self) -> None:
        self.conn.close()
