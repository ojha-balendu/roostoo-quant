"""Typed configuration: config/config.yaml for settings, .env / environment for secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class ExchangeConfig:
    base_url: str = "https://mock-api.roostoo.com"
    timeout_seconds: float = 10.0
    max_retries: int = 3
    min_request_interval_seconds: float = 0.2
    api_key: str = field(default="", repr=False)
    secret_key: str = field(default="", repr=False)


@dataclass
class FeeConfig:
    taker: float = 0.001
    maker: float = 0.0005
    short_open: float = 0.001
    short_close: float = 0.001


@dataclass
class ExecutionConfig:
    long_order_mode: str = "market"
    limit_timeout_seconds: float = 60.0
    min_trade_usd: float = 50.0
    min_close_usd: float = 2.0
    rebalance_threshold: float = 0.02
    cash_buffer: float = 0.005
    pause_between_orders_seconds: float = 0.3

    def __post_init__(self) -> None:
        if self.long_order_mode not in ("market", "passive_limit"):
            raise ValueError("execution.long_order_mode must be 'market' or 'passive_limit'")


@dataclass
class RiskConfig:
    max_gross_leverage: float = 1.0
    max_weight_per_asset: float = 0.2
    max_drawdown_kill: float = 0.25
    kill_cooldown_hours: float = 24.0


@dataclass
class LiveConfig:
    dry_run: bool = True
    poll_seconds: float = 10.0
    bar_interval: str = "15m"
    rebalance_every_bars: int = 1
    rebalance_utc_hours: list[int] | None = None  # e.g. [0]: rebalance on bars closing at 00:00 UTC
    warmup_bars: int = 500
    warmup_source: str = "binance"
    db_path: str = "state/bot.sqlite"
    record_ticks: bool = True
    heartbeat_minutes: float = 15.0
    log_dir: str = "logs"


@dataclass
class AccountingConfig:
    short_collateral_in_usd_balance: bool = False


@dataclass
class StrategyConfig:
    name: str = "flat"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    universe_path: str = "config/universe.yaml"
    market: str = "spot"
    interval: str = "1h"
    start: str = "2023-01-01"
    end: str | None = None
    max_workers: int = 8
    verify_checksum: bool = True
    symbol_overrides: dict[str, str | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # YAML turns 2023-01-01 into a date object; normalise to strings.
        self.start = str(self.start)
        self.end = None if self.end in (None, "") else str(self.end)


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    long_fee: str = "taker"
    spread_bps: float = 2.0
    execution_delay_bars: int = 1
    report_dir: str = "reports"

    def __post_init__(self) -> None:
        if self.long_fee not in ("taker", "maker"):
            raise ValueError("backtest.long_fee must be 'taker' or 'maker'")


@dataclass
class Config:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    fees: FeeConfig = field(default_factory=FeeConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    accounting: AccountingConfig = field(default_factory=AccountingConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


_SECTIONS = {
    "exchange": ExchangeConfig,
    "fees": FeeConfig,
    "execution": ExecutionConfig,
    "risk": RiskConfig,
    "live": LiveConfig,
    "accounting": AccountingConfig,
    "strategy": StrategyConfig,
    "data": DataConfig,
    "backtest": BacktestConfig,
}


def _build(cls: type, raw: Any, section: str) -> Any:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config section '{section}' must be a mapping")
    unknown = set(raw) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown keys in config section '{section}': {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load config; a typo in a key raises instead of being silently ignored."""
    load_dotenv()
    cfg_path = Path(path or os.environ.get("QUANTBOT_CONFIG", "config/config.yaml"))
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text()) or {}
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    cfg = Config(**{name: _build(cls, raw.get(name), name) for name, cls in _SECTIONS.items()})
    cfg.exchange.api_key = os.environ.get("ROOSTOO_API_KEY", cfg.exchange.api_key).strip()
    cfg.exchange.secret_key = os.environ.get("ROOSTOO_SECRET_KEY", cfg.exchange.secret_key).strip()
    return cfg
