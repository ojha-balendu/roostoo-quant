# roostoo-quant

Autonomous trading infrastructure for the Roostoo quant competition: exchange client,
execution engine, risk controls, Binance Vision data pipeline, a backtester that mirrors
Roostoo's fee and short mechanics, and an EC2/systemd deployment.

**The strategy shipped here (`xs_momentum`) is a placeholder** to exercise the plumbing.
It is not research-validated, and on random data it loses money mostly through fees.

## How the competition maps onto the API

| Action | Endpoint | Sizing | Fee |
|---|---|---|---|
| **LONG** (open/increase long) | `POST /v3/place_order` `side=BUY` | coin quantity | 0.1% market, 0.05% resting limit |
| **SELL** (reduce/close long) | `POST /v3/place_order` `side=SELL` | coin quantity | 0.1% market, 0.05% resting limit |
| **SHORT** (open/increase short) | `POST /v6/short_open` | **USD collateral**, qty = collateral / entry | 0.1% always, charged on acceptance |
| **CLOSE** (reduce/close short) | `POST /v6/short_close` | qty, pct, or all | 0.1%, always a market fill at the ask |

Strategies never call these directly. They output **target weights** per pair (`+0.1` = long
10% of equity, `-0.1` = short 10%), and `execution/decision.py` turns the difference between
target and actual exchange positions into these four actions. Live trading and the
backtester use the same function.

Things in the docs that shape the design:

* Failed requests return HTTP 200 with `Success: false`, so every response is checked.
* Zero-valued fields are omitted from responses and are read as 0.
* Signed requests need a timestamp within 60s of server time. The client tracks the offset
  and resyncs automatically, and `chrony` keeps the EC2 clock honest.
* Only documented parameters may be signed. Extra parameters break the signature.
* Shorts are 1x: loss is capped at collateral, and both longs and shorts consume USD,
  so gross exposure is limited to equity (`risk.max_gross_leverage: 1.0`).
* Limit shorts pay the same 0.1% as market shorts, so the bot always shorts at market.
  The maker discount only applies to spot longs (`execution.long_order_mode: passive_limit`).
* Roostoo has **no klines endpoint**. The bot builds its own bars from ticker polls and stores
  them, warming up from Binance at start.

## Layout

```
config/config.yaml          all settings (secrets go in .env)
config/universe.yaml        Roostoo pair -> Binance symbol map
src/quantbot/
  exchange/                 Roostoo client, signing, response models, precision rules
  execution/decision.py     target weights -> LONG/SELL/SHORT/CLOSE (shared with backtest)
  execution/executor.py     sends orders: precision, min size, cash scaling, dry-run, passive limits
  portfolio.py              account snapshot and equity from exchange state
  risk.py                   per-asset and gross caps, drawdown kill switch with auto-resume
  strategy/                 Strategy interface, registry, placeholder strategies
  data/                     Binance Vision downloader, parquet processing, panel loader, universe
  backtest/                 Roostoo-faithful backtester and metrics (Sharpe, Sortino, Calmar, ...)
  live/                     trading loop, bar builder, Binance warm-up, SQLite state store
  cli.py                    `quantbot` command
deploy/                     EC2 setup script, systemd unit, update script
tests/                      unit tests (signing is checked against the docs' published example)
research/                   notebooks and analysis
```

## Quickstart

```bash
git clone <your-repo-url> roostoo-quant && cd roostoo-quant
make install                       # venv + editable install with dev tools
cp .env.example .env               # add ROOSTOO_API_KEY / ROOSTOO_SECRET_KEY
source .venv/bin/activate

quantbot check                     # connectivity, pairs, spreads, account (if keys set)
quantbot universe                  # write config/universe.yaml from exchangeInfo, then review it
quantbot download                  # Binance Vision klines for the universe (config: data.*)
quantbot backtest                  # run config.strategy over the downloaded data
quantbot live --dry-run            # full loop, logs intended orders, sends nothing
pytest -q
```

Other commands: `quantbot smoke-test [--trade]` exercises every endpoint and, with `--trade`,
round-trips a tiny long and short. `quantbot flatten [--execute]` closes everything.

## Data (Binance Vision)

`quantbot download` fetches monthly archives for complete months and daily archives for the
current month, verifies each `.CHECKSUM`, and merges everything into
`data/processed/{market}/{interval}/{SYMBOL}.parquet`. It handles the switch of spot
timestamps from milliseconds to microseconds from 2025 onward, and archives with or without
header rows. Options: `--interval 1m`, `--start/--end`, `--market spot|um|cm`,
`--data-type klines|aggTrades|trades`, `--symbols BTCUSDT ETHUSDT`.

Symbols with no Binance data at all are listed at the end of the run. **The tokenized stocks
will most likely be among them.** Set those to `null` in `config/universe.yaml` for now; we
can add another data source for them during research.

## Running on EC2

1. Launch an Ubuntu 22.04/24.04 instance (t3.small is plenty). Pick a region close to Hong
   Kong (e.g. `ap-east-1` or `ap-southeast-1`), and avoid US regions because Binance
   restricts some endpoints there, which could break warm-up. Allow inbound SSH only.
2. On the instance:
   ```bash
   git clone <your-repo-url> roostoo-quant && cd roostoo-quant
   bash deploy/ec2_setup.sh
   nano .env
   .venv/bin/quantbot check
   .venv/bin/quantbot universe
   .venv/bin/quantbot smoke-test --trade
   ```
3. Set `live.dry_run: false` (and the accounting flag from the smoke test) in
   `config/config.yaml`, then:
   ```bash
   sudo systemctl start quantbot
   journalctl -u quantbot -f         # also logs/bot.log
   ```
4. Deploy new code with `bash deploy/update.sh`. It pulls, runs the tests, and restarts
   only if they pass.

### Why it can run unattended

* systemd restarts the process on any crash (`Restart=always`) and on reboot.
* The loop catches every exception, backs off and continues; startup retries until the
  exchange is reachable.
* No position state is kept locally. Every rebalance starts from the balances and short
  positions the exchange reports. After a restart or a lost order response, the next cycle
  converges to target without double-trading.
* Order placement is never blindly retried after a request may have reached the server.
* The drawdown kill switch flattens the book if equity falls `risk.max_drawdown_kill` below
  peak, then resumes after `kill_cooldown_hours` with a fresh peak, without a human.
* Peak equity and halt state persist in `state/bot.sqlite`, along with every tick, bar,
  equity point and order for post-mortems.

## Day-one checklist (things the docs leave open)

- [ ] Run `quantbot smoke-test --trade` and set `accounting.short_collateral_in_usd_balance`.
  It decides whether locked short collateral is already counted in the USD balance;
  getting it wrong double-counts or ignores short collateral in equity.
- [ ] Confirm the competition's rate limits and trading frequency rules, then set
  `exchange.min_request_interval_seconds` and `live.poll_seconds` accordingly.
- [ ] Check which commission currency BUY orders use (the bot already sells actual free
  balance, so it is safe either way).
- [ ] Confirm the starting balance (docs example shows 50k; ours should be 100k).
- [ ] Review `config/universe.yaml` for renamed tickers and the tokenized stocks.

## Writing a strategy

```python
# src/quantbot/strategy/my_strategy.py
class MyStrategy(Strategy):
    name = "my_strategy"

    @property
    def warmup_bars(self) -> int:
        return 200

    def generate_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        # rows = bars (UTC open time), columns = "BTC/USD", ...
        # row t may only use data up to row t
        ...
```

Register it in `strategy/registry.py`, set `strategy.name` / `strategy.params` in the config,
and run `quantbot backtest`. Keep the backtest interval and `live.bar_interval` the same.
