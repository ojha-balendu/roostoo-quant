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

## Competition strategy

Configuration: [`config/competition.yaml`](config/competition.yaml). Evaluation: `python research/08_final_portfolio.py`.

**What it trades.** The 15 most liquid majors (BTC, ETH, SOL, XRP, BNB, DOGE, ADA, LINK, AVAX, LTC, DOT,
SUI, NEAR, UNI, AAVE), long and short, rebalanced once a day at 00:00 UTC.

**How positions are set.** An equal-weight ensemble of two trend-following books (`strategy/trend_vol.py`):

1. *Trend signal.* For each coin, the return over the last 3, 7 and 14 days divided by its volatility,
   passed through `tanh` and averaged: a score between -1 and +1 that grows with trend strength.
2. *Risk balance.* Each coin is sized by 1 / volatility so every position carries similar risk.
3. *Volatility target.* The whole book is scaled so its expected volatility (from the last 7 days of
   correlated returns) is 30% a year, with at most 95% of equity in use and 25% in any coin.
4. *Two members.* One trades both directions freely; the other only goes long while BTC is above its
   7-day average and only short while it is below. Averaging them avoids choosing between two settings
   that performed similarly, which would mean fitting noise.

**Why this design.** It came out of the research log below: of every idea tested, only slow trend
following survived Roostoo's ~0.25% round-trip cost, and its clearest benefit is much smaller
drawdowns than holding crypto, which is what Sharpe, Sortino and Calmar reward. It trades rarely
(a 2% weight-change threshold and a daily schedule) because fast versions lost their edge to fees.

**Safety.** Orders are spaced 61 seconds apart (one per minute). A drawdown kill switch flattens the
book at 20% below peak and resumes automatically after 24 hours. Every cycle starts from the balances
and positions the exchange reports, so restarts and lost responses heal themselves. A consistency
check refuses to trigger the kill switch if the short-collateral accounting setting looks wrong.

### Running it for the competition

```bash
# on the EC2 instance, after deploy/ec2_setup.sh
echo "QUANTBOT_CONFIG=config/competition.yaml" >> .env
.venv/bin/quantbot check                      # connectivity, balance (expect $1,000,000)
.venv/bin/quantbot smoke-test --trade         # confirm short-collateral accounting on THIS account
sudo systemctl start quantbot                 # dry run (config default) until the round starts
# when the trading round starts: set live.dry_run: false in config/competition.yaml, then
sudo systemctl restart quantbot
journalctl -u quantbot -f
```

## Research log

Every hypothesis was tested with fees (0.1% per side), spread and realistic execution. Scripts are in
`research/`; each prints a plain verdict. Rules used throughout: profit after costs, statistical
confidence (t >= 3 unless noted), enough trades, profitable in both halves of the sample, and a
held-back final year where applicable.

| # | Hypothesis | Script | Result |
|---|---|---|---|
| 1 | Market structure: how correlated are the coins? | `01_correlations.py` | Average daily correlation 0.52 (about 2 independent bets among 65 coins); 0.29 after removing BTC; annual volatility ranges from 22% to 195%. Volatility sizing is essential. |
| 2 | A beta-neutral alt basket removes BTC risk | `02_beta_neutral.py` | Hedge works (beta to BTC 0.05 vs 0.57 unhedged) but leaves a volatile "alts vs BTC" bet; no edge by itself. |
| 3 | The ETH/BTC spread mean-reverts | `03_pairs_btc_eth.py` | Lost money: at the size needed to beat fees, the spread does not revert reliably. |
| 4 | Trend following with volatility targeting | `04_trend_vol.py` | Fast version: edge eaten by fees (-1.4% per 14-day window). Before fees: +0.7%. Slow version (3/7/14-day trends, daily rebalance): +0.7% to +1.1% per window after fees, worst drawdown -12% to -13% vs -29% for holding BTC. **Kept.** |
| 5 | BTC leads other coins | `05_btc_leadlag.py` | A real 1-minute lag, but no coin / bar size / holding period clears costs. |
| 6 | Coins over- or under-react relative to the market | `06_overreaction.py` | Nothing qualified on Jan 2023 to Oct 2025; the final year was left untouched. |
| 7 | Volatility management improves 14-day risk-adjusted ratios | `07_vol_managed.py` | No: within a 14-day window, Sharpe/Sortino/Calmar barely depend on exposure size, and timing exposure by predicted volatility did not help. |
| 8 | Technical indicators (RSI, Williams %R, MACD, CCI, OBV, Chaikin money flow, ... 22 + taker-buy ratio) rank coins | `13_indicators.py` | See its output. Also reports which indicators are the same signal in disguise, and which patterns are real but eaten by costs. |
| 9 | Composite indicator signal (31 weighted, normalised indicators): v1 trades the top/bottom 10% every bar; v2 trades only when the calibrated expected profit beats the fees | `14_composite.py` | See its output: both periods, every 14-day window, a shuffled-signal placebo and fee-free runs. |
| 10 | Final portfolio (ensemble of the slow trend books) | `08_final_portfolio.py` | See its output: every 14-day window since 2023 plus a continuous backtest, using exactly the live configuration. |

Main lesson: in liquid crypto at retail fees, small frequent price patterns do not survive costs;
what survives is slow, risk-controlled exposure that limits drawdowns.

## Reinforcement-learning sandbox (research)

`src/quantbot/rl/` contains a gymnasium environment in which a PPO agent (stable-baselines3) trades
the 15 majors every 15 minutes, with Roostoo's fees and spread, starting each 14-day episode flat
with $100k. It optimises `0.5*return% + 0.2*Sortino + 0.15*Sharpe + 0.15*Calmar` (ratios capped at
+/-10, Calmar with a 1% drawdown floor) as the yardstick for picking checkpoints and judging results.
The training reward is per step: `100 * (r - risk_aversion * min(r, 0)^2)` (mean-semivariance
utility, downside-only like Sortino). Two earlier reward designs were found to be broken and are
kept only as options: a linear loss penalty (it outweighs any realistic edge, so the agent's best
answer is to hold nothing) and adding the composite score at the end of each episode (the ratios
ignore position size, so a 1% position swings it as much as a 95% one: mostly noise to learn from).

```bash
pip install -e ".[rl]"
python research/10_rl_train.py                     # ~1M steps, a few minutes on a laptop CPU
python research/11_rl_evaluate.py                  # unseen test period vs benchmarks, with a chart
python research/12_rl_sweep.py                     # several configs x 3 seeds, ranked on validation only
```

Data is split in time: all earlier history for training, 30 days of validation (only to pick the best
checkpoint), and the last 60 days as an untouched test. The agent and every benchmark go through
the same accounting code (`rl/sim.py`, checked by tests). Lesson from building it: with 15-minute
decisions, PPO's default exploration noise trades randomly every bar; at the default noise level
every training episode lost about 50% to fees and the agent learned nothing. Small initial noise
(`--log-std-init -3`) and a 5% no-trade band keep exploration from churning the book.
