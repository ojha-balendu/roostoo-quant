# Research

Notebooks and scratch analysis live here. Install extras with `pip install -e ".[research]"`.

Workflow:
1. `quantbot download` puts parquet files in `data/processed/{market}/{interval}/`.
2. Load panels with `quantbot.data.loader.load_panel(...)` (columns are Roostoo pairs).
3. Prototype a signal, then implement it as a `Strategy` in `src/quantbot/strategy/`
   and register it in `registry.py`.
4. Evaluate with `quantbot backtest --strategy <name>`; the backtester uses the same
   order logic and fees as live trading.
5. Once live, `state/bot.sqlite` (tables `ticks`, `bars`) holds Roostoo's own prices and
   spreads, which is the best data for checking how well Binance history transfers.

Rules of thumb for this competition:
* Fees are about 0.2% per round trip on market orders (0.1% each side, shorts always 0.1%),
  so signal horizons must be long enough to beat that. Watch `annual_turnover`.
* Never let row t of a signal use data after bar t (the backtester trades it at t+1).
