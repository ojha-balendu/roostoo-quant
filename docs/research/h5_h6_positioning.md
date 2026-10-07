# H5 / H6: futures-positioning studies

Both use free Binance Vision USD-M data (funding rates, 5-minute open interest)
plus spot 1h klines, and follow the standard research protocol: fees and spread
included, net >= 5 bps per holding period, t >= 3, enough observations,
positive in both halves, last 365 days held out and evaluated once.

| | H5 liquidation-flush rebound | H6 cross-sectional funding |
|---|---|---|
| Mechanism | forced liquidations push price below where unforced traders would trade; liquidity takers are paid | crowded, high-funding longs unwind; low-funding coins outperform |
| Signal | 1h return <= -3 sigma and OI -5% in the same hour and prior funding > 0 | 7-day average daily funding, residualised on 7/30-day momentum |
| Trade | buy at bar close, hold 4/12/24 h | long bottom 20%, short top 20%, beta-neutral, gross 1, weekly |
| Cost | 25 bps round trip | 12.5 bps one way on turnover |
| Statistic | t on daily means of event returns (events cluster on crash days) | t over non-overlapping periods; rank IC |
| Pre-registered decision | condition `full`, default params | residual signal, `W-MON` |

## Run

```bash
python scripts/download_futures_data.py --symbols-from-roostoo --start 2023-01-01
python scripts/run_h5_liquidation_flush.py --grid
python scripts/run_h6_funding_xs.py
# once, after freezing parameters:
python scripts/run_h5_liquidation_flush.py --evaluate-holdout
python scripts/run_h6_funding_xs.py --evaluate-holdout --rebalance W-MON
```

Outputs go to `reports/`.  A marker file blocks a second look at the holdout
unless `--force` is passed.

## Reading the results

* H5: compare `price`, `price_oi` and `full`.  If the OI and funding filters
  add nothing, the effect is ordinary crash reversal, not liquidations.
  `excess_gross_bps` (versus the equal-weight market) shows whether the bounce
  is coin-specific or just the market rebounding.
* H6: if the raw signal passes but the residualised one fails, the result is
  momentum/reversal, which was already rejected.

## Known limitations

* Survivorship: the universe is today's Roostoo listings.
* H5 uses hourly bars; a 1-minute version would enter earlier, but would also
  pay more spread during the cascade.
* Binance stopped publishing liquidation snapshots consistently, so the OI drop
  is used as the liquidation proxy.
