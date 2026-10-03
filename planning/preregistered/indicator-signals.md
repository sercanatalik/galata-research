# Pre-registered: five indicator families on Hyperliquid BTC and ETH, daily

**Registered 2026-10-03, before any backtest of them was run.** This file is
committed alone, before the code that runs it, so its git history is the
evidence of that. Nothing below may change after the run without a new file
that says why.

## The claim being tested

The trading side's standard vocabulary (EMA crossovers, RSI and Bollinger
mean reversion, MACD, Supertrend) is what investing-algorithm-framework and
its `pyindicators` ship as strategies (`planning/investing-algorithm-framework.md`,
families A1–A5). The implicit claim is that **at least one common indicator
rule, with common parameters, earns a Sharpe that survives the search that
picked it** on liquid crypto perps.

The expectation, stated before the run: **not supported.** On this record
the 66 moving-average and momentum trials gave a DSR of 0.67
(`deflated_sharpe.py`), the Reality Check over 70 trials found nothing beyond
buy-and-hold (`whole_set.py`), and the best of the search sat below the
permuted median (`permuted_bars.py`). These families overlap that one. This is
a test of whether five more do better, not a search for one that does.

## The specification, frozen

- **Bars:** `gr.market.candles(["BTC", "ETH"], "1d", ...)`, traded and closed
  bars, from the first traded bar (2023-02-26) to the last closed bar at run
  time. Next-bar execution through `gr.studies.trial` (`gr.backtest.returns`),
  0.045% taker fee on every change of position. **Funding is not charged**
  (the record's settled funding does not cover the span). That flatters long
  exposure, and is stated beside every figure.
- **Indicators:** `gr.indicators`, each evaluated per contiguous stretch
  (none spans a hole), with Wilder's seeding for RSI and ATR and the
  mean-seeded EMA.
- **Sides:** every rule is run `long_flat` (flat in place of short) and
  `long_short`.

| Family | Rule (position from the close it is decided at) | Grid | Trials per ticker |
|---|---|---|---|
| EMA crossover | sign(EMA fast − EMA slow) | fast ∈ {5, 10, 20} × slow ∈ {50, 100, 200} | 18 |
| RSI reversion | long from RSI < lo until RSI > hi; short from RSI > hi until RSI < lo | n ∈ {7, 14} × (lo, hi) ∈ {(30, 70), (20, 80)} | 8 |
| Bollinger reversion | long from z < −k until z ≥ 0; short from z > k until z ≤ 0 (z over n bars, population σ) | n ∈ {20, 50} × k ∈ {1.5, 2.0, 2.5} | 12 |
| MACD | sign(MACD − signal) | (12, 26, 9), (8, 17, 9) | 4 |
| Supertrend | the direction, +1 or −1 | n ∈ {10, 20} × multiplier ∈ {2, 3} | 8 |

In `long_flat`, every short above is flat instead.

## The trials, and N

50 rules × {BTC, ETH} = **N = 100**, each `(trial, ticker)` a column, as in
`deflated_sharpe.py`. Buy-and-hold on each ticker is the benchmark, not a
trial. The report also reads DSR at N = 170 (these plus the 70 run before in
this repository).

## What would count as support

For the best `(trial, ticker)` by per-period Sharpe, all of:

1. **DSR ≥ 0.95 at N = 100**, with its own skewness and kurtosis
   (`gr.stats.deflate` over `gr.studies.summary`);
2. **PBO < 0.5** by CSCV with 16 blocks over the 100 columns, on the days all
   of them have a return (`gr.studies.matrix`);
3. **Reality Check p < 0.05 against buy-and-hold** on the same ticker
   (`gr.stats.reality_check`, 1,000 replicates, Politis–White block, seed 0);
4. **Permuted-bars p_best < 0.05**: the whole 100-trial search re-run on 200
   permuted markets (`gr.studies.permutation_test`, seed 0).

Anything short of all four is reported as **not supported on this record**.
No family is tuned, dropped or re-run with other parameters under this name.

**Described, not tested:** `gr.stats.describe` and `gr.trades.summary` for the
best trial of each family (win rate, profit factor, MAE/MFE, drawdown
length), and which family the best trial came from.

## Secondary: the same, on six years of Binance

The same 100 trials on Binance BTC and ETH 1m klines from `gr.reference.candles`,
resampled to whole 1d bars (`gr.timeseries.resample`), 2020-01-01 to the
last whole day held at run time, with the same four criteria. This sample is
longer but from another venue, so it is reported separately and **cannot
rescue a failure on the record**. It can only say whether a failure is the
short sample's.
