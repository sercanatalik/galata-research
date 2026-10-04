# trading-side: the features and signals to onboard

**NOT PROPOSED.** Written 2026-10-03. galata-research's research core (permutation tests,
walk-forward, DSR, PBO, the Reality Check, SPA, the Model Confidence Set, vol targeting) is
deep. Its trading side is thin: no standard indicator set, no trade-level metrics, no
position overlays, no cross-sectional ranking, no portfolio construction. This file lists
those features and the signals they enable, says which ones galata-research already has,
and orders the ones worth onboarding.

Each item is a `gr` function in polars, with a test that fails when its guard is removed,
and with a published figure where one exists. Nothing here talks to a venue: research
reaches a live system only through a person.

---

## Contents

- [Features](#features)
- [Signals](#signals)
- [Onboarding order](#onboarding-order)
- [Pitfalls to guard against](#pitfalls-to-guard-against)

---

## Features

✅ have it · ➕ onboard · ➖ leave

| Feature | galata-research today | Call |
|---|---|---|
| Monte Carlo permutation test | `studies.permute_bars`, `permutation_test`, `random_timing`, the whole search re-run per permutation | ✅ re-running one strategy instead of the search would ignore selection |
| Walk-forward / OOS windows | `timeseries.walk_forward_origins`, `fitted_through` on every forecast, registered splits | ✅ |
| Sharpe, PSR, DSR, PBO, RC, SPA, MCS | `gr.stats`, `gr.models.evaluate` | ✅ |
| Vol-targeting overlay | `gr.models.vol.target`, feedback control, ES sizing | ✅ |
| VaR / CVaR | `evaluate.value_at_risk` (FHS) with Kupiec, Christoffersen, DQ, FZ0 | ✅ |
| Max drawdown, performance fee | `stats.max_drawdown`, `performance_fee` | ✅ |
| Fees on turnover, funding charged | `backtest.returns(fee=, funding=)` | ✅ |
| Slippage from the book | `liquidity.cost_of_size`, `schedule_cost` | ✅ measured, never a flat % |
| **Sortino, Calmar, Omega, Ulcer index, recovery, drawdown duration** | max drawdown only | ➕ `gr.stats` |
| **Rolling Sharpe** | no | ➕ `gr.stats.rolling_sharpe` |
| **Monthly/yearly return table, % winning months** | no | ➕ for notebooks |
| **Trade-level metrics**: win rate, profit factor, win/loss, MAE/MFE, duration, consecutive wins/losses, exposure ratio | positions per bar only; `studies.runs` gives runs of position | ➕ `gr.trades` from `runs` |
| **Window consistency score** (1 − CV across windows, % profitable windows) | per-split verdicts in notebooks | ➕ small: a `stats.consistency` over a per-window frame |
| **Cross-study held-up ratio** (out-of-sample ÷ in-sample) | replication tables (`survival_table`) | ➖ registered verdicts say more than a ratio |
| **Cross-sectional factor pipeline** (rank, top-N, mask, neutralize, rolling beta) | none; D8 kept studies to one ticker | ➕ `gr.factors` in polars `over("ts")`, once a multi-ticker study is chosen |
| **Position overlays**: stop-loss (trailing), take-profit, cooldown, scale-in ladder | none | ➕ `gr.overlays`: each a function from a position series to a position series, so DSR still counts every variant |
| **Weighted-evidence scoring** (rules, weights, vetoes, a threshold) | none | ➖ for now. It is a rule engine, and every threshold is another trial to count. Revisit if a signal combiner is needed |
| **Portfolio construction**: inverse-vol risk parity, HRP, Markowitz | `gr.models.corr` gives Σ per origin | ➕ `gr.portfolio` on DCC Σ, after a multi-ticker study is chosen |
| Backtest bundles + an index to rank thousands of runs | trial frames returned whole (true N) | ➖ until trial counts outgrow memory. If they do, store Parquet per trial |
| Event-driven backtest, live/paper trading, deployment, credentials | out of scope by design | ➖ never: research reaches a live system only through a person |
| An HTML dashboard | marimo notebooks | ➖ |

### Metrics we lack, by name

The standard backtest report's metrics with no `gr` equivalent:
`sortino_ratio`, `calmar_ratio`, `omega_ratio`, `ulcer_index`, `max_drawdown_duration`,
`rolling_sharpe_ratio`, `profit_factor`, `gross_profit`, `gross_loss`, `win_rate`,
`win_loss_ratio`, `average_mae`, `average_mfe`, `mfe_mae_ratio`, `average_trade_duration`,
`average_win_duration`, `average_loss_duration`, `max_consecutive_wins`,
`max_consecutive_losses`, `exposure_ratio`, `trades_per_year`, `percentage_winning_months`,
`percentage_winning_years`, `best_month`, `worst_month`, `long_win_rate`, `short_win_rate`.

Each one goes into `gr.stats` (on returns) or `gr.trades` (on runs). Annualise with
`timeseries.periods_per_year`, never 252.

---

## Signals

Every signal is onboarded as a **position expression** for `studies.trial`, read on `close_ts`.
Each comes with a trial family that returns every parameter combination, so N is the true N for
DSR and PBO. A signal family is run through registered criteria
(`planning/preregistered/`) before any claim.

### A. Standard trading signals, which we lack

| # | Signal | Parameters | Note |
|---|---|---|---|
| A1 | EMA crossover (trend) | fast, slow | we have SMA crossover (`studies.moving_average`) |
| A2 | Bollinger + RSI mean reversion | BB 20/2σ, RSI 14, 30/70 | Wilder's RSI pinned to a published worked example |
| A3 | RSI alone (overbought/oversold) | 14, 30/70 | the A2 baseline |
| A4 | MACD signal-line cross | 12/26/9 | trend family |
| A5 | Supertrend (ATR bands) | ATR 10, ×3 | needs ATR. ATR on a hole must be null, the same rule as `returns` |
| A6 | Stochastic oscillator | %K 14, %D 3 | low priority |
| A7 | Volatility breakout (Donchian) | channel N | ✅ we have `donchian_ensemble` |
| A8 | Cross-sectional momentum, top-N | 30d, N | needs `gr.factors` and a multi-ticker universe |
| A9 | Multi-factor: momentum + low vol + liquidity gate | 30d returns, vol rank | the liquidity gate can use `gr.liquidity` spread |
| A10 | Pairs: z-score of log-price spread | BTC/ETH, lookback, ±2 | hedge ratio fitted on the window only; both legs on one `ts` grid |
| A11 | Beta-neutral residual (rolling beta) | window | datawatch already stores a `beta` signal: read it through `gr.signals` |
| A12 | Risk parity / HRP / Markowitz weights | rebalance | allocation, not timing; Σ from `gr.models.corr` |
| A13 | Vol-targeting overlay | target σ | ✅ `gr.models.vol.target` |
| A14 | DCA | — | ➖ a benchmark at most |

### B. Signals on perpetual and book data

These need data a spot OHLCV backtester does not have: funding, marks, the book and trades.

| # | Signal | Our data | Note |
|---|---|---|---|
| B1 | **Funding carry**: hold against the funding sign, net of fees | `market.funding` settled + premium, `funding_live` on `recv_ts`, Binance 8h | `backtest.returns(funding=)` already charges it. The predicted rate is known only on `recv_ts`: cross with `join_recv` |
| B2 | **Funding / premium mean reversion**: extreme premium → fade | `funding.premium`, `marks.premium` | as B1 |
| B3 | **Basis**: mark − oracle, perp vs spot | `marks` (mark, oracle, mid) | mark and oracle are on `recv_ts` only |
| B4 | **Order-flow imbalance** | `liquidity.flow`, `liquidity.ofi`, quotes, Bybit book | known at the bucket's end |
| B5 | **Micro-price / book imbalance** | top of book, `reference.book` depth within 2 and 10 bps | 1 s rows |
| B6 | **Lead–lag across venues** | `gr.leadlag` (HY, LLR) | a signal only if the lead exceeds latency; within ±50 ms is clock skew |
| B7 | **Jump-conditioned** (fade or follow after a Lee–Mykland jump) | `gr.jumps.lee_mykland` | the scale uses earlier returns only |
| B8 | **Scheduled-event windows** (CPI, FOMC, funding hour) | `reference.events`, `calendar` | flat or reduced exposure around releases |
| B9 | **Vol-regime switch** (trend on in low vol, off in high vol) | `gr.models.vol` forecasts, `msgarch` regimes | conditioning variable read at `fitted_through` |
| B10 | **Open interest change** | `marks` OI | `recv_ts` only |

---

## Onboarding order

One OpenSpec change per item, as in [`roadmap`](./roadmap.md). Each item lands with a notebook.

- [x] 1. **`score-the-trades`**: `gr.stats` adds Sortino, Calmar, Omega, Ulcer, drawdown duration, rolling Sharpe,
  monthly table. A new `gr.trades` turns `studies.runs` into trades with win rate, profit factor, MAE/MFE
  (from bar highs and lows), durations and streaks. Each figure is checked against a hand-worked case.
  No new data, no new trials: the cheapest gain.
  Done 2026-10-03: `gr.trades` (`table`, `summary`), the `gr.stats` additions, and `notebooks/trades.py`. `studies.trial`
  now carries `close_ts` and `cost`, which a trade needs.
- [x] 2. **`indicator-signals`**: `gr.indicators` (EMA, RSI, MACD, Bollinger, ATR, Supertrend, stochastic, z-score)
  as polars expressions over `ticker`. Each value is null for a window that spans a hole. Families A1–A5 go in
  `gr.studies`, and one notebook runs them through DSR, PBO and the permuted-bars test. Register first.
  Built 2026-10-03: `gr.indicators`, the five families in `gr.studies`, `notebooks/indicator_signals.py`. Registered alone
  first (`planning/preregistered/indicator-signals.md`, `020ef19`); **the run on the record is still to do**, where the record lives.
- [x] 3. **`overlay-the-positions`**: stop-loss (fixed, trailing), take-profit, cooldown, scale-in, as functions on a
  position series. Stops trigger on the bar's high or low, never on the close that decided the position.
  Every overlay variant counts as a trial.
  Built 2026-10-03: `gr.overlays` (`apply`, `grid`), `studies.shared_days`, `notebooks/overlays.py`. Registered alone first
  (`planning/preregistered/overlay-the-positions.md`, `f96b81a`): 12 overlays on three bases, N = 78. **The run on the
  record is still to do.** Two choices made after the registration, stated in the notebook: an overlay that never fired
  equals its base and is left out of that base's Reality Check (counted); and **scale-in is not built**, since these
  families emit position states, not repeated entry signals, so a ladder has nothing to trigger on.
- [x] 4. **`trade-the-funding`**: B1–B3 on settled funding and marks, through `join_recv`.
  Built 2026-10-03, on **Binance's published archives, not the record**: the record's settled funding is days long,
  and settled point 8 rules out a venue's API. `galata-fetch funding|premium binance-um` (`cdacf2c`), `gr.carry`
  (`b46db00`), `notebooks/carry.py`. Registered alone first (`planning/preregistered/trade-the-funding.md`, `d82ee2b`):
  H1 hedged carry (N = 26), H2 persistence, H3 the fade (N = 16). B3 (basis) is H1's hedged leg, the premium index;
  Hyperliquid's own mark − oracle waits for the record to hold months of marks.
  **Run once, 2026-10-03** (`42bd9de`): H1 supported, H2 supported, H3 not supported. The carry was real and is
  compressing (2026: ~2–3% a year); timing it on funding added nothing over holding it.
- [x] 5. **`trade-the-flow`**: B4–B6 on trades, quotes and the Bybit book. Costs come from `liquidity.effective`,
  not a flat fee.
  Built and run 2026-10-03 on Bybit's book and Binance's trades (reference archives, 12 days): `gr.flow`,
  `notebooks/flow.py`, registered alone first (`planning/preregistered/trade-the-flow.md`, `5c4e7c2`). **Every
  statistical part supported, no economic part**: the edges are real and under 1.2 bps gross against 11 bps of taker fees.
- [ ] 6. **`rank-the-universe`**: lifts D8 (operator's call). `gr.factors` adds cross-sectional `rank`, `top`, `zscore`,
  `neutralize` over `ts`, and A8–A11 run on BTC, ETH, HYPE and the reference venues.
  D8 lifted by the operator 2026-10-04 ("use all available tickers"). Registered alone first
  (`planning/preregistered/rank-the-universe.md`, `d333e55`): every Binance USDT perpetual, point in time with delisted
  coins, the 50 most liquid each day, 6 rankings × 2 sides (N = 12). Built: `galata-fetch daily|funding binance-um --all`,
  `gr.factors`, `notebooks/universe.py` (`068cd1f`). **The run follows the download.**
- [ ] 7. **`build-the-portfolio`**: A12 on DCC Σ per origin. Inverse-vol, HRP (López de Prado 2016, pinned to its
  example) and minimum variance, scored with `gr.stats`.

---

## Pitfalls to guard against

Each of these is a common way a trading-side backtest flatters itself. Every item onboarded here avoids it.

| The pitfall | The rule here |
|---|---|
| A signal read at bar *i* and filled at `close[i]` with stops tested on the close | a position decided at `close_ts` earns the next bar; a stop fills inside the bar from its high and low (`gr.overlays`) |
| A permutation test that re-runs one strategy, not the search that chose it | the whole search re-run per permutation (`studies.permutation_test`) |
| Sharpe on a fixed 252 or 365, a daily resample of equity | `periods_per_year(interval)`, calendar time |
| A pandas `shift(1)` that spans a hole silently | polars only, null across a hole |
| The last bar trusted as closed because the venue says so | closure by the record; indicators on `closed_only` bars |
| A flat % slippage | measured spread and depth (`gr.liquidity`) |
| A consistency ratio or held-up ratio read as a verdict | DSR, PBO and registered criteria; ratios only as descriptions |
