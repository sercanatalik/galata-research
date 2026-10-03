# investing-algorithm-framework: what to take, what to leave

**NOT PROPOSED.** Written 2026-10-03 from a read of
[coding-kitties/investing-algorithm-framework](https://github.com/coding-kitties/investing-algorithm-framework)
at `f8e82c7` (v9.0.0a22, 2026-09-30, Apache-2.0). This file lists its features and its signals,
says which ones galata-research already has, and names the ones worth onboarding, in order.

---

## Contents

- [The verdict](#the-verdict)
- [What the framework is](#what-the-framework-is)
- [Features, against what galata-research has](#features-against-what-galata-research-has)
- [Signals](#signals)
- [Onboarding order](#onboarding-order)
- [Mismatches to guard against](#mismatches-to-guard-against)

---

## The verdict

**Take the ideas. Do not take the dependency.** The framework is a strategy-to-live-trading
stack: ccxt, FastAPI, SQLAlchemy, AWS and Azure deployment, and pandas in its signatures.
galata-research never talks to a venue, keeps pandas out of every signature (D4), and owns
its rules once in polars. Each item below is reimplemented as a `gr` function. Each comes
with a test that fails when its guard is removed, and with a published figure where one exists.

Its research core is thinner than ours. We already have permutation testing, walk-forward,
vol targeting and robustness ranking, with DSR, PBO, Reality Check and SPA. What it adds is
**breadth on the trading side**. That means a standard indicator set, a cross-sectional
factor pipeline, trade-level metrics, position overlays such as stops and cooldowns, and
portfolio construction.

Its showcase marks three signal families "not possible", because the framework has no perp,
funding or order-book data: funding carry (🟡), basis (🟡) and microstructure (🔴). The
record has all three. Those are where galata-research has the edge.

---

## What the framework is

| Layer | What it ships |
|---|---|
| Strategy | `TradingStrategy` subclass: `data_sources`, `schedule`, `prepare_signal_data`, boolean `buy/sell/short/cover` signals, long and short, NETTING or HEDGE position modes |
| Vector backtest | Polars-backed, sweeps thousands of parameter variants; signals read at bar *i*, filled at `close[i]` |
| Event backtest | Bar by bar with orders, fills, portfolio state; the same strategy class as live |
| Studies | `Study` with `sample_type` (in-sample, OOS-time, OOS-universe, walk-forward, stress, Monte Carlo, exploratory); `BacktestWindow` (rolling, anchored, holdout) |
| Monte Carlo | `run_monte_carlo_test`: OHLCV relative moves shuffled from `start_index`, the strategy re-run, one-sided empirical p per metric |
| Metrics | 103 fields on `BacktestMetrics` (list below) |
| Robustness | `rank_by_cross_study_robustness` (OOS ÷ in-sample "held-up ratio"), CV consistency across windows, top selection |
| Pipeline | Zipline-style `Factor` on a long panel: `Returns`, `SMA`, `RSI`, `Volatility`, `AverageTradedValue`, `RollingBeta`, `CrossSectionalMean`, `Neutralize`, `StaticPerSymbol`; `.rank(mask)`, `.top(n)`, `.bottom(n)`, filters |
| Confluence cards | Explainable entries: required primary rules, weighted evidence groups, vetoes, score threshold |
| Risk rules | `ExposureRule`, `PositionSize`, `ScalingRule` (scale-in ladder), `StopLossRule` (trailing), `TakeProfitRule`, `CooldownRule` (side-blocking, in bars) |
| Costs | `TradingCost`: % fee, fixed fee, % slippage, or a `SlippageModel` (`VolumeShareSlippage`, `FixedBasisPointsSlippage`) |
| Storage | `.obtf` Open Backtest Format bundles (zip + Parquet), SQLite index to rank 10,000+ runs |
| Reporting | Self-contained HTML dashboard; an MCP server for AI tools to query backtests |
| Deployment | Local, AWS Lambda, Azure Functions, Finterion; ccxt order executors; paper trading |
| Indicators | `pyindicators` (same authors): `ema`, `sma`, `rsi`, `macd`, `bollinger_bands`, `supertrend`, `stochastic_oscillator`, `zscore`, `crossover`, `crossunder` |

---

## Features, against what galata-research has

✅ have it · ➕ onboard · ➖ leave · ⚠️ have it, theirs is weaker

| Feature | galata-research today | Call |
|---|---|---|
| Monte Carlo permutation test | `studies.permute_bars`, `permutation_test` (mcpt), `random_timing`, the whole search re-run per permutation | ✅ ⚠️ theirs re-runs one strategy, not the search, so its p ignores selection |
| Walk-forward / OOS windows | `timeseries.walk_forward_origins`, `fitted_through` on every forecast, registered splits | ✅ |
| Sharpe, PSR, DSR, PBO, RC, SPA, MCS | `gr.stats`, `gr.models.evaluate` | ✅ they have none of DSR, PBO, RC, SPA or MCS |
| Vol-targeting overlay | `gr.models.vol.target`, feedback control, ES sizing | ✅ |
| VaR / CVaR | `evaluate.value_at_risk` (FHS) with Kupiec, Christoffersen, DQ, FZ0 | ✅ |
| Max drawdown, performance fee | `stats.max_drawdown`, `performance_fee` | ✅ |
| Fees on turnover, funding charged | `backtest.returns(fee=, funding=)` | ✅ ⚠️ they have no funding |
| Slippage from the book | `liquidity.cost_of_size`, `schedule_cost` | ✅ ⚠️ theirs is a flat % or volume share |
| **Sortino, Calmar, Omega, Ulcer index, recovery, drawdown duration** | max drawdown only | ➕ `gr.stats` |
| **Rolling Sharpe** | no | ➕ `gr.stats.rolling_sharpe` |
| **Monthly/yearly return table, % winning months** | no | ➕ for notebooks |
| **Trade-level metrics**: win rate, profit factor, win/loss, MAE/MFE, duration, consecutive wins/losses, exposure ratio | positions per bar only; `studies.runs` gives runs of position | ➕ `gr.trades` from `runs` |
| **Window consistency score** (1 − CV across windows, % profitable windows) | per-split verdicts in notebooks | ➕ small: a `stats.consistency` over a per-window frame |
| **Cross-study held-up ratio** | replication tables (`survival_table`) | ➖ ours is stronger: registered verdicts, not a ratio |
| **Cross-sectional factor pipeline** (rank, top-N, mask, neutralize, rolling beta) | none; D8 kept studies to one ticker | ➕ `gr.factors` in polars `over("ts")`, once a multi-ticker study is chosen |
| **Position overlays**: stop-loss (trailing), take-profit, cooldown, scale-in ladder | none | ➕ `gr.overlays`: each a function from a position series to a position series, so DSR still counts every variant |
| **Confluence cards** (weighted evidence, vetoes) | none | ➖ for now. It is a rule engine, and every threshold is another trial to count. Revisit if a signal combiner is needed |
| **Portfolio construction**: inverse-vol risk parity, HRP, Markowitz | `gr.models.corr` gives Σ per origin | ➕ `gr.portfolio` on DCC Σ, after a multi-ticker study is chosen |
| Backtest bundles + SQLite index | trial frames returned whole (true N) | ➖ until trial counts outgrow memory. If they do, store Parquet per trial; .obtf is not needed |
| Event-driven backtest, live/paper, deployment, ccxt, credentials | out of scope by design | ➖ never: research reaches a live system only through a person |
| HTML dashboard, MCP server | marimo notebooks | ➖ |

### Metrics we lack, by name

From `BacktestMetrics`, the ones with no `gr` equivalent:
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

### A. From the framework, which we lack

| # | Signal | Their source | Parameters (theirs) | Note |
|---|---|---|---|---|
| A1 | EMA crossover (trend) | showcase 01, `pyindicators.ema` + `crossover` | fast, slow | we have SMA crossover (`studies.moving_average`); add `kind="ema"` |
| A2 | Bollinger + RSI mean reversion | showcase 02 | BB 20/2σ, RSI 14, 30/70 | a new family; Wilder's RSI pinned to its 1978 worked example |
| A3 | RSI alone (overbought/oversold) | README example, `RSI` factor | 14, 30/70 | the A2 baseline |
| A4 | MACD signal-line cross | `pyindicators.macd` | 12/26/9 | trend family |
| A5 | Supertrend (ATR bands), with EMA confirm | tutorial strategy | ATR 10, ×3 | needs ATR. ATR on a hole must be null, the same rule as `returns` |
| A6 | Stochastic oscillator | `pyindicators` | %K 14, %D 3 | low priority |
| A7 | Volatility breakout (Donchian) | showcase 07 | channel N | ✅ we have `donchian_ensemble` |
| A8 | Cross-sectional momentum, top-N | showcase 03, `Returns(30).rank().top(n)` | 30d, N | needs `gr.factors` and a multi-ticker universe |
| A9 | Multi-factor: momentum + low vol + liquidity gate | showcase 04 | 30d returns, vol rank, ATV filter | the liquidity gate can use `gr.liquidity` spread, not ATV |
| A10 | Pairs: z-score of log-price spread | showcase 05, `zscore` | BTC/ETH, lookback, ±2 | hedge ratio fitted on the window only; both legs on one `ts` grid |
| A11 | Beta-neutral residual (rolling beta, `Neutralize`) | `RollingBeta`, `Neutralize` factors | window | datawatch already stores a `beta` signal: read it through `gr.signals` |
| A12 | Risk parity / HRP / Markowitz weights | showcase 09–11 | rebalance | allocation, not timing; Σ from `gr.models.corr` |
| A13 | Vol-targeting overlay | showcase 12 | target σ | ✅ `gr.models.vol.target` |
| A14 | DCA | showcase 08 | — | ➖ a benchmark at most |

### B. Where the record beats the framework

These are the signal families the framework cannot run. The record holds their data.

| # | Signal | Their status | Our data | Note |
|---|---|---|---|---|
| B1 | **Funding carry**: hold against the funding sign, net of fees | 🟡 no funding provider | `market.funding` settled + premium, `funding_live` on `recv_ts`, Binance 8h | `backtest.returns(funding=)` already charges it. The predicted rate is known only on `recv_ts`: cross with `join_recv` |
| B2 | **Funding / premium mean reversion**: extreme premium → fade | — | `funding.premium`, `marks.premium` | as B1 |
| B3 | **Basis**: mark − oracle, perp vs spot | 🟡 spot proxy only | `marks` (mark, oracle, mid) | mark and oracle are on `recv_ts` only |
| B4 | **Order-flow imbalance** | 🔴 no book, no trades | `liquidity.flow`, `liquidity.ofi`, quotes, Bybit book | known at the bucket's end |
| B5 | **Micro-price / book imbalance** | 🔴 | top of book, `reference.book` depth within 2 and 10 bps | 1 s rows |
| B6 | **Lead–lag across venues** | 🔴 | `gr.leadlag` (HY, LLR) | a signal only if the lead exceeds latency; within ±50 ms is clock skew |
| B7 | **Jump-conditioned** (fade or follow after a Lee–Mykland jump) | — | `gr.jumps.lee_mykland` | the scale uses earlier returns only |
| B8 | **Scheduled-event windows** (CPI, FOMC, funding hour) | — | `reference.events`, `calendar` | flat or reduced exposure around releases |
| B9 | **Vol-regime switch** (trend on in low vol, off in high vol) | — | `gr.models.vol` forecasts, `msgarch` regimes | conditioning variable read at `fitted_through` |
| B10 | **Open interest change** | — | `marks` OI | `recv_ts` only |

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
- [ ] 4. **`trade-the-funding`**: B1–B3 on settled funding and marks, through `join_recv`. This is the first family
  the framework itself cannot run.
- [ ] 5. **`trade-the-flow`**: B4–B6 on trades, quotes and the Bybit book. Costs come from `liquidity.effective`,
  not a flat fee.
- [ ] 6. **`rank-the-universe`**: lifts D8 (operator's call). `gr.factors` adds cross-sectional `rank`, `top`, `zscore`,
  `neutralize` over `ts`, and A8–A11 run on BTC, ETH, HYPE and the reference venues.
- [ ] 7. **`build-the-portfolio`**: A12 on DCC Σ per origin. Inverse-vol, HRP (López de Prado 2016, pinned to its
  example) and minimum variance, scored with `gr.stats`.

---

## Mismatches to guard against

Each of these is a way their code would break one of ours if copied.

| Theirs | Ours | Consequence for a port |
|---|---|---|
| Vector backtest reads signal at bar *i* and fills at `close[i]` | position decided at `close_ts` earns the next bar | Same return, if the signal uses only data ≤ `close_ts`. Their stops test against `close[i]`, so no intrabar stop: ours must use high/low |
| Monte Carlo: one strategy re-run per permutation | the whole search re-run per permutation | their p ignores the parameter search, so do not port their p-values |
| `annual_volatility`, Sharpe on 252/365 conventions, daily resample of equity | `periods_per_year(interval)`, calendar time | recompute; never copy a constant |
| `pandas=True` data sources, pandas `shift(1)` in metrics | polars only, null across a hole | a pandas `shift` spans holes silently |
| `is_final` / last bar trusted as given by ccxt | closure by the record | indicators must run on `closed_only` bars |
| Flat % slippage | measured spread and depth | use `gr.liquidity` costs |
| Consistency = 1 − CV, held-up ratio | DSR, PBO, registered verdicts | add consistency as a description only, never as a verdict |
