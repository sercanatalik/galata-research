# Notebooks

One marimo notebook per question, grouped by what kind of study it is.
Filenames are unchanged from before the folders existed, so the paths that
`planning/preregistered/` recorded at run time still name the same files.

Run any of them with `uv run marimo edit notebooks/<folder>/<name>.py`, and
check them all as CI does with `uv run marimo check notebooks/*/*.py`.

## `record/` — what the record holds, and how the loaders read it

| Notebook | Asks |
|---|---|
| `candles.py` | the candles, deduplicated, closed by the record, with the frontier |
| `ticks.py` | trades and quotes, one row per execution, in the venue's order |
| `gaps.py` | where the capture has holes, loaded on the receipt clock |
| `clocks.py` | the two clocks: the venue's time and the receipt time |
| `account.py` | my margin snapshots, positions, fills and ledger history |

## `backtests/` — signal families, overlays, and the statistics that judge a backtest

| Notebook | Asks |
|---|---|
| `moving_average.py` | a Sharpe landscape for the moving-average crossover family |
| `momentum.py` | the same for time-series momentum |
| `indicator_signals.py` | five indicator families the frameworks ship, pre-registered |
| `carry.py` | does trading the funding pay? Pre-registered |
| `flow.py` | does the order flow pay a taker? Pre-registered |
| `overlays.py` | do stops, take-profits and cooldowns help? Pre-registered |
| `donchian_ensemble.py` | the pre-registered test of a published strategy |
| `deflated_sharpe.py` | does the best of the trials beat what luck would give? (DSR) |
| `overfitting.py` | does choosing on the past choose well? (PBO) |
| `random_timing.py` | is it the timing, or just the exposure? (shuffled-runs twins) |
| `whole_set.py` | does anything in the set beat its benchmark? (Reality Check, SPA) |
| `permuted_bars.py` | is there structure to find at all? (bar-permutation null) |
| `trades.py` | the trade-level report (win rate, payoff, MAE/MFE, drawdown length) for the trials the other studies run |

## `volatility/` — realized measures, the GARCH family and HAR, and their replays

| Notebook | Asks |
|---|---|
| `volatility.py` | what was the volatility? Realized volatility, five ways |
| `garch.py` | GARCH, GARCH-t and variations, walked forward, scored and traded |
| `replication.py` | does the BTC volatility study replicate on ETH and HYPE? |
| `vol_long.py` | does the volatility study hold on six years of BTC and ETH? |
| `har_long.py` | does *HAR beats GARCH* hold on six years? And the jump measures |
| `mcs_split.py` | does *GARCH outside the multi-horizon MCS* turn on the split? |
| `persistence_1h.py` | is 1h persistence of one neglected variance breaks? |
| `long_memory.py` | is that persistence true long memory, or level shifts? |

## `correlation/` — covariance forecasting from the volatility fit

| Notebook | Asks |
|---|---|
| `correlation.py` | DCC, cDCC and EWMA covariance from the volatility fit, walked forward |
| `correlation_study.py` | does DCC forecast covariance better than the simpler estimators? |

## `portfolio/` — the cross-section and the book

| Notebook | Asks |
|---|---|
| `universe.py` | ranking every Binance perpetual: momentum, low volatility, size, pre-registered |
| `portfolio.py` | weighting a book of perpetuals: three weightings × three covariance estimators against 1/N, pre-registered |

## `liquidity/` — the liquidity study, its claims and its forward tests

| Notebook | Asks |
|---|---|
| `liquidity.py` | when is the market liquid? (I: the liquidity week) |
| `liquidity_costs.py` | what trading costs: the spread, a size, the flow, the refill, depth against volatility |
| `liquidity_clock.py` | whose clock the market keeps, where it jumps, releases, the weekend reopen |
| `liquidity_venues.py` | across venues: who moves first, and whose price is the price |
| `liquidity_stress.py` | under stress and ahead of time: extreme hours, forecasting, working an order |
| `liquidity_claims.py` | what survives, set against the literature's and the vendors' claims |
| `liquidity_forward.py` | do the findings hold on data not yet seen? (pre-registered) |

`liquidity_claims.py` imports the other five study notebooks as modules and
embeds their results, so the liquidity notebooks stay in one folder together.
