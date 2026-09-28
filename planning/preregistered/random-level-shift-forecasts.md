# Registered: does a random level shift model forecast volatility better than GARCH and HAR?

**Written and committed on 2026-09-28, alone, before the model exists in
code or has forecast these series.** Item 32
(`long-memory-or-level-shifts.md`) found the volatility's persistence is
level shifts or a trend, not true long memory. If so, a model built for
level shifts should forecast better than GARCH, whose α+β ≈ 1 stands in for
them. Lu and Perron (2010) found their random level shift (RLS) model had
the smallest MSE in 64 of 72 cases on stock indices.

## The model

Lu and Perron's (2010) RLS model on y_t = log |r_t|:

- **Observation.** y_t = a + τ_t + c_t, with c_t ~ N(0, σ_c²) (quasi-ML:
  log |z| is not normal).
- **The level.** τ_t = τ_{t−1} + δ_t, where δ_t = 0 with probability 1 − p
  and N(0, σ_η²) with probability p. τ_0 is diffuse (variance 10⁴).
- **Estimation.** Maximum likelihood by a mixture Kalman filter: each
  step's two branches (shift or not) are updated, then collapsed to one
  Gaussian by moment matching. Parameters (a, p, σ_η, σ_c) are found by
  scipy's L-BFGS-B, with p ∈ [10⁻⁴, 0.5].
- **The variance forecast.** The level is a random walk, so its forecast is
  flat: v̂ = κ · exp(2(a + τ_{t|t})) for every h ≤ H, and the cumulative
  variance is h · v̂.
  - κ = mean of r_s² / exp(2(a + τ_{s|s−1})) over the fitting window, using
    only data up to the refit.
  - A zero return is missing in y: the filter predicts through it and
    does not update.

**Validation, before the run** (`tests/levels.py`). On a series simulated
from the model (T = 5,000, p = 0.01, σ_η = 1, σ_c = 0.5), p and σ_η must be
recovered within a factor of two. The filtered level must track the true
level with correlation above 0.9.

## Procedure

Item 24's setting, so the RLS forecasts are scored against the same GARCH
and HAR forecasts, already walked and cached in `notebooks/har_long.py`:

- **Data and split.** Binance BTC and ETH, 1d and 4h bars from 2020-01-01
  to 2026-09-27, out of sample from 2024-09-01.
- **The walk.** The origins are `gr.timeseries.walk_forward_origins`.
  Parameters are refitted every 30 bars at 1d and every 42 bars at 4h, on
  an expanding window. Between refits the filter runs with fixed
  parameters, updating the level at every bar.
- **Horizons.** 1d: 1, 7, 30. 4h: 1, 6, 42.
- **Scoring.** As item 24: `align` against `proxies(bars, "r2")`, then
  QLIKE per model and horizon on the rows every model shares. The HAR
  models use the record's RV construction (item 24's (A)).

## Hypotheses and rules, per cell {BTC, ETH} × {1d, 4h}

| # | claim | rule |
|---|---|---|
| R1 | RLS beats GARCH at every horizon | *consistent* if RLS QLIKE < GARCH QLIKE at every horizon; *contradicts* if at none; *mixed* otherwise |
| R2 | RLS beats HAR at every horizon | the same, against min(HAR, HARQ) |

## Predictions

- **R1:** *consistent* in all four cells (Lu and Perron 2010; item 32).
- **R2:** *consistent* in at least two cells. HAR's lags stand in for level
  changes too, so the gap should be smaller.

## What would count

- **Level shifts forecast better** if R1 is *consistent* in three or more
  cells.
- **Not in these forecasts** if R1 is *consistent* in at most one cell.

## References

- Lu, Y. K. and Perron, P. (2010). Modeling and forecasting stock return
  volatility using a random level shift model. *Journal of Empirical
  Finance* 17(1), 138–156.

---

## Results

*Appended after the run. Nothing above this line may change.*

### Run of 2026-09-28 (`notebooks/har_long.py`, item 33 section)

- **Validation, run first** (`tests/levels.py`). The fit recovers the
  simulation's p = 0.0055 (true 0.01), σ_η = 1.51 (1.0) and σ_c = 0.500
  (0.5), and the filtered level tracks the true one (ρ > 0.9).
- **The run.** 487 s. Item 24's cached GARCH and HAR forecasts were
  reused, and item 24's twelve verdicts reproduce unchanged.
- **Deviations.** None.

| # | BTC 1d | BTC 4h | ETH 1d | ETH 4h |
|---|---|---|---|---|
| R1 | **mixed** (1 of 3): RLS ÷ GARCH 1.057, 1.193, 0.961 | **contradicts**: 1.035, 1.141, 1.522 | **mixed** (2 of 3): 1.010, 0.988, 0.832 | **contradicts**: 1.039, 1.129, 1.486 |
| R2 | **contradicts** | **contradicts** | **contradicts** | **contradicts** |

**Against *What would count*: not in these forecasts.** R1 is consistent in
no cell. Both predictions failed:
- R1 was predicted consistent in four cells; it is consistent in none.
- R2 was predicted consistent in two or more; HAR beats RLS at every
  horizon in every cell.

**Read.**
- **One step ahead, RLS loses everywhere** (1.01–1.06× GARCH's loss). The
  likely reason is that the base model's noise c_t is i.i.d., so it has no
  short-run clustering; GARCH and HAR do. Later work (Xu and Perron 2014)
  adds mean reversion and time-varying jump probabilities; the registered
  base model had neither.
- **At 4h and 42 bars, the flat forecasts lose to GARCH badly.** Seven days
  ahead, GARCH's pull toward its mean wins:
  - RLS 1.49–1.52×;
  - EWMA 1.43–1.52×;
  - BTC's HAR and HARQ 1.48–1.49×.

  Only ETH's HAR is level with GARCH (1.001, HARQ 0.996).
- **Only at 1d and long horizons does RLS beat GARCH:** 0.96 (BTC, 30
  days), and 0.99 and 0.83 (ETH, 7 and 30 days). Even there HAR is better
  (0.80 and 0.62 at 30 days).
- **Item 32's level shifts are real in the spectrum, but a model of them
  alone does not forecast better.** It lacks the short-run dynamics, and
  HAR's weekly and monthly averages already track slow level changes.
