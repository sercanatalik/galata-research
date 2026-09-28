# Registered: does HAR beat GARCH on six years of BTC and ETH?

**Written and committed on 2026-09-28, alone, before any HAR or GARCH run on
the data below.** No forecast has been made on these bars, and the code that
builds them does not exist yet. Nothing here may change after this commit;
a deviation is reported as a deviation.

## Why

*HAR beats GARCH* (Bergsli, Lind, Molnár and Polasik 2022) is decided in
only two cells of the replication (`replication.md`): BTC 1d, and none at
4h, where the record's 1h bars start in 2026-03. Binance's published 1m
klines reach back to 2019-12-31, which gives years of out-of-sample
forecasts at both 1d and 4h, on BTC and ETH.

## Data

- **Source.** `gr.reference.candles`, Binance USDⓈ-M BTCUSDT and ETHUSDT 1m
  klines, fetched from `data.binance.vision`, as held on 2026-09-28 (2,462
  days each, 2019-12-31 to 2026-09-26, none missing).
- **Sample.** Bars opening from 2020-01-01 00:00 UTC to before
  2026-09-27 00:00 UTC.
- **Bars.** 1d and 4h OHLC bars built from the 1m klines on the UTC grid.
  A bar is kept only if all its 1m bars exist (1,440 or 240). Nothing is
  filled; a missing bar breaks the returns across it, as on the record.
- **Venue.** This is Binance, not Hyperliquid. The claim is about the
  method on BTC and ETH, not about one venue's prices.

## Procedure

As `notebooks/garch.py` walks and scores, with its functions, and no other
choice made after this file:

- **Split.** Out of sample from 2024-09-01 00:00 UTC (about 25 months: 755
  daily and ~4,530 4h forecast origins).
- **Models.** `ewma, garch, gjr, egarch, har, harq`. GARCH-family fits use
  Student-t innovations, constant mean, `min_obs = 250`,
  `simulations = 500`; EGARCH one step only (item 22). No deseasonalising.
- **Refits.** Every 5 bars at 1d and every 6 bars at 4h, as in `garch.py`.
- **Horizons.** 1d: 1, 7 and 30 bars. 4h: 1, 6 and 42 bars.
- **HAR's realized variance.** Two versions, each a separate hypothesis:
  - **(A) as the record does it**: RV from 4h bars at 1d, and from 1h bars
    at 4h (`gr.timeseries.realized_from`).
  - **(B) from 5-minute returns**: RV from 5m bars built the same way from
    the 1m klines. This is the literature's standard
    (Liu, Patton and Sheppard 2015).
- **Scoring.** `gr.models.evaluate`: `align` (cumulative, a gap drops the
  rows after it), then QLIKE per model and horizon by `scorecard`.
  - The proxy is squared returns (`proxies(bars, "r2")`), the default the
    record's verdict used.
  - QLIKE ranks forecasts consistently under any conditionally unbiased
    proxy (Patton 2011), so r² is noisy but not biased toward either model.

## Hypotheses and rules

| # | claim | rule |
|---|---|---|
| H1 | HAR beats GARCH, RV as (A) | *consistent* if min(HAR, HARQ) QLIKE < GARCH QLIKE at every horizon; *contradicts* if at none; *mixed* otherwise |
| H2 | HAR beats GARCH, RV as (B) | the same rule |
| H3 | HARQ beats GARCH at every horizon, RV as (A) | *yes* if `evaluate.uspa(aligned, model="harq", benchmark="garch", reps=499)` gives p < 0.05, else *no* |

Each is decided per cell: {BTC, ETH} × {1d, 4h}, giving 12 verdicts. A cell
the data cannot form (a model that fails to fit) is *can't tell*, with the
reason. It is never re-run with other settings under this number.

## Predictions

- **H1 and H2:** *consistent* in all four cells.
  - BTC 1d is the record's verdict: 3 of 3 horizons (replication.md).
  - The rest follow Bergsli et al. 2022, who found HAR on intraday RV
    ahead of daily GARCH for Bitcoin.
- **H3:** *yes* in all four cells. BTC 1d on the record gave uSPA p 0.004.
- **Expected weak spot (not a prediction):** the 30-day and 42-bar horizons.
  There GARCH's mean reversion and HAR's monthly term both pull toward the
  same long-run level, so the gap should narrow.

## What would count

- **The claim survives** on long history if H1 is *consistent* in all four
  cells.
- **It is venue- or sample-specific** if H1 is *contradicts* in two or more.
- **Anything else is mixed.** H2 says whether the finer RV changes the answer,
  and H3 whether HARQ's edge over GARCH is significant rather than merely
  lower.

## References

- Bergsli, L. Ø., Lind, A. F., Molnár, P. and Polasik, M. (2022).
  Forecasting volatility of Bitcoin. *Research in International Business
  and Finance* 59, 101540.
- Corsi, F. (2009). A simple approximate long-memory model of realized
  volatility. *Journal of Financial Econometrics* 7(2), 174–196.
- Bollerslev, T., Patton, A. J. and Quaedvlieg, R. (2016). Exploiting the
  errors: a simple approach for improved volatility forecasting. *Journal of
  Econometrics* 192(1), 1–18.
- Liu, L. Y., Patton, A. J. and Sheppard, K. (2015). Does anything beat
  5-minute RV? A comparison of realized measures across multiple asset
  classes. *Journal of Econometrics* 187(1), 293–311.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect
  volatility proxies. *Journal of Econometrics* 160(1), 246–256.

---

## Results

*Appended after the run. Nothing above this line may change.*

### Run of 2026-09-28 (`notebooks/har_long.py`)

- **Run.** All 32 walks completed; none skipped.
  - Forecast origins at h = 1: 756 daily and 4,536 4h.
  - `fitted_through ≤ close_ts` on every row.
- **Deviations.** None in the procedure.
- **A display correction, not a rule change.** The notebook's first
  "measured" text divided Patton's QLIKE means, which are negative at these
  scales, so its ratios read inverted. It now shows the scorecard's
  textbook QLIKE ratio (below 1: better). The verdicts, which compare raw
  QLIKE as `garch.py` does, did not change.

| # | BTC 1d | BTC 4h | ETH 1d | ETH 4h |
|---|---|---|---|---|
| H1 | **consistent**: best HAR ÷ GARCH 0.987, 0.957, 0.804 | **mixed** (1 of 3): 0.991, 1.088, 1.482 | **consistent**: 0.967, 0.870, 0.621 | **consistent**: 0.968, 0.952, 0.996 |
| H2 | **mixed** (1 of 3): 0.960, 1.046, 1.042 | **mixed** (1 of 3): 0.968, 1.079, 1.597 | **consistent**: 0.958, 0.891, 0.790 | **mixed** (2 of 3): 0.971, 0.985, 1.164 |
| H3 | **yes** (uSPA p 0.016) | **no** (p 1.000) | **yes** (p 0.000) | **no** (p 0.152) |

Ratios are at horizons 1, 7, 30 (1d) and 1, 6, 42 (4h).

**Against *What would count*: mixed.** H1 is consistent in three cells and
contradicts in none. Against the predictions:
- **H1:** held in 3 of 4 cells.
- **H2:** held in 1 of 4.
- **H3:** held in 2 of 4. It is significant at 1d, not at 4h.

**Read.**
- **At 1d, HAR beats GARCH on both tickers, and HARQ significantly.** The
  record's BTC verdict holds on six years of Binance bars. The gap
  **widens** with the horizon (0.80 and 0.62 at 30 days), against the
  registration's expectation that it would narrow.
- **At 4h it does not travel.** On BTC, HAR loses beyond one bar (1.48 at 42
  bars, 7 days). On ETH, HAR is lower at every horizon but not significantly.
- **5-minute RV makes HAR worse here, not better** (H2 below H1 in three
  cells). A diagnostic, *not registered*: over 2020-01 to 2024-08, mean
  5-minute RV is 1.09–1.23× the mean squared return it is scored against,
  while the record's coarser RV is 0.89–1.06×. HAR forecasts its own
  measure's level, and QLIKE against r² charges that bias. This contradicts
  Liu, Patton and Sheppard's (2015) ranking only if the proxy is taken as
  truth; it says the proxy and the RV must be built at the same frequency.
