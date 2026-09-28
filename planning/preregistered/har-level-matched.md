# Registered: does HAR on 5-minute RV lose to GARCH only through its level?

**Written and committed on 2026-09-28, alone, before the rescaled forecasts
exist.** It follows `har-vs-garch-long-history.md` (item 24). There, HAR on
5-minute RV (H2) did worse than HAR on the record's coarser RV (H1). A
diagnostic outside that registration found 5-minute RV 1.09–1.23× the mean
squared return it is scored against, over 2020-01 to 2024-08. Those ratios
are known here; the out-of-sample effect of removing them is not.

## Why the level, not the proxy

Roadmap item 27 was named *matched proxy*. Scoring against an RV proxy at
HAR's own frequency would also move GARCH's target: GARCH forecasts the
squared return. The comparison would then favour whichever model shares
the proxy's level. So this test keeps item 24's proxy (r²) and removes
HAR's level bias instead, with a scale fitted on the estimation period
alone.

## Procedure

Item 24's walks and data, unchanged: `notebooks/har_long.py`, the same
forecasts. The two HAR models (`har`, `harq`) under each RV version are
rescaled per (ticker, bars, RV version):

- **The scale** is c = Σ r² / Σ RV over the bars whose `close_ts` is at or
  before 2024-09-01 00:00 UTC.
  - r² is the squared log return of the bar.
  - RV is the version's realized variance of the same bar.
  - Only bars where both exist are used.
- **The rescaled forecast** is c × each forecast's `variance` and
  `cum_variance`. GARCH, EWMA, GJR and EGARCH are untouched.
- **Scoring** is item 24's: `align` against `proxies(bars, "r2")`, then
  QLIKE per model and horizon by `scorecard`.

## Hypotheses and rules

| # | claim | rule |
|---|---|---|
| H4 | rescaled HAR beats GARCH, RV from 5 minutes | *consistent* if min(HAR, HARQ) QLIKE < GARCH at every horizon; *contradicts* if at none; *mixed* otherwise |
| H5 | rescaled HAR beats GARCH, RV as the record builds it | the same rule |
| H6 | rescaled HARQ beats GARCH at every horizon, RV from 5 minutes | *yes* if `uspa(model="harq", benchmark="garch", reps=499)` p < 0.05 |

Cells: {BTC, ETH} × {1d, 4h}, twelve verdicts.

## Predictions

- **H4:** *consistent* in all four cells. The level bias explains H2.
- **H5:** at least as many consistent cells as H1 had (3 of 4). The record's
  RV is 0.89–1.06× r², so rescaling it should change little.
- **H6:** *yes* at 1d and *no* at 4h, as H3 was.

## What would count

- **The level explains H2** if H4 is *consistent* in at least three cells,
  including both at 1d.
- **The level does not explain it** if H4 has no more consistent cells than
  H2 had (1).

---

## Results

*Appended after the run. Nothing above this line may change.*

### Run of 2026-09-28 (`notebooks/har_long.py`, item 27 section)

- **Run.** Item 24's cached walks, unchanged; its twelve verdicts reproduce
  exactly.
- **Deviations.** None.
- **The scale c** (Σr² ÷ ΣRV, bars closing by 2024-09-01):

| | BTC 1d | BTC 4h | ETH 1d | ETH 4h |
|---|---|---|---|---|
| 5 minutes | 0.898 | 0.828 | 0.920 | 0.815 |
| the record's RV | 1.084 | 0.941 | 1.129 | 0.960 |

| # | BTC 1d | BTC 4h | ETH 1d | ETH 4h |
|---|---|---|---|---|
| H4 | **consistent**: best HAR ÷ GARCH 0.949, 0.958, 0.898 | **mixed** (2 of 3): 0.952, 0.982, 1.245 | **consistent**: 0.954, 0.859, 0.699 | **consistent**: 0.957, 0.925, 0.932 |
| H5 | **mixed** (1 of 3): 1.001, 1.015, 0.897 | **mixed** (1 of 3): 0.983, 1.054, 1.370 | **consistent**: 0.972, 0.897, 0.715 | **consistent**: 0.967, 0.943, 0.958 |
| H6 | **yes** (uSPA p 0.026) | **no** (p 1.000) | **yes** (p 0.000) | **yes** (p 0.002) |

**Against *What would count*: the level explains H2.** H4 is consistent in
three cells, both at 1d, against H2's one. Against the predictions:
- **H4:** held in 3 of 4 cells.
- **H5:** not held. It was consistent in 2 cells, below the predicted 3.
- **H6:** held in 3 of 4. ETH 4h is significant, where *no* was predicted.

**Read.**
- **Once its level is fixed, 5-minute RV is mostly the better HAR input.**
  Rescaled 5-minute HARQ has a lower ratio than item 24's best H1 HAR in 9
  of 12 cell-horizons. It loses at BTC 1d at 7 and 30 days and at ETH 1d at
  30 days. At 4h, where H1 failed on BTC, it is better at every horizon.
  That is mostly Liu, Patton and Sheppard's (2015) ranking, recovered.
- **Rescaling the record's coarser RV hurts BTC 1d** (1 of 3 horizons, from
  3 of 3). Its in-sample level ratio (1.08) did not hold out of sample, so the
  coarse RV's bias is not a stable constant to remove.
- **The finding item 24 left open is closed.** HAR on 5-minute RV lost
  through its level, not its information.
