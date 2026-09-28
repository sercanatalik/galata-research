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
