# Registered: does separating the jumps improve HAR, and does the proxy decide it?

**Written and committed on 2026-09-29, alone, before the jump measures or
the jump models exist in code or have forecast these series.** It is
roadmap item 34, from Phase 2's rows *HAR-CJ / HAR-RV-J with bipower
variation* and *5m RV as the proxy*.

## What is known before this commit

- **The literature is split.** Andersen, Bollerslev and Diebold (2007,
  *REStat*) split RV into a continuous part and significant jumps (the
  Barndorff-Nielsen and Shephard bipower test, α = 0.999) and found the jump
  part adds little to the forecast, often with a negative sign. Corsi,
  Pirino and Renò (2010, *J. Econometrics*) detect jumps with threshold
  bipower instead, and find their effect positive and mostly significant.
  On Bitcoin, Shen, Urquhart and Wang (2020, *European Financial
  Management*) find the jump coefficients of HAR-RV-CJ mixed in sign and
  significance across horizons, and HARQ-F-J best out of sample.
- **One diagnostic was run before this file**, on Binance BTC 5-minute
  log returns, 2020-01-01 to 2026-09-26 (2,461 whole days), with the
  bipower ratio statistic below: 10.6% of days have z above Φ⁻¹(0.999),
  the mean max(RV − BV, 0)/RV is 7.7%, and 0.18% of returns are exactly
  zero (so bipower's zero-return bias is negligible). No forecast was made.
- **Item 24 and item 27** already hold HAR and HARQ on 5-minute RV, walked
  on the split below and cached in `notebooks/har_long.py`. HARQ on
  5-minute RV, once its level is fixed, beats GARCH at 1d on both tickers.

## The measures, per bucket (1d or 4h) from its 5-minute log returns

The bucket's M returns are those `gr.timeseries.realized_from` uses (a whole
bucket only; nothing outside the bucket is read). μ₁ = √(2/π),
μ₄/₃ = 2^(2/3) Γ(7/6)/Γ(1/2), θ = π²/4 + π − 5.

- RV = Σ r².
- BV = μ₁⁻² (M/(M−1)) Σⱼ₌₂ |rⱼ||rⱼ₋₁| (Barndorff-Nielsen and Shephard 2004).
- TQ = M μ₄/₃⁻³ (M/(M−2)) Σⱼ₌₃ |rⱼ rⱼ₋₁ rⱼ₋₂|^(4/3).
- **The bipower test:** z_BNS = √M (RV − BV)/RV / √(θ max(1, TQ/BV²))
  (Huang and Tauchen 2005; ABD 2007 eq. 19–20).
- **The threshold versions** (Corsi, Pirino and Renò 2010): a local variance
  V̂ⱼ for each return from the bucket's own returns by CPR's iterated
  Gaussian-kernel filter (bandwidth L = 25, the return and its two
  neighbours excluded, returns above 3² V̂ removed each pass, until nothing
  more is removed); the threshold ϑⱼ = 3² V̂ⱼ; each |rⱼ|^γ above its
  threshold replaced by its expectation beyond the threshold under a normal
  (the corrected, "C-", form: 1.094 ϑⱼ^(1/2) at γ = 1). C-TBV and C-TTQ are
  BV and TQ built from those, and z_CTz is z_BNS with them in place of BV
  and TQ.
- **The split** for each test: J = 1{z > Φ⁻¹(0.999)} · max(RV − X, 0), with
  X = BV or C-TBV, and C = RV − J (ABD 2007 eq. 21–22).

## The models

HAR's walk-forward (`gr.models.vol.har`), the target RV, lags (1, 7, 30) at
1d and (1, 6, 42) at 4h, OLS, expanding window, `min_obs = 250`, the
insanity filter as it is:

- `har`: 1, RVₛ, RVʷ, RVᵐ (the benchmark; item 24's).
- `harj` (HAR-RV-J, ABD 2007): adds max(RVₛ − BVₛ, 0), untested.
- `harcj` (HAR-RV-CJ, ABD 2007): 1, Cₛ, Cʷ, Cᵐ, Jₛ, Jʷ, Jᵐ, from the
  bipower split.
- `hartcj` (HAR-TCJ, CPR 2010): the same from the threshold split.

## Procedure

Item 24's setting: Binance USDⓈ-M BTC and ETH 1m klines from
`gr.reference`, bars from 2020-01-01 to before 2026-09-27; the measures
from 5-minute bars built from them; out of sample from 2024-09-01 00:00 UTC;
refits every 5 bars at 1d and every 6 at 4h; horizons 1, 7, 30 at 1d and
1, 6, 42 at 4h, cumulative. Scoring: `align`, then QLIKE per model and
horizon by `scorecard`, on the rows the model and its benchmark share.

- **The primary proxy** is item 24's: squared bar returns,
  `proxies(bars, "r2")`.
- **The second proxy** is 5-minute RV scaled to the squared return's level:
  c · RV₅, with c = Σ r² / Σ RV₅ over the bars closing by 2024-09-01, per
  (ticker, bars): item 27's scale, fitted before the split. 5-minute RV sits
  9–23% above r² (item 24), so an unscaled RV₅ proxy would reward whichever
  forecast shares its level. QLIKE ranks forecasts consistently under any
  conditionally unbiased proxy (Patton 2011), so the two proxies should
  agree in expectation, and RV₅'s smaller noise should give sharper tests.

## Hypotheses and rules, per cell {BTC, ETH} × {1d, 4h}

| # | claim | rule |
|---|---|---|
| H7 | HAR-CJ beats HAR | *consistent* if harcj QLIKE < har QLIKE at every horizon; *contradicts* if at none; *mixed* otherwise |
| H8 | HAR-TCJ beats HAR | the same, for hartcj |
| H9 | HAR-TCJ beats HAR at every horizon, significantly | *yes* if `uspa(model="hartcj", benchmark="har", reps=499)` p < 0.05, else *no* |
| H10 | the verdict does not turn on the proxy | H7, H8 and item 24's H2 (min(HAR, HARQ) against GARCH, RV from 5 minutes) re-decided under the second proxy with their own rules: *robust* if at least 9 of the 12 (hypothesis, cell) verdicts equal those under r², else *proxy-dependent* |

Twelve cell verdicts for H7–H9, and one H10 verdict over twelve
comparisons. A cell the data cannot form (a model that fails, a refit the
walk refuses) is *can't tell*, with the reason, and is never re-run under
this number with other settings. `harj` is walked and reported, and decides
nothing.

## Predictions

- **H7:** *mixed* or *contradicts* in at least 3 of 4 cells. ABD's jump part
  added little, and on Bitcoin its signs are mixed.
- **H8:** *consistent* at 1d on both tickers, *mixed* at 4h (48 returns a
  bucket give the threshold little to work with).
- **H9:** *no* in every cell. The gain, if any, is small against HAR.
- **H10:** *robust*.

## What would count

- **Jumps pay** if H8 is *consistent* in at least 3 cells and H9 is *yes*
  in at least one.
- **Jumps do not pay** if neither H7 nor H8 is *consistent* in any cell.
- Otherwise **neither**, by this registration.

---

## Results

*Appended after the run. Nothing above this line may change.*
