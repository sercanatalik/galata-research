# Registered: does feedback targeting beat open-loop targeting on net Sharpe?

Written and committed on 2026-09-27, before any run of the procedure below
on the confirmatory span. Nothing here may change after that commit; a
deviation is reported as a deviation.

## What was already seen

Roadmap item 16 ran feedback targeting at the notebook's default split (70%
of BTC's daily history, 2025-08-30) and reported the year after it: every
model's net Sharpe rose (EWMA −0.38 → +0.14, GARCH −0.37 → −0.06, EGARCH
−0.23 → −0.07). That year, 2025-08-30 to 2026-09-27, is **seen**. The returns
before 2025-08-30 were fitted on, but no one has seen feedback or open-loop
targeting trade them.

## Hypothesis

**H1.** On BTC daily, net of the taker fee, feedback-controlled targeting has a
higher Sharpe ratio than open-loop inverse-volatility targeting built on the
same model's σ̂.

**H2** (the paper's own claim; descriptive, not tested). Feedback's realized
volatility is closer to the target: |ln(realized/τ)| is smaller than
open-loop's for every model.

## Data and procedure

- **Data.** `gr.market.candles(["BTC"], "1d", ...)`, the bars with
  `close_ts ≤ 2026-09-27T00:00Z`: 1,309 bars starting 2023-02-26.
- **Split.** `close_ts` of bar ⌊0.4 · 1309⌋ − 1 = 522, which is
  **2024-08-02T00:00Z**. It is the notebook's rule at share 0.4.
- **Forecasts.** `notebooks/garch.py`'s `walk` at that split, unchanged:
  `dist="t"`, refit every 5 bars, horizons (1, 7, 30), `min_obs=250`, no
  deseasonalising.
- **Models.** `ewma, garch, gjr, egarch`, and `har, harq` if the record lets
  them be walked from the split. If they are refused (4h realized variance
  starts 2024-06-14), they are reported as skipped and not replaced.
- **Targeting.** `vol.trials(bars, walked, split=split, rules=("inverse_vol",
  "feedback"), bands=(0.0,))`, with the defaults: τ = the estimation
  period's realized volatility, cap 2, `backtest.TAKER_FEE`. Feedback uses
  g = 55, θ = 0.6, half-life 126 and burn-in 10, as published. Nothing is
  tuned.
- **Confirmatory span.** Trial rows with `close_ts ≤ 2025-08-30T00:00Z`: from
  the first bar after the split to the old split, about 390 bars.

## Test

- **Pairing.** For each model, a = the feedback trial's `net` and b = the
  inverse_vol trial's `net`, on the same bars. A bar null in either is
  dropped from both.
- **Test.** `gr.models.evaluate.sharpe_difference(a, b)`, as Ledoit and Wolf
  (2008): the studentized circular block bootstrap, two-sided, M = 4,999
  replicates, seed 0. The block is automatic: the circular-block
  Politis–White length, 1.5^{1/3} × the median stationary length over the
  four moment columns, rounded up.
- **Multiplicity.** Holm's step-down at α = 0.05 across the models actually
  walked.

## Decision rule

Let k be the number of models walked.
- **Confirmed**: Holm rejects H0 with Δ̂ > 0 for more than k/2 models, and
  Δ̂ > 0 for every model.
- **Refuted**: Δ̂ ≤ 0 for at least k/2 models.
- **Not confirmed**: anything else. This includes a positive Δ̂ everywhere
  with too few rejections, which says the data cannot tell.

The verdict is stated in these words.

## Secondary (reported as such, never as the verdict)

- The same test on the seen year (the 70% split's own walk, 2025-08-30 to
  2026-09-27) and on 2024-08-02 to 2026-09-27 from the 40% walk.
- The block-size sensitivity: b ∈ {1, 2, 4, 6, 8, 10}, Ledoit and Wolf's grid.
- H2's |ln(realized/τ)| per model and span, and the HAC p-value beside the
  bootstrap's.

## References

- Ledoit, O. and Wolf, M. (2008). Robust performance hypothesis testing with
  the Sharpe ratio. *Journal of Empirical Finance* 15(5), 850–859.
- Nosek, B. A., Ebersole, C. R., DeHaven, A. C. and Mellor, D. T. (2018). The
  preregistration revolution. *PNAS* 115(11), 2600–2606.
- Harvey, C. R. (2017). Presidential address: the scientific outlook in
  financial economics. *Journal of Finance* 72(4), 1399–1440.
