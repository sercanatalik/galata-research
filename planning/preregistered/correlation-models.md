# Registered: does DCC forecast covariance better than the simpler estimators?

Written and committed on 2026-09-28, before any forecast below is scored on
any span. Nothing here may change after that commit; a deviation is reported
as a deviation.

## What was already seen

`add-dcc` (roadmap item 25) fitted DCC in sample on the whole 4h joint
sample (`notebooks/correlation.py`): all six instruments a = 0.0055,
b = 0.9838; the main dex (BTC, ETH, HYPE) a + b = 0.9998 under DCC against
0.9955 under cDCC. The same notebook walked GJR-t DCC forward from a 70% split
and showed only the last origin's Σ. galata-datawatch's signals were
computed from the last close. **No loss, of any model, on any origin, has
been computed**, and no other estimator below has been walked on this data.
The parameters were seen; the forecasts' accuracy was not.

## Hypotheses

On BTC, ETH and HYPE at 4h, one bar ahead, over the confirmatory span:

- **H1 (Σ).** DCC over GJR-t margins has a lower mean multivariate QLIKE
  than (a) RiskMetrics' EWMA covariance at λ = 0.94 and (b) the
  equal-weight sample covariance of the last 180 bars (30 days).
- **H2 (R alone).** With the same GJR-t margins, DCC has a lower mean
  multivariate QLIKE than (a) constant correlation (CCC) and (b) EWMA on
  the standardised returns at λ = 0.94 (`iewma`). With the σ shared, these
  differences are exactly differences in the correlation likelihood.
- **H3 (economic, secondary).** The global minimum-variance portfolio built
  from DCC's Σ has a lower mean realised squared return than those from
  CCC and from EWMA (Engle and Colacito 2006, with μ = 1: Patton and
  Sheppard 2009, §4.1).

cDCC is walked and reported beside DCC, and tested against nothing.

## Data and procedure

- **Data.** `gr.market.candles(["BTC", "ETH", "HYPE"], "4h", …)` through
  `gr.mask_gaps(…, "candles")`, bars with `close_ts ≤ 2026-09-28T00:00Z`;
  log returns (`gr.timeseries.returns(kind="log")`). The joint sample is
  3,969 returns from 2024-12-05 12:00Z (HYPE's first), counted on
  2026-09-28 before any forecast.
- **Split.** The close of joint row ⌊3969 / 2⌋ − 1 = 1983,
  **2025-11-01T04:00Z**. Origins from the split: 1,986; the last has no
  target bar, so 1,985 are scored.
- **Forecasts.** `gr.models.corr.walk_forward(returns, split=split,
  horizons=[1])`, expanding window, **refit every 6 bars** (daily), seed 0,
  `min_obs=500`, no deseasonalising, at the library commit this file is
  committed with:
  - `dcc`: `model="gjr", dist="t", corr="dcc"`
  - `cdcc`: the same with `corr="cdcc"`
  - `ccc`: the same with `corr="ccc"`
  - `iewma`: the same with `corr="iewma", lam=0.94`
  - `ewma`: `corr="ewma", lam=0.94`
  - `sample`: `corr="sample", sample_window=180`
- **Scoring.** `gr.models.corr.score(forecasts, returns)` on the origins every
  model forecast whose target bar is in the joint sample, with r rᵀ as the
  proxy: `stein` = ln|H| + rᵀH⁻¹r (multivariate QLIKE), `frobenius`, and
  `gmv` = (wᵀr)² with w = H⁻¹1 / 1ᵀH⁻¹1. Both losses are robust to a noisy,
  conditionally unbiased proxy (Patton and Sheppard 2009; Laurent,
  Rombouts and Violante 2013).

## Tests

- **DM.** `gr.models.corr.compare(scores, loss=…, benchmark=…)`: d = model −
  benchmark, the Newey–West variance at ⌊4(T/100)^(2/9)⌋ lags, the HLN
  factor, and a two-sided p from t(T − 1). Here the "model" is DCC and each
  alternative is the benchmark, so a negative statistic favours DCC.
- **Primary family:** H1a, H1b, H2a and H2b on `stein`, with Holm's
  step-down at α = 0.05 across the four.
- **Secondary family:** H3 against CCC and against EWMA on `gmv`, with Holm
  at α = 0.05 across the two.
- **MCS.** The Model Confidence Set at 90% on `stein` over all six models,
  10,000 stationary-bootstrap replicates, seed 0, block as `compare` chooses
  it. It is reported, not tested.
- **Frobenius** is reported for every comparison, not tested: large
  variances and heavy tails dominate it.

## Decision rule

For each hypothesis: **supported** when DCC's mean loss is lower and its
Holm-adjusted p < 0.05; **refuted** when it is higher and the adjusted
p < 0.05; **undecided** otherwise.

A refusal (a model that cannot be walked from the split) is reported as a
refusal, and its hypotheses as undecided. Nothing is replaced.

## Secondary, descriptive

All six instruments at 4h, split at joint row 799 (the first refit then has
800 returns, over `min_obs`), same models and scoring. Reported beside the
primary, with no test.

## What the literature expects

Laurent, Rombouts and Violante (2012, *JAE* 27(6)) find constant
correlation hard to reject in calm periods, and dynamic correlation mattering
in turbulent ones; the univariate specification matters more than the
correlation dynamics. Engle and Colacito (2006) find dynamic correlation
lowers minimum-variance risk, most for highly correlated assets: BTC and ETH
here. RiskMetrics' λ = 0.94 was set on daily data and is used at 4h
unchanged, because it is what Tier 16 declares; that is a limitation of
H1a and H2b, stated before the result.
