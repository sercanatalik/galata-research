# Pre-registered: weighting a fixed book of perpetuals, by three methods and three covariance estimators

**Registered 2026-10-04, before any portfolio of them was computed.** This
file is committed alone, before the code that runs it, so its git history is
the evidence of that. Nothing below may change after the run without a new
file that says why. The data is the reference store fetched for
`rank-the-universe` (`galata-fetch daily|funding binance-um --all`). No
statistic of these portfolios has been computed.

## The claims being tested

This is allocation, not timing: given a book of coins, how much of each?

- **H1, 1/N.** DeMiguel, Garlappi and Uppal ("Optimal Versus Naive
  Diversification", *Review of Financial Studies* 22, 2009) find that no
  optimised portfolio reliably beats equal weight out of sample, because
  estimation error eats the optimisation's gain. The claim tested is the
  opposite: **some optimised portfolio beats 1/N on net return after the
  search.** Expected: not supported.
- **H2, HRP.** López de Prado ("Building Diversified Portfolios that
  Outperform Out of Sample", *Journal of Portfolio Management* 42, 2016)
  reports that Hierarchical Risk Parity has a lower out-of-sample variance
  than the minimum-variance portfolio, because it never inverts the
  covariance matrix. The claim tested: **HRP's realised variance is lower
  than long-only minimum variance's under every covariance estimator.**
  Expected: not supported. A long-only minimum variance is itself a strong
  low-variance portfolio.
- **H3, DCC.** Engle (2002), and this repository's D13: a covariance fitted to
  each coin's volatility dynamics and their time-varying correlation forecasts
  better than a rolling sample. The claim tested: **the minimum-variance
  portfolio built on DCC's Σ has a lower realised variance than the one built
  on the 90-day sample Σ.** Expected: uncertain.

## The book, frozen

- **The 10 Binance USDT perpetuals with the highest mean dollar volume over
  the 30 days ending 2021-12-31**, among those with at least 500 daily bars by
  then, excluding the index perpetuals and USDC (`gr.factors.INDEX_AND_STABLE`).
  They are chosen on data known on that day and never revisited. A coin
  delisted later stays in the book until its last bar.
- **Span:** rebalances every 7 days from **2022-01-01**, held from the next
  day, to 2026-09-30.
- **The live set:** at a rebalance, the book's coins with a bar that day.
  After a delisting, every estimator is computed again on the survivors. For
  DCC that is a new walk-forward on the survivors' joint history.

## The portfolios

At each rebalance, from data up to its close:

**Three covariance estimators** of the 7-day Σ, all on daily returns
(`gr.timeseries` returns, null across a hole; the joint sample of the live set):
- `sample`: the last 90 days' covariance × 7;
- `ewma`: RiskMetrics, λ = 0.94, × 7;
- `dcc`: `gr.models.corr.walk_forward(model="gjr", dist="t", corr="dcc",
  window="expanding", every=28, horizons=(7,))`, its `cum_covariance` at the
  rebalance's close. Refitted every 28 days on the expanding window, as the
  library does.

**Three weighting methods**, all long-only and fully invested:
- `ivp`: inverse variance, w ∝ 1/σᵢ²;
- `hrp`: López de Prado's: distance √(½(1 − ρ)), single linkage on the
  Euclidean distances between the distance matrix's columns,
  quasi-diagonalisation, recursive bisection with inverse-variance cluster
  weights;
- `minvar`: the long-only minimum variance, w ≥ 0, Σw = 1.

**9 trials** (method × estimator), and **`1/N`**, equal weight on the live set.
All are held fixed between rebalances (`gr.factors.portfolio`), turnover pays
**Binance's 0.05% taker**, and funding is charged per day held.

## What would count as support

- **H1:** White's Reality Check over the 9 trials' daily net returns in
  excess of 1/N's (`stats.reality_check`, 1,000 replicates, Politis–White
  block, seed 0). **Supported if p < 0.05.**
- **H2:** for each estimator, Diebold–Mariano on the daily squared net
  returns, HRP's against minvar's (`gr.models.evaluate.dm`, h = 7). **Supported
  if HRP's are lower with a one-sided p < 0.05/3 under all three estimators.**
- **H3:** Diebold–Mariano on the daily squared net returns, `minvar dcc`
  against `minvar sample` (h = 7). **Supported if DCC's are lower with a
  one-sided p < 0.05.**

Anything short of a hypothesis's criterion is reported as **not supported**.

**Described, not tested:** each trial's annualised volatility, Sharpe,
maximum drawdown and Ulcer index (`gr.stats.describe`); its weights' effective
number of coins (1/Σw²); its mean turnover; the funding it paid; and how each
behaved through the coin delisted in the span, if any.

---

## Result — run once, 2026-10-04, `notebooks/portfolio.py`

*Appended after the run. Nothing above this line was changed.* Code at
`c73d595`, on the reference store fetched for `rank-the-universe` (899
perpetuals). The run took 157 s.

**The book**, chosen on 2021-12-31: **BTC, ETH, XRP, BNB, ADA, DOGE, ATOM,
LINK, EOS, LTC.** The registered 500-bar rule, there so DCC could be fitted,
left out several of that day's most liquid perpetuals that were younger than
500 days: LUNA (third by 30-day dollar volume, $1.64bn a day, 338 bars),
MATIC, SAND, NEAR and SOL. **So the book did not hold LUNA through its May 2022
collapse.** The rule was registered for another reason, but it is a selection
on age that kept the year's worst event out, and this result says nothing
about how the methods would have weathered it. **EOS** traded until
2025-05-21 (Binance's EOS perpetual ended there) and is the one delisting in
the span. 1,733 days, 2022-01-02 to 2026-09-30.

### H1, beating 1/N: **not supported**

White's Reality Check over the 9 against 1/N: **p = 0.167** (best by mean
excess: `ivp sample`). Every one of the 9 had a higher Sharpe than 1/N's 0.16,
and a smaller drawdown or one within two points of it. But no optimised book
beat equal weight by more than the search over 9 explains.

### H2, HRP below minimum variance: **not supported, and the opposite holds**

| estimator | DM (HRP's squared returns − minvar's), h = 7 | one-sided p for HRP lower |
|---|---|---|
| sample | +7.19 | 1.000 |
| ewma | +4.64 | 1.000 |
| dcc | +5.95 | 1.000 |

HRP's realised variance is significantly **higher** than long-only minimum
variance's under every estimator. HRP spreads the book over 7.1–7.6
effective coins, while minimum variance concentrates it in 1.8–2.1.

### H3, DCC below the sample for minimum variance: **not supported**

DM −1.42 for `minvar dcc` against `minvar sample`, a one-sided **p = 0.077**.
Lower, as Engle's model predicts, but short of 0.05.

### Described, not tested

| trial | volatility | Sharpe | CAGR | max drawdown | Ulcer | effective coins | turnover / rebalance | funding paid |
|---|---|---|---|---|---|---|---|---|
| minvar dcc | 50.1% | 0.39 | +7.0% | 64.7% | 0.35 | 1.84 | 0.38 | 19.5% |
| minvar sample | 51.5% | 0.47 | +11.6% | 60.5% | 0.30 | 2.09 | 0.21 | 14.3% |
| minvar ewma | 52.8% | 0.32 | +3.2% | 61.0% | 0.34 | 1.94 | 0.46 | 14.1% |
| ivp dcc | 58.2% | 0.18 | −6.5% | 66.6% | 0.42 | 7.46 | 0.16 | 20.1% |
| hrp ewma | 58.4% | 0.26 | −1.7% | 69.1% | 0.41 | 7.06 | 0.28 | 18.1% |
| ivp ewma | 58.4% | 0.22 | −4.1% | 67.3% | 0.41 | 7.53 | 0.15 | 19.8% |
| hrp sample | 58.6% | 0.25 | −2.6% | 68.1% | 0.41 | 7.41 | 0.18 | 18.4% |
| ivp sample | 58.7% | 0.26 | −2.3% | 66.8% | 0.40 | 7.76 | 0.07 | 19.6% |
| hrp dcc | 58.9% | 0.20 | −5.5% | 68.7% | 0.43 | 7.56 | 0.24 | 18.8% |
| 1/N | 64.0% | 0.16 | −10.1% | 68.7% | 0.47 | 9.70 | 0.47 | 21.1% |

Every long-only book of these coins lost 60–69% at its worst: the 2022 bear
market is in the span, and weighting only changed how much. Minimum
variance's lower volatility comes from concentration, about two effective
coins (mostly BTC with one other), not from diversification. Over the 10 days
to EOS's last bar, the minimum-variance books gained 1.5–2.2% while 1/N lost
5.3%. Minimum variance had already moved EOS's weight to 0.
