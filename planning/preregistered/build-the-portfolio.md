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
