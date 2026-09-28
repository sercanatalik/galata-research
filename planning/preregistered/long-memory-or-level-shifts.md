# Registered: is the volatility's persistence true long memory, or level shifts?

**Written and committed on 2026-09-28, alone, before the test exists in code
or has seen these series.** Items 29 and 31 found 1h GARCH persistence of
one on six years of Binance BTC and ETH:
- deseasonalising does not remove it (item 29);
- fitting between κ₂ variance breaks does not bring it below 0.99
  (item 31).

The candidates left are true long memory and level shifts too small or too
frequent for κ₂ at 30-day spacing. Both produce a slowly decaying
autocorrelation in volatility. They differ near frequency zero, which is
what Qu's (2011) test uses.

## The test

Qu's (2011) W statistic tests the null of stationary long memory against
spurious long memory from level shifts or a smooth trend. It uses the
derivatives of the profiled local Whittle likelihood:

- **The periodogram.** I_j = |Σ x_t e^{−iλ_j t}|² / (2πT) at
  λ_j = 2πj/T, j ≥ 1.
- **The memory estimate.** d̂ is the local Whittle estimate (Robinson 1995)
  from the first m frequencies, minimised over [−0.5, 2.5].
- **The statistic.** With Ĝ = mean(λ_j^{2d̂} I_j) and
  ν_j = log λ_j − mean(log λ_j),
  W = sup over r ∈ [ε, 1] of |Σ_{j ≤ ⌊mr⌋} ν_j (I_j / (Ĝ λ_j^{−2d̂}) − 1)| / √(Σ_{j ≤ m} ν_j²).
- **Settings.** m = ⌊1 + T^0.7⌋, ε = 0.02. Critical values 1.118 (10%),
  1.252 (5%), 1.517 (1%), from Qu (2011).
- **Reference implementation.** This follows `LongMemoryTS::Qu.test` and
  `local.W`.

**Validation, before the run.** The implementation goes in
`gr.models.memory` with tests that must pass first:
- **Size.** On 200 simulated fractional noises (d = 0.3, T = 5,000), it
  rejects at 5% in at most 10% of draws.
- **Power.** On 200 series of white noise plus random level shifts, it
  rejects in at least half.
- **The estimator.** The local Whittle estimate recovers d = 0.3 within 0.1
  on average.

## Data

- **Source.** Binance USDⓈ-M BTCUSDT and ETHUSDT 1m klines as held on
  2026-09-28, resampled to whole bars (`gr.timeseries.resample`), from
  2020-01-01 to before 2026-09-27.
- **Cells.** {BTC, ETH} × {1h, 1d}.
- **The series tested.** x = log |r|, with r the log return.
  - At 1h, r is deseasonalised by hour × weekday factors fitted on the
    whole sample, as in item 31.
  - At 1d, r is the raw log return.
  - Returns of exactly zero are dropped, and their count reported.

## Rule, per cell

- **Long memory not rejected** if W ≤ 1.252.
- **Spurious (level shifts or trend)** if W > 1.252.

Reading over the four cells:
- **true long memory**: not rejected in all four;
- **level shifts**: rejected in three or four;
- **mixed**: otherwise.

## Prediction

**Long memory not rejected in all four cells.** The JRFM (2020) study
"True versus Spurious Long Memory in Cryptocurrencies" (13(9), 186) found
true long memory in Bitcoin and Ethereum volatility with univariate tests,
Qu's among them. Item 31 found no breaks large enough to explain the
persistence.

**Context, not decided.**
- d̂ at m = T^0.5, T^0.6, T^0.7 and T^0.8. Under level shifts, d̂ falls
  as m grows (Perron and Qu 2010); under long memory it is stable.
- FIGARCH's d on the same returns.

## References

- Perron, P. and Qu, Z. (2010). Long-memory and level shifts in the
  volatility of stock market return indices. *Journal of Business &
  Economic Statistics* 28(2), 275–290.
- Qu, Z. (2011). A test against spurious long memory. *Journal of Business
  & Economic Statistics* 29(3), 423–438.
- Robinson, P. M. (1995). Gaussian semiparametric estimation of long range
  dependence. *Annals of Statistics* 23(5), 1630–1661.
- True versus spurious long memory in cryptocurrencies (2020). *Journal of
  Risk and Financial Management* 13(9), 186.

---

## Results

*Appended after the run. Nothing above this line may change.*

### Run of 2026-09-28 (`notebooks/long_memory.py`)

- **Validation, run first** (`tests/memory.py`). All three checks pass:
  - the local Whittle estimate recovers d = 0.304 against the true 0.3 (50
    draws);
  - Qu's test rejects true long memory in 3.5% of 200 fractional noises
    (size ≤ 10% required);
  - it rejects level shifts in 97% of 200 draws (power ≥ 50% required).
- **Deviations.** None. FIGARCH's d, listed as context, was not computed:
  the test answered without it.

| cell | T | zeros dropped | m | d̂ (m = T^0.7) | W | verdict |
|---|---|---|---|---|---|---|
| BTC 1h | 59,049 | 14 | 2,187 | 0.357 | **4.22** | spurious |
| BTC 1d | 2,460 | 0 | 237 | 0.225 | **1.76** | spurious |
| ETH 1h | 59,007 | 56 | 2,186 | 0.358 | **4.11** | spurious |
| ETH 1d | 2,459 | 1 | 237 | 0.239 | **1.78** | spurious |

**Reading: level shifts.** All four cells reject true long memory at 5%,
and at 1% (1.517) too. The prediction, long memory not rejected anywhere,
failed in every cell.

d̂ by bandwidth (context):

| cell | m = T^0.5 | T^0.6 | T^0.7 | T^0.8 |
|---|---|---|---|---|
| BTC 1h | 0.52 | 0.43 | 0.36 | 0.24 |
| BTC 1d | 0.44 | 0.35 | 0.23 | 0.14 |
| ETH 1h | 0.52 | 0.44 | 0.36 | 0.23 |
| ETH 1d | 0.40 | 0.31 | 0.24 | 0.16 |

**Read.**
- **The bandwidth pattern agrees.** d̂ falls steadily as more frequencies
  enter, in every cell. This is the pattern Perron and Qu (2010) show level
  shifts produce and long memory does not.
- **The persistence is spurious long memory.** Items 29 and 31 left it
  unexplained: not the daily cycle, not κ₂'s coarse breaks. It is low-
  frequency contamination:
  - level shifts smaller or more frequent than κ₂ found at 30-day spacing;
  - or a smooth trend, which Qu's alternative also covers. A secular change
    in the volatility level over 2020–2026 would be one; this run did not
    measure it.
- **This disagrees with the univariate result of the JRFM (2020) study** on
  an earlier sample. It agrees with that study's multivariate test, which
  found level shifts or trends.
- **For the models (inferred, not tested here).** GARCH's α+β ≈ 1 on these
  series likely measures the shifts, not memory, and a FIGARCH d would too.
  A random-level-shift variance model (Lu and Perron 2010, *Journal of
  Empirical Finance* 17(1), 138–156) is the natural next specification to
  forecast with.
