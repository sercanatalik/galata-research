# Pre-registered: funding carry and funding fades on Binance BTC and ETH, 2020–2026

**Registered 2026-10-03, before any backtest of them was run.** This file is
committed alone, before the code that runs it, so its git history is the
evidence of that. Nothing below may change after the run without a new file
that says why. The funding and premium archives were fetched only to check
their format (January–March 2024, and two days of premium); no return,
Sharpe or funding statistic was computed from them.

## The claims being tested

**H1, carry.** A perpetual's funding pays one side to hold it. Hedged against
the index, a position earns the funding and bears only the perp's premium
over the index. He, Manela, Ross and von Wachter ("Fundamentals of Perpetual
Futures", 2022, revised 2024) report that trading the perp–spot deviation
earns high Sharpe ratios after costs, and practitioners run the one-sided
form ("cash and carry": short the perp, long spot, while funding is
positive). The claim tested: **a hedged carry rule earns a net Sharpe that
survives the search over rules.** The expectation, stated before the run:
supported over 2020–2026 as a whole, since funding was persistently positive
in long stretches. If so, the open question is how much of it was one regime,
and that is described, not tested.

**H2, persistence.** Carry needs funding to persist. **Daily funding's
lag-one autocorrelation is positive** at p < 0.01, per ticker. Expected:
supported. This is the mechanism check on H1, not a separate edge.

**H3, the fade.** A common practitioner claim is that extreme funding marks
crowded positioning, so a perp position against it earns a return. The
claim tested: **an unhedged position against extreme funding beats buy-and-hold
after the search.** Expected: not supported. The record's directional searches
found nothing, and this is one more directional search.

## The data, frozen

- **Binance USDⓈ-M BTCUSDT and ETHUSDT** from `gr.reference` (`galata-fetch
  funding`, `premium` and `candles` on `binance-um`): settled funding (monthly
  files), the 1m premium index, and 1m klines. From 2020-01-01 to the end of
  the last whole month held at run time.
- **Daily bars, UTC.** A day's premium is the close of its last minute, kept
  only when all 1,440 minutes are held. A day's funding is the sum of the
  settlements in `(ts, close_ts]`, kept only when they cover its 24 hours. The
  perp's daily bars are whole buckets of the 1m klines (`gr.timeseries.resample`).
- A day missing any of these is a hole: nothing is earned across it, and the
  rules restart their windows after it (`gr.indicators.add`).

## H1: the carry rules

A position s ∈ {−1, 0, +1} is decided at a day's close and held through the
next day. s = −1 is short the perp and long the index (receives positive
funding); s = +1 is the reverse. Per day held, net return
= s × (Pₜ − Pₜ₋₁) / (1 + Pₜ₋₁) − s × Fₜ − 0.0015 × |Δs|, where P is the premium
index at the close and F the day's funding. 0.0015 is Binance's base taker fee
on both legs (0.05% the perp, 0.10% spot) per unit of turnover.

| Rule | Position | Grid |
|---|---|---|
| `always` | −1 every day | — |
| `carry d h both` | −sign(F̄_d) when \|F̄_d\| > h, else 0 | d ∈ {1, 7, 30} days, h ∈ {0, 0.0003} a day |
| `carry d h short_only` | −1 when F̄_d > h, else 0 | the same |

F̄_d is the mean daily funding over the last d days, known at the close.
0.0003 a day is 1 bp per 8-hour settlement. **13 rules × 2 tickers: N = 26.**

**Stated simplifications:** the hedge is the premium index, which Binance
builds from impact prices against a multi-venue spot index, not a fill on one
spot book. s = +1 needs spot borrowed and short, and its borrow cost is not
charged. `short_only` and `always` need no borrow. Margin and the capital on
the spot leg are not modelled: returns are per unit of notional.

**Support for H1 needs all three:**
1. the best of the 26 has **DSR ≥ 0.95 at N = 26**;
2. **Reality Check p < 0.05 against cash** over the 26 (`stats.reality_check`
   on the net returns, 1,000 replicates, Politis–White block, seed 0);
3. **PBO < 0.5** over the 26 (16 blocks).

**Described, not tested:** each rule's net Sharpe per calendar year, and the
share of its net return that was funding and not basis.

## H2: persistence

Per ticker, the lag-one autocorrelation of daily funding over the whole
sample, with a t-test on its Fisher z (n − 3 degrees of freedom).
**Supported if positive with p < 0.01 on both tickers.**

## H3: the fade

z is daily funding's z-score over the last w days (population σ, `gr.indicators.zscore`).
The perp position is −1 from z > k until z < 0.5, and +1 from z < −k until
z > −0.5 (`gr.indicators.hold`); `long_flat` replaces −1 with 0. The perp's
daily close-to-close return is charged its settled funding
(`gr.reference.funding_hours`) and 0.05% taker on turnover.

Grid: w ∈ {30, 90} × k ∈ {1.5, 2.0} × {long_short, long_flat}: **8 rules × 2
tickers, N = 16.**

**Support for H3 needs all three:** DSR ≥ 0.95 at N = 16, Reality Check
p < 0.05 against buy-and-hold on the same ticker (funding charged on it too),
and PBO < 0.5.

Anything short of a hypothesis's criteria is reported as **not supported**.
No rule is tuned, dropped or re-run with other parameters under this name.

---

## Result — run once, 2026-10-03, `notebooks/carry.py`

*Appended after the run. Nothing above this line was changed.* Code at
`b46db00`. Data fetched the same day into a fresh reference store with
`galata-fetch funding|premium|candles binance-um BTC ETH --from 2020-01-01
--to 2026-09-30`: every file `ok` against Binance's published checksum,
except **31 ETH premium days that Binance never published** (2021-01-18 to
01-24, 2021-02-02 to 02-20, and 2021-05-31, 06-02, 06-04, 06-27, 06-28). They are holes, as
registered. The span ran 2020-01-01 to 2026-09-29: **2,456 BTC days and 2,425
ETH days** with both a whole premium day and whole funding. (2026-09-30 is not
one, because its closing settlement is in October's file, which is not yet published.)

### H2, persistence: **supported**

| ticker | consecutive days | ρ (lag one) | Fisher z | p |
|---|---|---|---|---|
| BTC | 2,449 | 0.851 | 62.3 | < 1e-300 |
| ETH | 2,412 | 0.821 | 56.9 | < 1e-300 |

### H1, hedged carry: **supported**

| criterion | value | met |
|---|---|---|
| DSR of the best, N = 26 | 1.000 | yes |
| Reality Check p against cash | 0.002 | yes |
| PBO, 16 blocks, 2,151 shared days | 0.00 | yes |

The best trial is **`always` on BTC: short the perp, long the index, every
day.** Its annualised net Sharpe is 9.18, and it earned +118.6% on notional
over the span, almost all of it funding. Of the 26 trials, 24 have an annualised Sharpe above 2.6. **No rule beat
`always`** on BTC, and on ETH only `carry 30 0 short_only` did (8.62 against
8.45). Timing the carry on funding's recent sign or level added nothing over
holding it. The two-sided daily rule (`carry 1 0 both`) was the worst, at 0.56
and 0.46: flipping on one day's sign pays both legs' fees for noise.

### H3, the fade: **not supported**

| criterion | value | met |
|---|---|---|
| DSR of the best, N = 16 | 0.616 | no |
| Reality Check p against buy-and-hold | 0.934 | no |
| PBO, 16 blocks | 0.13 | yes |

The best is `fade 90 1.5 long_flat` on BTC, at an annualised Sharpe of 0.86
against buy-and-hold's 0.72 (both with funding charged). It is long-only and
flat after extreme funding, so it is mostly buy-and-hold. Every `long_short`
fade on ETH lost money.

### Described, not tested: the carry by year (`always`)

| year | BTC Sharpe | BTC funding earned | ETH Sharpe | ETH funding earned |
|---|---|---|---|---|
| 2020 | 9.6 | 16.7% | 12.0 | 26.6% |
| 2021 | 13.9 | 30.6% | 13.3 | 27.8% |
| 2022 | 7.2 | 4.1% | 0.9 | 0.9% |
| 2023 | 13.9 | 7.8% | 13.5 | 8.2% |
| 2024 | 15.8 | 11.9% | 17.2 | 13.0% |
| 2025 | 14.6 | 5.1% | 9.9 | 4.9% |
| 2026 (to 09-29) | 7.5 | 2.2% | 4.5 | 1.4% |

The income has compressed. It was 17–31% a year in 2020–2021 and is 1–2% so
far in 2026, and ETH's carry nearly vanished in 2022. The basis leg moved the
position by less than 0.3% in any year. **What H1 supports is that funding was
persistently paid to shorts over 2020–2026, and the evidence is strong. It
does not support a carry that pays today.** The stated simplifications stand,
and every one of them flatters: no borrow (unused by `always`), no margin or
capital on the spot leg, no exchange or liquidation risk, and the index
standing in for a spot fill. At 2026's pace, about 3% a year on BTC and 2% on
ETH, the income is a tenth of 2020–2021's, and that is before the costs left
out above.
