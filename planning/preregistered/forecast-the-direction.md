# Pre-registered: does momentum tell the next move's direction, on Binance BTC and ETH, from 5 minutes to a day

**Registered 2026-10-26, before any feature, target or forecast of it was
computed.** This file is committed alone, before the code that runs it, so its
git history is the evidence of that. Nothing below may change after the run
without a new file that says why. The archives below were being downloaded
when this was written. Nothing has been computed from them. The exploration
behind it is `research/notes/momentum-high-frequency.md` in the workspace, and
the change is `forecast-the-direction`.

## The claims being tested

Momentum on this record has been tested only as a trading rule, at daily
frequency, and has not held up: the indicator families
(`indicator-signals.md`), the Donchian ensemble (`donchian-ensemble.md`),
cross-sectional momentum (`rank-the-universe.md`, reversed in 2025–2026), and
galata-datawatch's Tier 16 check of 7- and 28-day TSMOM. This is a different
question. **Does short- and long-term momentum, measured on 1-minute data,
carry information about the direction of the next move beyond what
volatility and the time of day already carry?** It is a claim about
forecasts, judged by forecasting tests. It has an economic part only where
the arithmetic allows one.

- **S, statistical, at 5m, 30m, 5h and 1d.** A direction model that adds
  momentum (B) forecasts better out of sample than the same model without it
  (A). Expected: **supported at 5m and 30m, mixed at 5h, not supported at
  1d.** At 5m the out-of-sample span holds about 390,000 independent
  periods, so a hit rate of 50.2% is detectable. At 1d it holds about 1,370,
  so 52.7% is needed.
- **E, economic, at 5h and 1d only.** The direction forecast, traded as a
  sign, pays after costs and the search. Expected: **not supported.** At 5m
  and 30m a taker flipping each period needs a hit rate of 89% and 66% to
  break even, so E is not tested there.

Kim, Tse and Wald (2016) find that most of time-series momentum's published
alpha comes from volatility scaling. So E is judged against the repository's
own volatility-targeted book, not only against buy-and-hold.

## The specification, frozen

**Data.**
- Binance USDⓈ-M **BTCUSDT and ETHUSDT 1m klines**, `gr.reference.candles(…,
  venues="binance-um")`, 2020-01-01 to 2026-09-30.
- Funding from `gr.reference.funding(…, venues="binance-um")` over the same
  span.
- Fetched by `galata-fetch candles|funding binance-um BTC ETH --from
  2020-01-01 --to 2026-09-30`. Days the archive lacks are reported, not
  filled.

**Bars.**
- 1m bars with zero volume and zero trade count are refused, and the count
  is reported.
- 1m bars are aggregated to **5m decision bars** with `timeseries.resample`.
  A 5m bar missing any of its five minutes is a hole.
- Log returns throughout. Nothing is interpolated. No window spans a hole: a
  feature or target whose window contains a hole is null, and its row is
  dropped from fitting and scoring.

**Volatility, σ.**
- Each 5m bar's RV is the sum of its five squared 1m log returns.
- The RV is divided by the square of its `hour_x_weekday` seasonal factor
  (`timeseries.seasonal_factors(…, by="hour_x_weekday", stat="mean_abs")` on
  5m returns).
- σ is the square root of an EWMA of that deseasonalised RV, with a
  **half-life of 288 bars (one day)**.
- The seasonal factors are **refit every Monday 00:00 UTC on all bars closed
  before it**, and held for that week. Every quantity at a row is known at
  that row's close.
- Feature rows begin **2020-03-01**, which leaves room to warm up the 4-week
  lookback and the factors.

**Momentum features.**
- For lookback L (in 5m bars): z_L = (log close_t − log close_{t−L}) / (σ_t ·
  √L).
- **Short:** 15m (3), 1h (12), 4h (48). **Long:** 1d (288), 3d (864), 1w
  (2016), 4w (8064).
- **Encodings,** one per trial:
  - binary: sign(z), with sign(0) = +1;
  - ternary: 0 when |z| < 0.5, else sign(z);
  - score: z·exp(−z²/4)/0.89 (Baz, Granger, Harvey, Le Roux and Rattray
    2015).
- **Interaction:** the mean of the three short encodings times the mean of
  the four long ones.

**Targets.**
- For horizon h ∈ {**5m, 30m, 5h, 1d**} = {1, 6, 60, 288} bars, r =
  log close_{t+h} − log close_t.
- **(a) primary:** 1[r > 0].
- **(b) secondary:** up if r > +9.5 bps, down if r < −9.5 bps, flat
  otherwise. 9.5 bps is Hyperliquid's 0.045% taker each way plus a 0.5 bps
  spread, and it is part of the target's definition.

**Models,** logistic regression, one per horizon × ticker × encoding:
- **A:** an intercept, log σ_t, sin and cos of 2π·hour/24 and 4π·hour/24, and
  sin and cos of 2π·weekday/7 (UTC). 7 columns.
- **B:** A plus the seven ladder encodings and the interaction. 15 columns.
- Fitted by Newton–IRLS, with columns other than the intercept standardised
  on the training rows only, and a **fixed L2 penalty of 1e-6** that is not
  searched.
- Target (b) uses the multinomial logistic regression with the same columns,
  method and penalty.

**Walk-forward.**
- Split **2023-01-01 00:00 UTC**: 2020-03 → 2022-12 is the estimation
  period, which may be looked at to check the code.
- 2023-01-01 → 2026-09-30 is walked forward and scored. Its halves are
  **2023-01-01 → 2024-12-31** and **2025-01-01 → 2026-09-30**.
- Refits every **2016 bars (weekly)** on an **expanding** window
  (`timeseries.walk_forward_origins`).
- **Purge:** a refit at R trains only on rows with close_t + h ≤ R.
- **Embargo:** its model applies to rows with close_t ≥ R + h until the next
  refit. The rows in between take the previous refit's model.
- **Climatology:** at each refit, the training rows' share of up targets.

**Trials, and N.** 4 horizons × 3 encodings × 2 tickers = **N = 24** cells
for target (a). Each cell compares B with A. Target (b) is secondary, scored
on S only, and adds no trials to the verdict.

## What would count as support

**S, per cell (horizon × encoding × ticker).** All of the following:

1. **B beats A on out-of-sample log loss.** Diebold–Mariano on per-row
   loss_B − loss_A, every 5m origin, long-run variance over h − 1 lags,
   Harvey–Leybourne–Newbold corrected (`gr.models.evaluate.dm`, h in bars).
   The mean difference must be negative, with **Holm-adjusted p < 0.05
   across the 24 cells**.
2. **Pesaran–Timmermann** (1992) on B's calls (P(up) > 0.5) against the
   outcomes: **p < 0.05.**
3. **The same sign** of the mean log-loss difference on **both tickers** (the
   same horizon and encoding) and in **both halves**.

A **horizon** is supported on S if at least one of its encodings meets 1–3 on
both tickers. Otherwise it is reported as **not supported**.

**E, at 5h and 1d, per cell.**
- The rule holds +1 if B's P(up) > 0.5, else −1. It is decided on every h-th
  origin only (non-overlapping) and held for h.
- It pays **4.75 bps per unit of position change**, so a flip pays the
  9.5 bps round trip, and Binance funding at each settlement it holds
  through (`studies.trial`).
- All of the following, for the best cell by net Sharpe:
  1. **DSR ≥ 0.95 at N = 24** (`stats.deflate`), with its own skewness and
     kurtosis;
  2. **PBO < 0.5** by CSCV with 16 blocks over the 12 E cells' daily net
     returns (`stats.pbo`);
  3. **Annualised net Sharpe above both** buy-and-hold of the same ticker
     and the inverse-volatility book on the same σ (`gr.models.vol.target`,
     target set on the estimation period), over the same days.

Anything short of that is reported as **not supported**.

**Secondary, (b):** S criteria 1 and 3 on multinomial log loss, B against A.
Reported beside (a). It cannot change a verdict on (a).

**Described, not tested:**
- each cell's Brier skill against climatology, and its hit rate;
- the reliability table (ten equal-width bins of P(up));
- B's coefficients;
- the S statistics repeated on non-overlapping origins only;
- the 2×2 table of P(up) by (short sign, long sign) on the estimation
  period;
- the share of rows lost to holes, per year.

---

## Result — run once, 2026-10-26, `notebooks/forecasting/direction.py`

*Appended after the run. Nothing above this line was changed.* Code at
`a0960a7`. The run took 35 minutes (08:44–09:20 UTC), on a reference store
fetched the same day: 4,930 1m kline days for BTC and ETH, all `ok`, and 81
funding months each. The E table was recomputed after one fix, `e944497`,
described under E.

**The data.** Binance's archive missed no minute: 3,549,600 of 3,549,600 per
ticker. The registered refusal of untraded bars dropped **273 BTC and 302
ETH minutes**. They have zero volume and zero trades at a frozen price, and
are exchange halts (40 episodes on ETH, for example 74 minutes on
2024-10-28). Each one breaks a 5m bar, and with no window spanning a hole,
it nulls the 4-week z for the four weeks after it. Out-of-sample rows lost:
**15% in 2023, 8% in 2024, 19% in 2025 and 0% in 2026**, the same on both
tickers. That share was found before the run, and the operator chose to run
as registered. The rows lost are chosen on information known at the time
(a halt in the last four weeks), so nothing leaks, but the verdicts below
are about the rows after no recent halt.

### S, statistical: **supported at 5m, 30m and 5h; not at 1d**

| horizon | cells met (of 6) | DM t, range | Holm p, worst cell met | B's hit rate | Brier skill of B against climatology |
|---|---|---|---|---|---|
| 5m | 6 | −9.3 to −16.5 | < 1e-15 | 52.0–52.4% | 0.20–0.38% |
| 30m | 6 | −7.4 to −10.9 | < 1e-10 | 52.3–53.2% | 0.35–0.55% |
| 5h | 5 | −2.7 to −3.8 | 0.012 (`ternary BTC` 0.050, not met) | 52.2–53.3% | 0.28–0.53% |
| 1d | 0 | −0.01 to −0.98 | 1.0 | 51.3–52.2% | −0.01 to 0.19% |

At 1d, B's mean log loss is below A's in every cell, but no cell is close to
significant, and on BTC the second half's sign flips. At 5h every encoding
met criteria 1–3 on both tickers except `ternary` on BTC (Holm p 0.0504).
Against the expectation (mixed at 5h), 5h is supported as registered.

**Read with these, all described rather than tested:**
- **Pesaran–Timmermann decided nothing.** On every 5m origin the targets
  overlap h-fold, and the test assumes independent rows: it gave p < 1e-30
  in every cell, including 1d at a 51.3% hit rate. Criteria 1 and 3 carried
  the verdict.
- **On non-overlapping origins only, 5h does not hold.** 5m and 30m are
  still met in every cell. At 5h, Holm p is 0.09–1.0 (raw DM p
  0.007–0.33), on 5,821 origins per cell. **The 5h verdict rests on the
  overlapping comparison's HAC correction**, and it is the claim most in
  need of the forward data.
- **The information is mostly reversal, not continuation.** At each cell's
  last refit (score encoding), the short lookbacks' coefficients are
  negative on both tickers at 5m, 30m and 5h (15m and 1h around −0.07 at 5m
  and 30m; 4h and 1d around −0.08 at 5h). The 4-week coefficient is
  positive at 5h and 1d (+0.04 to +0.07). The ladder is collinear, so no
  single coefficient is read alone. But the signs agree across tickers and
  horizons: what B adds over A at 5m to 5h is that **recent moves tend to
  reverse**.
- **Calibration.** B's forecasts sit in 0.4–0.6 almost everywhere. At 5m
  to 5h each well-populated bin's share up is within about 1.5 points of
  its mean forecast. At 1d B is overconfident: its 0.4–0.5 bin forecast
  47.3–47.5% and realised 49.7–49.9% up.

### E, economic, at 5h and 1d: **not supported**

| criterion | value | met |
|---|---|---|
| best cell by net Sharpe | `1d score BTC`, annualised 0.06 | |
| DSR at N = 24 | 0.011 | no |
| PBO, 16 blocks, the 12 cells | 0.12 | yes |
| Sharpe above hold (0.87) and inverse vol (0.71), same days | 0.06 | no |

Every 5h cell lost: annualised net Sharpe −1.28 to −1.74. Only one of the
six 1d cells was positive. The PBO is low because the cells' ranking
persists from one half of a split to the other (the 1d cells above the 5h
cells, which flip more often). That says nothing about whether any of them
pays.

**The fix behind this table.** The first computation scored buy-and-hold
and inverse vol over all 1,369 out-of-sample days, while the rule, which has
no call on the days lost to halts, covered 1,208 (1d) and 1,218 (5h). The
registration says "over the same days". `e944497` makes the benchmarks hold
only where the rule has a call, and adds the test that would have caught it.
Hold BTC moved from 1.00 to 0.87, inverse vol BTC from 0.84 to 0.71, ETH
hold from 0.54 to 0.59. The rule's figures, the DSR and the PBO are
unchanged, and so is the verdict.

### (b), secondary: the same three horizons

On the three-class target (flat inside ±9.5 bps), B beats A on multinomial
log loss with Holm p < 0.05 and the same sign on both tickers and halves in
every cell at 5m, 30m and 5h, and in none at 1d. It changes no verdict on
(a).

### What survives

- **Short- and long-term momentum on 1m data carries information about the
  next 5 minutes to 5 hours' direction beyond volatility and the clock.**
  It is statistically clear at 5m and 30m on every encoding and both
  tickers, and supported but fragile at 5h.
- **Most of that information is reversal of recent moves.**
- **It is far too small to trade as a taker.** Brier skill is 0.2–0.55% and
  hit rates are 52–53%, against the 55% (5h) to 89% (5m) needed. At 5h and
  1d the sign rule loses after 9.5 bps a flip, as the expectation said.
- Use it, if anywhere, as a conditioning input (execution timing, a regime
  label), as `trade-the-flow.md` concluded for the order flow. That is a
  claim for a new registration, on data after 2026-09-30.
