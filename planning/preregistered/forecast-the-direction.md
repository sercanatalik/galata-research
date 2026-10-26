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
