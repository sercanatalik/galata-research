# roadmap: a common research library, and the GARCH study that is its first user

**NOT PROPOSED.** Written 2026-09-27 from an explore session: five research
threads (about 80 sources), fits on the record, and tooling timed on this
machine. It argues for two OpenSpec changes, in order:

1. **`shape-the-common-library`**: a refactor that changes no behaviour and
   turns `galata_research` into the shared research library (`gr.utils`,
   `gr.timeseries`, public names for what notebooks now reach for privately).
2. **`study-garch-volatility`**: `gr.models.vol`, `gr.models.evaluate`, and
   `notebooks/garch.py`. The GARCH family, walk-forward forecasts at
   several horizons, a forecast scorecard, and a vol-targeting backtest,
   with the theory and the literature in the notebook.

Later work is listed under [Phase 2](#phase-2-follow-ups).

---

## Contents

- [Decisions taken](#decisions-taken)
- [What the record holds](#what-the-record-holds)
- [First fits on the record](#first-fits-on-the-record)
- [Phase 0: shape-the-common-library](#phase-0-shape-the-common-library)
- [Phase 1: study-garch-volatility](#phase-1-study-garch-volatility)
  - [Library surface](#library-surface)
  - [The polars boundary](#the-polars-boundary)
  - [Gap policy](#gap-policy)
  - [Models in scope](#models-in-scope)
  - [The timeline: estimation period, split, window](#the-timeline-estimation-period-split-window)
  - [Walk-forward mechanics and cost](#walk-forward-mechanics-and-cost)
  - [Horizons](#horizons)
  - [Realized measures and the scoring proxy](#realized-measures-and-the-scoring-proxy)
  - [Evaluation suite](#evaluation-suite)
  - [The economic backtest](#the-economic-backtest)
  - [The notebook](#the-notebook)
  - [Guards and tests](#guards-and-tests)
- [Phase 2: follow-ups](#phase-2-follow-ups)
- [Risks and open questions](#risks-and-open-questions)
- [References](#references)

---

## Decisions taken

| # | Decision | Why |
|---|---|---|
| D1 | `galata_research` becomes the **common research library**. Notebooks consume it and do not reimplement it. | Six notebooks copy `PER_YEAR`; eight call sites reach into private functions. |
| D2 | Add **`gr.utils`** (domain-free, used by at least two modules) and **`gr.timeseries`** (bars to series, polars only). Refactor where needed. | See [Phase 0](#phase-0-shape-the-common-library). |
| D3 | Fitted models live in **`gr.models`**, the first being **`gr.models.vol`**, as an optional extra `galata-research[models]`. | Keeps the core free of numpy. `models`, not `vol`, because regime and correlation models follow. |
| D4 | **Polars in, polars out.** pandas never appears in a signature, a return value or a notebook, and touches only `gr/models/_arch.py`. | `arch` returns pandas (`params`, `forecast().variance`, `MCS.pvalues`) and requires it as a dependency. |
| D5 | **Gap policy (a), bridge:** a null return across a hole is dropped, the recursion continues, and rows after a gap carry `after_gap`. | Gaps are rare and restated bars are whole. The flag lets scoring exclude those rows. Nothing is interpolated. |
| D6 | **Estimation period and window both:** a date range where the models are looked at and chosen, then an out-of-sample part walked forward with a rolling (L) or expanding window. | Choosing on the same data that is scored is the leak this repo exists to prevent. |
| D7 | **Forecasts at several horizons**, chosen in time (1 bar, 1 day, 1 week, 1 month) and converted to bars. **Cumulative** variance by default, **point** as a toggle. | Sizing asks for vol over the holding period, and a sum is a less noisy proxy. |
| D8 | **One ticker at a time.** No multi-ticker view for now. | Run time scales with the number of tickers. |
| D9 | **Full model scope.** The hand-written likelihoods go in `gr.models.vol`, with tests, not in the notebook. | They can be tested only in the library. |
| D10 | The backtest is **`gr.backtest.returns`**: a forecast made at close t is the position held through bar t+1, with **no extra shift**. | `returns()` already applies the one shift (`backtest.py:62`). |
| D11 | The notebook carries **theory, literature and a reference list**. Library docstrings cite their paper, and a test reproduces a published figure where the paper gives one. | The house style, as in `stats.py` (DSR pinned to 0.9004). |
| D12 | Explicitly in v1: **SHAR and HARQ** beside HAR-RV; **hour-of-week deseasonalization** before the 1h GARCH fit; **CARR**; a **signature-plot** diagnostic. | Operator's list, 2026-09-27. Each is cheap and supported by the literature (below). |

---

## Delivery order

One OpenSpec change per item, each explored (with a literature and docs pass), proposed, applied
and archived before the next starts. Per `openspec/config.yaml`, each lands with a marimo notebook
that uses it. The GARCH study grows in `notebooks/garch.py` from item 4 onward.

- [x] 1. **`shape-the-common-library`**: Phase 0. `gr.utils`, `gr.timeseries` (periods_per_year,
  returns, bootstrap moved from `stats`), public `studies.trial` / `replay` / `runs`, notebooks
  off private names. No behaviour changes.
- [x] 2. **`measure-realized-volatility`**: `timeseries.realized` (CC, Parkinson, GK, RS, YZ,
  realized range, RV from finer bars), `ewma_vol`, `ewma_max`, `signature`. Notebook
  `notebooks/volatility.py`.
- [x] 3. **`walk-forward-origins-and-seasonality`**: `timeseries.walk_forward_origins`
  (rolling/expanding, refit every k) and `deseasonalize` (hour-of-week, fitted on a window).
- [x] 4. **`fit-the-garch-family`**: the `[models]` extra, lazy `gr.models`, the `_arch` polars
  boundary, `gr.models.vol.garch` in-sample fits and summaries. `notebooks/garch.py` ①–⑤b.
- [x] 5. **`forecast-walking-forward`**: `gr.models.vol.walk_forward`, horizons,
  `fitted_through`, `after_gap`, simulation for EGARCH/APARCH, caching. Notebook ⑥–⑦.
- [x] 6. **`forecast-from-realized-measures`**: `gr.models.vol.har` (HAR-RV, SHAR, HARQ) in the
  walk-forward.
- [x] 7. **`hand-written-likelihoods`**: `gr.models.vol.custom` (component GARCH, Beta-t-EGARCH,
  CARR), each tested against a nested case or a reference.
- [x] 8. **`score-the-forecasts`**: `gr.models.evaluate` (QLIKE, MSE, FZ0, DM+HLN, MZ-GLS, MCS,
  SPA, Kupiec, Christoffersen, DQ, fluctuation test). Notebook ⑧, ⑨, ⑪.
- [x] 9. **`target-the-volatility`**: `gr.models.vol.target` (band, cap, conditional), the trials,
  the performance fee. Notebook ⑩, ⑫.
- [x] 10. **`say-what-survives`**: theory and literature blocks, the claims table filled, references
  verified, ⑬, the README study entry and screenshot.

Phase 2, in the follow-up table's order, skipping the two that wait for 1m history (HAR-CJ, 5m RV):

- [x] 11. **`fit-realized-garch`**: the log-linear Realized GARCH (Hansen, Huang, Shek 2012) on daily
  RV from 4h bars, fitted and walked forward like the other models.
- [x] 12. **`switch-regimes`**: a two-regime Markov-switching GARCH (Haas, Mittnik, Paolella 2004 form)
  with a Hamilton filter, fitted and walked forward.
- [x] 13. **`compare-across-horizons`**: Quaedvlieg's (2021) multi-horizon SPA, uniform and average,
  one verdict across the horizon path.
- [x] 14. **`ask-when-models-win`**: the Giacomini–White (2006) conditional predictive ability test,
  "does GARCH win specifically in high vol?".

Phase 2, continued in the table's order. Multi-ticker views wait (the operator's call, 2026-09-27),
revisiting legacy's EWMA/GARCH exclusion is the operator's decision, and HAR-CJ and 5m RV still wait
for 1m history:

- [x] 15. **`size-on-expected-shortfall`**: ES-target sizing for the t models: exposure moves with ν̂,
  which a σ target cannot show; `walk_forward` carries each refit's ν.
- [x] 16. **`control-the-volatility`**: feedback-controlled vol targeting (Devanathan, Rueter, Boyd et al.
  2026) against open-loop 1/σ̂.
- [x] 17. **`break-the-persistence`**: variance breaks detected (ICSS, Inclán and Tiao 1994) and GARCH
  refitted with them, testing whether daily persistence is breaks (Lamoureux and Lastrapes 1990).
- [ ] 18. **`confidence-across-horizons`**: Quaedvlieg's multi-horizon Model Confidence Set (uMCS, aMCS).

---

## What the record holds

Measured with `gr.market.candles(None, interval, ...)` on 2026-09-27:

| ticker | 1d bars | 1d from | 4h bars | 4h from | 1h bars | 1h from |
|---|---|---|---|---|---|---|
| BTC | 1,309 | 2023-02-26 | 5,011 | 2024-06-14 | 5,046 | 2026-03-01 |
| ETH | 1,309 | 2023-02-26 | 5,011 | 2024-06-14 | 5,046 | 2026-03-01 |
| HYPE | 661 | 2024-12-05 | 3,966 | 2024-12-05 | 5,044 | 2026-03-01 |
| XYZ100 | 349 | 2025-10-13 | 2,093 | 2025-10-13 | 5,046 | 2026-03-01 |
| GOLD | 279 | 2025-12-22 | 1,673 | 2025-12-22 | 5,046 | 2026-03-01 |
| CL | 264 | 2026-01-06 | 1,583 | 2026-01-06 | 5,045 | 2026-03-01 |

- **1h and 4h stop at about 5,000 bars**, the venue's history limit. 1h covers only 2026-03 onward.
- **1m** exists only from about 2026-09-23 (the recovered bars). A realized variance from 1m bars is not
  usable in a backtest yet; see [the proxy](#realized-measures-and-the-scoring-proxy).
- **GOLD, CL and XYZ100 are 24/7 perps on underlyings that close.** Weekend bars are near flat,
  and Monday opens gap.
- A stable ν needs about 500 observations. CL and GOLD daily do not have them.

## First fits on the record

`arch` 8.0, constant mean, (1,1), close-to-close log returns × 100, full sample (in sample only):

```
              n     kurt   N→t ΔBIC    ν     α+β     GJR γ
BTC 1d      1308     6.4    −141     3.2   0.990   0.064
BTC 4h      5010     9.4   −1059     2.9   0.979   0.093
BTC 1h      5045    14.4   −1164     3.3   1.000   0.026   ← integrated
HYPE 4h     3965     9.2    −273     5.8   0.996   0.005   ← no leverage
GOLD 1h     5045    16.2   −2094     3.7   ~1.0    0.138   ← session effects
```

1. **Student-t wins decisively everywhere.** ν ≈ 3 on BTC is near the edge where the fourth moment stops existing.
2. **Asymmetry barely pays** on BTC and HYPE. BIC does not reward GJR's γ.
3. **α+β = 1.000 at 1h.** This is likely seasonality or breaks the model cannot represent, not true
   long memory (Andersen and Bollerslev 1997; Lamoureux and Lastrapes 1990). The notebook tests this; it does not assume it.
4. **GOLD at 1h** is shaped by the underlying's sessions.
5. EGARCH's persistence is β alone, not α+β. A first script got this wrong, and the fit table must not.

---

## Phase 0: shape-the-common-library

A refactor that changes no behaviour. **Every existing test passes unchanged**, and that is the
evidence it is safe. It ships before anything new is built on it.

### What the code shows

| Finding | Where | Consequence |
|---|---|---|
| `PER_YEAR = {"1d": 365, "4h": 2190}` copied into 6 notebooks, with no 1h or 1m | `moving_average.py:48`, `deflated_sharpe.py:53`, `overfitting.py:47`, `random_timing.py:43`, `permuted_bars.py:45`, `momentum.py` | `gr.timeseries.periods_per_year(interval)`, derived from `market.INTERVALS` |
| Notebooks call private names: `studies._trial`, `_replay`, `_runs`, `_donchian_signal`, `gr._root.root` | 8 call sites | These are the missing public API |
| Contiguous-bar returns (`close_ts.shift == ts`, null across a hole) are inline in `backtest.returns` | `backtest.py:62-66` | Extract to `timeseries.returns(kind="simple"|"log")`; `backtest` uses it |
| The "required columns or refuse with a hint" check is written twice | `backtest.py:50`, `:111` | `utils.require(frame, cols, hint)` |
| `stationary_bootstrap_indices` and `optimal_block` sit in `stats`, next to Sharpe | `stats.py:195`, `:295` | Move to `timeseries`: resampling, which MCS and the fluctuation test will use too |
| `instant()` and `window()` (refuse naive datetimes) are private in `_scan` | `_scan.py:21` | Public in `utils`; notebooks parsing a split date need them |

### Target layout

```
galata_research/                     numpy-free (the core rule, now stated)
├── market · account · clocks · gaps   the record, unchanged
├── utils.py        require · lazy · instant · window
├── timeseries.py   periods_per_year · returns(kind) · stationary_bootstrap_indices · optimal_block
├── backtest.py     returns(), built on timeseries.returns, same output
├── stats.py        Sharpe, PSR, DSR, PBO, Reality Check: performance only
├── studies.py      trial() · replay() · runs() public; families unchanged
└── models/         (Phase 1)
```

**The `utils` rule:** a helper enters `utils` only if it is domain-free **and** used by at least two
modules. Anything about bars or series belongs in `timeseries`. This keeps `utils` from becoming a
junk drawer.

### Acceptance

- `uv run pytest` passes with no test edited except import paths.
- Every notebook runs (`marimo export` headless) and produces the same numbers.
- No notebook calls a name that starts with `_`.
- Spec deltas: `backtest` (built on `timeseries.returns`); `study-statistics` (the bootstrap moves);
  new `timeseries` and `utils` capabilities.

---

## Phase 1: study-garch-volatility

### Library surface

```
galata_research/
├── timeseries.py   + realized(bars, estimator, window)         cc · parkinson · garman_klass ·
│                                                               rogers_satchell · yang_zhang ·
│                                                               realized_range · rv(finer bars)
│                   + ewma_vol(returns, lam) · ewma_max(fast, slow)
│                   + deseasonalize(returns, by="hour_of_week", fit_on=window)
│                   + walk_forward_origins(split, rolling(L)|expanding, every=k)
│                   + signature(bars_1m, samplings=[1m, 5m, 15m, 1h])
└── models/                          extra: galata-research[models] = arch, scipy, numpy
    ├── __init__.py   loaded lazily through gr.__getattr__; without the extra, a named refusal
    ├── _arch.py      the ONLY module that touches pandas
    ├── vol/
    │   ├── garch.py    EWMA, RiskMetrics2006, GARCH, GJR, EGARCH, APARCH, FIGARCH;
    │   │               normal, t, skew-t, GED
    │   ├── custom.py   component GARCH · Beta-t-EGARCH · CARR (hand-written likelihoods, scipy)
    │   ├── har.py      HAR-RV · SHAR · HARQ (least squares)
    │   ├── walk.py     walk_forward(): σ̂ at every horizon, fitted_through on every row
    │   └── target.py   σ̂ → position: band, cap, conditional; for gr.backtest.returns
    └── evaluate.py   losses (QLIKE, MSE, FZ0) · DM + HLN · MZ-GLS · MCS · SPA ·
                      Kupiec · Christoffersen · DQ · fluctuation test · performance fee
```

- **Realized estimators go in `timeseries`, not `models`.** They measure; they do not fit. The
  rolling-σ and EWMA baselines use them, and legacy's risk code needed them without GARCH
  (`legacy/galata-legacy/crates/statistics/src/lib.rs`).
- **`evaluate` sits in `models`** because MCS and SPA come from `arch.bootstrap`. If the numpy-free
  core ever needs DM or QLIKE, that half is split out then.

### The polars boundary

```
 notebook / caller                gr.models                          arch
 ─────────────────                ─────────                          ────
 pl.DataFrame  ──────▶  vol.walk_forward(bars, ...)
                           ▼
                        models/_arch.py
                        ├ to_numpy(series)   polars → float64 ndarray; refuses nulls by name
                        ├ fit / forecast  ────────────────────▶  pandas objects
                        └ from_arch(...)  ◀───────────────────
                             │  reattaches ticker, ts, close_ts by position
                             ▼
 pl.DataFrame  ◀──────  forecasts: ticker, ts, close_ts, h, sigma_hat, fitted_through, after_gap
 dict / float  ◀──────  fit summaries: params as {name: float}
```

Checked on arch 8.0: numpy input gives `params` as a pandas Series, `conditional_volatility` as an
ndarray, `forecast().variance` as a pandas DataFrame, `MCS.pvalues` as a pandas DataFrame. Altair 6.3
reads polars through narwhals 2.26. Returns are scaled ×100 before fitting (unscaled returns raise
`DataScaleWarning` and hurt convergence), and σ is converted back before annualizing.

**The risk here:** numpy input drops the timestamps, so `from_arch` reattaches them by position. One
row off in either direction is a silent lookahead or lag. It gets its own test: a synthetic series
with a known variance shock at bar t must first move the forecast at t+1.

### Gap policy

**(a) Bridge**, decided 2026-09-27. `timeseries.returns` yields a null across a hole (as
`backtest.returns` does today). Before a fit, the null is dropped and the recursion continues. The
first row after each hole carries `after_gap = true`, and scoring can exclude those rows (a toggle,
excluded by default). Charts draw gaps as grey bands from `gr.mask_gaps`. Legacy's rule, "a gap is a
null and is never interpolated", holds: nothing is filled. Range estimators must use `mask_gaps`,
because a bar that spans a gap has a meaningless high and low.

Rejected: **(b) restart per segment**, whose segments can be too short for ν; **(c) refuse**, where
one gap blocks a whole run.

### Models in scope

| Model | Where | Source | Why it's here |
|---|---|---|---|
| EWMA λ=0.94 / RiskMetrics2006 | arch | J.P. Morgan 1996; Zumbach 2006 | The baseline to beat; IGARCH with no fit |
| max(fast, slow) EWMA | timeseries | Bloomberg 2021 | The practitioner rule: rises fast, falls slowly |
| Rolling σ (L) | timeseries | — | What vol targeting uses by default |
| GARCH(1,1) normal | arch | Bollerslev 1986 | Quasi-MLE; the foil for tails |
| GARCH-t | arch | Bollerslev 1987 | Fat tails; the strongest single baseline |
| GARCH skew-t / GED | arch | Hansen 1994; Nelson 1991 | Cheap extras; the literature finds little gain |
| GJR-t | arch | Glosten, Jagannathan, Runkle 1993 | Asymmetry. **γ unconstrained**: crypto shows a reversed effect (Cheikh et al. 2020) |
| EGARCH-t | arch | Nelson 1991 | Asymmetry in logs; multi-step forecasts by simulation |
| APARCH-t | arch | Ding, Granger, Engle 1993 | Nests several of the others; multi-step by simulation |
| FIGARCH-t | arch | Baillie, Bollerslev, Mikkelsen 1996 | Long memory; the best BTC model in Chkili 2021 |
| **Component GARCH-t** | custom (~40 lines) | Engle and Lee 1999 | **The winner on BTC daily** (Katsiampa 2017) |
| **Beta-t-EGARCH** | custom (~50 lines) | Harvey and Chakravarty 2008 | Score-driven: at ν≈3 one outlier barely moves σ (GAS evidence on BTC: Troster et al. 2019) |
| **CARR** | custom (~30 lines) | Chou 2005 | The only range-based conditional model; uses the OHLC already loaded |
| **HAR-RV** | har | Corsi 2009 | The "does GARCH win at all" check (Bergsli et al. 2022) |
| **SHAR** | har | Patton and Sheppard 2015 | Negative semivariance predicts much better than positive |
| **HARQ** | har | Bollerslev, Patton, Quaedvlieg 2016 | Downweights the lagged RV when it was measured noisily |

About 16 models. Each hand-written likelihood is tested against `arch` on the nested case where one
exists (component GARCH with a constant long-run level is GARCH; Beta-t-EGARCH's recursion is compared
with EGARCH's form).

**Deseasonalization (D12).** At 1h and 4h, `timeseries.deseasonalize` divides returns by an
hour-of-week scale (the mean |r| per hour-of-week cell), **estimated on the estimation period
only**. GARCH is fitted to what remains, and σ̂ is multiplied back by the scale of the target bar
(known in advance, so no lookahead). A toggle in the notebook shows α+β before and after. This is the
direct test of finding 3 (Andersen and Bollerslev 1997; Hansen, Kim, Kimbrough 2021 on crypto
periodicity). Hyperliquid funds hourly, so no eight-hour funding spike is expected. Spillover from
eight-hour venues is plausible but unverified.

### The timeline: estimation period, split, window

```
 record start                    split                              frontier
     │                             │                                    │
     ├──── ESTIMATION PERIOD ──────┤──────── OUT-OF-SAMPLE ─────────────┤
     │  [start ━━━━━━━━━━━━ end]   │                                    │
     │  • one fit per model        │  at each close t (refit every k    │
     │  • params, ν, half-life     │  bars; filter forward between):    │
     │  • BIC, Ljung-Box, QQ       │   ┌─ rolling:   [t−L ━━━ t] ─┐     │
     │  • news-impact curves       │   └─ expanding: [start ━━ t] ─┘    │
     │  • deseasonal scale fitted  │         ↓                          │
     │  • model choice made here   │    σ̂(t+h | close_ts(t)), h=1..H    │
     │                             │         ↓                          │
     │                             │    scored + traded on t+1..t+h     │
```

Refused by name: an estimation period shorter than a model needs (about 500 bars for a t model);
a rolling L longer than the estimation period; an out-of-sample part shorter than the longest
horizon plus a scoring minimum.

**Window choice is a trial.** Evaluating several L and reporting the best is data snooping (Inoue and
Rossi 2012). Every L tried counts toward the DSR's N, or the result is shown as a curve over L. No
window is best in general (Pesaran and Timmermann 2007), so rolling and expanding are both offered.

### Walk-forward mechanics and cost

The pattern, checked on arch 8.0: `model.fit(last_obs=e)`, then
`res.forecast(horizon=H, start=e, reindex=False)` filters forward **with fixed parameters** to the
end of the data. An h=24 forecast from 2,045 origins took 56 ms. Filtering between refits costs
almost nothing; the cost is the number of refits.

Measured on BTC 1h (5,045 obs), one fit:

| GARCH-t | GJR-t | EGARCH-t | APARCH-t | FIGARCH-t |
|---|---|---|---|---|
| 20 ms | 22 ms | 35 ms | 172 ms | 814 ms |

200 GARCH-t refits with an expanding window took 3.1 s. **EGARCH multi-step by simulation** (h=24,
1,000 paths) costs about 0.2 s per 145 origins, about 7 s at every 1h origin. That is affordable, so it
simulates at every origin.

Default `refit_every` (overridable; refitting every k bars is safe per Ardia and Hoogerheide 2014):

| interval | GARCH family | FIGARCH |
|---|---|---|
| 1d | 1 (~1.3k refits, ~30 s across the models) | 5 |
| 4h | 6 | 24–42 |
| 1h | 24 (~200 refits, a few s) | 168 (~30 refits, ~25 s) |

Everything expensive sits behind `mo.ui.run_button`, in a function cached with `mo.persistent_cache`
and keyed on plain values
`(ticker, interval, split, window, L, k, model, H, deseasonalize)`, never on the bars frame. Arch
result objects are not cached; they may not pickle.

### Horizons

Chosen in time and converted to bars:

| choice | 1h | 4h | 1d |
|---|---|---|---|
| 1 bar | 1 | 1 | 1 |
| 1 day | 24 | 6 | 1 |
| 1 week | 168 | 42 | 7 |
| 1 month | 720 | 180 | 30 |

- **Cumulative** (Σ σ̂²(t+i|t), i ≤ h, against realized variance over t+1..t+h) is the default.
  **Point** is a toggle.
- The last h out-of-sample bars have nothing to be scored against yet. They are dropped, and the
  notebook says so.
- Errors from overlapping multi-step forecasts are autocorrelated by construction, so DM uses
  Newey-West with h−1 lags and the Harvey-Leybourne-Newbold correction.
- σ̂ is plotted at its **target** date, not its origin.

### Realized measures and the scoring proxy

- **The proxy.** The 1m realized-variance proxy **is not feasible** (1m history from 2026-09-23 only).
  Daily targets use **realized variance from 24 hourly returns plus the hourly realized range** (Σ
  Parkinson over 24 bars; Christensen and Podolskij 2007, Martens and van Dijk 2007, unverified this
  session). Realized range is several times more efficient than RV at the same sampling. Intraday
  targets use the bar's range estimator. 5m RV (the conventional optimum; Liu, Patton, Sheppard 2015)
  becomes the proxy once enough 1m data has built up.
- **Yang-Zhang** only matters where the underlying closes. On 24/7 perps the next open equals the
  last close, so it collapses toward Rogers-Satchell. It is kept for GOLD, CL and XYZ100 and labelled
  as redundant elsewhere.
- **Range estimators are biased downward** (discrete sampling misses the true extremes; Jensen's
  inequality; Molnár 2012). The notebook shows them next to close-to-close and does not pick a winner.
- **Signature plot (D12).** Mean RV against sampling interval (1m, 5m, 15m, 1h) over the recent 1m
  window. On BTC, reported RV drops about 9% from 5m to 10m sampling, which is microstructure noise at
  high frequency. This is a diagnostic, and it tells us when 5m RV becomes usable as a proxy.
- **Annualization** is calendar time, labelled: ×365 at 1d, ×2,190 at 4h, ×8,760 at 1h. There is no
  settled convention for perps on underlyings that close. For GOLD, CL and XYZ100 the notebook shows a
  weekend flag and weekend and weekday realized vol side by side.

### Evaluation suite

| Test | What it answers | Source | Cost |
|---|---|---|---|
| **QLIKE**, as `proxy/h + log h` | The primary loss: ranks correctly under a noisy but unbiased proxy, with the most power | Patton 2011; Patton and Sheppard 2009 | trivial |
| MSE | Secondary; also robust in Patton's sense | Patton 2011 | trivial |
| **Model Confidence Set**, per horizon | The headline: the set of models that contains the best one; a large set is an honest answer | Hansen, Lunde, Nason 2011 (`arch.bootstrap.MCS`) | low |
| SPA vs EWMA | Does anything beat the baseline? | Hansen 2005; Hansen and Lunde 2005 | low |
| DM + HLN, HAC h−1 | Pairwise stars on the heatmap | Diebold and Mariano 1995; Harvey, Leybourne, Newbold 1997 | low |
| **MZ-GLS** | Bias and efficiency: proxy/h on 1/h and 1. R² is descriptive only; nothing is regressed in logs | Patton and Sheppard 2009 | low |
| **Fluctuation test** | *When* a model wins: rolling DM with critical bands on chart ⑨ | Giacomini and Rossi 2010 (critical values from its Table 1) | ~30 lines |
| Kupiec, Christoffersen | VaR coverage and independence | Kupiec 1995; Christoffersen 1998 | low |
| **DQ** | Clustered VaR breaches, with more power | Engle and Manganelli 2004 | low |
| **FZ0** | VaR and ES scored jointly; separates GARCH-n from GARCH-t in the tail | Patton, Ziegel, Chen 2019 | low |

Why QLIKE is written `proxy/h + log h`: the textbook form `proxy/h − log(proxy/h) − 1` is infinite
when the proxy is 0, which happens on flat weekend bars for GOLD and CL. The two differ by a term in
the proxy alone, so rankings and DM statistics are unchanged.

### The economic backtest

`position = clip(target / σ̂, 0, cap)` joined onto bar t, then `gr.backtest.returns` (earns t+1,
0.045% taker fee on turnover, funding when passed).

**Trials**, all counted in the DSR's N:

| trial | position |
|---|---|
| hold | 1 |
| rolling-σ target | target / rolling std(L) |
| EWMA target | target / EWMA |
| max(fast, slow) EWMA | target / max(fast EWMA, slow EWMA) |
| conditional targeting | scale only in the top and bottom quintiles of lagged vol, else 1× (Bongaerts, Kang, van Dijk 2020); thresholds from the estimation period only |
| each vol model | target / σ̂ |
| 1/σ̂² (labelled Moreira-Muir) | one extra variant, for comparison |

**Turnover control.** A rebalance band of 0% and 25% (trade only when the target moves more than that
from the current position), both reported, with a **2× cap**. Without a band the backtest measures
forecast noise converted into fees; conventional vol targeting turns over more than 200% a year
(Bongaerts et al. 2020).

**Reported per trial:** net Sharpe · max drawdown · **max drawdown per unit of realized vol** (Harvey
et al. 2018; Bloomberg 2021 on BTC: 0.90 → 1.28 at a 10% target) · turnover · fees · **performance
fee Δ against EWMA**, in bp a year at γ ∈ {1, 5, 10} (Fleming, Kirby, Ostdiek 2001, 2003; utility
form from memory, unverified) · the DSR across all trials.

**Guards:**
- The target vol is a user constant, or estimated in-sample only. **Never from the full sample.**
  That is the look-ahead Liu, Tang and Zhou (2019) found in Moreira and Muir (2017).
- 1/σ̂ is the primary weight. 1/σ̂² doubles the swings in leverage.
- No extra shift: the position at row t is the forecast made at close t.

**The known gap between accuracy and money.** Better σ forecasts do not reliably give better
portfolios (Becker, Clements, Doolan, Hurn 2015; Bianco and Bernardi 2022), and economic losses can
reward wrong forecasts. So Δ sits **next to** QLIKE, never in place of it, and panel ⑫ shows when
the two rankings disagree. No peer-reviewed study was found comparing GARCH-t with EWMA as the
targeting input on BTC (an absence, unverified), so this comparison is modestly new.

### The notebook

`notebooks/garch.py`, one ticker at a time.

```
 controls: ticker · bars · estimation [start━━end] · split · window rolling(L)|expanding
           refit k · horizons {1 bar, 1 day, 1 week, 1 month} · cumulative|point
           deseasonalize ☐ · band {0, 25%} · target vol · ⟦run⟧
 ─── the claims under test (a table, filled in at ⑬) ──────────────────────
 ─────────────────────────────── in-sample ────────────────────────────────
 ①  overview strip with a brush (every panel below follows it)
 ②  returns · histogram against N and the fitted t · ACF(r) against ACF(r²)
 ③  realized vol: CC · Parkinson · GK · RS · [YZ] · realized range; signature plot
 ④  fitted σ_t against the proxy; z_t strip beneath
 ⑤  per model: QQ of z · news-impact curve · ACF(z²); the fit table
     (ω α β γ ν, persistence by the right formula per model, half-life, LL, AIC/BIC, Ljung-Box)
 ⑤b persistence artifacts: rolling α+β against full-sample; rolling GJR γ (sign and drift);
     α+β before and after deseasonalizing
 ────────────────────────────── out-of-sample ─────────────────────────────
 ⑥  σ̂ against the proxy at the target date, at the chosen horizon; refit ticks
 ⑦  forecast fan from a clicked origin: σ̂(t+h|t), long-run level dashed, realized dots;
     "every Nth origin" toggle
 ⑧  VaR/ES bands on returns, breaches marked; Kupiec · Christoffersen · DQ · FZ0
 ⑨  cumulative QLIKE difference against EWMA, with fluctuation-test bands; horizon selector
 ⑪  model × horizon heatmap: QLIKE / QLIKE_EWMA, DM stars, MCS membership, n origins
 ⑩  backtest: equity (log) · position strip · drawdown strip │ the trial table
 ⑫  rank agreement: QLIKE rank against net-Sharpe and Δ rank (Spearman)
 ⑬  what survives
 ─── references ────────────────────────────────────────────────────────────
```

**Chart rules:**
- One time axis, with the split drawn as a labelled rule.
- Gaps drawn as grey bands, never bridged by a line.
- Vol annualized, on a log y-axis.
- Three models shown by default (EWMA, GARCH-t and the best one), the rest greyed out; a legend
  click brings any model forward.
- The brush lives in Vega-Lite across the stacked panels. A separate `mo.ui.altair_chart` returns the
  one click that Python needs (the fan's origin).
- About 50–150k rows fit within marimo's CSV transport; fans are sent only for the selected origin.

**Every section follows the same four blocks:**

```
 THEORY          the equation in KaTeX; its symbols match the fit table's columns
                 (arch's parameterization, not a textbook's; EGARCH differs)
 LITERATURE      two or three sentences, short citations: (Patton 2011)
 ▸ derivation    collapsed in mo.accordion: why QLIKE is robust, why h−1 HAC lags,
                 why EGARCH's persistence is β
 ON THIS RECORD  one sentence generated from the results: consistent with / contradicts /
                 can't tell (n too small)
```

**The claims under test** (the top table; ⑬ fills its right column):

| Claim | Source | Checked in |
|---|---|---|
| Fat tails: t beats normal | Troster et al. 2019 | ⑤, ⑧ |
| No leverage effect in crypto, or a reversed one | Cheikh et al. 2020 | ⑤, ⑤b |
| α+β≈1 at 1h is seasonality or breaks, not memory | Andersen and Bollerslev 1997; Lamoureux and Lastrapes 1990; Mikosch and Stărică 2004 | ⑤b |
| Component GARCH fits BTC daily best | Katsiampa 2017 | ⑤, ⑪ |
| HAR beats GARCH when intraday data exists | Bergsli et al. 2022 | ⑨, ⑪ |
| Does anything beat GARCH(1,1)? | Hansen and Lunde 2005 | ⑪ (SPA, MCS) |
| Better σ does not mean better P&L | Becker et al. 2015 | ⑫ |
| Vol targeting cuts drawdown, but not drawdown per unit of vol | Harvey et al. 2018; Bloomberg 2021 | ⑩ |

**References cell.** Full citations grouped by section. Anything marked *unverified* below is checked
before it goes in, or keeps the mark.

### Guards and tests

In `tests/`, named after the claim they defend, as the repo does. **Each must fail when its guard is
removed.**

| Test | Defends |
|---|---|
| `fitted_through ≤ close_ts` on every forecast row | No lookahead in the walk-forward |
| A shock at bar t first moves σ̂ at t+1 | `_arch.from_arch` reattaches timestamps with no off-by-one |
| One bar's P&L by hand equals `gr.backtest.returns` | No extra shift from forecast to position |
| The deseasonal scale is fitted on the estimation window only | No lookahead in deseasonalization |
| Band and conditional thresholds come from the estimation period only | No lookahead in the trials |
| Target vol set from the full sample is refused | The Liu, Tang and Zhou bug |
| Component GARCH with a constant long-run level equals arch's GARCH | The hand-written likelihood |
| Beta-t-EGARCH and CARR match reference values (published or R) | The hand-written likelihoods |
| QLIKE is finite on a zero proxy and ranks as the textbook form does | The `proxy/h + log h` form |
| MCS reproduces arch's own example; DQ and FZ0 on a known case | The evaluation suite |
| Every public `gr.models` function returns polars or plain Python | D4 |
| Nothing outside `models/_arch.py` imports pandas (grep) | D4 |
| `import galata_research` works without the `[models]` extra; `gr.models.*` refuses by name | D3 |
| Range estimators against their papers' figures on synthetic Brownian paths | The estimators |

---

## Phase 2: follow-ups

Each is its own change, once Phase 1 has said something.

| Follow-up | Why | Cost / blocker |
|---|---|---|
| Realized GARCH (log-linear) | The best-documented out-of-sample on BTC with jump-robust measures (Hung, Liu, Yang 2020) | ~50 lines scipy; needs daily RV from intraday bars |
| HAR-CJ / HAR-RV-J with bipower variation | BTC jumps are frequent and cluster (Scaillet, Treccani, Trevisan 2020) | Needs 1m history depth |
| 5m RV as the proxy | The conventional optimum | Waits for 1m history |
| MS-GARCH, two regimes | Strong VaR and ES evidence (Ardia et al. 2019; Caporale and Zekokh 2019) | No maintained Python package; own Hamilton filter |
| Multi-horizon SPA | One verdict across the horizon path (Quaedvlieg 2021) | Port from R |
| Giacomini-White conditional test | "Does GARCH win specifically in high vol?" | Medium |
| ES-target sizing for t models | Exposure moves with ν̂, which a σ target cannot show | Medium |
| Feedback-controlled vol targeting | Hits the target better than open-loop 1/σ̂ (Devanathan, Boyd et al. 2026, simulation only) | Medium |
| FIGARCH or component GARCH with break dummies | Separates true long memory from breaks (Mensi et al. 2019) | Medium; needs break detection |
| Multi-ticker small multiples of ⑨ | Does a win carry over from BTC to ETH and HYPE? | Run time × tickers |
| Revisit legacy's exclusion of EWMA and GARCH from the risk derive step | `legacy/galata-legacy/design/tower/econometrics.md`: "a model that quietly reweights it is the opposite of a review that exists to make choices explicit" | A decision for the operator, informed by ⑪ and ⑩ |

---

## Risks and open questions

- **HAR will probably win the short horizons.** The literature expects it on BTC. The notebook's
  "what survives" is written to accept that result, not to protect GARCH.
- **Fees may decide the backtest at 1h.** The band and the conditional trial exist for this. If
  every trial loses to hold after fees, that is the finding.
- **Hand-written likelihoods** are where silent errors hide. Each has a nested-case or reference test,
  or it does not ship.
- **Short samples:** CL and GOLD daily; 1h only from 2026-03. The refusals name the minimum.
- **Unverified in the research:** resolved 2026-09-27. Every † citation was checked against publisher or
  repository pages. Chu et al. (2017) found IGARCH-normal best for Bitcoin (stated against the fat-tails claim),
  Rambaccussing and Mazibas (2020) find crypto volatility long memory genuine (stated in tension with the 1h
  result), and arXiv 2404.04962 was dropped.
- **Legacy context:** `legacy/galata-legacy/design/tower/econometrics.md` (estimator choice, the
  close-to-close default, range estimators mostly *over*-reading σ on testnet: BTC 0.40 CC against
  0.43 Parkinson and 0.44 GK); `design/gaps-vs-literature.md` §2.8 (vol targeting: Moreira and Muir
  against Cederburg et al.).


### Outcomes (2026-09-27)

What each risk turned into, on BTC (`notebooks/garch.py`, ⑬, which computes these):

- **HAR did win the short horizons**, and the longer ones: HARQ, HAR and CARR lead QLIKE at 1, 7 and 30 days
  at 1d (0.95, ~0.75, ~0.45 of EWMA's). The Model Confidence Set cannot separate the nine models at one day.
- **Fees and the market decided the backtest**: no targeting trial made money out of sample (hold −0.31, best
  −0.23, DSR 0.33 over 29 trials), and the forecast ranking did not carry over (ρ 0.20–0.26 at 1h and 1d).
- **The hand-written likelihoods passed their checks**, and L-BFGS-B was replaced by Nelder–Mead after it
  stalled at the start values.
- **The persistence finding depends on the bar**: deseasonalising lowers α+β at 1h (1.0000 → 0.9891) but not
  at 4h (0.961 → 0.982), and GJR's γ crosses the leverage rule at 4h only (0.106, just over 2 se).
- **An intermittent test failure**: twice, a full run failed one test that five clean reruns did not
  reproduce (after a heavy run; not identified). Open.
- **Short samples bit where expected**: HAR cannot be walked at 4h because 1h realized variance starts in
  2026-03; the notebook reports it as skipped rather than failing.

---

## References

Grouped by where they are used. Every † item was verified on 2026-09-27 (`say-what-survives`), and three changed: Chu et al.'s finding is stated, Rambaccussing and Mazibas is added, and arXiv 2404.04962 is dropped (it does not report ν ≈ 3–4).

**Models**
- Engle, R. (1982). Autoregressive conditional heteroscedasticity. *Econometrica* 50(4), 987–1007.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *J. Econometrics* 31, 307–327.
- Bollerslev, T. (1987). A conditionally heteroskedastic time series model for speculative prices and rates of return. *REStat* 69(3), 542–547.
- Nelson, D. (1991). Conditional heteroskedasticity in asset returns: a new approach. *Econometrica* 59(2), 347–370.
- Glosten, L., Jagannathan, R., Runkle, D. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *J. Finance* 48(5), 1779–1801.
- Ding, Z., Granger, C., Engle, R. (1993). A long memory property of stock market returns and a new model. *J. Empirical Finance* 1, 83–106.
- Hansen, B. (1994). Autoregressive conditional density estimation. *Int. Economic Review* 35(3), 705–730.
- Baillie, R., Bollerslev, T., Mikkelsen, H. (1996). Fractionally integrated GARCH. *J. Econometrics* 74, 3–30.
- Engle, R., Lee, G. (1999). A long-run and short-run component model of stock return volatility. In *Cointegration, Causality and Forecasting*, Oxford UP.
- J.P. Morgan (1996). *RiskMetrics Technical Document*, 4th ed. · Zumbach, G. (2006). The RiskMetrics 2006 methodology.
- Harvey, A., Chakravarty, T. (2008). Beta-t-(E)GARCH. Cambridge Working Papers in Economics 0840.
- Creal, D., Koopman, S. J., Lucas, A. (2013). Generalized autoregressive score models. *J. Applied Econometrics* 28(5), 777–795.
- Chou, R. (2005). Forecasting financial volatilities with extreme values: the CARR model. *J. Money, Credit and Banking* 37(3), 561–582.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *J. Financial Econometrics* 7(2), 174–196.
- Andersen, T., Bollerslev, T., Diebold, F. (2007). Roughing it up: including jump components in the measurement, modeling and forecasting of return volatility. *REStat* 89(4), 701–720.
- Patton, A., Sheppard, K. (2015). Good volatility, bad volatility. *REStat* 97(3), 683–697.
- Bollerslev, T., Patton, A., Quaedvlieg, R. (2016). Exploiting the errors: a simple approach for improved volatility forecasting. *J. Econometrics* 192(1), 1–18.
- Hansen, P., Huang, Z., Shek, H. (2012). Realized GARCH. *J. Applied Econometrics* 27(6), 877–906.
- Shephard, N., Sheppard, K. (2010). Realising the future: forecasting with high-frequency-based volatility (HEAVY) models. *J. Applied Econometrics* 25(2), 197–231.

**Crypto evidence**
- Katsiampa, P. (2017). Volatility estimation for Bitcoin. *Economics Letters* 158, 3–6.
- Chu, J., Chan, S., Nadarajah, S., Osterrieder, J. (2017). GARCH modelling of cryptocurrencies. *JRFM* 10(4), 17. **IGARCH(1,1) with normal innovations fits Bitcoin best**, against the fat-tails claim.
- Rambaccussing, D., Mazibas, M. (2020). True versus spurious long memory in cryptocurrencies. *JRFM* 13(9), 186. Long memory in volatility mostly genuine.
- Ardia, D., Bluteau, K., Rüede, M. (2019). Regime changes in Bitcoin GARCH volatility dynamics. *Finance Research Letters* 29, 266–271.
- Caporale, G. M., Zekokh, T. (2019). Modelling volatility of cryptocurrencies using Markov-switching GARCH models. *RIBAF* 48, 143–155.
- Troster, V., Tiwari, A., Shahbaz, M., Macedo, D. (2019). Bitcoin returns and risk: a general GARCH and GAS analysis. *Finance Research Letters* 30, 187–193.
- Mensi, W., Al-Yahyaee, K., Kang, S. H. (2019). Structural breaks and double long memory of cryptocurrency prices. *Finance Research Letters* 29, 222–230.
- Cheikh, N. B., Ben Zaied, Y., Chevallier, J. (2020). Asymmetric volatility in cryptocurrency markets. *Finance Research Letters* 35, 101293.
- Chi, Y., Hao, W. (2020). Volatility models for cryptocurrencies. arXiv 2010.07402.
- Scaillet, O., Treccani, A., Trevisan, C. (2020). High-frequency jump analysis of the Bitcoin market. *J. Financial Econometrics* 18(2), 209–232.
- Hung, J.-C., Liu, H.-C., Yang, J. J. (2020). Improving the realized GARCH's volatility forecast for Bitcoin with jump-robust estimators. *North American J. Economics and Finance* 52.
- Chkili, W. (2021). Modeling Bitcoin price volatility: long memory vs Markov switching. *Eurasian Economic Review* 11, 433–448.
- Hansen, P. R., Kim, C., Kimbrough, W. (2021). Periodicity in Cryptocurrency Volatility and Liquidity. arXiv 2109.12142.
- Bergsli, L., Lind, A., Molnár, P., Polasik, M. (2022). Forecasting volatility of Bitcoin. *RIBAF* 59.

**Persistence, seasonality, breaks**
- Lamoureux, C., Lastrapes, W. (1990). Persistence in variance, structural change, and the GARCH model. *JBES* 8(2), 225–234.
- Andersen, T., Bollerslev, T. (1997). Intraday periodicity and volatility persistence in financial markets. *J. Empirical Finance* 4, 115–158.
- Mikosch, T., Stărică, C. (2004). Nonstationarities in financial time series, the long-range dependence, and the IGARCH effects. *REStat* 86(1), 378–390.

**Realized measures**
- Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *J. Business* 53(1), 61–65.
- Garman, M., Klass, M. (1980). On the estimation of security price volatilities from historical data. *J. Business* 53(1), 67–78.
- Rogers, L. C. G., Satchell, S. (1991). Estimating variance from high, low and closing prices. *Annals of Applied Probability* 1(4), 504–512.
- Yang, D., Zhang, Q. (2000). Drift-independent volatility estimation based on high, low, open and close prices. *J. Business* 73(3), 477–491.
- Christensen, K., Podolskij, M. (2007). Realized range-based estimation of integrated variance. *J. Econometrics* 141(2), 323–349.
- Martens, M., van Dijk, D. (2007). Measuring volatility with the realized range. *J. Econometrics* 138(1), 181–207.
- Barndorff-Nielsen, O., Hansen, P., Lunde, A., Shephard, N. (2008). Designing realized kernels. *Econometrica* 76(6), 1481–1536.
- Molnár, P. (2012). Properties of range-based volatility estimators. *Int. Review of Financial Analysis* 23, 20–29.
- Liu, L., Patton, A., Sheppard, K. (2015). Does anything beat 5-minute RV? *J. Econometrics* 187(1), 293–311.

**Evaluation**
- Diebold, F., Mariano, R. (1995). Comparing predictive accuracy. *JBES* 13(3), 253–263.
- Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models. *J. Derivatives* 3(2), 73–84.
- Harvey, D., Leybourne, S., Newbold, P. (1997). Testing the equality of prediction mean squared errors. *Int. J. Forecasting* 13(2), 281–291.
- Christoffersen, P. (1998). Evaluating interval forecasts. *Int. Economic Review* 39(4), 841–862.
- Engle, R., Manganelli, S. (2004). CAViaR. *JBES* 22(4), 367–381.
- Hansen, P. (2005). A test for superior predictive ability. *JBES* 23(4), 365–380.
- Hansen, P., Lunde, A. (2005). A forecast comparison of volatility models: does anything beat a GARCH(1,1)? *J. Applied Econometrics* 20(7), 873–889.
- Hansen, P., Lunde, A. (2006). Consistent ranking of volatility models. *J. Econometrics* 131, 97–121.
- Giacomini, R., White, H. (2006). Tests of conditional predictive ability. *Econometrica* 74(6), 1545–1578.
- Pesaran, M. H., Timmermann, A. (2007). Selection of estimation window in the presence of breaks. *J. Econometrics* 137(1), 134–161.
- Patton, A., Sheppard, K. (2009). Evaluating volatility and correlation forecasts. In *Handbook of Financial Time Series*, Springer.
- Giacomini, R., Rossi, B. (2010). Forecast comparisons in unstable environments. *J. Applied Econometrics* 25(4), 595–620.
- Patton, A. (2011). Volatility forecast comparison using imperfect volatility proxies. *J. Econometrics* 160(1), 246–256.
- Hansen, P., Lunde, A., Nason, J. (2011). The model confidence set. *Econometrica* 79(2), 453–497.
- Gneiting, T. (2011). Making and evaluating point forecasts. *JASA* 106(494), 746–762.
- Inoue, A., Rossi, B. (2012). Out-of-sample forecast tests robust to the choice of window size. *JBES* 30(3), 432–453.
- Ardia, D., Hoogerheide, L. (2014). GARCH models for daily stock returns: impact of estimation frequency on Value-at-Risk and Expected Shortfall forecasts. *Economics Letters* 123(2), 187–190.
- Acerbi, C., Székely, B. (2014). Back-testing expected shortfall. *Risk* 27(11), 76–81.
- Patton, A., Ziegel, J., Chen, R. (2019). Dynamic semiparametric models for expected shortfall (and Value-at-Risk). *J. Econometrics* 211(2), 388–413.
- Quaedvlieg, R. (2021). Multi-horizon forecast comparison. *JBES* 39(1), 40–53.

**Volatility targeting and economic value**
- Fleming, J., Kirby, C., Ostdiek, B. (2001). The economic value of volatility timing. *J. Finance* 56(1), 329–352.
- Fleming, J., Kirby, C., Ostdiek, B. (2003). The economic value of volatility timing using "realized" volatility. *JFE* 67(3), 473–509.
- Engle, R., Colacito, R. (2006). Testing and valuing dynamic correlations for asset allocation. *JBES* 24(2), 238–253.
- Becker, R., Clements, A., Doolan, M., Hurn, S. (2015). Selecting volatility forecasting models for portfolio allocation purposes. *Int. J. Forecasting* 31(3), 849–861.
- Kim, A., Tse, Y., Wald, J. (2016). Time series momentum and volatility scaling. *J. Financial Markets* 30, 103–124.
- Moreira, A., Muir, T. (2017). Volatility-managed portfolios. *J. Finance* 72(4), 1611–1644.
- Harvey, C., Hoyle, E., Korgaonkar, R., Rattray, S., Sargaison, M., Van Hemert, O. (2018). The impact of volatility targeting. *J. Portfolio Management* 45(1), 14–33.
- Liu, F., Tang, X., Zhou, G. (2019). Volatility-managed portfolio: does it really work? *J. Portfolio Management* 46(1), 38–51.
- Cederburg, S., O'Doherty, M., Wang, F., Yan, X. (2020). On the performance of volatility-managed portfolios. *JFE* 138(1), 95–117.
- Bongaerts, D., Kang, X., van Dijk, M. (2020). Conditional volatility targeting. *Financial Analysts Journal* 76(4), 54–71.
- Barroso, P., Detzel, A. (2021). Do limits to arbitrage explain the benefits of volatility-managed portfolios? *JFE* 140(3), 744–767.
- Ghia, K., Hou, S. (2021). Crypto Insights: The Impact of Volatility Targeting. Bloomberg. BTC at a 10% target: return/vol 1.47 → 1.57, max drawdown/vol 0.90 → 1.28.
- Bianco, N., Bernardi, M. (2022). Smoothing volatility targeting. arXiv 2212.07288.
- Grobys, K., Kolari, J., Sandretto, D., Shahzad, S. J. H., Äijö, J. (2025). Cryptocurrency momentum has (not) its moments. *Financial Markets and Portfolio Management* 39(4).
- Devanathan, N., Rueter, D., Boyd, S., Candès, E., Hastie, T., Kochenderfer, M. et al. (2026). Single-Asset Adaptive Leveraged Volatility Control. arXiv 2603.01298.

**Resampling**
- Politis, D., Romano, J. (1994). The stationary bootstrap. *JASA* 89(428), 1303–1313.
- Politis, D., White, H. (2004). Automatic block-length selection for the dependent bootstrap. *Econometric Reviews* 23(1), 53–70; corrected by Patton, Politis, White (2009).
