# Registered: does the volatility study hold at 1h on six years of BTC and ETH?

**Written and committed on 2026-09-28, alone, before `garch.py` was run at
1h on the archive's bars.** It extends `vol-study-long-history.md` (item 26,
1d and 4h) to the one bar it left out. The record's 1h bars cover only
2026-03 onward (about 5,000 bars, the venue's history limit). Binance's
archived 1m klines give about 58,000 whole 1h bars from 2020-01.

## Procedure

Item 26's, unchanged, at 1h:

- **The replay.** `notebooks/vol_long.py` replays `notebooks/garch.py`
  through `App.embed`, with `replication.settings(ticker, "1h")`, plus
  `source = "binance 1m klines"` and `until = "2026-09-27T00:00Z"`.
- **Settings.**
  - share 0.7, no deseasonalising, h = 1;
  - models `ewma, garch, gjr, egarch` (`garch.py` walks no HAR at 1h);
  - refits every 24 bars;
  - horizons 1, 24 and 168 bars.
- **Cells.** {BTC, ETH} × {1h}.
- **Data.** Binance USDⓈ-M BTCUSDT and ETHUSDT 1m klines as held on
  2026-09-28, resampled to whole 1h bars on the UTC grid; a partial hour is
  dropped.
- **Code.** The library and notebooks at this commit.

## Claims, rules and predictions

The eleven claims and rules of item 26. Each cell's prediction is **that
ticker's 1h verdict on the record**, from the README's survival table (run
2026-09-28). *HAR beats GARCH* and *HARQ beats GARCH* are undecided at 1h on
both sides, since `garch.py` walks no HAR there.

| claim | BTC 1h | ETH 1h |
|---|---|---|
| t beats normal | yes | yes |
| no leverage effect | yes | yes |
| α+β≈1 intraday is the daily cycle | yes | yes |
| something beats GARCH(1,1) | no | yes |
| better σ ≠ better P&L | yes | yes |
| targeting does not cut drawdown per vol | no | yes |
| GARCH outside the multi-horizon MCS | no | yes |
| feedback tracks the target better | yes | yes |
| feedback's Sharpe gain is not significant | yes | yes |

## Rules for the result

Item 26's: *repeats*, *differs* or *can't tell* per cell.

**Headline.** Repeats out of the 18 cells above.

A claim's reading combines items 26 and 29, over every cell the record
decided at 1d, 4h and 1h:
- **holds on long history**: it repeats everywhere;
- **sample-specific**: it differs in half or more;
- **mixed**: anything else.

**Expectation (stated, not tested).**
- ***α+β≈1 at 1h is the daily cycle*** should hold more clearly. On six
  years the hour-by-weekday factors are fitted on ~41,000 in-sample bars,
  not ~3,500.
- ***something beats GARCH*** and the MCS claim should move, as at 1d and 4h.

---

## Results

*Appended after the run. Nothing above this line may change.*

### Run of 2026-09-28 (`notebooks/vol_long.py`, 1h section)

- **Samples.** Both tickers ran from 2020-01-01 to 2026-09-27, split at
  2024-09-18 16:00 UTC.
- **Deviations.** None.
- **Run time.** 1,629 s.

| claim | BTC 1h | ETH 1h |
|---|---|---|
| t beats normal | yes → yes (ΔBIC −14,196) | yes → yes (−11,711) |
| no leverage effect | yes → yes (γ +0.008, se 0.005) | yes → yes (γ +0.013, se 0.004) |
| α+β≈1 intraday is the daily cycle | **yes → no** (1.0000 → 0.9997) | **yes → no** (1.0000 → 1.0001) |
| something beats GARCH(1,1) | no → no | yes → yes (outside at h = 24) |
| better σ ≠ better P&L | yes → yes | yes → yes |
| targeting does not cut drawdown per vol | no → no | **yes → no** (0.94 vs hold 1.05) |
| GARCH outside the multi-horizon MCS | no → no (uMCS p 0.191) | **yes → no** (p 1.000) |
| feedback tracks the target better | **yes → no** (3 of 4 models) | **yes → no** (2 of 4) |
| feedback's Sharpe gain is not significant | yes → yes | yes → yes |

**Headline: 12 of 18 repeat.**

Every claim over 1d, 4h and 1h (items 26 and 29, by the registered rule):

| claim | repeats / decided | reading |
|---|---|---|
| t beats normal | 6 / 6 | **holds on long history** |
| feedback's Sharpe gain is not significant | 6 / 6 | **holds on long history** |
| no leverage effect | 5 / 6 | mixed |
| something beats GARCH(1,1) | 5 / 6 | mixed |
| better σ ≠ better P&L | 5 / 6 | mixed |
| targeting does not cut drawdown per vol | 5 / 6 | mixed |
| feedback tracks the target better | 4 / 6 | mixed |
| α+β≈1 intraday is the daily cycle | 2 / 4 | **sample-specific** |
| GARCH outside the multi-horizon MCS | 3 / 6 | **sample-specific** |

**Against the expectation.**
- ***α+β≈1 at 1h is the daily cycle*: the expectation was wrong.** On six
  years, deseasonalising leaves 1h persistence at 1.0000 on both tickers.
  The record's 1h finding (1.0000 → 0.9895) came from seven months. On a long
  sample, near-integrated 1h variance is not the daily cycle; breaks or long
  memory remain candidates.
- ***feedback tracks the target better* fails at 1h on both tickers.** It
  held at BTC 1d, BTC 4h and ETH 4h (ETH 1d was *no* on the record too).
  Its tracking gain, like its Sharpe gain, is not general.
- **Two claims survive every bar and both tickers on six years:**
  - Student-t beats normal;
  - feedback targeting's Sharpe gain is not significant.

  The replication's other everywhere-claim on the record, *better σ ≠
  better P&L*, now fails once (ETH 1d).
