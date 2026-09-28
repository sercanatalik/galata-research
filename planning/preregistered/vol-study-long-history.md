# Registered: does the volatility study hold on six years of BTC and ETH?

**Written and committed on 2026-09-28, alone, before `garch.py` was run on
the archive's bars.** The only change to `garch.py` is a data-source choice:
the record by default, or Binance's 1m klines resampled, routed through one
loader. No rule, model or default moved. Nothing here may change after this
commit; a deviation is reported as a deviation.

## Why

Every verdict of the volatility study rests on the record. The record's 1d
bars start in 2023-02 and its 4h bars in 2024-06. Binance's archived 1m
klines give 1d and 4h bars from 2020-01, three to four times the history.
The replication (`replication.md`) asked whether the study travels across
tickers. This asks whether it travels across **samples and a venue**.

## Procedure

- **The driver.** `notebooks/vol_long.py` replays `notebooks/garch.py`
  through `App.embed`, as `replication.py` does and with its functions
  (`settings`, `extra_claims`).
- **Settings.** Replication's settings, plus two new values:
  - share 0.7, no deseasonalising, h = 1;
  - models `ewma, garch, gjr, egarch, har, harq`;
  - refits every 5 (1d) or 6 (4h) bars;
  - `source = "binance 1m klines"`, `until = "2026-09-27T00:00Z"`.
- **Cells.** {BTC, ETH} × {1d, 4h}.
- **Data.** Binance USDⓈ-M BTCUSDT and ETHUSDT 1m klines from
  `gr.reference`, as held on 2026-09-28 (2,462 days each, none missing).
  - Bars from 2020-01-01, resampled to whole 1d, 4h and 1h bars on the UTC
    grid by `gr.timeseries.resample`; a partial bucket is dropped.
  - HAR's RV uses the record's construction: 4h bars at 1d, 1h bars at 4h.
- **Code.** The library and notebooks at this commit.

## Claims and rules

The eleven claims of `replication.md`: ⑬'s seven rows, and the four in
`replication.extra_claims`. Each is decided by the rule printed beside it in
the notebook, unchanged.

## Predictions

The prediction for each cell is **that ticker's verdict on the record**, from
the survival table in the README (run 2026-09-28, record through 2026-09-27).
A cell the record left undecided (`—`) is not a prediction.

| claim | BTC 1d | BTC 4h | ETH 1d | ETH 4h |
|---|---|---|---|---|
| t beats normal | yes | yes | yes | yes |
| no leverage effect | yes | no | yes | yes |
| α+β≈1 intraday is the daily cycle | — | no | — | no |
| HAR beats GARCH | yes | — | yes | — |
| something beats GARCH(1,1) | no | yes | yes | yes |
| better σ ≠ better P&L | yes | yes | yes | yes |
| targeting does not cut drawdown per vol | no | no | no | no |
| GARCH outside the multi-horizon MCS | yes | yes | yes | no |
| HARQ beats GARCH at every horizon | yes | — | yes | — |
| feedback tracks the target better | yes | yes | no | yes |
| feedback's Sharpe gain is not significant | yes | yes | yes | yes |

**Not blind.** *HAR beats GARCH* and *HARQ beats GARCH* were run on nearly
this data by item 24 (`har-vs-garch-long-history.md`): fixed split
2024-09-01 there, share 0.7 here. Their verdicts here are reported, not
counted in the headline.

## Rules for the result

Per (ticker, bars, claim):
- **repeats**: the long-history verdict equals the record's;
- **differs**: both decided and unequal;
- **can't tell**: either side undecided.

**Headline.** Repeats out of the record's decided cells, excluding the two
claims above. That is 34 cells: the other nine claims' 36, less the two
where *α+β* is undecided.
- **A claim holds on long history** if it repeats in every one of its
  decided cells.
- **It is sample-specific** if it differs in half or more.
- **Otherwise it is mixed.**

**Expectation (stated, not tested).** The claims that rest on
out-of-sample ranking are the least stable:
- *something beats GARCH*;
- *GARCH outside the MCS*;
- *feedback tracks the target*.

They should move most. The in-sample ones (*t beats normal*, *no leverage
effect*) should hold, and should hold more firmly on a sample three times
longer.

---

## Results

*Appended after the run. Nothing above this line may change.*
