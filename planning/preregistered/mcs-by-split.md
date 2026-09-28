# Registered: does *GARCH outside the multi-horizon MCS* turn on where the sample is split?

**Written and committed on 2026-09-28, alone, before any replay at a share
other than 0.7.** Item 26 (`vol-study-long-history.md`) found this the one
claim that is sample-specific on six years: the record's verdict flipped in
2 of 4 cells. Those 0.7 verdicts are known here, the other shares' are not.

## Why

A verdict from one out-of-sample window can be an artifact of where the
window starts. Rossi and Inoue (2012) show forecast-comparison tests can be
driven by the choice of window. Hansen and Timmermann (2012) show that
choosing the split can create apparent predictability. The MCS claim is
decided on one split (share 0.7). If it moves with the split alone, its
flips in item 26 say nothing about the record against long history.

## Procedure

- **The replay.** Item 26's, unchanged except for `share`. `garch.py` is
  replayed through `App.embed` with `replication.settings(ticker, bars)`,
  `source = "binance 1m klines"` and `until = "2026-09-27T00:00Z"`.
  - The share is each of **0.5, 0.6, 0.7 and 0.8** in turn.
  - At 0.7 the replay is item 26's own run, cached, not repeated.
- **Cells.** {BTC, ETH} × {1d, 4h}; 16 replays in all.
- **The verdict at each share.** `replication.extra_claims`' rule: *yes* if
  GARCH is outside the 90% uniform multi-horizon MCS (Quaedvlieg 2021) at
  that split, *no* if inside, *can't tell* if the set cannot be formed.

## Rule

- **Per cell:**
  - *stable* if the verdict is the same at all four shares;
  - *split-dependent* if both *yes* and *no* occur.
- **The claim:**
  - **turns on the split** if split-dependent in two or more cells;
  - **does not** if stable in all four;
  - **mixed** otherwise.

## Prediction

**The claim turns on the split.** Item 26's two flips (BTC 1d, ETH 4h)
make it likely. The uMCS p-values at 0.7 were 0.156, 0.090, 0.020 and
0.005 (BTC 1d, BTC 4h, ETH 1d, ETH 4h). The two nearest the 0.10 line
(BTC) should be the first to move.

## Reported, not decided

At each share the replay also gives *something beats GARCH(1,1)* and *HARQ
beats GARCH at every horizon*. They are shown for context, with no verdict
drawn from them here.

## References

- Hansen, P. R. and Timmermann, A. (2012). Choice of sample split in
  out-of-sample forecast evaluation. EUI Working Paper ECO 2012/10.
- Quaedvlieg, R. (2021). Multi-horizon forecast comparison. *Journal of
  Business & Economic Statistics* 39(1), 40–53.
- Rossi, B. and Inoue, A. (2012). Out-of-sample forecast tests robust to the
  choice of window size. *Journal of Business & Economic Statistics* 30(3),
  432–453.

---

## Results

*Appended after the run. Nothing above this line may change.*

### Run of 2026-09-28 (`notebooks/mcs_split.py`)

- **The run.** Twelve new replays (shares 0.5, 0.6, 0.8) took 4,712 s. The
  0.7 row is read from item 26's cached run and matches it (BTC 1d uMCS p
  0.156).
- **The splits.** 2023-05-15, 2024-01-16, 2024-09-18 and 2025-05-22, the
  4h splits falling 12–16 hours later.
- **Deviations.** None.
- **A coincidence, checked.** BTC 1d at 0.5 gives uMCS p 0.191, the value
  item 29 found at BTC 1h. The frames differ: here GARCH is outside the
  single-horizon MCS at h = 30, while at 1h it was in every set.

*GARCH outside the 90% uniform multi-horizon MCS* (uMCS p in brackets):

| cell | 0.5 | 0.6 | 0.7 | 0.8 | reading |
|---|---|---|---|---|---|
| BTC 1d | no (0.191) | **yes** (0.035) | no (0.156) | no (0.704) | split-dependent |
| BTC 4h | no (0.317) | no (0.447) | **yes** (0.090) | no (1.000) | split-dependent |
| ETH 1d | yes (0.030) | yes (0.005) | yes (0.020) | yes (0.045) | **stable** |
| ETH 4h | yes (0.085) | yes (0.020) | yes (0.005) | **no** (0.111) | split-dependent |

**Result: the claim turns on the split** (3 of 4 cells split-dependent), as
predicted. Both BTC cells, the two nearest the line at 0.7, moved. The
record's BTC verdicts ("yes" at 1d and 4h) are not a finding about GARCH.
Only ETH 1d says the same thing at every split.

**Context (not decided).**
- ***HARQ beats GARCH at every horizon*:**
  - ETH 1d: *yes* at all four shares.
  - BTC 1d: *yes* at 0.5–0.7, *no* at 0.8 (p 0.234, the shortest
    out-of-sample window).
  - BTC 4h: *no* at all four.
  - ETH 4h: *yes* only at 0.6.
- ***Something beats GARCH(1,1)*:**
  - *yes* at all four shares on BTC 1d and ETH 1d. The record's BTC 1d *no*
    was the record's sample, not the split.
  - BTC 4h: *yes* only at 0.7.
  - ETH 4h: *yes* except at 0.8.
- **The pattern in all three:** 1d verdicts are steadier than 4h, and the
  shortest window (0.8, 16 months) most often loses a rejection. This is the
  power loss Hansen and Timmermann (2012) describe for short evaluation
  samples.
