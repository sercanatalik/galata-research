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
