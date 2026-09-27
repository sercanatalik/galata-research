# Registered: does the BTC volatility study replicate on ETH and HYPE?

Written and committed on 2026-09-27, before any ETH or HYPE run of the
procedure below. Nothing here may change after that commit; a deviation is
reported as a deviation.

## Procedure

`notebooks/replication.py` replays `notebooks/garch.py` (unchanged) through
`App.embed`. For each ticker in {ETH, HYPE} and bars in {1d, 4h, 1h}, only the
UI values are substituted:
- share 0.7, no deseasonalising, horizon h = 1;
- the default models (`ewma, garch, gjr, egarch`, plus `har, harq` except at
  1h), refitted every 5 (1d), 6 (4h) or 24 (1h) bars;
- the walk pressed.

The record is the one on disk at the run, the same day as the BTC baseline
below; each row reports its frontier. The code is the same as at the commit
of this file: `garch.py`, `replication.py`, and the library at `8011f7a`,
which fixed `mcs_horizons` before any ETH or HYPE run.

## Claims and predictions

Eleven claims: ⑬'s seven rows, then four from the run's frames. Those four
are GARCH outside the 90% uniform multi-horizon MCS; HARQ beating GARCH by
uSPA (p < 0.05, 499 replicates); feedback's vol error lower for every model;
and no model's Ledoit–Wolf p for feedback against open loop rejecting after
Holm at 5% (M = 4,999, seed 0, automatic block). Each verdict uses the rule
printed beside it in the notebook.

The prediction is BTC's verdict, from the same code run on BTC today
(frontier 2026-09-27; 1d, 4h and 1h). Cells marked can't tell are not
predictions.

| claim | 1d | 4h | 1h |
|---|---|---|---|
| t beats normal | consistent (ΔBIC -97) | consistent (ΔBIC -710) | consistent (ΔBIC -701) |
| no leverage effect | consistent (γ +0.057 (se 0.053)) | contradicts (γ +0.106 (se 0.052)) | consistent (γ +0.052 (se 0.033)) |
| α+β≈1 intraday is the daily cycle | can't tell | contradicts (0.9607 → 0.9816) | consistent (1.0000 → 0.9895) |
| HAR beats GARCH | consistent (3 of 3 horizons) | can't tell | can't tell |
| something beats GARCH(1,1) | no (GARCH in every MCS) | yes (GARCH outside the MCS at h = [1, 6, 42]) | no (GARCH in every MCS) |
| better σ ≠ better P&L | consistent (ρ = 0.26 over 6) | consistent (ρ = -0.40 over 4) | consistent (ρ = 0.20 over 4) |
| targeting does not cut drawdown per vol | contradicts (1.03 vs hold 1.21) | contradicts (0.83 vs hold 0.87) | contradicts (0.21 vs hold 0.21) |
| GARCH outside the multi-horizon MCS | yes (uMCS p 0.075) | yes (uMCS p 0.005) | no (uMCS p 1.000) |
| HARQ beats GARCH at every horizon | yes (uSPA p 0.004) | can't tell | can't tell |
| feedback tracks the target better | yes (6 of 6 models) | yes (4 of 4 models) | yes (4 of 4 models) |
| feedback's Sharpe gain is not significant | yes (min p 0.234, 0 Holm rejections) | yes (min p 0.173, 0 Holm rejections) | yes (min p 0.033, 0 Holm rejections) |

The BTC 1d drawdown verdict now reads *contradicts*, while the README says
*consistent*. Since item 16 the best trial by Sharpe is a feedback trial,
and its drawdown per vol (1.03) is below hold's (1.21). The prediction is
the verdict this code gives.

## Replication rule

For each (ticker, bars, claim):
- **replicates**: the verdict equals BTC's;
- **differs**: both sides decided and the verdicts are unequal;
- **can't tell**: either side is can't tell.

For each claim:
- **generalises**: it replicates on both ETH and HYPE at every bar where
  BTC decided it;
- **BTC-specific**: it differs on both tickers at a majority of those bars;
- **mixed**: anything else.

Per ticker, the headline is the count of replicating cells out of the 28 BTC
decided.

## Expectation (stated, not tested)

ETH co-moves with BTC (Katsiampa 2019), so most of ETH's cells should
replicate. HYPE daily starts 2024-12-05 (661 bars, so about 200 days out of
sample at 1d). Expect more can't-tell cells there, and the claims that rest
on out-of-sample ranking to be the least stable.

## References

- Hou, K., Xue, C. and Zhang, L. (2020). Replicating anomalies. *Review of
  Financial Studies* 33(5), 2019–2133.
- Katsiampa, P. (2019). Volatility co-movement between Bitcoin and Ether.
  *Finance Research Letters* 30, 221–227.
- Quaedvlieg, R. (2021). Multi-horizon forecast comparison. *JBES* 39(1),
  40–53.
