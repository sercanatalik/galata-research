# Registered: is 1h persistence of one on six years neglected variance breaks?

**Written and committed on 2026-09-28, alone, before any segment is fitted.**
Item 29 (`vol-study-long-history-1h.md`) found deseasonalised GARCH-t
persistence at 1h stays at 1.0000 on six years of Binance BTC and ETH, so
the record's reading (the daily cycle) fails there. The literature's next
candidate is neglected breaks in the unconditional variance: they push
estimated persistence toward one (Lamoureux and Lastrapes 1990; Hillebrand
2005; Mikosch and Stărică 2004).

## Why a placebo

Fitting between breaks shortens every sample. A short GARCH sample can pull
α+β down on its own, whether or not the cuts sit at real breaks. κ₂ is
also known to over-detect under persistent GARCH (17–41% at a nominal 5%,
measured in this library). So segments at detected breaks are compared with
segments at **random** cuts of the same number. Breaks explain the
persistence only if cutting there lowers α+β more than cutting anywhere.

## Procedure

- **Data.** Binance USDⓈ-M BTCUSDT and ETHUSDT 1m klines as held on
  2026-09-28, resampled to whole 1h bars (`gr.timeseries.resample`), from
  2020-01-01 to before 2026-09-27.
- **Returns.** Log returns, deseasonalised by `gr.timeseries.seasonal_factors`
  (hour × weekday) fitted on the whole sample. This is a question about
  in-sample structure, not a forecast, so no split.
- **Breaks.** `gr.timeseries.variance_breaks(statistic="kappa2",
  min_segment=720)`, at least 30 days between breaks, on the deseasonalised
  returns.
- **Fits.** `gr.models.vol.segmented(model="garch", dist="t", min_obs=2000)`:
  GARCH(1,1)-t on the full sample and on every segment of at least 2,000
  returns. Shorter segments are skipped.
- **The statistic.** M = the median α+β over the fitted segments.
  - M_break is M for the κ₂ breaks.
  - The placebo takes 20 draws (seeds 0–19), each with as many cuts as κ₂
    found, at bars chosen uniformly without replacement. Draws are
    re-drawn only when a segment would be shorter than 720 bars. Each
    draw's M is computed the same way.

## Rule, per ticker

- **Breaks explain the persistence** if the full-sample α+β ≥ 0.999,
  M_break < 0.99, and M_break is below at least 19 of the 20 placebo M's.
- **Segment length explains it** if M_break < 0.99 but it is not below 19
  of the 20 placebo M's.
- **Neither** if M_break ≥ 0.99.

## Prediction

**Breaks explain the persistence, on both tickers** (the literature's
expectation). Stated alongside: the record's 1d test found no κ₂ break at
all (roadmap item 17), so this is the first sample on which the question
can be decided.

## References

- Hillebrand, E. (2005). Neglecting parameter changes in GARCH models.
  *Journal of Econometrics* 129(1–2), 121–138.
- Lamoureux, C. G. and Lastrapes, W. D. (1990). Persistence in variance,
  structural change, and the GARCH model. *Journal of Business & Economic
  Statistics* 8(2), 225–234.
- Mikosch, T. and Stărică, C. (2004). Nonstationarities in financial time
  series, the long-range dependence, and the IGARCH effects. *Review of
  Economics and Statistics* 86(1), 378–390.

---

## Results

*Appended after the run. Nothing above this line may change.*
