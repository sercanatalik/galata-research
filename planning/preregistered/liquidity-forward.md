# Pre-registered: seven forward claims from the intraday-liquidity study

**Registered 2026-09-28, before any of the data they are tested on exists.**
This file is committed alone, before the code that scores it, so its git
history is the evidence. Every claim is tested only on data with `ts` at or
after **2026-09-29 00:00 UTC**. Nothing below may change after a result
without a new file that says why.

## Where the claims come from

The study (`notebooks/liquidity*.py`, 2026-09-27/28) found these on data up
to 2026-09-28. Most also agree with published work (`notebooks/liquidity_claims.py`).
A finding that held in-sample can still be an artifact of the sample; this
file says, in advance, what the next months must show for each to stand.

## The claims, frozen

Every claim is on Binance USDⓈ-M BTCUSDT unless stated, from the reference
store (`gr.reference`). Any seasonal profile or periodicity a claim needs is
fitted **only** on data before 2026-09-29.

| # | claim | data and measure | sample needed | supported if |
|---|---|---|---|---|
| F1 | ±1% depth is deepest in the European morning | ±1% notional (both sides), hourly median of snapshots, each day ÷ its own mean, averaged by UTC hour | the first 60 complete days | the best hour is 9, 10, 11 or 12 UTC |
| F2 | volume peaks around the US open | Σ volume × close per hour from 1m klines, each day ÷ its mean | the first 60 complete days | the best hour is 13, 14, 15 or 16 UTC |
| F3 | in US standard time the most volatile hour is 15 UTC | Σ r² of 1m log returns per complete hour, each day ÷ its mean | the first 60 complete days from 2026-11-01 (US standard time) | the most volatile hour is 15 UTC |
| F4 | price jumps cluster at US macro releases | Lee–Mykland on 5m returns after Boudt–Croux–Laurent periodicity (fitted 2019-12-31 → 2026-09-28), α = 1% a day | the first 90 complete days | 08:30 New York is among the three busiest New York 5-minute slots |
| F5 | an FOMC statement moves BTC | the 5m \|r\| of the statement bin ÷ the mean at the same New York time on weekdays within ±10 days that are neither FOMC days nor NYSE closures (⑧'s rule) | the first three scheduled statements after registration | the ratio is at least 2 at two of the three |
| F6 | Binance moves before Bybit | shifted Hayashi–Yoshida on trades per UTC hour (`gr.leadlag.lead_lag`, ⑨'s lag grid) | at least 10 days on which both venues' trades are fetched | Binance leads (lead > 0) in at least 85% of hours |
| F7 | a seasonal-plus-HAR decomposition beats persistence one hour ahead | `gr.models.intraday` as in ⑯ (refit every 720 h on all hours before the origin) on ±1% depth and on dollar volume | the first 60 complete days | on both series, R²oos against persistence > 0 with Diebold–Mariano p < 0.05 |

## What would count

- **Per claim, one verdict:** *supported*, *not supported*, or *not yet decidable* (the sample is not there). Nothing is tuned, and no claim is re-run with different parameters under its number.
- **The study as a whole:** read as the count supported out of seven, with no single claim carrying it.
- **Data gaps:** a day the archive lacks is left out, never filled. A claim whose sample needs it waits for more days rather than using fewer.

---

## Results

*Appended after each claim's sample is complete. Nothing above this line may change.*
