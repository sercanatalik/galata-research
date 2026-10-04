# roadmap: the intraday-liquidity study, after the reference data

**Written 2026-09-27**, after `fetch-reference-market-data` (archived) put Binance depth, klines and Bybit trades
and books beside the record. The loop runs five changes, one at a time, each explored (the web and the legacy
repos), proposed, applied and archived before the next starts. Each lands in `notebooks/liquidity/liquidity.py`.

- [x] 1. **`estimate-the-spread`**: `gr.liquidity`, which covers:
  - quoted spread from quotes and the rebuilt book;
  - effective, realized and price-impact spreads from trades against the prevailing mid;
  - the bar estimators (Roll, Corwin–Schultz, Abdi–Ranaldo, EDGE), pinned to published figures;
  - calibrating those estimators by hour against the measured spread, on the same venue and the same days.
- [x] 2. **`profile-on-three-clocks`**: the liquidity week on the UTC, New York and London clocks, US DST and standard
  time apart, US-holiday days apart. Legacy's rule holds: fetch sessions, do not maintain a holiday file
  (`legacy/galata-legacy/design/datawatch/trading-hours.md:68`).
- [x] 3. **`detect-jumps-by-hour`**: a robust intraday periodicity (Boudt–Croux–Laurent WSD), Lee–Mykland with
  false discovery rate control, and the jump map by hour of the week.
- [x] 4. **`study-scheduled-events`**: an event calendar covering:
  - US macro releases and FOMC decisions, sourced, not typed in;
  - hourly funding (Hyperliquid, Bybit) and 8h funding (Binance);
  - Deribit expiries.

  Then an event study of volume, spread, depth and jumps in ±30 min windows against matched quiet days.
- [x] 5. **`measure-price-discovery`**: who leads between Hyperliquid, Binance and Bybit, and in which hour. This
  covers Hayashi–Yoshida lead-lag on trades and information shares on 1 s mids. Clock skew is stated
  (`legacy/galata-legacy/design/measured.md:249-256`).

## What the legacy repos hold

A survey of `legacy/` and `galata-tower` on 2026-09-27 found **no implementation** of any of the five. They hold
lessons to carry:
- **The mark is not a tradeable price:** a post-only order at the mark is rejected on a one-tick spread, and an
  IOC at the mark does not cross (`legacy/vade-trader/docs/decisions.md` D-036, D-052).
- **Range estimators read above close-to-close** on 4 of 5 markets at 1h, from "a thin book reverting" inside the
  hour (`legacy/galata-legacy/design/measured.md:1537-1565`). That is the spread in the range, which item 1
  measures.
- **A simulator guesses against the strategy**, and market impact is declared absent rather than ignored
  (`legacy/vade/dyno/src/model.rs:6-30`).
- **Hyperliquid funding settles hourly**, so 8h-stamp findings do not transfer
  (`legacy/galata-legacy/design/gaps-vs-literature.md` §2.9).
- **Hyperliquid's `l2Book` is 20 levels every 5.27 s.** Microstructure measures computed on it are not the papers'
  quantities (`legacy/galata-legacy/planning/declare-the-book-cadence.md`). The tape's `bbo` is event-driven.

## The second five (2026-09-27)

- [x] 6. **`price-a-trade-by-size`**: the cost of a market order of $10k, $100k and $1M by hour.
  - From Binance's depth bands, with the cost curve interpolated between bands.
  - From Bybit's book within its reach.
  - Compared with the measured effective spread and impact.
- [x] 7. **`split-the-order-flow`**: signed order flow by hour and venue, from Binance aggTrades and Bybit trades:
  - order-flow imbalance;
  - Kyle's λ (the return per signed dollar);
  - Amihud.
- [x] 8. **`study-the-weekend-reopen`**: the xyz HIP-3 perps (GOLD, CL, XYZ100), which trade 24/7 on underlyings that close.
  - Liquidity and volatility over the weekend.
  - The gap and the jumps at the underlying's Sunday reopen.
  - GOLD against Binance's XAUUSDT.
- [x] 9. **`measure-information-shares`**: Hasbrouck's information-share bounds and Gonzalo–Granger's component share, from a VECM on 1 s mids (the `[models]` extra).
- [ ] 10. **Blocked (2026-09-27): rerun the Hyperliquid results on the rebuilt tape.** The operator chose this over the AWS fetch. It cannot proceed, because the record lost days.
  - After `galata-tape-rebuild --replace hyperliquid 2026-09-24 2026-09-28`, datawatch's **archive** holds Hyperliquid trades, quotes and candles for 2026-09-20, 22, 25, 26 and 27 only.
  - 2026-09-21, 23 and 24 are gone, and the 20th and 22nd are fragments. At the start of the session the tape held the whole week (480k BTC trades).
  - Raised against galata-datawatch, not worked around here.
  - Every Hyperliquid figure in ⑤, ⑨, ⑪ and ⑬ rests on 2026-09-25 and 26.

  The original item 10, Hyperliquid's own depth history: its requester-pays S3 archive, if AWS credentials are given. If not, the item is decided at that point.: its requester-pays S3 archive, if AWS credentials are given. If not, the item is decided at that point.

## The third five (2026-09-28)

Item 10 stays blocked, since the archive still lacks 2026-09-21/23/24. Item 11 was deferred by the operator: Hyperliquid's funding history needs datawatch's `walk_funding_days`.

- [ ] 11. **`watch-funding-across-venues`**: **deferred**, until datawatch walks Hyperliquid's settled funding.
- [x] 12. **`measure-book-resilience`**: how fast Bybit's book refills after the largest one-second flows, by hour.
  - The book's depth within 2 bps, its top size and its spread, before and for 60 s after.
  - Binance's ±1% depth at 30 s beside it.
- [x] 13. **`study-liquidity-crashes`**: depth, spread, volume and λ through the largest liquidation cascades in the sample (for example 2025-10-10/11), against the days around them.
- [x] 14. **`forecast-the-liquidity-day`**: next-hour depth and volume from the hour-of-week profile and recent levels, scored out of sample against the profile alone.
- [x] 15. **`schedule-an-execution`**: the cheapest schedule for a day's order, given the hour costs (⑩) and λ (⑪), against a uniform TWAP, out of sample.
- [x] 16. **`relate-liquidity-and-volatility`**: the elasticity of depth and λ to realized volatility, across hours and days.

## The fourth five (2026-09-28)

Items 10 and 11 are still blocked on galata-datawatch: the archive lacks 2026-09-21/23/24, and `walk_funding_days` is commented.

- [x] 17. **`split-the-liquidity-notebook`**: five notebooks, one per question. The same 18 results, checked frame by frame.
- [x] 18. **`say-what-survives-liquidity`**: the claims table (every finding, its test, the tickers and years it holds on), the references checked, and the README entry.
- [x] 19. **`preregister-liquidity-claims`**: forward claims frozen under `planning/preregistered/`, scored only on days after registration.
- [x] 20. **`add-okx-trades`**: a third CEX from OKX's trade archive, for the day's shape, λ and lead-lag. OKX peaks at 14 UTC like the others; on BTC it sits between Binance and Bybit in who moves first, and moves most per $1M.
- [ ] 21. **Items 10/11**, if datawatch has unblocked them by then.
