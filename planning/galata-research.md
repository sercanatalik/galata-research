# galata-research

**NOT PROPOSED.** Redesigned 2026-09-25, replacing the earlier plan (a Rust
backtest harness; see commit `a753c23`). Nothing here has an OpenSpec artifact
or tasks.

---

*galata-research is a Python research environment over the record
galata-datawatch keeps: a library that loads market data and my own fills,
for every venue, as polars (by default) or DuckDB, and marimo notebooks that
use it. It reads the record; it never captures, and it never talks to a
venue.*

---

## The shape

```
  galata-datawatch/var                              galata-research
  ────────────────────                              ───────────────
  tape/    typed, rebuildable ──────────────▶  gr.market.*   ─┐
  ledger/  my accounts, per venue ──────────▶  gr.account.*  ─┼─▶ pl.LazyFrame (default)
  archive/ verbatim payloads (never parsed here)              │   duckdb relation (opt-in)
                                                              ▼
                                                       marimo notebooks
```

- **polars by default, DuckDB optionally.** Every loader returns a
  `pl.LazyFrame`; `engine="duckdb"` returns a DuckDB relation over the same
  rules. The rules live in one place, not once per engine.
- **f64 at the edge.** The tape is `DECIMAL(38,18)` end to end. The library
  casts prices, sizes and rates to `Float64` on load. The record stays exact;
  research does not need it to be.
- **The full series.** From the oldest walked bar to the tape's frontier,
  across closed days and today.

## Time: `at_micros`, exactly

```
  ts := at_micros  as Datetime("us", "UTC")      the one time axis
  as_of=T  ⇒  ts ≤ T        candles: close_ts ≤ T
  no at_micros  ⇒  not on the axis; never filled from recv_micros
  recv_micros    kept as a column, for latency studies
```

Measured on the tape, 2026-09-25:

| dataset | rows | `at_micros` null | venue→recv lag p50 / p99 | on the axis |
|---|---|---|---|---|
| quotes | 1.12M | 0% | 349 ms / 871 ms | yes |
| trades | 460k | 0% | 389 ms / 32 s | yes |
| gaps | 288 | 0% | — | yes, by `from`/`to` |
| candles | 330k | 0% | `at` is the bar's **open** | yes, by `close_ts` |
| funding, settled | 630 | 0% | settlement hour + ms | yes |
| funding, live | 214k | 100% | — | **no** |
| marks | 214k | 100% | — | **no** |

**A candle is known at its close, not its open.** Every candle carries
`close_ts = ts + interval`, and `as_of` filters on it. Filtering on `ts` would
hand every as-of join one bar of the future.

**Marks and live funding have no venue time, and none can be recovered.**
Hyperliquid's `activeAssetCtx` carries no time field (2,000 archived payloads
sampled: `funding, premium, openInterest, oraclePx, markPx, midPx, impactPxs,
prevDayPx, dayNtlVlm, dayBaseVlm`, nothing else), pushed about every 1,020 ms
per ticker. They live on a **second, named clock**, `recv_ts`, in their own
loaders. Joining them to the `at_micros` axis is an explicit `join_asof`
whose output says `recv`. The two are never merged silently. The figure a
position is charged, settled funding, is on the exact axis.

## What the library owns

DuckDB and polars already read the tape with no flags, so a wrapper over
`read_parquet` is not the product. The product is the record's semantics,
which a naive read gets wrong without any error:

| in the raw tape | the rule |
|---|---|
| candles re-fetched: 1h holds 90,027 rows for ~30,000 bars | dedupe on `(venue, ticker, interval, at_micros)`, keep the latest `recv_micros` |
| live 1m bars: 127,307 rows with `is_final = false` | dropped by default |
| trades replayed on reconnect (1.55% of a 24-min run) | dedupe on `trade_id` |
| `activeAssetCtx` re-sent unchanged (24% byte-identical) | consecutive identical rows collapsed |
| daily bars before 2023-02-26 carry `trade_count = 0` | `traded_only=True` by default |
| `funding` holds settled and live rows under one name | two loaders: `funding` (settled, `ts`) and `funding_live` (`recv_ts`) |
| gaps are a separate dataset | `gr.mask_gaps(df)` marks rows inside one; nothing is interpolated |
| where the record lives | one root, from `GALATA_VAR` or config, defaulting to `../galata-datawatch/var` |
| how current it is | `gr.frontier()`: how far each dataset is durable |

## A first surface, for discussion

```
  gr.market.candles(tickers, interval, start, end, *, as_of=None, traded_only=True)
  gr.market.quotes | trades | funding(tickers, start, end, *, as_of=None)
  gr.market.marks | funding_live(tickers, start, end)          → recv_ts, not ts
  gr.market.gaps(...)  ·  gr.mask_gaps(df)
  gr.account.snapshots(account, venue, ...)                    → ledger payloads decoded
  gr.account.fills(account, venue, ...)                        → see Open
  gr.frontier()
  every loader: engine="polars" (default) | "duckdb"
```

## Intraday: the tape is up to an hour behind

The projection runs hourly, so the newest ≤60 minutes exist only in the
archive. The library reads the tape and states its frontier. It never parses
the archive, because that would be a second normaliser drifting from datawatch's
one ingestion path. Projecting today on demand (`refresh=True`, running
`galata-tape-rebuild` under the `galata-record` lock) is the option if an
hour is too old.

## Owed by galata-datawatch

Found while designing this. Each is fixed at the source, because the archive
still holds the bytes and a rebuild recovers every captured day:

- **`marks.index` is `midPx`** (`adapters/hyperliquid/normalise.rs:245`).
  Hyperliquid sends no index price here. Rename it `mid`.
- **`premium`, `impactPxs`, `prevDayPx` and `dayNtlVlm` are dropped** by the
  adapter. Premium is what funding is computed from.
- **`funding` mixes settled history and the live predicted rate**, and
  `next_micros` is null on every live row.
- **My fills are not in the record.** The ledger holds accounts and margin
  snapshots (roadmap Tier 12). Fills, funding paid and transfers are Tier 13,
  planned for Hyperliquid, and there is no ledger plan yet for Robinhood Crypto
  or Robinhood Chain.

## Open

- **Fills.** Wait for datawatch Tier 13 and read `ledger/kind=fills`, or land
  Tier 13 first with the loader beside it. Research does not fetch from a
  venue. There is no legacy history worth importing: the only fills on disk
  are 3,270 Hyperliquid **testnet** fills, 2026-09-03 → 09-06
  (`legacy/galata-legacy/var/testnet/archive/account=main/kind=fills`), and
  none from Robinhood. Legacy's reader rule carries over: collapse by
  `(venue, trade_id)`, keep the first-recorded row, and count differing
  payloads under one id as conflicting
  (`legacy/galata-legacy/crates/tower/src/past.rs:40-50`).
- **`refresh=True`**: needed, or is an hour-old frontier enough?
- **What stays of the old plan's research methods** (random baselines, Deflated
  Sharpe, PBO): as notebooks over this library, or not at all.

## Depends on

The tape and ledger schemas of `galata-datawatch`, read as Parquet. No Rust
crate is linked.
