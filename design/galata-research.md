# galata-research — design

*The mechanism behind [`planning/galata-research.md`](../planning/galata-research.md),
which argues for it. Rewritten 2026-09-25 for the redesign: a Python research
environment, not a backtest harness. The chart is to be redrawn. Nothing here
is proposed yet; the OpenSpec changes it implies are listed at the end, in
order.*

```
  ../galata-datawatch/var            galata_research                         you
  ───────────────────────            ───────────────                         ───
  tape/kind=<k>/date=<d>/*.parquet ─▶ scan ─▶ rules ─▶ cast ─▶ clock ─┬─▶ pl.LazyFrame  ─▶ marimo
  ledger/venue=/account=/kind=/…  ─▶ decode ─┘                        └─▶ duckdb relation
                 │
                 └─ segment names ─▶ frontier()   (a listing, no decode)
```

---

## One loader, end to end

Every `gr.market.*` and `gr.account.*` call is the same five steps. Only the
rules differ per dataset.

1. **Scan.** `pl.scan_parquet` over `kind=<k>/date=<d>/*.parquet`, with
   `hive_partitioning=True`. Only the `date=` directories the window touches
   are listed. `date` is the event's venue-time day, or its receipt day when
   it has no venue time. A rule that needs the rest of the record (candle
   closure: is there a later bar?) reads the columns it needs from every
   segment, not a widened window.
2. **Rules.** The dataset's dedupe and filters (below), as polars
   expressions, before anything is collected.
3. **Cast.** Every `DECIMAL(38,18)` column becomes `Float64`, once, here.
   Integers stay integers: `trade_count`, `stream_seq`, `*_micros`.
4. **Clock.** `at_micros` becomes `ts: Datetime("us", "UTC")`. Candles gain
   `close_ts = ts + interval`. Datasets without a venue time get `recv_ts`
   instead, and **no `ts` column at all**, so a join on `ts` fails to resolve
   rather than joining the wrong clock.
5. **Bound.** `start ≤ ts < end`, then `as_of`: `ts ≤ as_of`, or
   `close_ts ≤ as_of` for candles.

The result is a `pl.LazyFrame`. `engine="duckdb"` hands the same lazy plan to
DuckDB (`duckdb.from_arrow` over its streamed batches), so **the rules exist
once, in polars**, and DuckDB is a place to run SQL over their output, not a
second implementation of them.

## The rules, per dataset

| dataset | clock | dedupe | default filters |
|---|---|---|---|
| `candles` | `ts` (open) · `close_ts` | `(venue, ticker, interval, at_micros)`, latest `recv_micros` | `is_final`; `trade_count > 0` (`traded_only`) |
| `quotes` | `ts` | none: each is a distinct top of book | — |
| `trades` | `ts` | `(venue, ticker, trade_id)`, first receipt | — |
| `funding` | `ts` | `(venue, ticker, at_micros)` | rows **with** `at_micros`: settled |
| `funding_live` | `recv_ts` | consecutive identical rows collapsed | rows **without** `at_micros`: predicted |
| `marks` | `recv_ts` | consecutive identical rows collapsed | — |
| `gaps` | `from_ts`, `to_ts` | none | — |

Each default is an argument, so it can be turned off, and a turned-off
default is visible in the call. The measurements that chose them are in the
planning file's table.

**Why the latest receipt for candles:** a bar is re-fetched by the history
walk and pushed live, and the latest receipt of a final bar is the venue's
last word on it. **Why the first receipt for trades:** a replayed execution
is the same execution, and its first receipt is when it was first known.

## The two clocks

```
  EXACT: ts (venue time)                 ARRIVAL: recv_ts
  candles · quotes · trades              marks (mark, oracle, mid, OI)
  funding (settled) · gaps · fills       funding_live (predicted rate)
            │                                       │
            └──── gr.join_recv(left, right, on="ticker", tolerance=…) ────┘
                   an explicit backward asof of left.ts onto right.recv_ts;
                   every joined column is suffixed _recv
```

`join_recv` is the only function that crosses the clocks, and its output
names what it did. A notebook that wants "the mark at this trade" writes it,
and the `_recv` suffix follows the column into every chart.

## Gaps

`gr.mask_gaps(df, dataset)` adds `in_gap: bool` and `gap_cause`: a row is in
a gap when its `ts` falls inside a `gaps` row for the same venue, ticker and
series. Nothing is dropped or interpolated. A caller who wants holes to
refuse a window filters on `in_gap`. For candles, a bar is in a gap when any
part of `[ts, close_ts)` is.

## Where the record is, and how far it goes

- **The root.** `GALATA_VAR`, else `var_root` in `galata-research.toml`,
  else `../galata-datawatch/var`. A missing root is refused by name, never
  treated as empty.
- **`gr.frontier()`**, one row per dataset:
  `(dataset, last_stream_seq, max_ts, max_recv_ts, days)`. A segment's
  filename is its `stream_seq` range (`s-<first>_<last>.parquet`, measured
  2026-09-25), so the durable position is a directory listing. `max_ts` comes
  from the newest day's Parquet footer statistics. The tape keeps statistics
  for `venue`, `ticker` and `at_micros` only, so `max_recv_ts` reads one
  column of that day. Older days are listed, never opened.
- **Intraday.** The tape is projected hourly, so the frontier is up to an
  hour behind capture. The library reads the tape and states it. It never
  parses the archive.

## The account half

`gr.account.snapshots(account, venue)` reads `ledger/…/kind=margin` and
`kind=accounts`. Those are verbatim payloads (a `payload BLOB` per row), so
the library decodes them. **That is a second parser, and it is temporary**:
it ends when datawatch projects the ledger into typed datasets, as it
projects the tape. Until then it decodes Hyperliquid's `clearinghouseState`
only, and refuses any other channel by name.

`gr.account.fills(account, venue)` reads `ledger/…/kind=fills` once datawatch
Tier 13 writes it. It has `ts` (the fill's venue time), is deduped on the
venue's fill id, and is on the exact clock. It is not built until the data
exists.

## What goes where

```
  galata-research/
    pyproject.toml                 uv · polars · duckdb · marimo · pyarrow
    src/galata_research/
      __init__.py                  gr.market · gr.account · gr.frontier · gr.join_recv · gr.mask_gaps
      _root.py                     locating the record
      _scan.py                     partition listing, window widening, cast, clock
      market.py                    one function per dataset, its rules
      account.py                   snapshots now, fills later
      frontier.py
    notebooks/                     marimo, one per question
    tests/                         against a fixture tape written in the test
    design/ · planning/ · openspec/
```

Tests follow datawatch's Python convention: named after the claim
(`a_refetched_candle_is_counted_once`, `no_marks_row_has_a_ts`), collected
by `python_functions = ["a_*", "an_*", "the_*", "every_*", "no_*"]`.

## Decisions

| | chosen | instead of |
|---|---|---|
| the rules' one home | polars expressions | SQL views run by both engines (two SQL dialects, so the rules would exist twice and drift) |
| money | `Float64` on load | `Decimal` (exact, slow, and unsupported by much of polars) |
| time | `at_micros`, as `ts` | `recv_micros` filled in where venue time is missing |
| data without venue time | `recv_ts`, no `ts` column | an estimated `ts` (recv minus measured socket lag) |
| intraday | the tape's frontier, stated | parsing today's archive in Python |
| fills | read from the ledger | fetched from each venue by research |

## The OpenSpec changes, in order

1. **`load-the-candles`**: the package, the root, the scan-rules-cast-clock
   pipeline, candles with dedupe, `close_ts` and `as_of`, and `frontier()`.
2. **`load-the-ticks`**: quotes and trades.
3. **`keep-two-clocks`**: settled `funding`, `funding_live`, `marks`,
   `recv_ts` and `join_recv`.
4. **`mask-the-gaps`**: `gaps` and `mask_gaps`.
5. **`decode-the-snapshots`**: the account half, interim decode.
6. **`load-my-fills`**: after datawatch Tier 13.

Each lands with a marimo notebook that uses it.

## Open

- **`refresh=True`**, projecting today on demand under the `galata-record`
  lock: needed, or is an hour-old frontier enough?
- **Legacy fills**: a one-off import of existing fills history, or none.
- **Research methods** from the earlier plan (random baselines, Deflated
  Sharpe, PBO): notebooks over this library, or not at all.
- **Multi-venue fills**: datawatch has no ledger plan yet for Robinhood Crypto
  or Robinhood Chain.
