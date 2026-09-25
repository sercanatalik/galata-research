# galata-research

**Galata's Python research environment: the record galata-datawatch keeps,
loaded as polars or DuckDB, explored in marimo.**

> **Status: candles, trades and quotes load, and gaps mark them.** The rest
> of the market data and the account half follow the roadmap below. Argued in
> [`planning/galata-research.md`](planning/galata-research.md); the mechanism is
> [`design/galata-research.md`](design/galata-research.md).

## Where it fits in Galata

Galata is a low-latency algorithmic trading framework: multi-venue market data
capture, signal generation, deterministic portfolio risk controls, and agentic
strategy execution. Research reads the record that
[galata-datawatch](https://github.com/sercanatalik/galata-datawatch) keeps. It
never captures, and it never talks to a venue.

```text
  galata-datawatch   capture · archive · tape · ledger      built
  galata-research    load · explore · study                 ← this
  signals · risk · decision model · execution               planned
```

## What it will provide

- **A Python library** that loads market data (candles, quotes, trades,
  funding, marks, gaps) and **my own fills, per venue**, as a polars
  `LazyFrame` by default or a DuckDB relation on request.
- **The full series**, from the oldest walked bar to the tape's frontier,
  with `gr.frontier()` saying how far each dataset is durable.
- **Exact time.** `at_micros`, the venue's time, is the one time axis, as
  `ts`. A candle is known at `close_ts`, not its open. Data with no venue time
  (marks, the live funding rate) is on a second, named clock, `recv_ts`, and is
  joined to `ts` only explicitly.
- **The record's semantics, applied once**: re-fetched candles and replayed
  trades deduped, trade-less bars filtered, settled and live funding split,
  gaps marked, never interpolated. Decimals become `f64` on load.
- **marimo notebooks**, one per question, over the library.

## A first surface

```python
import galata_research as gr

gr.market.candles(["BTC", "ETH"], "4h", start, end, as_of=t)   # pl.LazyFrame
gr.market.trades(["BTC"], start, end, engine="duckdb")          # duckdb relation
gr.market.quotes(["BTC"], start, end, as_of=t)                  # the book's top, as received
gr.mask_gaps(gr.market.trades(["BTC"], start, end), "trades")  # + in_gap, gap_cause
gr.market.marks(["BTC"], start, end)                            # recv_ts, not ts
gr.account.fills("main", "hyperliquid", start, end)             # after datawatch Tier 13
gr.frontier()
```

## Roadmap

| Step | Scope |
|---|---|
| 1 · candles ✓ | the package, the record's root, the loader pipeline, candles, `frontier()` |
| 2 · ticks ✓ | quotes and trades |
| 3 · two clocks | settled funding, live funding, marks, `join_recv` |
| 4 · gaps ✓ | `gaps` and `mask_gaps` |
| 5 · snapshots | the account half, decoded from the ledger |
| 6 · fills | my fills, once datawatch records them |

## Running it

```bash
uv sync
uv run pytest                     # fixture tapes, plus the real record when found (`-m record` for only those)
uv run marimo edit notebooks/candles.py
```

The record is found at `GALATA_VAR`, else `var_root` in `./galata-research.toml`,
else `../galata-datawatch/var`.

## Depends on

The tape and ledger layouts of galata-datawatch, read as Parquet from its
`var/` (`GALATA_VAR`). No Rust crate is linked. Python ≥ 3.11, uv, polars,
duckdb, marimo.
