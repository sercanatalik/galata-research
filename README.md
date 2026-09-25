# galata-research

**Research and replay for Galata: find out whether a strategy is worth trading
before building the thing that trades it.**

> **Status: designed, not proposed.** No code yet. Argued in
> [`planning/galata-research.md`](planning/galata-research.md); the mechanism is
> [`design/galata-research.md`](design/galata-research.md), with its chart
> [`design/charts/galata-research.html`](design/charts/galata-research.html).
> Next: the OpenSpec changes the design lists, in order.

## Where it fits in Galata

Galata is a low-latency algorithmic trading framework in Rust: multi-venue
market data capture, signal generation, deterministic portfolio risk
controls, and agentic strategy execution driven by a fine-tuned decision
model. Research is the second layer. It reads the record that
[galata-datawatch](https://github.com/sercanatalik/galata-datawatch) keeps,
and it comes **before** the trading half. Three earlier rewrites built
trading first and deferred research, and two of them never got back to it.

```text
  galata-datawatch   the record and the tape          built
  galata-research    replay, fills, runs, verdicts     ← this
  signals · risk · decision model · execution         planned
```

## What it will provide

- **Point-in-time replay.** A strategy is handed events in order by a virtual
  clock and only ever sees what that clock has reached. The host's lookbacks
  use a view of the tape *as it stood at time T*
  ([`bound-the-replay`](https://github.com/sercanatalik/galata-datawatch/blob/main/planning/bound-the-replay.md),
  planned in galata-datawatch).
- **One strategy seam, shared with live.** A backtest that runs a different
  implementation from the live one measures that other implementation.
- **One fill model**, pessimistic by default: queue position defaults to the
  back of the queue, funding is charged, liquidation is simulated, and every
  figure it produces is labelled *modelled*.
- **A run manifest**: study, trial, and selection recorded as an event, with
  a hash per row. Failed and pruned trials count toward N, because the
  Deflated Sharpe Ratio needs the true N.
- **Coverage and gaps as reported figures.** Data coverage, not the harness,
  was the binding constraint every time before.

Rust runs the strategies, because the seam is shared with live. Python and
polars judge the results: PBO, Deflated Sharpe, the stationary bootstrap.

## Roadmap: the first cut

| Step | Scope |
|---|---|
| 0 · history | walk candles and funding back as far as each venue serves; report coverage and holes per instrument before any strategy runs |
| 1 · the seam | a `Strategy` trait, a virtual clock, and a hashed decision log with no prices in it |
| 2 · fills | the one `FillModel`, every parameter swept; a parameter whose range fits inside one observation interval is reported as unmeasurable |
| 3 · runs | the study/trial manifest and its storage |
| 4 · first runs | score against random agents first, then permute the lookback, then the trend family |

## The line it holds

**It reports a landscape; it never deploys.** A research result reaches a
live configuration only through a person. Selecting a winner is a timed event
in the run store, never an overwrite, and nothing here gates, sizes or
retires anything live.

## Settled in planning (2026-09-25)

- **The seam** is a shared `galata-strategy` crate: `evaluate(now, inputs) ->
  Book`, where strategies return **views in units of R** (never sizes) and
  receive **declared measures**, handed by the host from one pure measures
  library shared with live. The clock is the cadence grid.
- **History is measured, and leaving.** The venue serves ~5,000 bars per
  interval on a rolling window, and datawatch keeps only `1m`. Keeping `1h`,
  `4h` and `1d` (`walk-the-coarse-candles`, in galata-datawatch) is step 0.
  Daily bars before 2023-02-26 carry no trades and are filtered out of every
  run.
- **Runs** live in `galata-segments` under their own root, not in the
  deletable scheduler store or beside the tape, which is a cache.
- **Data** is the mainnet record, so the first execution a verdict points at
  is a paper venue on mainnet quotes, sharing this fill model.

Still open: a view with two legs (funding carry).

## Depends on

`galata-wire`, `galata-segments` and the tape schema, published with
galata-datawatch 0.1.0.
