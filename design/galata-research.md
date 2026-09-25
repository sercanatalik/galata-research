# galata-research — design

*The mechanism behind [`planning/galata-research.md`](../planning/galata-research.md),
which argues for it and settles its four questions. Chart:
[`charts/galata-research.html`](charts/galata-research.html), with its source
[`charts/galata-research.architecture.json`](charts/galata-research.architecture.json).
Written 2026-09-25. Nothing here is proposed yet; the OpenSpec changes it
implies are listed at the end, in order.*

```
  record + tape ──in order──▶ replay host ──bars──▶ measures ──held/absent──▶ strategy
        │                          ▲                                      ▲       │
        └──────▶ view as of T ─────┘ lookbacks            live host ─ ─ ─┘       │ Book
                 (bound-the-replay)                        (later, same seam)      ▼
  a person ◀──landscape── judge ◀──all N── run store ◀──trips── fill model ◀─ ─ random agents
                          (polars)         (var/research)        │  (one)        (matched exposure)
                                                                  └ ─ ─▶ paper venue (later, same model)
```

---

## The run, end to end

A **study** is a declared question: one strategy family, a parameter space, a
window, a market set, and a sample split. A **trial** is one point of that
space, run over the window. A trial is this, and only this:

1. **The replay host** reads the record's candles for the declared markets
   and widths, filtered to `trade_count > 0` (the filter is written into the
   study). It deduplicates on `(ticker, interval, at_micros)`, keeping the
   final bar: the tape holds a re-fetched bar more than once, by design.
2. **The virtual clock is the cadence grid.** The host steps through `due`
   positions (`start + k × cadence`). At each one it has handed the strategy
   everything up to `due` and nothing after. **An overrun skips and never
   queues** (galata-legacy's runner), so a backtest reproduces the live rule
   exactly.
3. **The measures library** turns the bars up to `due` into each declared
   measure, `held(value)` or `absent(reason)`: *not enough bars*, *a gap in
   the window*, *trade-less bars*. A strategy that declared nothing gets
   nothing; one that reads an undeclared measure fails to compile, not to run.
4. **The strategy** evaluates: `evaluate(&mut self, due, &Inputs) -> Book`.
   The book is complete over its declared markets, and each market's view is
   `Held { facing, conviction, ratio, entry, protect } | Flat | Absent`.
5. **The fill model** turns the sequence of books into **modelled trips**. It
   enters by `entry`, exits by `protect` (stop, trail, take) or by the next
   `Flat`, and charges fees and funding, each in R.
6. **The run store** appends the trial row and its per-period return series.
   The trial counter is the store's own sequence.

The **judge** reads the store after the study is sealed. The **person** reads
the judge.

## The components

### `galata-strategy` (new crate, shared with live)

The trading vocabulary `galata-wire` left behind, plus the seam. It depends on
`galata-wire` only.

```rust
pub trait Algo {
    /// What this algo needs and how often, checked before the first evaluate.
    fn declares(&self) -> Declaration;
    /// One cadence position. `due` is the virtual or live clock; the algo reads no other.
    fn evaluate(&mut self, due_micros: i64, inputs: &Inputs<'_>) -> Book;
}

pub struct Declaration {
    pub markets: Vec<Market>,
    pub measures: Vec<MeasureDecl>,    // e.g. range(high/low, 20 bars, 1d)
    pub cadence_micros: i64,           // the grid
    pub warm_up_micros: i64,           // finite; checked by the Divergence harness
}

pub enum View {
    Held { facing: Facing, conviction: Unit, ratio: RMultiple, entry: Entry, protect: Protect },
    Flat { reason: String },
    Absent { reason: String },
}
```

`Inputs` holds the declared measures by `(market, measure)`, each `held` or
`absent(reason)`, and nothing else: no positions, no fills, no other
algo's views. Carried from galata-legacy `crates/algo/src/seam.rs:166` and
`crates/wire/src/trading.rs`, with two departures. **No history accessor in
the first cut**: slow-tier lookbacks are measures, so the strategy never holds
a reader at all. **No `StatisticsHanded`**: σ and ρ are the allocator's, and
there is no allocator here.

**Two proofs of determinism, both in the crate:**
- `conform(build, positions)`: a continuous instance against one warmed over
  `warm_up` only, position by position, returning `Divergence { at, market,
  continuous, warmed }` on the first disagreement (vade `spotter/src/replay.rs`,
  galata-legacy `conformance.rs`). It proves the declared warm-up is enough.
- **A decision log**: each position's `(due, market, view)` with **no prices
  in it**, hashed with sha256 per trial (vade-trader `lab/src/rig.rs`). It
  proves a rerun is the same run, and it's the log `diff-the-views` compares
  against what a live host published.

### The measures library (new crate, shared with live)

`Vec<Bar> + MeasureDecl → Measure`, pure: no I/O, no clock, no reader.
galata-legacy's `crates/statistics` was already this shape. The first measures
are the ones the trend family needs (gaps-vs-literature §2.1):
`range(high, low, n)`, `ret(horizon)`, `atr(n)`, and funding over a horizon
(the long side's carry, measured at about 11.6% a year at a neutral premium).
**Each states its own floor**: fewer bars than `n`, a gap inside the window,
or a trade-less bar in it gives `absent` with that reason, never a value
computed over a hole.

### The replay host (galata-research)

It owns the virtual clock, reads the record, calls the measures library and
the strategy, and writes the decision log. It is **the only thing that reads
the tape** in a run. Its lookbacks go through `bound-the-replay`'s view at `T`
once that exists; until then, over **closed** days only, it reads the tape
whole and cuts at `due` itself. That's sound because a closed day's tape
doesn't change, and the cut is tested position by position against the
warmed instance.

### The fill model (galata-research, later shared with the paper venue)

`Book` sequence + bars → trips. It **guesses against the strategy** (v/D-025):
- an `Entry::Market` fills at the bar's worse extreme plus
  `max_slippage_bps`, not at the mark;
- a `Limit` fills only if the bar trades **through** it, never just touches it
  (the back of the queue);
- a stop that sits inside a bar with both stop and take inside it is taken as
  the **stop**;
- funding is charged hourly on the position, from the record's funding rows;
- liquidation is simulated at the venue's maintenance margin, because without
  it the model assumes infinite leverage (v/D-026).

**Every figure is labelled `modelled`.** Every parameter above is swept per
study, and a parameter whose swept range sits inside one bar is reported as
**unmeasurable** at that width, not as a number (vade-trader's carried-forward
simrig requirements). Output is in R: a trip's R-multiple is its P&L over the
risk its `protect.stop` stated at entry.

### Random agents (galata-research)

`score-against-random`'s baseline, as ordinary `Algo`s through the same host,
fill model and store. Each is **matched** to the strategy's realised trade
count, holding-time distribution, side mix and exposure, and seeded, with the
seed in the trial row. The report is the strategy's percentile,
`(1 + #{random ≥ observed}) / (N + 1)`, with the null named. A permuted-bar
null (Masters) is a second null, stated beside the first, never merged with it.

### The run store (`galata-segments` under `var/research/`)

Three append-only datasets, each segment footer-labelled with its `study` and
`schema_version`:

```
  trials       (study, seq) · params · seed · state (ok | failed | pruned)
               · decision-log sha256 · T · mean R · SR · skew · kurtosis
  returns      (study, seq, period) · return in R        ← PBO and DSR read this
  selections   (study, at) · seq chosen · by whom · why  ← an event, never an overwrite
```

A study has its own manifest row (the search-space hash, the window, the
markets, the split, whether the out-of-sample half has been read, the binary
hash, and the filters) and is **sealed** before the judge reads it. Failed and
pruned trials are rows, because the Deflated Sharpe Ratio needs the true `N`.

### The judge (Python, polars)

It reads the sealed study, and computes and renders: the landscape across the
parameter space (System Parameter Permutation's median, not its peak), PBO by
combinatorially symmetric cross-validation, the Deflated Sharpe Ratio, the
stationary bootstrap of trips, and the percentile against random. It writes a
dated report. **It writes no configuration anywhere.** The only output that
reaches a live declaration is a person's.

## What goes where

```
  galata-research/                 (this repo: a Rust workspace plus py/)
    crates/galata-strategy/        the seam and vocabulary   → published, the live host links it
    crates/galata-measures/        the pure measures         → published, the live host links it
    crates/galata-research/        host · fills · random · store · the `research` binary
    py/judge/                      polars: PBO · DSR · bootstrap · reports
    design/ · planning/ · openspec/
```

The two shared crates live here first and are published before the live host
exists. That's the order the planning file argues for: research defines the
seam, the live host links it later.

## Decisions carried, and departed from

| | carried from | here |
|---|---|---|
| handed, never read (time and inputs) | all three predecessors | same |
| views in R, never sizes | galata-legacy `View` | same; vade-trader's sized `Target` not carried |
| declared inputs, refused at boot | vade, galata-legacy | as measures from one library |
| cadence grid, skip never queue | galata-legacy runner | same, as the virtual clock |
| warm-up Divergence **and** decision-log hash | vade · vade-trader | both |
| one fill model, guessing against the strategy | v/D-025, vt/D-083 | same, swept and labelled |
| a history accessor on the slow tier | galata-legacy `View` via reader | **departed**: lookbacks are measures |
| Python replayed | never (vt/D-078) | the judge is Python; strategies are Rust |

## The OpenSpec changes, in order

1. **`add-galata-strategy`**: the crate, the vocabulary, `conform`, the
   decision log. No host yet; tested by fixtures.
2. **`add-galata-measures`**: the four trend measures, their floors, and the
   `absent` reasons.
3. **`replay-a-closed-window`**: the host, the virtual clock, the dedup and
   filter rules, one strategy end to end into a decision log.
4. **`model-the-fills`**: the fill model, its sweep and the `modelled` label.
5. **`record-the-runs`**: the store, the study manifest and sealing.
6. **`score-against-random`**: the first study, and the judge's first report.

`bound-the-replay` (in galata-datawatch) is needed by 3 only for open days,
and is proposed when the host first needs a view at an open `T`.

## Open

- **A view with two legs** (funding carry): undecided, deliberately. The first
  family has one leg.
- **Where the judge's reports live** (beside the run store, or the tower):
  decided with `score-against-random`.
- **`bound-the-replay`'s lag** (declare a lag, record the commit time, or state
  it): decided when the host first reads an open day.
