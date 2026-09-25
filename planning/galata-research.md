# galata-research

**NOT PROPOSED.** Named 2026-09-25, from an exploration of the three legacy
stacks' research plans against what `galata-datawatch` now provides. Nothing
here has an OpenSpec artifact or tasks. Next in the pipeline: `design/` with an
Archify chart, then `openspec/changes/`.

---

*Three rewrites have come before this one. Each promised research as the tier
after next, and in two of them it never arrived. The only backtest ever built,
vade-trader's rig, ran seven times and was the one piece of research tooling
that paid for itself: **momentum was turned down for the cost of a replay
instead of an account.** galata is about to rebuild the trading half, and has
the choice the other two did not take: build the thing that says whether a
strategy is worth trading **before** building the thing that trades it.*

---

## The lineage, and what each one left

```
  vade-trader  08-11→08-22   BUILT  lab/src/rig.rs · Rig<S: Strategy>, 696 lines
                             RAN    7 runs in lab-runs/, identical sha on rerun
                             ON     one testnet window, ~31 h, BTC only, 17 capture holes

  vade         08-23→08-27   BUILT  dyno, the paper venue (4.1k lines), a ledger fold
                             NOT    the scorecard (spotter/src/scorecard.rs, never wired)
                                    "history is handed, and nothing hands it yet" (v/D-085)

  galata-legacy 08-27→09-11  PLANNED planning/open-the-research-lane.md and nine others
                             NOT    any runner. "A backtest is those four in a row" —
                                    replay, reader-replay, algo-slow, scorecard — and
                                    the four were never put in a row
```

**What the rig proved:** the mechanism. `Rig<S: Strategy>` feeds envelopes
through `grid::feed::observe_envelope`, the function the live runtime calls, and
`the_rig_and_the_live_feed_agree` holds that. The decision log carries no
prices and is hashed, so a rerun is byte-identical. Fills are priced
**afterwards** by `FillModel`, so a strategy never meets its own cost model.

**What it did not prove:** that any strategy is worth trading. vt/D-067 turned
momentum down at −6.74 modelled. vt/D-071 kept fade's +3.66 from becoming a
grant, because it was measured on its own tuning window: *"a live binary is
granted by a measured verdict, not by existing."* The fill model was mark ± 2
bps, with no queue, no latency and no funding. The sweep and the
pessimistic-beside-central report, both carried forward from the stack before
it (`docs/carried-forward/capabilities/simrig-*.md`), were never built.

## The lessons, which are the requirements

1. **Data coverage is the constraint, not the harness.** Every legacy
   measurement came from one 31-hour window with 17 holes in it. An
   out-of-sample run was demanded (vt/D-071) and there was never a second
   window to run it on. **Coverage, and the holes in it, are reported
   figures.**
2. **The tuning set becomes the evaluation set** unless the harness forbids it.
   A study declares its split, and records whether the out-of-sample half has
   been read.
3. **One strategy seam, shared with live.** That is why the rig paid for
   itself. A backtest that runs a different implementation from the live one
   measures that implementation instead.
4. **A virtual clock end to end.** vt/D-067: the rig and the bus replay
   *"cannot match tick-for-tick, because `vade-replay` is unpaced"*. A strategy
   that reads a wall clock can't be replayed.
5. **One fill model, and it guesses against the strategy** (vt/D-083, v/D-025,
   v/D-026). A second implementation doesn't fail, it disagrees. Queue
   position is a required parameter and defaults to the back of the queue.
   Liquidation is simulated, otherwise the model is assuming infinite leverage.
6. **Simulator state is a ledger fold, never a cache** (vt/D-087: the paper
   account double-counted positions on every restart).
7. **History is handed, never scanned** (v/D-029: *"the backtest that justified
   the strategy was measuring the future"*).
8. **Modelled means labelled.** Every figure a simulation produces says so.

## What already exists in the new world

- **`galata-wire`**: the vocabulary, with no float in any type that crosses a
  contract.
- **`galata-segments`**: durable parquet, with cursors over time, block and
  sequence.
- **The one ingestion path and `replay.rs`**: the record re-fed through the
  same normalisation live capture uses. The tape rebuild is byte-identical.
- **The tape**: `kind=candles|funding|gaps|marks|quotes|trades`, readable by
  DuckDB and polars with no flags. It holds 2026-09-20 → 09-22 today, because
  capture stopped on 09-22 and was relaunched under launchd on 09-25.
- **The history walk** (`history-walk`, `venue-history`): it resumes from the
  record's own watermark and keeps no bookmark. Hyperliquid serves 5,000 candle
  rows per interval, about 208 days at 1h, and a funding walk exists in legacy.
- **`gaps` as a dataset**: every loss is an event with a cause. A simulator
  can refuse to trade across one instead of guessing through it.

**What does not exist:** an algo host, views, fills, a scorecard, a producer.
The legacy lane assumed all of them.

**And one thing that does not carry over.** The new reader's bound is the
store's **durable frontier**, a stream position per venue, and nothing
corresponds to legacy's `reader-replay`, which stated a replay position. A
backtest at simulated time *t* therefore cannot take a bounded view "as of
*t*". The **strategy** doesn't need one: lesson 7 says it is handed events in
order, not given a reader, and only ever sees what the clock has reached. The
type-level no-lookahead property moves from the reader to the seam. The
**host** that hands them does need one, for lookbacks at *t*. That is
[`bound-the-replay`](../../galata-datawatch/planning/bound-the-replay.md),
planned in `galata-datawatch`, where the reader lives.

## The first cut

```
  0  history     datawatch's walk-the-coarse-candles: 1h, 4h and 1d kept before the
                 venue's window drops them. Coverage, holes, trade-less bars and the
                 independent periods per horizon are the first output, before any
                 strategy runs.
  1  the seam    galata-strategy: the Algo trait, Declaration, Inputs, Book and View
                 (in R), the cadence-grid clock, the warm-up Divergence harness and a
                 hashed, price-free decision log. The live host links it later.
  1b measures    one pure measures library (Vec<Bar> + declaration in, figures out),
                 shared with live; the host hands its outputs as held or absent.
  2  fills       ONE FillModel, turning views in R into modelled trips. Pessimistic by
                 default, every parameter swept, and a parameter whose swept range sits
                 inside one observation interval is reported as unmeasurable. Funding
                 charged. Output labelled modelled.
  3  runs        the legacy manifest (open-the-research-lane.md) on galata-segments
                 under var/research: study · trial · per-period returns · selection
                 as an event · a hash per row. Failed and pruned trials count toward
                 N, because the Deflated Sharpe Ratio needs N.
  4  first runs  score-against-random (the zero on every axis) → permute-the-lookback
                 → the trend family (gaps-vs-literature §4, rank 1)
```

**The split by role, kept from legacy's Tier 8.** Rust runs the strategy,
because the seam is shared with live and the seam is what made the rig pay.
Python and polars judge the landscape: PBO, Deflated Sharpe, the stationary
bootstrap, VaR against the grant. Nothing is computed twice, so the float
tolerance question never touches money.

## What it carries from galata-legacy/planning, and where

| legacy file | here |
|---|---|
| `open-the-research-lane` | this file: the runner, the manifest, the sample split |
| `score-against-random` | first run. Random agents are matched on trade count, holding time, side and exposure, and the null is named |
| `permute-the-lookback` | second run. System Parameter Permutation, PBO |
| `simulate-the-variance` | trips come from the simulator until real fills exist |
| `measure-the-edge-ratio` | the same, labelled modelled; re-run on real fills later |
| `measure-the-tail` · `measure-the-shortfall` · `diff-the-views` | **deferred**: each needs the live trading half |
| `report-the-intelligence` | deferred: it narrates scorecards that don't exist yet |
| `signal-the-trend` · `signal-the-perp-structure` | the first strategy family's inputs; funding netted from the long side |

`galata-legacy/planning/decisions.md` carries over whole. There is no genetic
search, no mean-variance optimisation, no market making, and no forward-filling
of bars.

## The line to hold

**It reports a landscape; it never writes a declaration.** A run's output
reaches an algo's configuration only through a person. Selecting a winner is a
timed event in the run store, never an overwrite. Nothing here gates, sizes or
retires anything live. The moment a research run can choose a declaration,
this has become the data-mining machine that
`gaps-vs-buildalpha §4` exists to resist.

And the second line, which the lineage adds: **research does not wait for the
trading half.** Each of the three predecessors built the trading half first and
deferred research; two of them never came back to it. If the seam is shared,
the live host links it later, and `diff-the-views` stops being a hope and
becomes a test.

## Settled, 2026-09-25

### The seam: a shared `galata-strategy` crate

Read against all three predecessors' seams: vade-trader's `Strategy`
(`driver/src/strategy.rs:114`), vade's `FastAlgo`/`SlowAlgo`
(`spotter/src/algo.rs:357`) and galata-legacy's `Algo`
(`crates/algo/src/seam.rs:166`). All three agree on one shape:

```
  fn evaluate(&mut self, now_micros: i64, inputs: &Inputs) -> Book
```

Time and inputs are **handed**, never read. Only the host builds the inputs.
The book is **complete**: a market left out is refused, never "no opinion".
Transport, stamping and sequencing are the host's. Python code is never
replayed; only its recorded output is (vt/D-078). The crate is shared, research
links it first, and the live host links it later. It is also where the trading
vocabulary lives, because `galata-wire` deliberately left legacy's
`trading.rs` behind. Where the three disagree, the choices:

- **A strategy returns views in units of R, not sizes** (galata-legacy's
  `View`): `Held { facing, conviction, ratio in R, entry, protect { stop,
  trail, take } } | Flat { reason } | Absent { reason }`. The strategy never
  sizes; vade-trader's sized `Target` is not carried. So **research scores in
  R**: expectancy, SQN, the R-multiple of each trip. That is also what makes a
  random agent matched on exposure comparable (`score-against-random`). A live
  risk layer sizes later; research needs no allocator.
- **A strategy receives declared measures, handed**, not raw bars. It declares
  the measures it needs (a high and low over a lookback, a return over a
  horizon, ATR, funding), and the host computes them with **one pure measures
  library, shared with live**, handing each as *held* or *absent with a
  reason*. An undeclared input is refused at boot, which vade-trader could
  never do. The cost is a small library before the first strategy: legacy's
  `statistics` crate was already this shape (`Vec<Bar>` + config in, figures
  out, clockless, no I/O).
- **The clock is the cadence grid** (galata-legacy): evaluate at each `due`
  position, and an overrun **skips, never queues**. In a backtest the virtual
  clock *is* that grid, so the rule replays exactly.
- **Determinism is proven twice.** A declared, finite `warm_up`, checked by a
  continuous-versus-warmed `Divergence` harness (vade, galata-legacy), proves
  the warm-up is enough. A hash of the price-free decision log (vade-trader)
  proves the run is reproducible. They test different claims.
- **The slow tier first.** At 4h and daily bars there is no fast tier; a slow
  strategy gets history through the host's bounded view
  (`bound-the-replay`). The fast tier's *no history*, which galata-legacy made
  a type (`NoHistory`), is carried when the first fast strategy arrives, not
  before.

Still open, deliberately: **a view with two legs** (funding carry: long one
market, short another, with the stop on the spread). It is undecided in
galata-legacy's `planning/decisions.md` too, and the trend family, research's
first target, has one leg.

### How much history there is: measured, and it is leaving

The question was "is 208 days of `1h` enough?". That was the wrong unit. Asked
of the venue (`candleSnapshot` from `startTime = 0`), every interval serves its
**most recent ~5,000 bars, and the window rolls**:

```
                  1m        1h                   4h                   1d
  BTC · ETH     3.6 days   208 days (2026-03-01)  833 days (2024-06-14)  from 2020-08-19
  HYPE                     208 days              —                     from listing, 2024-12-05
  xyz:GOLD                 208 days              —                     from 2025-12-22
  funding       BTC from 2023-05-12 · HYPE from 2024-12-05
```

Three consequences, in order:

1. **History is leaving the venue, and datawatch keeps none of it.** It walks
   only `1m`. The oldest day of `1h` and of `4h` leaves every day, for good.
   That is `galata-datawatch/planning/walk-the-coarse-candles.md`, and by the
   rule *what cannot be recovered comes first* it goes before anything here.
   Step 0, *history*, is that change landing, not a research feature.
2. **The daily series is not what it looks like.** BTC's first **921** daily
   bars (2020-08-19 → 2023-02-25) have `volume = 0` and `n = 0`: prices from
   before the venue traded. Real trading starts on **2023-02-26**, and
   funding on 2023-05-12. So the honest depth is **~2.6 years of daily BTC
   and ETH, 2.3 years of `4h`**. The tape already carries `trade_count`.
   **Every run filters `trade_count > 0` and states the filter in its
   manifest**, because a backtest across the trade-less years would measure
   an index, not this venue.
3. **The trend family is measurable at daily and 4h, and not much finer.**
   Weekly-return predictability (Liu & Tsyvinski) and a multi-horizon trend
   factor want years of daily bars, and ~2.6 years is thin but real. 12- and
   24-week momentum over 2.6 years is **about six to thirteen independent
   periods**, so the harness must say *unmeasurable at this depth* for those
   horizons and mean it. Coverage is a reported figure (lesson 1), and so is
   the number of independent periods behind a horizon.

### Where runs live: galata-segments, under their own root

Not cereyan's run store. Its first invariant is that the store is
**deletable**, and a research store deleted is every trial's `N` gone, the one
input the Deflated Sharpe Ratio cannot do without. Not beside the tape either:
the tape is a cache, removed and rebuilt at will (it was, today). Not SQLite,
which is a second storage engine for the same append-only shape.

`galata-segments` under `var/research/`, the record's own machinery: rename
is the commit; `Cursor::Seq` gives the study its monotonic trial counter;
footer labels state the study and the schema version as a *yes*; and DuckDB
and polars read it with no flags, which is where PBO and Deflated Sharpe are
computed. Trials, each trial's per-period return series, and selection events
are three datasets, all append-only. A selection is a row, never an overwrite.

### Which market data drives a run: the mainnet record

That is what datawatch captures, it is the only side with depth, and testnet is
missing symbols (no XRP or LINK there, and an unknown symbol resets the
whole subscription feed rather than being refused). So the first execution a verdict can point at is a **paper venue on
mainnet quotes**. The fill model built here is the one that paper venue will
share (v/D-025, *one fill model*), not a research-only approximation.

## Depends on, and depended on by

- **Depends on** `galata-wire`, `galata-segments` and the tape schema, all to
  be published at `galata-datawatch` Tier 10.
- **Depended on by** the future algo host (through the shared seam), a paper
  venue (through the one fill model) and every "worth live size" verdict.
