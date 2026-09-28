"""Market-data signals as galata-datawatch stored them (`kind=signals`, its Tier 16), read point-in-time.

    gr.signals.history("varcov", "2026-09-01T00:00Z", "2026-10-01T00:00Z", horizon="4h")
    gr.signals.known_at("beta", "2026-09-28T03:00Z")      # what a reader had then
    gr.signals.matrix(rows, "correlation")                  # one asof, square

A signal has **two clocks**, and both are columns:

- `ts`, its **grid position**: the close of the bar the figure stands on
  (`asof_micros`). This is the one time axis, as for every dataset here
  (settled design 3).
- `computed_ts`, **when the figure became known**: our clock when the flow
  computed it. It is not a venue time, so it says so in its name
  (settled design, the design rule on clocks).

A figure is never known before it is computed, and a backtest that reads one
at its `ts` looks up to the flow's cadence into the future. `as_of` filters on
`computed_ts`, and `known_at` is the reader's view at an instant.

**Stored, not recomputed.** A signal is a record of what the flow computed
under the model it declared, and is read as it stands: a value, or the reason
it is absent. The arithmetic behind it is `gr.models.corr`, but these rows are
the flow's, with its `code` (galata-research's commit) on each.
"""

from datetime import datetime

import polars as pl

from . import _root, _scan, utils
from ._errors import Refused

_READ = [
    "signal", "horizon", "measure", "ticker_i", "ticker_j", "h", "value", "absent", "n_eff",
    "asof_micros", "target_micros", "computed_micros", "fitted_through_micros", "fit_from_micros",
    "model", "params", "fitted", "after_gap", "code", "run_id",
]  # fmt: skip
_KEY = ["signal", "horizon", "measure", "ticker_i", "ticker_j", "h", "asof_micros"]
SCHEMA = {
    "signal": pl.String,
    "horizon": pl.String,
    "measure": pl.String,
    "ticker_i": pl.String,
    "ticker_j": pl.String,
    "h": pl.Int64,
    "ts": _scan.UTC_US,
    "target_ts": _scan.UTC_US,
    "computed_ts": _scan.UTC_US,
    "value": pl.Float64,
    "absent": pl.String,
    "n_eff": pl.Float64,
    "fitted_through": _scan.UTC_US,
    "fit_from": _scan.UTC_US,
    "model": pl.String,
    "params": pl.String,
    "fitted": pl.Boolean,
    "after_gap": pl.Boolean,
    "code": pl.String,
    "run_id": pl.String,
}


def _dataset():
    dataset = _root.root() / "tape" / "kind=signals"
    every = _scan.segments(dataset)
    if not every:
        raise Refused(f"the record has no signals under {dataset}: galata-datawatch's derive-the-signals flow writes them")
    _scan.require_columns(every[-1], set(_READ))
    return dataset, every


def _rows(files, signal: str, horizon: str | None, measure: str | None, lo: int | None, hi: int | None, bound: int | None) -> pl.LazyFrame:
    lf = _scan.scan(files, _READ).filter(pl.col("signal") == signal)
    if horizon is not None:
        lf = lf.filter(pl.col("horizon") == horizon)
    if measure is not None:
        lf = lf.filter(pl.col("measure") == measure)
    if lo is not None:
        lf = lf.filter((pl.col("asof_micros") >= lo) & (pl.col("asof_micros") < hi))
    if bound is not None:
        lf = lf.filter(pl.col("computed_micros") <= bound)
    # One figure computed twice (a run repeated after a failed commit): the
    # latest computation known by the bound is the figure; run_id breaks a tie.
    return lf.sort(["computed_micros", "run_id"]).group_by(_KEY).agg(pl.all().last())


def _out(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.select(
        "signal", "horizon", "measure", "ticker_i", "ticker_j", "h",
        _scan.clock("asof_micros", "ts"),
        _scan.clock("target_micros", "target_ts"),
        _scan.clock("computed_micros", "computed_ts"),
        "value", "absent", "n_eff",
        _scan.clock("fitted_through_micros", "fitted_through"),
        _scan.clock("fit_from_micros", "fit_from"),
        "model", "params", "fitted", "after_gap", "code", "run_id",
    ).sort(["horizon", "ts", "measure", "ticker_i", "ticker_j"])  # fmt: skip


def history(
    signal: str,
    start: datetime | str,
    end: datetime | str,
    *,
    horizon: str | None = None,
    measure: str | None = None,
    as_of: datetime | str | None = None,
    engine: str = "polars",
):
    """One signal's stored rows with `ts` (the grid position) in `[start, end)`.

    `as_of` keeps the rows computed at or before it: the record as a reader
    could have read it then. A figure computed more than once is its latest
    computation within that bound. A value or the reason it is absent, never
    both; nothing is recomputed.
    """
    lo, hi = utils.window(start, end)
    bound = None if as_of is None else utils.instant("as_of", as_of)
    if engine not in _scan.ENGINES:
        raise Refused(f"engine={engine!r} is not one of {', '.join(_scan.ENGINES)}")
    _, every = _dataset()
    # Partitions are by asof date, the grid position, so the window picks them.
    files = _scan.partitions(_root.root() / "tape" / "kind=signals", lo, hi)
    if not files:
        return _scan.finish(pl.LazyFrame(schema=SCHEMA), engine)
    return _scan.finish(_out(_rows(files, signal, horizon, measure, lo, hi, bound)), engine)


def known_at(signal: str, when: datetime | str, *, horizon: str | None = None, lookback_days: int = 14) -> pl.DataFrame:
    """Each horizon's newest figures that had been computed by `when`: what a reader held at that instant.

    Reads the asof dates within `lookback_days` before `when`, twice the widest
    horizon the flow declares (1w), as the tower does.
    """
    at = utils.instant("when", when)
    lo = at - lookback_days * 86_400_000_000
    _, every = _dataset()
    files = _scan.partitions(_root.root() / "tape" / "kind=signals", lo, at + 1)
    if not files:
        return pl.DataFrame(schema=SCHEMA)
    rows = _rows(files, signal, horizon, None, lo, at + 1, at)
    newest = rows.group_by("horizon").agg(pl.col("asof_micros").max().alias("_newest"))
    return _out(rows.join(newest, on="horizon").filter(pl.col("asof_micros") == pl.col("_newest")).drop("_newest")).collect()


def matrix(rows: pl.DataFrame | pl.LazyFrame, measure: str) -> pl.DataFrame:
    """One horizon's one asof as a square frame: `ticker` and a column per instrument, both halves filled.

    An absent cell is null. The rows must hold exactly one (horizon, ts).
    """
    frame = utils.lazy(rows).filter(pl.col("measure") == measure).collect()
    keys = frame.select("horizon", "ts").unique()
    if keys.height != 1:
        raise Refused(f"a matrix is one horizon at one asof; these rows hold {keys.height}")
    if frame["ticker_j"].null_count():
        raise Refused(f"{measure} is about one instrument, not a pair: it has no matrix")
    both = pl.concat(
        [
            frame.select(pl.col("ticker_i").alias("a"), pl.col("ticker_j").alias("b"), "value"),
            frame.select(pl.col("ticker_j").alias("a"), pl.col("ticker_i").alias("b"), "value"),
        ]
    ).unique(subset=["a", "b"], keep="first")
    tickers = sorted(set(both["a"]) | set(both["b"]))
    wide = both.pivot(on="b", index="a", values="value").rename({"a": "ticker"}).sort("ticker")
    return wide.select("ticker", *[pl.col(t) if t in wide.columns else pl.lit(None, pl.Float64).alias(t) for t in tickers])
