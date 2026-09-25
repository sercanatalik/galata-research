"""Two clocks: settled funding on the venue's, marks and live funding on ours, and the one join between them.

Hyperliquid's asset context carries no venue time (every marks and live
funding row, 2026-09-25), so the mark, oracle, mid, open interest, premium and
predicted rate exist only as received: `recv_ts`, and never `ts`. A settled
rate is a venue event at its hour, on `ts`. `join_recv` is how a venue-timed
row meets a receipt-timed one, and every column it brings says `_recv`.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta

import polars as pl

from . import _root, _scan
from ._errors import Refused

MARK_VALUES = ["mark", "oracle", "mid", "index", "open_interest", "premium"]
_KEYS = ["venue", "ticker"]
_BASE = ["venue", "ticker", "at_micros", "recv_micros", "stream_seq"]

FUNDING_SCHEMA = {"venue": pl.String, "ticker": pl.String, "ts": _scan.UTC_US, "rate": pl.Float64, "recv_ts": _scan.UTC_US}
LIVE_SCHEMA = {"venue": pl.String, "ticker": pl.String, "recv_ts": _scan.UTC_US, "rate": pl.Float64}
MARK_SCHEMA = {"venue": pl.String, "ticker": pl.String, "recv_ts": _scan.UTC_US, **{c: pl.Float64 for c in MARK_VALUES}}


def funding(
    tickers: Sequence[str] | str | None,
    start: datetime | str,
    end: datetime | str,
    *,
    as_of: datetime | str | None = None,
    engine: str = "polars",
):
    """Settled funding: the rate charged at each hour, on the venue's clock.

    Only rows the venue timed (`fundingHistory`). The live prediction is
    `funding_live`. At a neutral premium Hyperliquid's rate is its interest
    floor, 0.0000125 an hour (0.01% per 8 h). The record's settled history is
    only as deep as datawatch's funding walk.
    """
    lo, hi = _scan.window(start, end)
    bound = None if as_of is None else _scan.instant("as_of", as_of)
    dataset, wanted = _dataset("funding", ["rate"], tickers, engine)
    files = _scan.partitions(dataset, lo, hi)
    if not files:
        return _scan.finish(pl.LazyFrame(schema=FUNDING_SCHEMA), engine)
    lf = _scan.scan(files, [*_BASE, "rate"], position=True).filter(
        pl.col("at_micros").is_not_null()
        & pl.col("ticker").is_in(wanted)
        & (pl.col("at_micros") >= lo)
        & (pl.col("at_micros") < hi)
    )
    if bound is not None:
        lf = lf.filter(pl.col("at_micros") <= bound)
    out = (
        lf.sort(["recv_micros", "stream_seq", "_row"])
        .group_by([*_KEYS, "at_micros"])
        .agg(pl.all().first())
        .sort([*_KEYS, "at_micros"])
        .with_columns(_scan.clock("at_micros", "ts"), _scan.clock("recv_micros", "recv_ts"), *_scan.floats("rate"))
        .select(list(FUNDING_SCHEMA))
    )
    return _scan.finish(out, engine)


def funding_live(
    tickers: Sequence[str] | str | None,
    start: datetime | str,
    end: datetime | str,
    *,
    collapse: bool = False,
    engine: str = "polars",
):
    """The venue's current predicted rate, as received: `recv_ts`, no `ts`.

    Pushed with every asset context, about once a second per ticker; the rate
    changed on 8% of pushes. `collapse=True` keeps the first receipt of each
    rate. It is off by default, so that "unchanged" never reads as "not
    received" in `join_recv`.
    """
    return _received("funding", ["rate"], LIVE_SCHEMA, tickers, start, end, collapse, engine)


def marks(
    tickers: Sequence[str] | str | None,
    start: datetime | str,
    end: datetime | str,
    *,
    collapse: bool = False,
    engine: str = "polars",
):
    """Mark, oracle, mid, index, open interest and premium, as received: `recv_ts`, no `ts`.

    `index` is null on Hyperliquid, which prints none; `mid` is the book's
    midpoint as the venue printed it. `collapse=True` drops consecutive
    identical rows (30% of pushes), keeping the first receipt of each state.
    """
    return _received("marks", MARK_VALUES, MARK_SCHEMA, tickers, start, end, collapse, engine)


def join_recv(left: pl.LazyFrame | pl.DataFrame, right: pl.LazyFrame | pl.DataFrame, *, tolerance: str | timedelta | None = "5s"):
    """Each `left` row with the latest `right` row received by its `ts`, per venue and ticker.

    The one place a venue-timed row meets a receipt-timed one. A value
    received by `ts` was produced before it, so nothing is looked ahead;
    staleness is the venue's lag plus the push interval (1.4 s median).
    Every joined column `c` becomes `c_recv`, and the matched row's receipt is
    `matched_recv_ts`. A match older than `tolerance` (default 5 s, over three
    times the p99 push interval of 1.54 s) is null, so a value is never
    carried across a gap. `tolerance=None` removes the bound.
    """
    eager = isinstance(left, pl.DataFrame)
    lhs, rhs = left.lazy(), right.lazy()
    lcols, rcols = lhs.collect_schema().names(), rhs.collect_schema().names()
    missing = [c for c in ("venue", "ticker", "ts") if c not in lcols]
    if missing:
        raise Refused(f"left needs {', '.join(missing)}: join_recv matches a venue-timed frame")
    if "ts" in rcols:
        raise Refused("right has ts, a venue clock: join two venue-timed frames with join_asof, not join_recv")
    missing = [c for c in ("venue", "ticker", "recv_ts") if c not in rcols]
    if missing:
        raise Refused(f"right needs {', '.join(missing)}: join_recv matches a receipt-timed frame")
    values = [c for c in rcols if c not in ("venue", "ticker", "recv_ts")]
    names = {**{c: f"{c}_recv" for c in values}, "recv_ts": "matched_recv_ts"}
    taken = [n for n in names.values() if n in lcols]
    if taken:
        raise Refused(f"left already has {', '.join(taken)}")

    joined = (
        lhs.with_row_index("_join_row")
        .sort("ts")
        .join_asof(
            rhs.select("venue", "ticker", "recv_ts", *values).rename(names).sort("matched_recv_ts"),
            left_on="ts",
            right_on="matched_recv_ts",
            by=_KEYS,
            strategy="backward",
            tolerance=tolerance,
            # Both sides are sorted on their keys just above; polars cannot check that per group.
            check_sortedness=False,
        )
        .sort("_join_row")
        .select([*lcols, *names.values()])
    )
    return joined.collect() if eager else joined


def _dataset(kind: str, values: list[str], tickers, engine):
    if engine not in _scan.ENGINES:
        raise Refused(f"engine={engine!r} is not one of {', '.join(_scan.ENGINES)}")
    dataset = _root.root() / "tape" / f"kind={kind}"
    every = _scan.segments(dataset)
    if not every:
        raise Refused(f"the record has no {kind} under {dataset}")
    _require(every[-1], values, kind)
    return dataset, _scan.wanted(tickers, sorted(_scan.tickers(every)), kind)


def _require(file, values: list[str], kind: str) -> None:
    missing = {*_BASE, *values} - set(pl.read_parquet_schema(file))
    if missing:
        remedy = " — it predates datawatch de22b3c; rebuild the tape" if {"mid", "premium"} & missing else ""
        raise Refused(f"{file} lacks {', '.join(sorted(missing))}{remedy}")


def _received(kind, values, schema, tickers, start, end, collapse, engine):
    lo, hi = _scan.window(start, end)
    dataset, wanted = _dataset(kind, values, tickers, engine)
    # An untimed row is partitioned by its receipt day, so the window's days are receipt days.
    files = _scan.partitions(dataset, lo, hi)
    if not files:
        return _scan.finish(pl.LazyFrame(schema=schema), engine)
    for file in files:
        _require(file, values, kind)
    lf = (
        _scan.scan(files, [*_BASE, *values], position=True)
        .filter(
            pl.col("at_micros").is_null()
            & pl.col("ticker").is_in(wanted)
            & (pl.col("recv_micros") >= lo)
            & (pl.col("recv_micros") < hi)
        )
        .with_columns(*_scan.floats(*values))
        .sort([*_KEYS, "recv_micros", "stream_seq", "_row"])
    )
    if collapse:
        changed = [pl.col(c).ne_missing(pl.col(c).shift(1)).over(_KEYS) for c in values]
        first = pl.int_range(pl.len()).over(_KEYS) == 0
        lf = lf.filter(first | pl.any_horizontal(changed))
    out = lf.with_columns(_scan.clock("recv_micros", "recv_ts")).select(list(schema))
    return _scan.finish(out, engine)
