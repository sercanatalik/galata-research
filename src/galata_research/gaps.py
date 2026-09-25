"""The record's gaps, and marking the rows that sit inside one.

A gap is capture's own statement that it was not receiving: every restart
publishes the window it missed, dated from the last durable receipt
(datawatch `coverage-and-gaps`). Its bounds are receipt times, so they are
loaded as `from_recv_ts` and `to_recv_ts`, never as `ts`.
"""

import re
from collections.abc import Sequence
from datetime import datetime, timedelta

import polars as pl

from . import _root, _scan
from ._errors import Refused

MASKABLE = ("candles", "trades", "quotes")

GAP_SCHEMA = {
    "venue": pl.String,
    "ticker": pl.String,
    "series": pl.String,
    "cause": pl.String,
    "clipped": pl.String,
    "from_recv_ts": _scan.UTC_US,
    "to_recv_ts": _scan.UTC_US,
}
_READ = ["venue", "ticker", "series", "cause", "clipped", "from_micros", "to_micros"]


def _raw() -> pl.LazyFrame:
    # Read whole: a gap that starts before a window lives in an earlier
    # partition, and there are 72 rows per restart.
    dataset = _root.root() / "tape" / "kind=gaps"
    files = _scan.segments(dataset)
    if not files:
        raise Refused(f"the record has no gaps under {dataset}")
    _scan.require_columns(files[-1], set(_READ))
    return _scan.scan(files, _READ)


def gaps(
    tickers: Sequence[str] | str | None,
    start: datetime | str,
    end: datetime | str,
    *,
    series: str | None = None,
    engine: str = "polars",
):
    """Every gap overlapping `[start, end)`, on the receipt clock.

    `tickers` keeps those tickers' gaps and every venue-wide gap (a null
    ticker). A ticker with no gaps gives no rows, which is a true answer.
    """
    lo, hi = _scan.window(start, end)
    if engine not in _scan.ENGINES:
        raise Refused(f"engine={engine!r} is not one of {', '.join(_scan.ENGINES)}")
    lf = _raw().filter((pl.col("from_micros") < hi) & (pl.col("to_micros") > lo))
    if tickers is not None:
        wanted = [tickers] if isinstance(tickers, str) else list(tickers)
        lf = lf.filter(pl.col("ticker").is_in(wanted) | pl.col("ticker").is_null())
    if series is not None:
        lf = lf.filter(pl.col("series") == series)
    out = (
        lf.with_columns(_scan.clock("from_micros", "from_recv_ts"), _scan.clock("to_micros", "to_recv_ts"))
        .select(list(GAP_SCHEMA))
        .sort(["venue", "ticker", "series", "from_recv_ts"], nulls_last=True)
    )
    return _scan.finish(out, engine)


def mask_gaps(frame: pl.LazyFrame | pl.DataFrame, dataset: str, *, margin: str | timedelta = "1s"):
    """`frame` with `in_gap` and `gap_cause`, every row kept.

    A tick is in a gap when `ts` is in `[from − margin, to)`. A bar is in a
    gap when `[ts, close_ts)` overlaps it **and** its kept receipt (`recv_ts`)
    came before the gap's end: a candle is the venue's own aggregate, so one
    restated after the gap (the walk after every restart) is complete, and 91%
    of 1m bars on 2026-09-25 were restated that way. The bounds are receipt times, and
    venue time trails receipt (quotes p99 0.87 s on 2026-09-25), so `margin`
    widens the start, and only the start: it overstates a loss, and never
    narrows one.
    """
    if dataset not in MASKABLE:
        raise Refused(f"dataset={dataset!r} is not one of {', '.join(MASKABLE)}")
    widen = _micros(margin)
    eager = isinstance(frame, pl.DataFrame)
    lf = frame.lazy()
    columns = lf.collect_schema().names()
    needed = ["venue", "ticker", "ts", *(["close_ts", "recv_ts"] if dataset == "candles" else [])]
    missing = [c for c in needed if c not in columns]
    if missing:
        raise Refused(f"a {dataset} frame to mask needs {', '.join(missing)}")
    taken = [c for c in ("in_gap", "gap_cause") if c in columns]
    if taken:
        raise Refused(f"the frame already has {', '.join(taken)}; mask it once")

    windows = _merged(_raw().filter(pl.col("series") == dataset), widen)
    probe = (pl.col("close_ts") - pl.duration(microseconds=1)) if dataset == "candles" else pl.col("ts")
    lf = lf.with_row_index("_mask_row").with_columns(probe.alias("_probe")).sort("_probe")

    hits = []
    for by, part in (
        (["venue", "ticker"], windows.filter(pl.col("ticker").is_not_null())),
        (["venue"], windows.filter(pl.col("ticker").is_null()).drop("ticker")),
    ):
        # Windows are disjoint and sorted, so the last one starting at or
        # before the probe is the only one that can hold it. Both sides are
        # sorted on the join key just above, which polars cannot check per group.
        extra = ["recv_ts"] if dataset == "candles" else []
        found = lf.select("_mask_row", "_probe", "ts", *extra, *by).join_asof(
            part.sort("_lo"), left_on="_probe", right_on="_lo", by=by, strategy="backward", check_sortedness=False
        )
        # A tick needs ts < to; a bar needs to > ts (its open), since its probe
        # is its close, and a bar the venue restated after the gap is whole.
        hit = pl.col("_hi").is_not_null() & (pl.col("_hi") > pl.col("ts"))
        if dataset == "candles":
            hit = hit & (pl.col("recv_ts") < pl.col("_hi"))
        hits.append(found.select("_mask_row", pl.when(hit).then(pl.col("_cause")).alias("_c")))

    causes = (
        pl.concat(hits)
        .group_by("_mask_row")
        .agg(pl.col("_c").drop_nulls().unique().sort().str.join(",").alias("gap_cause"))
        .with_columns(pl.when(pl.col("gap_cause") != "").then(pl.col("gap_cause")).alias("gap_cause"))
    )
    out = (
        lf.join(causes, on="_mask_row", how="left")
        .sort("_mask_row")
        .with_columns(pl.col("gap_cause").is_not_null().alias("in_gap"))
        .select([*columns, "in_gap", "gap_cause"])
    )
    return out.collect() if eager else out


def _merged(raw: pl.LazyFrame, widen: int) -> pl.LazyFrame:
    """Disjoint windows per (venue, ticker): overlapping or touching gaps become one, causes joined."""
    keys = ["venue", "ticker"]
    return (
        raw.select(*keys, (pl.col("from_micros") - widen).alias("lo"), pl.col("to_micros").alias("hi"), "cause")
        .sort([*keys, "lo"], nulls_last=True)
        .with_columns(pl.col("hi").cum_max().shift(1).over(keys).alias("_reach"))
        .with_columns((pl.col("_reach").is_null() | (pl.col("lo") > pl.col("_reach"))).cum_sum().over(keys).alias("_run"))
        .group_by([*keys, "_run"])
        .agg(
            pl.col("lo").min(),
            pl.col("hi").max(),
            pl.col("cause").unique().sort().str.join(",").alias("_cause"),
        )
        .select(
            *keys,
            pl.col("lo").cast(pl.Datetime("us")).dt.replace_time_zone("UTC").alias("_lo"),
            pl.col("hi").cast(pl.Datetime("us")).dt.replace_time_zone("UTC").alias("_hi"),
            "_cause",
        )
    )


_MARGIN = re.compile(r"^(-?\d+)(us|ms|s|m)$")
_UNIT = {"us": 1, "ms": 1_000, "s": 1_000_000, "m": 60_000_000}


def _micros(margin: str | timedelta) -> int:
    if isinstance(margin, timedelta):
        value = (margin.days * 86_400 + margin.seconds) * 1_000_000 + margin.microseconds
    else:
        m = _MARGIN.match(str(margin).strip())
        if not m:
            raise Refused(f"margin={margin!r} is not a duration like '1s', '500ms' or '0s'")
        value = int(m.group(1)) * _UNIT[m.group(2)]
    if value < 0:
        raise Refused(f"margin={margin!r} is negative; a margin widens a loss and never narrows it")
    return value
