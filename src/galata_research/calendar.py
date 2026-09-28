"""Exchange sessions and closures, from a calendar maintained upstream: the `[calendars]` extra.

    gr.calendar.sessions("XNYS", start, end)        # date, open_ts, close_ts, early_close
    gr.calendar.closures("XNYS", start, end)        # the weekdays it did not open
    gr.calendar.mark_sessions(frame, "XNYS")        # xnys_open, xnys_closed_day

Legacy's rule holds: fetch sessions, do not maintain a holiday file
(`legacy/galata-legacy/design/datawatch/trading-hours.md:68`). The calendar
is `exchange_calendars`, which records moving holidays, early closes and
ad-hoc closures (the national day of mourning of 2025-01-09) as they are
announced; its version is pinned in `uv.lock`. It returns pandas, which is
taken apart here into polars and plain values and never reaches a caller.
"""

from datetime import UTC, date, datetime, timedelta

import polars as pl

from . import utils
from ._errors import Refused

try:
    import exchange_calendars as _xc
except ImportError as missing:
    raise Refused(f"gr.calendar needs the calendars extra ({missing.name} is missing): uv sync --extra calendars") from None

_UTC_US = pl.Datetime("us", "UTC")


def _calendar(exchange: str, start: date, end: date):
    """The calendar over a margin of 10 days each side: its own bounds must be sessions."""
    if exchange not in _xc.get_calendar_names():
        raise Refused(f"exchange={exchange!r} is not a calendar exchange_calendars lists, e.g. XNYS, XLON, CMES")
    pad = timedelta(days=10)
    return _xc.get_calendar(exchange, start=(start - pad).isoformat(), end=(end + pad).isoformat())


def _days(start, end) -> tuple[date, date]:
    lo, hi = utils.window(start, end)
    first = datetime.fromtimestamp(lo / 1e6, UTC).date()
    last = datetime.fromtimestamp((hi - 1) / 1e6, UTC).date()
    return first, last


def sessions(exchange: str, start, end) -> pl.DataFrame:
    """One row per session whose date falls in `[start, end)`: `date, open_ts, close_ts, early_close`, times in UTC.

    `early_close` is true when the session closes before that weekday's
    regular close: XNYS's day after Thanksgiving, 13:00 New York.
    """
    first, last = _days(start, end)
    cal = _calendar(exchange, first, last)
    names = [d for d in cal.sessions if first <= d.date() <= last]
    early = {d.date() for d in cal.early_closes}
    rows = [
        {
            "date": d.date(),
            "open_ts": cal.session_open(d).to_pydatetime(),
            "close_ts": cal.session_close(d).to_pydatetime(),
            "early_close": d.date() in early,
        }
        for d in names
    ]
    schema = {"date": pl.Date, "open_ts": _UTC_US, "close_ts": _UTC_US, "early_close": pl.Boolean}
    return pl.DataFrame(rows, schema=schema)


def closures(exchange: str, start, end) -> pl.DataFrame:
    """The Monday–Friday dates in `[start, end)` on which the exchange did not open: `date`."""
    first, last = _days(start, end)
    opened = set(sessions(exchange, start, end)["date"].to_list())
    days = pl.date_range(first, last, eager=True)
    return pl.DataFrame({"date": [d for d in days.to_list() if d.weekday() < 5 and d not in opened]}, schema={"date": pl.Date})


def mark_sessions(frame: pl.LazyFrame | pl.DataFrame, exchange: str):
    """The frame with `<x>_open` (ts inside a session) and `<x>_closed_day` (the exchange's local date is a closure).

    `<x>` is the exchange code in lower case. The exchange's zone is the
    calendar's own, never guessed from an abbreviation.
    """
    utils.require(frame, ("ts",), "a frame with a UTC `ts`")
    lf = utils.lazy(frame)
    bounds = lf.select(pl.col("ts").min().alias("lo"), pl.col("ts").max().alias("hi")).collect().row(0)
    if bounds[0] is None:
        return frame.with_columns(pl.lit(False).alias(f"{exchange.lower()}_open"), pl.lit(False).alias(f"{exchange.lower()}_closed_day"))
    lo, hi = bounds[0].replace(hour=0, minute=0, second=0, microsecond=0), bounds[1]
    hi = hi.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=2)
    zone = str(_calendar(exchange, lo.date(), hi.date()).tz)
    x = exchange.lower()
    open_ = sessions(exchange, lo, hi).rename({"date": "_d"})
    shut = closures(exchange, lo, hi).rename({"date": "_d"}).with_columns(pl.lit(True).alias(f"{x}_closed_day"))
    out = (
        lf.with_columns(pl.col("ts").dt.convert_time_zone(zone).dt.date().alias("_d"))
        .join(open_.lazy(), on="_d", how="left")
        .join(shut.lazy(), on="_d", how="left")
        .with_columns(
            ((pl.col("ts") >= pl.col("open_ts")) & (pl.col("ts") < pl.col("close_ts"))).fill_null(False).alias(f"{x}_open"),
            pl.col(f"{x}_closed_day").fill_null(False),
        )
        .drop("_d", "open_ts", "close_ts", "early_close")
    )
    return out if isinstance(frame, pl.LazyFrame) else out.collect()


def reopenings(exchange: str, start, end) -> pl.DataFrame:
    """Each session open in `[start, end)` that follows a closure of more than 24 hours.

    `open_ts, closed_from, closed_hours, kind`: `weekend` when the closure
    spans a Saturday and lasts at most 60 hours, `weekend+holiday` when it
    spans a Saturday and lasts longer, `holiday` otherwise. The calendar is
    upstream's: for COMEX, NYMEX and CMES it has no daily break, and closes
    at 22:00 UTC in US daylight time where CME's Globex closes at 21:00.
    """
    lo, hi = utils.window(start, end)
    pad = 10 * 86_400_000_000
    s = sessions(exchange, datetime.fromtimestamp((lo - pad) / 1e6, UTC), datetime.fromtimestamp(hi / 1e6, UTC))
    s = s.with_columns(pl.col("close_ts").shift(1).alias("closed_from")).drop_nulls("closed_from")
    hours = (pl.col("open_ts") - pl.col("closed_from")).dt.total_seconds() / 3600
    saturday = pl.date_ranges(pl.col("closed_from").dt.date(), pl.col("open_ts").dt.date()).list.eval(pl.element().dt.weekday() == 6).list.any()
    out = (
        s.with_columns(hours.alias("closed_hours"), saturday.alias("_sat"))
        .filter((pl.col("closed_hours") > 24) & (pl.col("open_ts").dt.epoch("us") >= lo) & (pl.col("open_ts").dt.epoch("us") < hi))
        .with_columns(
            pl.when(pl.col("_sat") & (pl.col("closed_hours") <= 60)).then(pl.lit("weekend"))
            .when(pl.col("_sat")).then(pl.lit("weekend+holiday"))
            .otherwise(pl.lit("holiday")).alias("kind")
        )  # fmt: skip
    )
    return out.select("open_ts", "closed_from", "closed_hours", "kind")
