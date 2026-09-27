"""Helpers that know nothing about the record: frames, columns, times.

A helper belongs here only if it is domain-free and used by at least two
modules. Anything about bars or series is `timeseries`'.
"""

from collections.abc import Iterable
from datetime import UTC, datetime

import polars as pl

from ._errors import Refused


def lazy(frame: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """The frame as a LazyFrame, whichever it was given as."""
    return frame.lazy()


def require(frame: pl.LazyFrame | pl.DataFrame, columns: Iterable[str], hint: str) -> None:
    """Refuse a frame missing any of `columns`, naming them, then `hint`: where the right frame comes from."""
    names = frame.collect_schema().names()
    missing = [c for c in columns if c not in names]
    if missing:
        raise Refused(f"needs {', '.join(missing)}; {hint}")


def instant(name: str, value: datetime | str) -> int:
    """A zone-aware time as micros since the epoch. A naive one is refused."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            raise Refused(f"{name}={value!r} is not an ISO-8601 time") from None
    if not isinstance(value, datetime):
        raise Refused(f"{name} must be a datetime or an ISO-8601 string, not {type(value).__name__}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise Refused(f"{name}={value.isoformat()} has no zone; give it one, e.g. tzinfo=UTC or a +00:00 offset")
    delta = value - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def window(start: datetime | str, end: datetime | str) -> tuple[int, int]:
    """`[start, end)` in micros; an empty or reversed window is refused."""
    lo, hi = instant("start", start), instant("end", end)
    if lo >= hi:
        raise Refused(f"start ({start}) must be before end ({end})")
    return lo, hi
