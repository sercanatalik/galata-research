"""The pipeline every loader shares: times in, partitions listed, decimals cast, clock put on, engine chosen.

The rules live here and in each loader once, as polars expressions.
`engine="duckdb"` runs DuckDB over their output rather than restating them.
"""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl

from ._errors import Refused

ENGINES = ("polars", "duckdb")
UTC_US = pl.Datetime("us", "UTC")


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


def _day(micros: int) -> date:
    return (datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=micros)).date()


def partitions(dataset: Path, lo: int, hi: int) -> list[Path]:
    """Every segment in the `date=` directories `[lo, hi)` touches.

    `date` is the UTC day of the row's venue time (0 of 342,768 candles
    disagreed on 2026-09-25), so a window's rows are all in these days.
    """
    first, last = _day(lo), _day(hi - 1)
    files: list[Path] = []
    for d in range((last - first).days + 1):
        directory = dataset / f"date={first + timedelta(days=d)}"
        files.extend(sorted(directory.glob("*.parquet")))
    return files


def segments(dataset: Path) -> list[Path]:
    return sorted(dataset.glob("date=*/*.parquet"))


def require_columns(file: Path, columns: set[str]) -> None:
    """Refuse, by column name, a segment written with a schema this loader doesn't read."""
    missing = columns - set(pl.read_parquet_schema(file))
    if missing:
        raise Refused(f"{file} lacks {', '.join(sorted(missing))}: the tape's schema is not the one this loader reads")


def scan(files: list[Path], columns: list[str]) -> pl.LazyFrame:
    # Paths are listed here, so the partition columns are not read from them.
    return pl.scan_parquet(files, hive_partitioning=False).select(columns)


def floats(*columns: str) -> list[pl.Expr]:
    """DECIMAL(38,18) → Float64, once. Research does not need the record's exactness."""
    return [pl.col(c).cast(pl.Float64) for c in columns]


def clock(micros: str, name: str) -> pl.Expr:
    """An int64 microsecond column as a UTC datetime, to the microsecond."""
    return pl.col(micros).cast(pl.Datetime("us")).dt.replace_time_zone("UTC").alias(name)


def finish(lf: pl.LazyFrame, engine: str):
    """The same rows, as a polars LazyFrame or a DuckDB relation."""
    if engine == "polars":
        return lf
    if engine == "duckdb":
        con = duckdb.connect()
        # TIMESTAMPTZ renders in the session's zone; the record's clock is UTC.
        con.execute("SET TimeZone = 'UTC'")
        return con.from_arrow(lf.collect().to_arrow())
    raise Refused(f"engine={engine!r} is not one of {', '.join(ENGINES)}")
