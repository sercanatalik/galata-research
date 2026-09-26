"""The pipeline every loader shares: times in, partitions listed, decimals cast, clock put on, engine chosen.

The rules live here and in each loader once, as polars expressions.
`engine="duckdb"` runs DuckDB over their output rather than restating them.
"""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl
import pyarrow.parquet as pq

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


def tickers(files: list[Path]) -> set[str]:
    """Every ticker the segments hold, from footers where a row group holds one ticker.

    Most row groups hold one ticker, but not all (9 of 78 quote groups mixed
    them on 2026-09-25), so a mixed group's `ticker` column is read, alone.
    """
    held: set[str] = set()
    for f in files:
        parquet = pq.ParquetFile(f)
        meta = parquet.metadata
        index = meta.schema.to_arrow_schema().get_field_index("ticker")
        for g in range(meta.num_row_groups):
            stats = meta.row_group(g).column(index).statistics
            if stats is not None and stats.has_min_max and stats.min == stats.max:
                held.add(stats.min)
            else:
                column = parquet.read_row_group(g, columns=["ticker"]).column("ticker")
                held.update(v for v in column.unique().to_pylist() if v is not None)
    return held


def wanted(tickers, held: list[str], what: str) -> list[str]:
    """The tickers asked for, or every held one; an unheld one is refused with the list."""
    if tickers is None:
        return held
    asked = [tickers] if isinstance(tickers, str) else list(tickers)
    unknown = [t for t in asked if t not in held]
    if unknown:
        raise Refused(f"the record holds no {what} for {', '.join(unknown)}; it holds {', '.join(held) or 'none'}")
    return asked


def require_columns(file: Path, columns: set[str]) -> None:
    """Refuse, by column name, a segment written with a schema this loader doesn't read."""
    missing = columns - set(pl.read_parquet_schema(file))
    if missing:
        raise Refused(f"{file} lacks {', '.join(sorted(missing))}: the tape's schema is not the one this loader reads")


def scan(files: list[Path], columns: list[str], *, position: bool = False) -> pl.LazyFrame:
    """The listed segments' columns; `position=True` adds `_row`, each row's place in the scan.

    One message can carry many events under one stream_seq (up to 983 trades),
    and their order within it is only the order of the rows in the segment.
    """
    # Paths are listed here, so the partition columns are not read from them.
    #
    # The tape's schemas are additive-only (datawatch `market-vocabulary`): a
    # column is appended, and files written before it lack it until a rebuild.
    # polars refuses a scan whose later file has a column its first file does
    # not (measured, polars 1.44.2), which breaks every window spanning the
    # append. So a column beyond the first file's is ignored, and one a later
    # file lacks reads as null. Selecting a column the FIRST file lacks still
    # fails: the scan takes its schema from that file, so a loader that reads
    # a newly appended column passes the schema explicitly.
    lf = pl.scan_parquet(
        files,
        hive_partitioning=False,
        row_index_name="_row" if position else None,
        missing_columns="insert",
        extra_columns="ignore",
    )
    return lf.select([*(["_row"] if position else []), *columns])


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
