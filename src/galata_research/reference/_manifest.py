"""The store's truth: one row per (kind, venue, ticker, date), saying where it came from.

A day is present only when its row is `ok` (or `partial`: some of its hours
are gaps, stated under `kind=gaps`) and its Parquet file exists. The raw
archive is not kept (one BTC book day is 638 MB unzipped, 2026-09-26);
`url` and `sha256` make it refetchable and prove a refetch is the same file.

A day built from many files (hyperliquid-s3: 25 hour files, every ticker at
once) also has one row per file in the object ledger, `objects.parquet`:
its key, ETag, size and sha256, so a refetch names the hour that changed.
"""

import fcntl
import os
from datetime import date
from pathlib import Path

import polars as pl

NAME = "manifest.parquet"
KEY = ["kind", "venue", "ticker", "date"]
STATUSES = ("ok", "partial", "absent", "mismatch")
# The statuses whose day has bars to read.
HELD = ("ok", "partial")
SCHEMA = {
    "kind": pl.String,
    "venue": pl.String,
    "ticker": pl.String,
    "date": pl.Date,
    "url": pl.String,
    "bytes": pl.Int64,
    "sha256": pl.String,
    "published_sha256": pl.String,
    "fetched_at_recv": pl.Datetime("us", "UTC"),
    "rows": pl.Int64,
    "status": pl.String,
}


OBJECTS = "objects.parquet"
OBJECT_KEY = ["bucket", "key"]
OBJECT_SCHEMA = {
    "bucket": pl.String,
    "key": pl.String,
    "etag": pl.String,
    "bytes": pl.Int64,
    "sha256": pl.String,
    "fetched_at_recv": pl.Datetime("us", "UTC"),
    "status": pl.String,
}


def read(root: Path) -> pl.DataFrame:
    return _read(root / NAME, SCHEMA)


def read_objects(root: Path) -> pl.DataFrame:
    return _read(root / OBJECTS, OBJECT_SCHEMA)


def _read(path: Path, schema: dict) -> pl.DataFrame:
    if not path.is_file():
        return pl.DataFrame(schema=schema)
    # One open, then parse: a fetch may replace the file between the reads a
    # path-based parse makes, and the halves of two files do not parse.
    return pl.read_parquet(path.read_bytes()).select(list(schema)).cast(schema)


def upsert(root: Path, rows: list[dict]) -> pl.DataFrame:
    """Replace or add these days' rows, written whole to a temporary file then renamed.

    Read and write happen under an exclusive lock on the store, so two fetches
    running at once (depth in one terminal, candles in another) cannot each
    rewrite the manifest from a copy that lacks the other's rows.
    """
    return _upsert(root, NAME, SCHEMA, KEY, rows)


def upsert_objects(root: Path, rows: list[dict]) -> pl.DataFrame:
    """Replace or add these files' rows in the object ledger, under the same lock as the manifest."""
    return _upsert(root, OBJECTS, OBJECT_SCHEMA, OBJECT_KEY, rows)


def _upsert(root: Path, name: str, schema: dict, key: list[str], rows: list[dict]) -> pl.DataFrame:
    new = pl.DataFrame(rows, schema=schema)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".manifest.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        out = pl.concat([_read(root / name, schema).join(new.select(key), on=key, how="anti"), new]).sort(key)
        tmp = root / f".{name}.{os.getpid()}.tmp"
        out.write_parquet(tmp)
        os.replace(tmp, root / name)
    return out


def gaps_path(root: Path, venue: str, day: date) -> Path:
    """A day's gaps for a venue that states them: every ticker's, in one file."""
    return root / "kind=gaps" / f"venue={venue}" / f"date={day.isoformat()}.parquet"


def day_path(root: Path, kind: str, venue: str, ticker: str, day: date) -> Path:
    interval = "interval=1m/" if kind == "candles" else ""
    return root / f"kind={kind}" / f"venue={venue}" / f"ticker={ticker}" / f"{interval}date={day.isoformat()}.parquet"
