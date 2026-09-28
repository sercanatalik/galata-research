"""The store's truth: one row per (kind, venue, ticker, date), saying where it came from.

A day is present only when its row is `ok` and its Parquet file exists. The
raw archive is not kept (one BTC book day is 638 MB unzipped, 2026-09-26);
`url` and `sha256` make it refetchable and prove a refetch is the same file.
"""

import fcntl
import os
from datetime import date
from pathlib import Path

import polars as pl

NAME = "manifest.parquet"
KEY = ["kind", "venue", "ticker", "date"]
STATUSES = ("ok", "absent", "mismatch")
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


def read(root: Path) -> pl.DataFrame:
    path = root / NAME
    if not path.is_file():
        return pl.DataFrame(schema=SCHEMA)
    # One open, then parse: a fetch may replace the file between the reads a
    # path-based parse makes, and the halves of two files do not parse.
    return pl.read_parquet(path.read_bytes()).select(list(SCHEMA)).cast(SCHEMA)


def upsert(root: Path, rows: list[dict]) -> pl.DataFrame:
    """Replace or add these days' rows, written whole to a temporary file then renamed.

    Read and write happen under an exclusive lock on the store, so two fetches
    running at once (depth in one terminal, candles in another) cannot each
    rewrite the manifest from a copy that lacks the other's rows.
    """
    new = pl.DataFrame(rows, schema=SCHEMA)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".manifest.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        out = pl.concat([read(root).join(new.select(KEY), on=KEY, how="anti"), new]).sort(KEY)
        tmp = root / f".{NAME}.{os.getpid()}.tmp"
        out.write_parquet(tmp)
        os.replace(tmp, root / NAME)
    return out


def day_path(root: Path, kind: str, venue: str, ticker: str, day: date) -> Path:
    interval = "interval=1m/" if kind == "candles" else ""
    return root / f"kind={kind}" / f"venue={venue}" / f"ticker={ticker}" / f"{interval}date={day.isoformat()}.parquet"
