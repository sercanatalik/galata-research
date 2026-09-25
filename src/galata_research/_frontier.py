"""How far the record goes, without scanning a dataset.

A segment's filename is its stream_seq range, `s-<first>_<last>.parquet`
(measured 2026-09-25), so the durable position is a directory listing.
`max_ts` comes from the footer statistics of the newest day's segments. The
tape keeps statistics for `venue`, `ticker` and `at_micros` only, so
`max_recv_ts` reads one column of that one day.
"""

import polars as pl
import pyarrow.parquet as pq

from . import _root, _scan

# The datasets this library loads. Each later loader adds its own.
DATASETS = ("candles", "gaps", "quotes", "trades")


def frontier() -> pl.DataFrame:
    """One row per dataset: `last_stream_seq`, `max_ts`, `max_recv_ts`, `days`."""
    tape = _root.root() / "tape"
    rows = []
    for dataset in DATASETS:
        files = _scan.segments(tape / f"kind={dataset}")
        if not files:
            continue
        days = sorted({f.parent.name for f in files})
        newest = [f for f in files if f.parent.name == days[-1]]
        rows.append(
            {
                "dataset": dataset,
                "last_stream_seq": max(int(f.stem.rsplit("_", 1)[1]) for f in files),
                "max_ts": _footer_max(newest, "at_micros"),
                "max_recv_ts": _column_max(newest, "recv_micros"),
                "days": len(days),
            }
        )
    schema = {
        "dataset": pl.String,
        "last_stream_seq": pl.UInt64,
        "max_ts": pl.Int64,
        "max_recv_ts": pl.Int64,
        "days": pl.UInt32,
    }
    return pl.DataFrame(rows, schema=schema).with_columns(
        _scan.clock("max_ts", "max_ts"), _scan.clock("max_recv_ts", "max_recv_ts")
    )


def _footer_max(files, column: str) -> int | None:
    best = None
    for f in files:
        meta = pq.ParquetFile(f).metadata
        index = meta.schema.to_arrow_schema().get_field_index(column)
        for g in range(meta.num_row_groups):
            stats = meta.row_group(g).column(index).statistics
            if stats is not None and stats.has_min_max:
                best = stats.max if best is None else max(best, stats.max)
    return best


def _column_max(files, column: str) -> int | None:
    return pl.scan_parquet(files, hive_partitioning=False).select(pl.col(column).max()).collect().item()
