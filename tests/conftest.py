"""A fixture tape written by each test, in the record's own layout and schema.

Every test states its own rows, so what a claim rests on is in the test.
Segments are named as datawatch names them, `s-<first>_<last>.parquet` over
`stream_seq`, under `tape/kind=candles/date=<UTC day of at_micros>/`.
"""

from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from galata_research import market

DECIMAL = pa.decimal128(38, 18)

# The candle columns as the tape wrote them on 2026-09-25.
CANDLES = pa.schema(
    [
        ("venue", pa.string()),
        ("ticker", pa.string()),
        ("at_micros", pa.int64()),
        ("recv_micros", pa.int64()),
        ("stream_seq", pa.uint64()),
        ("interval", pa.string()),
        ("open", DECIMAL),
        ("high", DECIMAL),
        ("low", DECIMAL),
        ("close", DECIMAL),
        ("volume", DECIMAL),
        ("trade_count", pa.uint32()),
        ("is_final", pa.bool_()),
    ]
)


def us(iso: str) -> int:
    """An ISO time, read as UTC when it has no offset, in micros."""
    t = datetime.fromisoformat(iso)
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    delta = t - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


class Tape:
    def __init__(self, root: Path):
        self.root = root
        (root / "tape").mkdir(parents=True, exist_ok=True)
        self.seq = 0
        self.rows: list[dict] = []

    def bar(
        self,
        ticker: str,
        interval: str,
        opens: str,
        received: str,
        *,
        close: str = "100",
        trade_count: int = 10,
        is_final: bool = True,
        venue: str = "hyperliquid",
        at_micros: int | None = None,
        seq: int | None = None,
    ) -> "Tape":
        self.seq += 1
        price = Decimal(close)
        self.rows.append(
            {
                "venue": venue,
                "ticker": ticker,
                "at_micros": us(opens) if at_micros is None else at_micros,
                "recv_micros": us(received),
                "stream_seq": self.seq if seq is None else seq,
                "interval": interval,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": Decimal("1.5"),
                "trade_count": trade_count,
                "is_final": is_final,
                "_day": opens[:10],
            }
        )
        return self

    def write(self, kind: str = "candles", schema: pa.Schema = CANDLES) -> "Tape":
        """One segment per day for the rows added since the last write."""
        by_day = defaultdict(list)
        for row in self.rows:
            by_day[row.pop("_day")].append(row)
        for day, rows in by_day.items():
            seqs = [r["stream_seq"] for r in rows]
            directory = self.root / "tape" / f"kind={kind}" / f"date={day}"
            directory.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pylist(rows, schema=schema)
            pq.write_table(table, directory / f"s-{min(seqs)}_{max(seqs)}.parquet")
        self.rows = []
        return self


@pytest.fixture
def tape(tmp_path, monkeypatch) -> Tape:
    monkeypatch.setenv("GALATA_VAR", str(tmp_path))
    return Tape(tmp_path)


def load(*args, **kwargs) -> pl.DataFrame:
    return market.candles(*args, **kwargs).collect()
