"""A fixture tape written by each test, in the record's own layout and schema.

Every test states its own rows, so what a claim rests on is in the test.
Segments are named as datawatch names them, `s-<first>_<last>.parquet` over
`stream_seq`, under `tape/kind=<dataset>/date=<UTC day of at_micros>/`. One
write of one day is one segment of one row group, so rows written together
share a row group, tickers mixed.
"""

import json
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


_TICK = [
    ("venue", pa.string()),
    ("ticker", pa.string()),
    ("at_micros", pa.int64()),
    ("recv_micros", pa.int64()),
    ("stream_seq", pa.uint64()),
]
TRADES = pa.schema(
    [*_TICK, ("price", DECIMAL), ("size", DECIMAL), ("aggressor", pa.string()), ("trade_id", pa.string())]
)
QUOTES = pa.schema(
    [*_TICK, *[(c, DECIMAL) for c in ["bid_px", "ask_px", "bid_sz", "ask_sz", "bid_spread", "ask_spread"]]]
)
GAPS = pa.schema(
    [*_TICK, ("series", pa.string()), ("from_micros", pa.int64()), ("to_micros", pa.int64()),
     ("cause", pa.string()), ("clipped", pa.string())]
)  # fmt: skip
SCHEMAS = {"candles": CANDLES, "trades": TRADES, "quotes": QUOTES, "gaps": GAPS}


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
                "_kind": "candles",
            }
        )
        return self

    def _tick(self, kind, ticker, at, received, venue, seq, at_micros, fields) -> "Tape":
        self.seq += 1
        self.rows.append(
            {
                "venue": venue,
                "ticker": ticker,
                "at_micros": us(at) if at_micros is None else at_micros,
                "recv_micros": us(received),
                "stream_seq": self.seq if seq is None else seq,
                **fields,
                "_day": at[:10],
                "_kind": kind,
            }
        )
        return self

    def trade(
        self, ticker, at, received, trade_id, *, price="100", size="1", aggressor="bid",
        venue="hyperliquid", seq=None, at_micros=None,
    ) -> "Tape":  # fmt: skip
        fields = {"price": Decimal(price), "size": Decimal(size), "aggressor": aggressor, "trade_id": trade_id}
        return self._tick("trades", ticker, at, received, venue, seq, at_micros, fields)

    def quote(
        self, ticker, at, received, *, bid="99", ask="101", bid_sz="1", ask_sz="2",
        venue="hyperliquid", seq=None, at_micros=None,
    ) -> "Tape":  # fmt: skip
        fields = {
            "bid_px": Decimal(bid), "ask_px": Decimal(ask), "bid_sz": Decimal(bid_sz), "ask_sz": Decimal(ask_sz),
            "bid_spread": None, "ask_spread": None,
        }  # fmt: skip
        return self._tick("quotes", ticker, at, received, venue, seq, at_micros, fields)

    def gap(self, ticker, series, since, until, *, cause="downtime", venue="hyperliquid", seq=None) -> "Tape":
        """A gap as datawatch publishes one: at = from, received at the restart."""
        fields = {"series": series, "from_micros": us(since), "to_micros": us(until), "cause": cause, "clipped": "continuous"}
        self._tick("gaps", ticker, since, until, venue, seq, None, fields)
        return self

    def write(self, kind: str | None = None, schema: pa.Schema | None = None) -> "Tape":
        """One segment per dataset and day for the rows added since the last write.

        `kind` files every pending row under that dataset instead of its own.
        """
        grouped = defaultdict(list)
        for row in self.rows:
            own = row.pop("_kind")
            grouped[(kind or own, own, row.pop("_day"))].append(row)
        for (target, own, day), rows in grouped.items():
            seqs = [r["stream_seq"] for r in rows]
            directory = self.root / "tape" / f"kind={target}" / f"date={day}"
            directory.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pylist(rows, schema=schema or SCHEMAS[own])
            pq.write_table(table, directory / f"s-{min(seqs)}_{max(seqs)}.parquet")
        self.rows = []
        return self


@pytest.fixture
def tape(tmp_path, monkeypatch) -> Tape:
    monkeypatch.setenv("GALATA_VAR", str(tmp_path))
    return Tape(tmp_path)


def load(*args, **kwargs) -> pl.DataFrame:
    return market.candles(*args, **kwargs).collect()


LEDGER = pa.schema(
    [
        ("seq", pa.uint64()),
        ("recv_micros", pa.int64()),
        ("venue", pa.string()),
        ("channel", pa.string()),
        ("symbol", pa.string()),
        ("origin", pa.string()),
        ("payload", pa.binary()),
        ("schema_version", pa.uint16()),
    ]
)


def position(coin="BTC", szi="-0.83", value="53120.0", liquidation="71000", **overrides) -> dict:
    """A position in legacy's fixture shape (galata-legacy wire.rs:614-617)."""
    p = {
        "coin": coin, "szi": szi, "entryPx": "60000", "positionValue": value, "unrealizedPnl": "12.5",
        "liquidationPx": liquidation, "marginUsed": "200", "leverage": {"type": "cross", "value": 20},
        "maxLeverage": 50, "returnOnEquity": "0.01",
        "cumFunding": {"allTime": "3.21", "sinceOpen": "1.5", "sinceChange": "1.5"},
    }  # fmt: skip
    p.update(overrides)
    return {"type": "oneWay", "position": p}


def snapshot(dex="", time_ms=1790347182376, positions=(), mode="unifiedAccount", value="0.0") -> dict:
    summary = {"accountValue": value, "totalNtlPos": "0.0", "totalRawUsd": "0.0", "totalMarginUsed": "0.0"}
    return {
        "dex": dex,
        "mode": {"answer": mode, "recv_micros": 1790347182465855},
        "clearinghouseState": {
            "marginSummary": summary, "crossMarginSummary": dict(summary), "crossMaintenanceMarginUsed": "0.0",
            "withdrawable": "0.0", "assetPositions": list(positions), "time": time_ms,
        },
    }  # fmt: skip


def ledger(root: Path, rows: list[dict], *, account="main", venue="hyperliquid", day="2026-09-25") -> None:
    """Ledger rows as datawatch writes them; the alias is a directory level only."""
    directory = root / "ledger" / f"venue={venue}" / f"account={account}" / "kind=margin" / f"date={day}"
    directory.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [
            {
                "seq": n + 1, "recv_micros": r.get("recv", 1790347182465855), "venue": venue,
                "channel": r.get("channel", "clearinghouseState"), "symbol": None, "origin": "fetched",
                "payload": json.dumps(r["payload"]).encode(), "schema_version": r.get("schema_version", 1),
            }  # fmt: skip
            for n, r in enumerate(rows)
        ],
        schema=LEDGER,
    )
    pq.write_table(table, directory / f"t-{len(list(directory.iterdir()))}.parquet")
