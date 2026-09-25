"""Market data from the tape, one function per dataset, each with its rules."""

from collections.abc import Sequence
from datetime import datetime

import polars as pl

from . import _root, _scan
from ._errors import Refused
from .gaps import gaps

INTERVALS = {
    "1m": 60_000_000,
    "1h": 3_600_000_000,
    "4h": 14_400_000_000,
    "1d": 86_400_000_000,
}

_SERIES = ["venue", "ticker", "interval"]
_BAR = [*_SERIES, "at_micros"]
_PRICES = ["open", "high", "low", "close", "volume"]
_READ = [*_BAR, "recv_micros", "stream_seq", *_PRICES, "trade_count"]

CANDLE_SCHEMA = {
    "venue": pl.String,
    "ticker": pl.String,
    "interval": pl.String,
    "ts": _scan.UTC_US,
    "close_ts": _scan.UTC_US,
    **{c: pl.Float64 for c in _PRICES},
    "trade_count": pl.UInt32,
    "recv_ts": _scan.UTC_US,
}


def candles(
    tickers: Sequence[str] | str | None,
    interval: str,
    start: datetime | str,
    end: datetime | str,
    *,
    as_of: datetime | str | None = None,
    traded_only: bool = True,
    closed_only: bool = True,
    engine: str = "polars",
):
    """OHLCV bars with `ts` in `[start, end)`, one row per closed bar.

    - **One row per bar**, the latest receipt: the tape holds a re-fetched bar
      about three times (90,027 rows for 30,015 1h bars, 2026-09-25).
    - **Closed by the record**: a later bar exists in the series, or one of the
      bar's own receipts arrived at or after its close. `is_final` is not read;
      the walk stamps a still-forming bar final.
    - **Known at its close**: `close_ts = ts + interval`, and `as_of` keeps
      `close_ts <= as_of`. At 1m the kept value is the latest receipt, which
      can arrive up to ~1.8 s after `close_ts`.
    - **Trade-less bars dropped** by default: BTC and ETH daily bars before
      2023-02-26 have `trade_count = 0`.

    `closed_only=False` returns the open bar too, with a `closed` column.
    """
    if interval not in INTERVALS:
        raise Refused(f"interval={interval!r} is not one of {', '.join(INTERVALS)}")
    width = INTERVALS[interval]
    lo, hi = _scan.window(start, end)
    bound = None if as_of is None else _scan.instant("as_of", as_of)
    if engine not in _scan.ENGINES:
        raise Refused(f"engine={engine!r} is not one of {', '.join(_scan.ENGINES)}")

    dataset = _root.root() / "tape" / "kind=candles"
    every = _scan.segments(dataset)
    if not every:
        raise Refused(f"the record has no candles under {dataset}")
    _scan.require_columns(every[-1], set(_READ))

    # Each series' latest bar across the whole record: it validates the
    # tickers, and it is how a bar is known to be followed by another,
    # wherever in the record that later bar is.
    latest = (
        _scan.scan(every, _BAR)
        .filter(pl.col("interval") == interval)
        .group_by(_SERIES)
        .agg(pl.col("at_micros").max().alias("latest_at"))
        .collect()
    )
    wanted = _wanted(tickers, sorted(latest["ticker"].unique()), f"{interval} candles")

    columns = dict(CANDLE_SCHEMA)
    if not closed_only:
        columns["closed"] = pl.Boolean
    files = _scan.partitions(dataset, lo, hi)
    if not files:
        return _scan.finish(pl.LazyFrame(schema=columns), engine)

    bars = (
        _scan.scan(files, _READ)
        .filter(
            (pl.col("interval") == interval)
            & pl.col("ticker").is_in(wanted)
            & (pl.col("at_micros") >= lo)
            & (pl.col("at_micros") < hi)
        )
        # The latest receipt is the venue's last word on a bar; stream_seq
        # makes the choice total when two receipts share a microsecond.
        .sort(["recv_micros", "stream_seq"])
        .group_by(_BAR)
        .agg(pl.all().last())
        .join(latest.lazy(), on=_SERIES, how="left")
        .with_columns(
            (
                (pl.col("at_micros") < pl.col("latest_at"))
                | (pl.col("recv_micros") >= pl.col("at_micros") + width)
            ).alias("closed")
        )
    )
    if closed_only:
        bars = bars.filter(pl.col("closed"))
    if traded_only:
        bars = bars.filter(pl.col("trade_count") > 0)
    if bound is not None:
        bars = bars.filter(pl.col("at_micros") + width <= bound)

    out = (
        bars.with_columns(
            _scan.clock("at_micros", "ts"),
            _scan.clock("at_micros", "close_ts").dt.offset_by(f"{width}us"),
            _scan.clock("recv_micros", "recv_ts"),
            *_scan.floats(*_PRICES),
            pl.col("trade_count").cast(pl.UInt32),
        )
        .select(list(columns))
        .sort(["venue", "ticker", "ts"])
    )
    return _scan.finish(out, engine)


_TRADE_FLOATS = ["price", "size"]
TRADE_SCHEMA = {
    "venue": pl.String,
    "ticker": pl.String,
    "ts": _scan.UTC_US,
    "price": pl.Float64,
    "size": pl.Float64,
    "aggressor": pl.String,
    "trade_id": pl.String,
    "recv_ts": _scan.UTC_US,
}

_QUOTE_FLOATS = ["bid_px", "ask_px", "bid_sz", "ask_sz", "bid_spread", "ask_spread"]
QUOTE_SCHEMA = {
    "venue": pl.String,
    "ticker": pl.String,
    "ts": _scan.UTC_US,
    **{c: pl.Float64 for c in _QUOTE_FLOATS},
    "recv_ts": _scan.UTC_US,
}


def trades(
    tickers: Sequence[str] | str | None,
    start: datetime | str,
    end: datetime | str,
    *,
    as_of: datetime | str | None = None,
    engine: str = "polars",
):
    """Executions with `ts` in `[start, end)`, one row per execution.

    - **Counted once**: a venue that replays recent history on reconnect sends
      an execution again (2.75% of rows, re-sent 17 s to 708 s later,
      2026-09-25). One row per `(venue, ticker, trade_id)`, the first receipt.
    - **In a total order**: up to 983 trades share one millisecond and one
      message (one `stream_seq`), and `trade_id` is not ordered, so ties keep
      the venue's order: the message's, then the row's within it.
    - **Exact to the venue**: Hyperliquid stamps trades to the millisecond.

    `engine="duckdb"` materialises the result; narrow the window for months.
    """
    lf = _ticks("trades", ["trade_id", "aggressor", *_TRADE_FLOATS], tickers, start, end, as_of, engine)
    if lf is None:
        return _scan.finish(pl.LazyFrame(schema=TRADE_SCHEMA), engine)
    out = (
        # A replay is the same execution; its first receipt is when it was known.
        lf.sort(["recv_micros", "stream_seq", "_row"])
        .group_by(["venue", "ticker", "trade_id"])
        .agg(pl.all().first())
        .sort(["venue", "ticker", "at_micros", "stream_seq", "_row"])
    )
    return _scan.finish(_ticks_out(out, _TRADE_FLOATS, TRADE_SCHEMA), engine)


def quotes(
    tickers: Sequence[str] | str | None,
    start: datetime | str,
    end: datetime | str,
    *,
    as_of: datetime | str | None = None,
    engine: str = "polars",
):
    """The book's top with `ts` in `[start, end)`, every row as received.

    No dedupe: none of 1,242,843 quotes repeated a `(ticker, ts)` on
    2026-09-25, and a record test holds that. `bid_spread` and `ask_spread`
    are a broker's stated spread, null where the venue states none
    (Hyperliquid). Exact to the venue's millisecond.

    `engine="duckdb"` materialises the result; a day is ~1.2M rows.
    """
    lf = _ticks("quotes", _QUOTE_FLOATS, tickers, start, end, as_of, engine)
    if lf is None:
        return _scan.finish(pl.LazyFrame(schema=QUOTE_SCHEMA), engine)
    out = lf.sort(["venue", "ticker", "at_micros", "stream_seq", "_row"])
    return _scan.finish(_ticks_out(out, _QUOTE_FLOATS, QUOTE_SCHEMA), engine)


def _ticks(kind, fields, tickers, start, end, as_of, engine) -> pl.LazyFrame | None:
    """The shared half of a tick loader: checks, validation, the window. None when no partition is touched."""
    lo, hi = _scan.window(start, end)
    bound = None if as_of is None else _scan.instant("as_of", as_of)
    if engine not in _scan.ENGINES:
        raise Refused(f"engine={engine!r} is not one of {', '.join(_scan.ENGINES)}")

    dataset = _root.root() / "tape" / f"kind={kind}"
    every = _scan.segments(dataset)
    if not every:
        raise Refused(f"the record has no {kind} under {dataset}")
    read = ["venue", "ticker", "at_micros", "recv_micros", "stream_seq", *fields]
    _scan.require_columns(every[-1], set(read))

    held = sorted(_scan.tickers(every))
    wanted = _wanted(tickers, held, kind)
    files = _scan.partitions(dataset, lo, hi)
    if not files:
        return None
    lf = _scan.scan(files, read, position=True).filter(
        pl.col("ticker").is_in(wanted) & (pl.col("at_micros") >= lo) & (pl.col("at_micros") < hi)
    )
    if bound is not None:
        lf = lf.filter(pl.col("at_micros") <= bound)
    return lf


def _ticks_out(lf: pl.LazyFrame, floats: list[str], schema: dict) -> pl.LazyFrame:
    return lf.with_columns(
        _scan.clock("at_micros", "ts"),
        _scan.clock("recv_micros", "recv_ts"),
        *_scan.floats(*floats),
    ).select(list(schema))


def _wanted(tickers, held: list[str], what: str) -> list[str]:
    if tickers is None:
        return held
    wanted = [tickers] if isinstance(tickers, str) else list(tickers)
    unknown = [t for t in wanted if t not in held]
    if unknown:
        raise Refused(f"the record holds no {what} for {', '.join(unknown)}; it holds {', '.join(held) or 'none'}")
    return wanted
