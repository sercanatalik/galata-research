"""Each archive's layout and its parse into the store's columns. No network here: bytes in, frame out.

MEASURED 2026-09-27 on the files themselves:

- Binance CSVs carry a header from about 2022; older ones (aggTrades and
  klines of 2020-01-01) have none, so the header is detected, not assumed.
- Binance `bookDepth.timestamp` is `YYYY-MM-DD HH:MM:SS` with no zone, and
  writes `-5` as well as `-5.00`. The file for a day holds that UTC day.
- Binance `transact_time` and kline times are epoch milliseconds.
- Bybit trade `timestamp` is epoch seconds as text with up to 4 decimals
  (`1790380800.0232`); 2020 files run newest first, 2026 files oldest first;
  `RPI` appears only in later files.
- Bybit's book is JSON lines, `snapshot` then `delta`, prices and sizes as
  text, size "0" removing a level. HYPE's first day (2024-12-04) holds only
  two empty snapshots.
- OKX's trade file for D is zipped CSV `instrument_name, trade_id, side,
  price, size, created_time[, source]`, epoch milliseconds, D-1 16:00 to
  D 16:00 UTC (cut by `trade_id`, so up to 15 s either side in early 2023),
  one row per contiguous `trade_id` from 2021-11-01. `side` is
  the taker's (`buy`, or `BUY` in older files); `size` is contracts.
  `source` (0 normal, 1 Enhanced Liquidity Program) appears only in later files.
"""

import gzip
import io
import json
import zipfile
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta

import polars as pl

from .._errors import Refused
from ._instruments import DAY_ENDS_EARLY, OB200_FROM, instrument

UTC_US = pl.Datetime("us", "UTC")
_MINUTE_US = 60_000_000
_DAY_US = 86_400_000_000
_OKX_SPILL_US = 60_000_000
# A time outside these years is a misread unit, not a trade: Binance moved
# its spot archive from milliseconds to microseconds in 2025.
_YEARS = (2017, 2100)

TRADES = {
    "venue": pl.String, "ticker": pl.String, "symbol": pl.String, "ts": UTC_US,
    "price": pl.Float64, "size": pl.Float64, "aggressor": pl.String, "trade_id": pl.String,
    "rpi": pl.Boolean,
}  # fmt: skip
DEPTH = {
    "venue": pl.String, "ticker": pl.String, "symbol": pl.String, "ts": UTC_US,
    "band_pct": pl.Float64, "depth": pl.Float64, "notional": pl.Float64,
}  # fmt: skip
BOOK_FLOATS = [
    "bid_px", "ask_px", "bid_sz", "ask_sz",
    "bid_depth_2bps", "ask_depth_2bps", "bid_depth_10bps", "ask_depth_10bps",
    "bid_reach_bps", "ask_reach_bps",
]  # fmt: skip
BOOK = {
    "venue": pl.String,
    "ticker": pl.String,
    "symbol": pl.String,
    "ts": UTC_US,
    **{c: pl.Float64 for c in BOOK_FLOATS},
}
CANDLES = {
    "venue": pl.String, "ticker": pl.String, "symbol": pl.String, "interval": pl.String,
    "ts": UTC_US, "close_ts": UTC_US,
    **{c: pl.Float64 for c in ["open", "high", "low", "close", "volume"]},
    "trade_count": pl.UInt32,
}  # fmt: skip
SCHEMAS = {"trades": TRADES, "depth": DEPTH, "book": BOOK, "candles": CANDLES}

_BINANCE = "https://data.binance.vision/data/futures/um/daily"
_BINANCE_KIND = {"depth": "bookDepth", "trades": "aggTrades", "candles": "klines"}
_BINANCE_COLUMNS = {
    "bookDepth": ["timestamp", "percentage", "depth", "notional"],
    "aggTrades": ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"],
    "klines": ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
               "taker_buy_volume", "taker_buy_quote_volume", "ignore"],
}  # fmt: skip


def url(kind: str, venue: str, ticker: str, day: date) -> str:
    """Where the archive keeps one day of a kind."""
    symbol = instrument(venue, ticker).symbol
    d = day.isoformat()
    if venue == "binance-um":
        name = _BINANCE_KIND[kind]
        if kind == "candles":
            return f"{_BINANCE}/klines/{symbol}/1m/{symbol}-1m-{d}.zip"
        return f"{_BINANCE}/{name}/{symbol}/{symbol}-{name}-{d}.zip"
    if venue == "okx-swap":
        return f"https://static.okx.com/cdn/okex/traderecords/trades/daily/{day:%Y%m%d}/{symbol}-trades-{d}.zip"
    if kind == "trades":
        return f"https://public.bybit.com/trading/{symbol}/{symbol}{d}.csv.gz"
    depth = 200 if day >= OB200_FROM else 500
    return f"https://quote-saver.bycsi.com/orderbook/linear/{symbol}/{d}_{symbol}_ob{depth}.data.zip"


def checksum_url(venue: str, archive: str) -> str | None:
    """Binance publishes a sha256 beside each file; Bybit publishes none."""
    return f"{archive}.CHECKSUM" if venue == "binance-um" else None


def parse(kind: str, venue: str, ticker: str, day: date, blob: bytes, name: str) -> pl.DataFrame:
    """One archived day in the store's columns, sorted by `ts` in the venue's order."""
    symbol = instrument(venue, ticker).symbol
    if venue == "binance-um":
        frame = _binance(kind, _unzip(blob, name), name, day)
    elif venue == "okx-swap":
        frame = _okx_trades(_unzip(blob, name), name, day, instrument(venue, ticker).contract)
    elif kind == "trades":
        frame = _bybit_trades(gzip.decompress(blob), name)
    else:
        # Streamed from the zip: an ob500 day unzips past a gigabyte (BTC 2024-06-05 is 167 MB zipped).
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            members = z.namelist()
            if len(members) != 1:
                raise Refused(f"{name} holds {len(members)} files, not one")
            with z.open(members[0]) as lines:
                frame = book(lines, day)
    return frame.with_columns(
        pl.lit(venue).alias("venue"),
        pl.lit(ticker).alias("ticker"),
        pl.lit(symbol).alias("symbol"),
    ).select(list(SCHEMAS[kind]))


def _unzip(blob: bytes, name: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        members = z.namelist()
        if len(members) != 1:
            raise Refused(f"{name} holds {len(members)} files, not one")
        return z.read(members[0])


def _csv(text: bytes, columns: list[str]) -> pl.DataFrame:
    has_header = not text[:1].isdigit() and text[:1] != b"-"
    frame = pl.read_csv(
        text,
        has_header=has_header,
        new_columns=columns,
        infer_schema=False,
        truncate_ragged_lines=True,
    )
    return frame.select(columns)


def _epoch(column: str, unit_us: int) -> pl.Expr:
    return (pl.col(column).cast(pl.Int64) * unit_us).alias(column)


def _check_years(frame: pl.DataFrame, column: str, name: str) -> None:
    if frame.is_empty():
        return
    lo, hi = frame[column].min(), frame[column].max()
    for micros in (lo, hi):
        year = 1970 + micros / (365.2425 * _DAY_US)
        if not _YEARS[0] <= year < _YEARS[1]:
            raise Refused(
                f"{name}: {column} implies the year {year:.0f}; the file is not in the unit this parser reads"
            )


def _binance(kind: str, text: bytes, name: str, day: date) -> pl.DataFrame:
    archive = _BINANCE_KIND[kind]
    raw = _csv(text, _BINANCE_COLUMNS[archive])
    if kind == "depth":
        frame = raw.select(
            pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="us", time_zone="UTC").alias("ts"),
            pl.col("percentage").cast(pl.Float64).alias("band_pct"),
            pl.col("depth").cast(pl.Float64),
            pl.col("notional").cast(pl.Float64),
        )
        lo = datetime(day.year, day.month, day.day, tzinfo=UTC)
        outside = frame.filter((pl.col("ts") < lo) | (pl.col("ts") >= lo + timedelta(days=1)))
        if outside.height:
            raise Refused(f"{name}: {outside.height} row(s) fall outside {day} UTC, first at {outside['ts'][0]}")
        return frame.sort("ts", maintain_order=True)
    if kind == "trades":
        frame = raw.with_columns(_epoch("transact_time", 1000))
        _check_years(frame, "transact_time", name)
        return frame.select(
            pl.col("transact_time").cast(UTC_US).alias("ts"),
            pl.col("price").cast(pl.Float64),
            pl.col("quantity").cast(pl.Float64).alias("size"),
            # The buyer made the market, so the seller crossed: the tape's `ask`.
            pl.when(pl.col("is_buyer_maker").str.to_lowercase() == "true")
            .then(pl.lit("ask"))
            .otherwise(pl.lit("bid"))
            .alias("aggressor"),
            pl.col("agg_trade_id").alias("trade_id"),
            pl.lit(None, pl.Boolean).alias("rpi"),
        ).sort("ts", maintain_order=True)
    frame = raw.with_columns(_epoch("open_time", 1000))
    _check_years(frame, "open_time", name)
    return frame.select(
        pl.lit("1m").alias("interval"),
        pl.col("open_time").cast(UTC_US).alias("ts"),
        (pl.col("open_time") + _MINUTE_US).cast(UTC_US).alias("close_ts"),
        *[pl.col(c).cast(pl.Float64) for c in ["open", "high", "low", "close", "volume"]],
        pl.col("count").cast(pl.UInt32).alias("trade_count"),
    ).sort("ts", maintain_order=True)


def _bybit_trades(text: bytes, name: str) -> pl.DataFrame:
    raw = pl.read_csv(text, infer_schema=False)
    if "RPI" not in raw.columns:
        raw = raw.with_columns(pl.lit(None, pl.String).alias("RPI"))
    # Seconds as text, into integer micros without passing through a float:
    # 1790380800.0232 is 1790380800023200 exactly.
    parts = pl.col("timestamp").str.split_exact(".", 1)
    fraction = parts.struct.field("field_1").fill_null("").str.pad_end(6, "0").str.slice(0, 6)
    micros = parts.struct.field("field_0").cast(pl.Int64) * 1_000_000 + fraction.cast(pl.Int64)
    frame = raw.select(
        micros.alias("micros"),
        pl.col("price").cast(pl.Float64),
        pl.col("size").cast(pl.Float64),
        # The taker's side, in the tape's words: a buy crossing is `bid`.
        pl.col("side").replace_strict({"Buy": "bid", "Sell": "ask"}, default=None).alias("aggressor"),
        pl.col("side").alias("_side"),
        pl.col("trdMatchID").alias("trade_id"),
        (pl.col("RPI") == "1").alias("rpi"),
    )
    _check_years(frame, "micros", name)
    bad = frame.filter(pl.col("aggressor").is_null())
    if bad.height:
        raise Refused(f"{name}: side {bad['_side'][0]!r} is neither Buy nor Sell")
    frame = frame.drop("_side")
    # Older files run newest first; the venue's order within a timestamp is
    # the file's order read forwards in time.
    if frame.height > 1 and frame["micros"][0] > frame["micros"][-1]:
        frame = frame.reverse()
    return (
        frame.sort("micros", maintain_order=True).with_columns(pl.col("micros").cast(UTC_US).alias("ts")).drop("micros")
    )


def _okx_trades(text: bytes, name: str, day: date, contract: float) -> pl.DataFrame:
    raw = pl.read_csv(text, infer_schema=False)
    if "source" not in raw.columns:
        raw = raw.with_columns(pl.lit(None, pl.String).alias("source"))
    raw = raw.with_columns(pl.col(c).str.strip_chars() for c in raw.columns)
    repeated = raw.filter(pl.col("trade_id").is_duplicated())
    if repeated.height:
        raise Refused(
            f"{name}: {repeated['trade_id'].n_unique():,} trade_id(s) repeat, the first {repeated['trade_id'][0]}; "
            "a file that lists each trade on both sides does not say who crossed"
        )
    frame = raw.select(
        (pl.col("created_time").cast(pl.Int64) * 1000).alias("micros"),
        pl.col("price").cast(pl.Float64),
        (pl.col("size").cast(pl.Float64) * contract).alias("size"),
        # The taker's side, in the tape's words: a buy crossing is `bid`.
        pl.col("side").str.to_lowercase().replace_strict({"buy": "bid", "sell": "ask"}, default=None).alias("aggressor"),
        pl.col("side").alias("_side"),
        pl.col("trade_id"),
        # An ELP fill, like Bybit's RPI, took liquidity only retail takers could reach.
        pl.col("source").cast(pl.Int8).eq(1).alias("rpi"),
    )
    _check_years(frame, "micros", name)
    bad = frame.filter(pl.col("aggressor").is_null())
    if bad.height:
        raise Refused(f"{name}: side {bad['_side'][0]!r} is neither buy nor sell")
    # The file for D ends DAY_ENDS_EARLY before the end of UTC day D.
    midnight = datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(days=1) - DAY_ENDS_EARLY["okx-swap"]
    end = int(midnight.timestamp()) * 1_000_000
    # Files are cut by trade_id, not by time: in early 2023 a file begins up to
    # 15 s before its 16:00 (BTC 2023-01-20: 45 rows, ids contiguous with the
    # file before, none repeated). A minute's spill is the cut; more is not.
    outside = frame.filter((pl.col("micros") < end - _DAY_US - _OKX_SPILL_US) | (pl.col("micros") >= end + _OKX_SPILL_US))
    if outside.height:
        raise Refused(f"{name}: {outside.height} row(s) fall more than a minute outside the archive's day, 16:00 to 16:00 UTC")
    return (
        frame.drop("_side")
        .sort("micros", maintain_order=True)
        .with_columns(pl.col("micros").cast(UTC_US).alias("ts"))
        .drop("micros")
    )


def book(lines: Iterable[bytes | str], day: date) -> pl.DataFrame:
    """Bybit's book replayed, one row per whole second: the state after every message at or before it.

    - A `snapshot` replaces both sides; a `delta` sets levels; size "0" removes one.
    - No row before the day's first snapshot, and none while either side is empty.
    - Depth bands sum the size within 2 and 10 bps of the mid. A band past the
      deepest level held on its side is null, never a truncated sum: ob200
      reaches 4.1 bps (bid) and 3.7 bps (ask) on BTC (2026-09-26).
    - Rows run to the day's last second; the next day's file starts its own book.
    """
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    rows: dict[str, list] = {c: [] for c in ["second", *BOOK_FLOATS]}
    start = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())
    end = start + 86_400
    state: tuple | None = None
    changed = False
    next_second: int | None = None

    def emit(second: int) -> None:
        nonlocal state, changed
        if changed:
            state = _top(bids, asks)
            changed = False
        if state is None:
            return
        rows["second"].append(second)
        for c, v in zip(BOOK_FLOATS, state, strict=True):
            rows[c].append(v)

    for line in lines:
        if not line.strip():
            continue
        message = json.loads(line)
        ms = int(message["ts"])
        if next_second is not None:
            while next_second < end and next_second * 1000 < ms:
                emit(next_second)
                next_second += 1
        data = message["data"]
        if message["type"] == "snapshot":
            bids.clear()
            asks.clear()
            if next_second is None:
                next_second = max(start, -(-ms // 1000))
        elif next_second is None:
            continue  # a delta before any snapshot changes a book not yet known
        for side, levels in ((bids, data.get("b", ())), (asks, data.get("a", ()))):
            for px, sz in levels:
                price, size = float(px), float(sz)
                if size == 0.0:
                    side.pop(price, None)
                else:
                    side[price] = size
        changed = True
    if next_second is not None:
        while next_second < end:
            emit(next_second)
            next_second += 1

    frame = pl.DataFrame(rows, schema={"second": pl.Int64, **{c: pl.Float64 for c in BOOK_FLOATS}})
    return frame.with_columns((pl.col("second") * 1_000_000).cast(UTC_US).alias("ts")).drop("second")


def _top(bids: dict[float, float], asks: dict[float, float]) -> tuple | None:
    if not bids or not asks:
        return None
    best_bid, best_ask = max(bids), min(asks)
    mid = (best_bid + best_ask) / 2
    bid_reach = (mid - min(bids)) / mid * 1e4
    ask_reach = (max(asks) - mid) / mid * 1e4

    def band(side: dict[float, float], bps: float, reach: float, below: bool) -> float | None:
        if bps > reach:
            return None
        edge = mid * (1 - bps / 1e4) if below else mid * (1 + bps / 1e4)
        return sum(s for p, s in side.items() if (p >= edge if below else p <= edge))

    return (
        best_bid, best_ask, bids[best_bid], asks[best_ask],
        band(bids, 2, bid_reach, True), band(asks, 2, ask_reach, False),
        band(bids, 10, bid_reach, True), band(asks, 10, ask_reach, False),
        bid_reach, ask_reach,
    )  # fmt: skip
