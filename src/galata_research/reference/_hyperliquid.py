"""Hyperliquid's published fills, read into trades and 1m bars. No network here: bytes in, frames out.

The archive is three formats in turn (`_instruments.CUTS`, and the MEASURED
block there): `node_trades`, one trade per line; `node_fills`, one
`[address, fill]` per line; `node_fills_by_block`, one block per line with
its `events`. A trade in the fills formats is two fills under one
`(coin, tid)`, the taker's `crossed`.

- **One row per trade.** A fills trade is its taker fill; a `(coin, tid)`
  without exactly one taker fails the hour, since a taker guessed is a
  side invented.
- **Placed by its own time.** A file holds what the node received in its
  hour, so a trade stamped in an hour's last second can be in the next
  file: an hour is read from its own file and the next.
- **Each hour from one format.** Formats overlap at each handover, and
  `node_trades` has no `tid` to match a trade across them, so an hour of
  trade time is read from the format `CUTS` gives it and from no other.
- **Bars as the venue's.** Volume, trade count, open and close take every
  trade; `high` and `low` leave out liquidation prices, as the venue's
  candles do, and `high_all` and `low_all` keep them. `node_trades`'
  `NetChildVaultPositions` rows are not trades the venue's candles count.
- **An hour not read is a gap**, with its cause, for every ticker: no bar
  is written in it, and a bar is never shortened by one.
"""

import io
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import polars as pl

from .._errors import Refused
from ._instruments import HYPERLIQUID, prefix_at

BUCKET = "hl-mainnet-node-data"
REGION = "ap-northeast-1"
HOST = f"{BUCKET}.s3.{REGION}.amazonaws.com"
PREFIXES = ("node_trades", "node_fills", "node_fills_by_block")
CAUSES = ("archive-missing", "unreadable", "unparsed")

UTC_US = pl.Datetime("us", "UTC")
_HOUR = timedelta(hours=1)
_MINUTE_US = 60_000_000
_SECOND = timedelta(seconds=1)
# node_trades' trade_dir_override: what each says, and whether the venue counts the row as a trade.
_OVERRIDES = {"Na": (False, True), "LiquidatedMarket": (True, True), "NetChildVaultPositions": (False, False)}

TRADES = {"ticker": pl.String, "micros": pl.Int64, "price": pl.Float64, "size": pl.Float64,
          "taker_buy": pl.Boolean, "liquidation": pl.Boolean, "seq": pl.Int64}  # fmt: skip
CANDLES = {
    "venue": pl.String, "ticker": pl.String, "symbol": pl.String, "interval": pl.String,
    "ts": UTC_US, "close_ts": UTC_US,
    **{c: pl.Float64 for c in ["open", "high", "low", "close", "volume"]},
    "trade_count": pl.UInt32,
    "notional": pl.Float64, "taker_buy_volume": pl.Float64, "liquidations": pl.UInt32,
    "high_all": pl.Float64, "low_all": pl.Float64,
}  # fmt: skip
GAPS = {"venue": pl.String, "ticker": pl.String, "series": pl.String, "cause": pl.String,
        "from_ts": UTC_US, "to_ts": UTC_US}  # fmt: skip


class HourFailed(Exception):
    """An hour file that could not be read, with the gap cause it stands for."""

    def __init__(self, cause: str, detail: str):
        super().__init__(f"{cause}: {detail}")
        self.cause = cause


def key(prefix: str, hour: datetime) -> str:
    """The bucket's key for one hour file: the hour is not zero-padded."""
    return f"{prefix}/hourly/{hour:%Y%m%d}/{hour.hour}.lz4"


def directory(prefix: str, day: date) -> str:
    return f"{prefix}/hourly/{day:%Y%m%d}/"


def allowed(path: str) -> bool:
    """Whether a key or listing prefix is one this source may ask for: under a prefix, and climbing out of none."""
    if any(part in ("..", ".") for part in path.split("/")) or any(c in path for c in "?#\\"):
        return False
    return any(path.startswith(f"{p}/hourly/") for p in PREFIXES)


@dataclass(frozen=True)
class Hour:
    """One hour of trade time: the format it is read from, its own file, and the next (its last second's spill)."""

    start: datetime
    prefix: str | None
    own: str | None
    spill: str | None


def hours(day: date) -> list[Hour]:
    """The 24 hours of a UTC day, each with the files that hold it."""
    midnight = datetime(day.year, day.month, day.day, tzinfo=UTC)
    out = []
    for h in range(24):
        start = midnight + h * _HOUR
        prefix = prefix_at(start)
        if prefix is None:
            out.append(Hour(start, None, None, None))
        else:
            out.append(Hour(start, prefix, key(prefix, start), key(prefix, start + _HOUR)))
    return out


def keys(day: date) -> list[str]:
    """Every file a day needs: 25 on an ordinary day, more on a handover day."""
    return sorted({k for h in hours(day) for k in (h.own, h.spill) if k is not None})


def read(blob: bytes, name: str, coins: dict[str, str]) -> pl.DataFrame:
    """One hour file's trades in `coins` (the venue's coin → the store's ticker), one row per trade.

    `seq` is the file's order, which breaks a tie in time. A file that does
    not decompress is `unreadable`; a line of no known shape, a side or
    override of another word, or a fills trade without exactly one taker is
    `unparsed`.
    """
    import lz4.frame

    rows: dict[str, list] = {c: [] for c in TRADES}
    fills: dict[tuple[str, int], list[dict]] = {}
    try:
        with lz4.frame.open(io.BytesIO(blob), "rb") as lines:
            for n, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError as e:
                    raise HourFailed("unparsed", f"{name} line {n + 1} is not JSON ({e})") from None
                if isinstance(record, dict) and "events" in record:
                    for _, fill in record["events"]:
                        _keep_fill(fill, coins, fills, n)
                elif isinstance(record, list) and len(record) == 2 and isinstance(record[1], dict):
                    _keep_fill(record[1], coins, fills, n)
                elif isinstance(record, dict) and "side_info" in record:
                    _keep_trade(record, coins, rows, n, name)
                else:
                    raise HourFailed("unparsed", f"{name} line {n + 1} is no shape this reader knows")
    except HourFailed:
        raise
    except (RuntimeError, EOFError, OSError, ValueError) as e:
        raise HourFailed("unreadable", f"{name} does not decompress ({e})") from None
    for (coin, tid), group in fills.items():
        takers = [f for f in group if f["crossed"] is True]
        if len(takers) != 1:
            raise HourFailed("unparsed", f"{name}: {coin} trade {tid} has {len(takers)} taker fills, not one")
        taker = takers[0]
        rows["ticker"].append(coins[coin])
        rows["micros"].append(int(taker["time"]) * 1000)
        rows["price"].append(float(taker["px"]))
        rows["size"].append(float(taker["sz"]))
        rows["taker_buy"].append(_side(taker["side"], name))
        rows["liquidation"].append(any(f.get("liquidation") for f in group))
        rows["seq"].append(taker["_seq"])
    return pl.DataFrame(rows, schema=TRADES).sort("micros", "seq")


def _keep_fill(fill: dict, coins: dict[str, str], fills: dict, n: int) -> None:
    coin = fill.get("coin")
    if coin not in coins:
        return
    fills.setdefault((coin, fill["tid"]), []).append({**fill, "_seq": n})


def _keep_trade(record: dict, coins: dict[str, str], rows: dict, n: int, name: str) -> None:
    coin = record.get("coin")
    if coin not in coins:
        return
    override = record.get("trade_dir_override")
    if override not in _OVERRIDES:
        raise HourFailed("unparsed", f"{name} line {n + 1}: trade_dir_override {override!r} is not one of {', '.join(_OVERRIDES)}")
    liquidation, counted = _OVERRIDES[override]
    if not counted:
        return
    rows["ticker"].append(coins[coin])
    rows["micros"].append(_iso_micros(record["time"], name))
    rows["price"].append(float(record["px"]))
    rows["size"].append(float(record["sz"]))
    rows["taker_buy"].append(_side(record["side"], name))
    rows["liquidation"].append(liquidation)
    rows["seq"].append(n)


def _side(side: str, name: str) -> bool:
    if side not in ("B", "A"):
        raise HourFailed("unparsed", f"{name}: side {side!r} is neither B nor A")
    return side == "B"


def _iso_micros(text: str, name: str) -> int:
    """`2025-04-01T12:00:00.092565137` (UTC, nanoseconds) as integer micros, the nanoseconds cut."""
    whole, _, fraction = text.partition(".")
    try:
        second = datetime.fromisoformat(whole).replace(tzinfo=UTC)
    except ValueError:
        raise HourFailed("unparsed", f"{name}: time {text!r} is not ISO") from None
    return int(second.timestamp()) * 1_000_000 + int((fraction + "000000")[:6])


def assemble(day: date, read_hours: dict[str, pl.DataFrame | HourFailed | None], symbols: dict[str, str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """A day's 1m bars for each ticker in `symbols` (ticker → coin), and its gaps.

    `read_hours` maps each of `keys(day)` to its trades, its failure, or
    None when the archive does not hold it. An hour whose own file is not
    read is a gap; an hour whose next file is not read loses its last
    second, which is a gap too. A bar that overlaps a gap is not written.
    """
    gaps: list[tuple[datetime, datetime, str]] = []
    frames = []
    for hour in hours(day):
        end = hour.start + _HOUR
        own = read_hours.get(hour.own) if hour.own else None
        if hour.prefix is None or not isinstance(own, pl.DataFrame):
            gaps.append((hour.start, end, own.cause if isinstance(own, HourFailed) else "archive-missing"))
            continue
        lo, hi = _micros(hour.start), _micros(end)
        if own.filter(pl.col("micros") >= hi).height:
            # A file holds what arrived in its hour: a trade stamped after it would be lost by every hour's read.
            gaps.append((hour.start, end, "unparsed"))
            continue
        spill = read_hours.get(hour.spill)
        if not isinstance(spill, pl.DataFrame):
            gaps.append((end - _SECOND, end, spill.cause if isinstance(spill, HourFailed) else "archive-missing"))
        for part, rank in ((own, 0), (spill, 1)):
            if isinstance(part, pl.DataFrame):
                frames.append(part.filter((pl.col("micros") >= lo) & (pl.col("micros") < hi)).with_columns(pl.lit(rank).alias("_file")))
    trades = pl.concat(frames) if frames else pl.DataFrame(schema={**TRADES, "_file": pl.Int32})
    bars = _bars(trades.sort("micros", "_file", "seq"), symbols)
    for lo, hi, _ in gaps:
        # A bar overlapping a gap would be a minute with trades missing: none is written.
        bars = bars.filter(~((pl.col("ts") < hi) & (pl.col("close_ts") > lo)))
    gap_frame = pl.DataFrame(
        {"venue": [HYPERLIQUID] * len(gaps), "ticker": [None] * len(gaps), "series": ["candles"] * len(gaps),
         "cause": [c for _, _, c in gaps], "from_ts": [lo for lo, _, _ in gaps], "to_ts": [hi for _, hi, _ in gaps]},
        schema=GAPS,
    )  # fmt: skip
    return bars, gap_frame


def _micros(t: datetime) -> int:
    return int(t.timestamp()) * 1_000_000


def _bars(trades: pl.DataFrame, symbols: dict[str, str]) -> pl.DataFrame:
    """Trades in time order into one bar per traded minute."""
    ordinary = ~pl.col("liquidation")
    minute = (pl.col("micros") // _MINUTE_US) * _MINUTE_US
    out = (
        trades.filter(pl.col("ticker").is_in(list(symbols)))
        .with_columns(minute.alias("_minute"))
        .group_by("ticker", "_minute", maintain_order=True)
        .agg(
            pl.col("price").first().alias("open"),
            pl.col("price").filter(ordinary).max().alias("high"),
            pl.col("price").filter(ordinary).min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("size").sum().alias("volume"),
            pl.len().cast(pl.UInt32).alias("trade_count"),
            (pl.col("price") * pl.col("size")).sum().alias("notional"),
            pl.col("size").filter(pl.col("taker_buy")).sum().alias("taker_buy_volume"),
            pl.col("liquidation").sum().cast(pl.UInt32).alias("liquidations"),
            pl.col("price").max().alias("high_all"),
            pl.col("price").min().alias("low_all"),
        )
        .with_columns(
            # A minute of liquidations only: the venue's high and low are then its liquidation prices.
            pl.col("high").fill_null(pl.col("high_all")),
            pl.col("low").fill_null(pl.col("low_all")),
            pl.lit(HYPERLIQUID).alias("venue"),
            pl.col("ticker").replace_strict(symbols, return_dtype=pl.String).alias("symbol"),
            pl.lit("1m").alias("interval"),
            pl.col("_minute").cast(UTC_US).alias("ts"),
            (pl.col("_minute") + _MINUTE_US).cast(UTC_US).alias("close_ts"),
        )
    )
    return out.select(list(CANDLES)).sort("ticker", "ts")


def symbols_for(tickers: list[str]) -> dict[str, str]:
    """Ticker → the venue's coin, for tickers of this venue; another is refused with the list held."""
    from ._instruments import instrument

    return {t: instrument(HYPERLIQUID, t).symbol for t in tickers}


def require_lz4() -> None:
    try:
        import lz4.frame  # noqa: F401
    except ImportError:
        raise Refused("hyperliquid-s3 reads LZ4 hour files: install the extra with uv sync --extra hyperliquid") from None
