"""What each archive lists, and from when: declared once, measured from the archives.

MEASURED 2026-09-27, the first file each archive holds:

    binance-um    bookDepth   BTCUSDT, ETHUSDT 2023-01-01   HYPEUSDT 2025-05-30   XAUUSDT 2025-12-11
                  fundingRate (monthly files only) BTCUSDT, ETHUSDT 2020-01   HYPEUSDT 2025-06
                  premiumIndexKlines 1m   BTCUSDT 2020-01-01   HYPEUSDT by 2025-06-01
                  klines 1d (monthly files)   900 USDT perpetuals listed 2026-10-04, the delisted
                              among them (LUNAUSDT, FTTUSDT, SRMUSDT to 2024-05)
                  aggTrades   BTCUSDT, ETHUSDT 2019-12-31   HYPEUSDT 2025-05-30   XAUUSDT 2025-12-11
                  klines 1m   BTCUSDT, ETHUSDT 2019-12-31   HYPEUSDT 2025-05-30   XAUUSDT 2025-12-11
    bybit-linear  trading     BTCUSDT 2020-03-25   ETHUSDT 2020-10-21   HYPEUSDT 2024-12-05   XAUUSDT 2026-03-09
                  orderbook   BTCUSDT, ETHUSDT 2023-01-18   HYPEUSDT 2024-12-04   XAUUSDT 2026-03-09
                              ob500 through 2025-08-20, ob200 from 2025-08-21 (BTC, ETH, HYPE)
    okx-swap      trades      BTC-USDT-SWAP, ETH-USDT-SWAP 2021-10-01, but October 2021 lists
                              every trade twice (BUY and SELL), so the archive is taken from 2021-11-01

OKX's file for a date runs 16:00 to 16:00 UTC, the Beijing day; its sizes are
contracts, 0.01 BTC or 0.1 ETH each (OKX help, "USDT margined perpetual swap";
okx/agent-trade-kit). HYPE and XAU are listed by OKX but their contract
values were not confirmed from a published page, so they are not mapped.

The record's ticker is kept, and the venue's symbol travels on every row: a
USDT book is not a USD book, and XAUUSDT is not Hyperliquid's GOLD.

MEASURED 2026-10-25, Hyperliquid's own fills (`hyperliquid-s3`), by a full
listing of the requester-pays bucket `hl-mainnet-node-data` (ap-northeast-1)
and one sample hour per prefix:

    node_trades           2025-03-22 10:00 .. 2025-05-25 14:00 used, no hour missing in it;
                          15.1 GB all told, 8.5 MB the median hour. It runs on to 06-21 with
                          26 whole days missing, which is not read.
    node_fills            2025-05-25 14:00 .. 2025-07-27 08:00, no hour missing; 31.8 GB, 18.3 MB
    node_fills_by_block   2025-07-27 08:00 .. now, no hour missing in 456 days; 321.4 GB,
                          25.5 MB the median hour, 359 MB the largest

- Keys are `<prefix>/hourly/YYYYMMDD/H.lz4`, the hour not zero-padded.
- Each handover hour has a file in both formats (05-25 hour 14, 07-27 hour
  8), but neither format holds the whole hour: node_trades' last trade is
  14:23:41.017 and node_fills' first 14:32:53.109; node_fills' last is
  08:45:48.664 and node_fills_by_block's first 08:50:10.273. `CUTS` reads
  each later format from its first trade, and the holes between are
  silences. The first backfill's code read those hours from the later
  format alone, and lost 14:00..14:23 and 08:00..08:45 of those days.
- The archiving node goes silent inside files that exist. Across every
  coin, the longest step between two trades in an ordinary hour is 2.5 s
  (9 hours sampled, 2025-04 to 2026-09, all three formats); 2025-07-27's
  hours 8 to 12 hold steps of 48 to 178 s, many an hour. A step of 10 s or
  more is `archive-silent`. BTC, which trades every minute, was short of
  1,440 bars on 25 of the backfill's first 120 days, mostly by 2 to 6
  minutes on Sundays.
- A file holds the trades the node received in its hour: up to ~30 fills
  stamped in the hour's last second land in the next file (4 in node_fills
  2025-07-01 12, 30 in by_block 2026-09-24 14), none in an earlier one.
- Every trade in node_fills and by_block is two fills under one
  `(coin, tid)`, exactly one `crossed`, for the 13 coins measured (the 11
  below, and xyz:JPY and xyz:EUR, since left out as too thin). node_trades
  has no `tid`: one row per trade, the taker's `side`, and rows that share
  every field are distinct trades of one block (1,183 in the sample hour).
- The venue's candles count liquidations in volume and trade count but
  leave their prices out of high and low: the 2026-09-24 14:00 hour built
  from by_block equals the venue's 1h candle on open, close, volume and
  trade count for all 13 coins measured, and on high and low except BTC (83,350
  against 83,351) and SILVER (63.275 against 63.28), where every fill below
  the venue's low was a liquidation. Over the week 2026-09-21..27 (the
  first fetch, 175 files, 8.2 GB, ~$0.94), every one of the 1,848 hours of
  the 11 tickers below, built from the 1m bars, equals the venue's 1h
  candle on open, high, low, close, volume and trade count, the same hours
  held on both sides, no gap; `high_all` or `low_all` differs in 25 of them,
  each a liquidation wick.
- The venue's candles leave out node_trades' `NetChildVaultPositions` rows:
  on 2025-04-01, VVV, AI16Z, COMP, TST and TURBO (842 down to 1 such rows)
  match the venue's 1d candle in volume and trade count exactly without
  them, and BTC (377,931 ordinary trades and 922 `LiquidatedMarket`) matches
  with its liquidations in.
- node_trades also writes a liquidation as an object naming its margin
  mode and side, `{"LiquidatedCross": "B"}` or `{"LiquidatedIsolated": "B"}`
  (first seen 2025-04-06, one of each on ETH); both are read as
  liquidations. That day's ETH counts 237,510 trades without them and
  237,512 with, against the venue's 1d candle's 237,511 (1.1248 ETH more
  volume than without, matching neither row), so a one-trade residual
  remains that these rows do not explain; the venue no longer serves the
  1h candles that would place it.
- The xyz coins are named in the fills as the venue's `meta` spells them
  (`xyz:CL`, `xyz:SKHX`, ...); spot coins are `@<index>`. Listings are each
  market's first 1d candle on the venue, read 2026-10-25.
- Transfer is ~2 GB a month in 2025-03, ~30 GB in 2026-09; the whole archive
  is ~360 GB, about $41 at ap-northeast-1's $0.114/GB to the internet
  (AWSDataTransfer price list published 2026-09-16), and GETs are $0.0037
  per 10,000, LISTs $0.0047 per 1,000 (AmazonS3, published 2026-10-15).
"""

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from .._errors import Refused

VENUES = ("binance-um", "bybit-linear", "okx-swap", "hyperliquid-s3")
# The venue built from fills rather than fetched as files: one pass over an hour serves every ticker.
HYPERLIQUID = "hyperliquid-s3"
# The store's kind for a venue's own hours missing or unreadable, written beside its candles.
GAPS = "gaps"
KINDS = ("trades", "depth", "book", "candles", "funding", "premium", "daily")
# Kinds archived one file per month, held under the month's first day.
MONTHLY = frozenset({"funding", "daily"})
# Kinds whose files are a few kilobytes: fetched with more parallel requests.
LIGHT = frozenset({"funding", "daily"})


@dataclass(frozen=True)
class Instrument:
    symbol: str
    listed: date  # the first day any of its archives holds
    contract: float = 1.0  # base units in one unit of the archive's `size`


INSTRUMENTS: dict[tuple[str, str], Instrument] = {
    ("binance-um", "BTC"): Instrument("BTCUSDT", date(2019, 12, 31)),
    ("binance-um", "ETH"): Instrument("ETHUSDT", date(2019, 12, 31)),
    ("binance-um", "HYPE"): Instrument("HYPEUSDT", date(2025, 5, 30)),
    ("binance-um", "GOLD"): Instrument("XAUUSDT", date(2025, 12, 11)),
    ("bybit-linear", "BTC"): Instrument("BTCUSDT", date(2020, 3, 25)),
    ("bybit-linear", "ETH"): Instrument("ETHUSDT", date(2020, 10, 21)),
    ("bybit-linear", "HYPE"): Instrument("HYPEUSDT", date(2024, 12, 4)),
    ("bybit-linear", "GOLD"): Instrument("XAUUSDT", date(2026, 3, 9)),
    ("okx-swap", "BTC"): Instrument("BTC-USDT-SWAP", date(2021, 10, 1), contract=0.01),
    ("okx-swap", "ETH"): Instrument("ETH-USDT-SWAP", date(2021, 10, 1), contract=0.1),
    # Hyperliquid under the record's tickers, the venue's coin as the symbol. The crypto perps
    # start where the archive does; each xyz market at its first 1d candle on the venue.
    (HYPERLIQUID, "BTC"): Instrument("BTC", date(2025, 3, 22)),
    (HYPERLIQUID, "ETH"): Instrument("ETH", date(2025, 3, 22)),
    (HYPERLIQUID, "HYPE"): Instrument("HYPE", date(2025, 3, 22)),
    (HYPERLIQUID, "XYZ100"): Instrument("xyz:XYZ100", date(2025, 10, 13)),
    (HYPERLIQUID, "NVDA"): Instrument("xyz:NVDA", date(2025, 11, 12)),
    (HYPERLIQUID, "GOLD"): Instrument("xyz:GOLD", date(2025, 12, 22)),
    (HYPERLIQUID, "SILVER"): Instrument("xyz:SILVER", date(2025, 12, 26)),
    (HYPERLIQUID, "CL"): Instrument("xyz:CL", date(2026, 1, 6)),
    (HYPERLIQUID, "SKHX"): Instrument("xyz:SKHX", date(2026, 2, 19)),
    (HYPERLIQUID, "BRENTOIL"): Instrument("xyz:BRENTOIL", date(2026, 3, 4)),
    (HYPERLIQUID, "SP500"): Instrument("xyz:SP500", date(2026, 3, 18)),
}

# Which prefix each instant of trade time is read from: the last prefix whose cut is at or before it.
# Each cut is the later format's own first trade (MEASURED above); the earlier format is read up to it.
CUTS = (
    (datetime(2025, 3, 22, 10, tzinfo=UTC), "node_trades"),
    (datetime(2025, 5, 25, 14, 32, 53, 109000, tzinfo=UTC), "node_fills"),
    (datetime(2025, 7, 27, 8, 50, 10, 273000, tzinfo=UTC), "node_fills_by_block"),
)


def prefix_at(instant: datetime) -> str | None:
    """The prefix an instant of trade time is read from, or None before the archive begins."""
    chosen = None
    for cut, prefix in CUTS:
        if instant >= cut:
            chosen = prefix
    return chosen

# The venue whose every USDT perpetual can be named by its base asset, for a point-in-time universe.
OPEN_VENUE = "binance-um"
OPEN_LISTED = date(2019, 9, 1)  # BTCUSDT, the first USDⓈ-M perpetual
# Binance lists symbols in other scripts too (币安人生USDT, 2025): anything without a space, a slash or a dot.
_SYMBOL = re.compile(r"[^\s/\\.]{1,30}")
# The record's HIP-3 markets: their names on Binance, if any, are other assets.
RECORD_ONLY = frozenset({"CL", "XYZ100"})

# The day each venue's archive of a kind begins, whatever the symbol.
ARCHIVE_FROM: dict[tuple[str, str], date] = {
    ("binance-um", "depth"): date(2023, 1, 1),
    ("binance-um", "trades"): date(2019, 12, 31),
    ("binance-um", "candles"): date(2019, 12, 31),
    ("binance-um", "funding"): date(2020, 1, 1),
    ("binance-um", "premium"): date(2020, 1, 1),
    ("binance-um", "daily"): date(2019, 9, 1),
    ("bybit-linear", "trades"): date(2020, 3, 25),
    ("bybit-linear", "book"): date(2023, 1, 18),
    ("okx-swap", "trades"): date(2021, 11, 1),
    (HYPERLIQUID, "candles"): date(2025, 3, 22),
}

# How far a venue's archive day ends before the UTC day of the same date:
# OKX's file for D holds D-1 16:00 to D 16:00 UTC, so a UTC day needs D and D+1.
DAY_ENDS_EARLY: dict[str, timedelta] = {"okx-swap": timedelta(hours=8)}


def archive_days(venue: str, day: date) -> list[date]:
    """The archive dates that together hold one UTC day."""
    return [day, day + timedelta(days=1)] if venue in DAY_ENDS_EARLY else [day]

# Bybit's book archive changed depth on this day: ob500 before, ob200 from.
OB200_FROM = date(2025, 8, 21)


def instrument(venue: str, ticker: str) -> Instrument:
    """The venue's listing of a ticker; a venue or ticker it has not is refused with what it has."""
    if venue not in VENUES:
        raise Refused(f"venue={venue!r} is not one of {', '.join(VENUES)}")
    if (venue, ticker) not in INSTRUMENTS and venue == OPEN_VENUE and ticker in RECORD_ONLY:
        held = sorted(t for v, t in INSTRUMENTS if v == venue)
        raise Refused(
            f"{venue} lists no {ticker}; it lists {', '.join(held)} and any USDT perpetual by its base asset, "
            f"but {ticker} is the record's HIP-3 market, and a Binance symbol of that name is another asset"
        )
    if (venue, ticker) not in INSTRUMENTS and venue == OPEN_VENUE:
        # Binance's whole USDⓈ-M list is fetchable: any ticker is its USDT perpetual, and an archive
        # that lacks it answers 404, recorded absent. The record's own tickers keep their mapping above.
        if not _SYMBOL.fullmatch(ticker):
            raise Refused(f"{ticker!r} is not a Binance base asset name")
        return Instrument(f"{ticker}USDT", OPEN_LISTED)
    try:
        return INSTRUMENTS[(venue, ticker)]
    except KeyError:
        held = sorted(t for v, t in INSTRUMENTS if v == venue)
        raise Refused(f"{venue} lists no {ticker}; it lists {', '.join(held)}") from None


def first_day(venue: str, kind: str, ticker: str) -> date:
    """The first day worth asking for: the later of the listing and the archive's start."""
    listed = instrument(venue, ticker).listed
    if (venue, kind) not in ARCHIVE_FROM:
        offered = sorted(k for v, k in ARCHIVE_FROM if v == venue)
        raise Refused(f"{venue} publishes no {kind} archive; it publishes {', '.join(offered)}")
    return max(listed, ARCHIVE_FROM[(venue, kind)])
