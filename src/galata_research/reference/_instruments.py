"""What each archive lists, and from when: declared once, measured from the archives.

MEASURED 2026-09-27, the first file each archive holds:

    binance-um    bookDepth   BTCUSDT, ETHUSDT 2023-01-01   HYPEUSDT 2025-05-30   XAUUSDT 2025-12-11
                  fundingRate (monthly files only) BTCUSDT, ETHUSDT 2020-01   HYPEUSDT 2025-06
                  premiumIndexKlines 1m   BTCUSDT 2020-01-01   HYPEUSDT by 2025-06-01
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
"""

from dataclasses import dataclass
from datetime import date, timedelta

from .._errors import Refused

VENUES = ("binance-um", "bybit-linear", "okx-swap")
KINDS = ("trades", "depth", "book", "candles", "funding", "premium")
# Kinds archived one file per month, held under the month's first day.
MONTHLY = frozenset({"funding"})


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
}

# The day each venue's archive of a kind begins, whatever the symbol.
ARCHIVE_FROM: dict[tuple[str, str], date] = {
    ("binance-um", "depth"): date(2023, 1, 1),
    ("binance-um", "trades"): date(2019, 12, 31),
    ("binance-um", "candles"): date(2019, 12, 31),
    ("binance-um", "funding"): date(2020, 1, 1),
    ("binance-um", "premium"): date(2020, 1, 1),
    ("bybit-linear", "trades"): date(2020, 3, 25),
    ("bybit-linear", "book"): date(2023, 1, 18),
    ("okx-swap", "trades"): date(2021, 11, 1),
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
