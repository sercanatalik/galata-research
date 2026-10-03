"""Other venues' published history, fetched by `galata-fetch` into the reference store and read here.

    gr.reference.depth(["BTC"], start, end, venues=["binance-um"])  # ±0.2..5% depth, every 30 s
    gr.reference.trades(["HYPE"], start, end, rpi=False)            # signed trades, no RPI fills
    gr.reference.book(["BTC"], start, end)                          # Bybit's top and near depth, per second
    gr.reference.candles(["ETH"], start, end, as_of=t)              # Binance 1m, known at close_ts
    gr.reference.funding(["BTC"], start, end)                       # Binance's settled funding, each with its interval
    gr.reference.premium(["BTC"], start, end)                       # Binance's 1m premium index: the perp over its index
    gr.reference.coverage()                                         # which days each series holds

The rules are the record's where the meaning is the same: `ts` is the
venue's time on UTC to the microsecond, a candle is known at `close_ts`, an
unknown ticker is refused with what is held, and a day the archive lacked is
stated, never filled. Archives carry no receipt clock, so no frame here has
`recv_ts`, and none is ever derived.

Loading never opens a socket: the fetch command is not imported by this
module (`galata_research.reference.fetch`, run as `galata-fetch`).
"""

from collections.abc import Sequence
from datetime import datetime, timedelta

import polars as pl

from .. import _scan, utils
from .._errors import Refused
from . import _manifest, _sources
from ._instruments import DAY_ENDS_EARLY, INSTRUMENTS, KINDS, MONTHLY, VENUES
from ._root import root

__all__ = [
    "INSTRUMENTS",
    "KINDS",
    "VENUES",
    "book",
    "candles",
    "coverage",
    "depth",
    "funding",
    "funding_hours",
    "premium",
    "root",
    "trades",
]

_OUT = {kind: {c: t for c, t in schema.items() if c != "rpi"} for kind, schema in _sources.SCHEMAS.items()}


def trades(
    tickers: Sequence[str] | str | None,
    start: datetime | str,
    end: datetime | str,
    *,
    venues: Sequence[str] | str | None = None,
    as_of: datetime | str | None = None,
    rpi: bool | None = None,
    engine: str = "polars",
):
    """Executions with `ts` in `[start, end)`, in each venue's order.

    - **Binance** rows are aggTrades: one taker order's fills at one price,
      under its `agg_trade_id`. A row is not one execution; its `size` is.
    - **Bybit** rows are single executions under `trdMatchID`, with `ts` to
      the 100 µs the archive states. `rpi=False` drops Bybit's
      retail-price-improvement fills, which the public book never offered;
      `rpi=True` keeps only them. Binance has none.
    - **OKX** rows are single executions under `trade_id`, to the
      millisecond; `size` is in BTC or ETH, not OKX's contracts. Its
      Enhanced Liquidity Program fills count as `rpi`: like Bybit's, they
      took liquidity only retail takers could reach.
    """
    lf = _load("trades", tickers, start, end, venues, as_of)
    if isinstance(lf, pl.LazyFrame) and rpi is not None:
        lf = lf.filter(pl.col("rpi").fill_null(False) == rpi)
    return _finish(lf, "trades", engine)


def depth(tickers, start, end, *, venues=None, as_of=None, engine: str = "polars"):
    """Binance's cumulative book depth at ±0.2, 1, 2, 3, 4 and 5%, one snapshot every 30 s.

    `band_pct` is signed, negative on the bid; `depth` is base quantity and
    `notional` USDT within the band. Binance does not document whether a band
    is measured from the mid or the touch.
    """
    return _finish(_load("depth", tickers, start, end, venues, as_of), "depth", engine)


def book(tickers, start, end, *, venues=None, as_of=None, engine: str = "polars"):
    """Bybit's book replayed to one row per second: the top, and the depth within 2 and 10 bps.

    A band past the deepest level the archive holds is null: ob200 reaches
    about 4 bps on BTC (2026-09-26), so there 10 bps is always null.
    `*_reach_bps` says how far each side's archive reached. Before
    2025-08-21 the archive is ob500, which reaches further.
    """
    return _finish(_load("book", tickers, start, end, venues, as_of), "book", engine)


def candles(tickers, start, end, *, venues=None, as_of=None, engine: str = "polars"):
    """Binance 1m bars with `ts` in `[start, end)`, known at `close_ts = ts + 1m`: `as_of` filters on it."""
    return _finish(_load("candles", tickers, start, end, venues, as_of), "candles", engine)


def events(start, end, *, sources: Sequence[str] | str | None = None) -> pl.DataFrame:
    """Scheduled US releases and FOMC decisions dated in `[start, end)`: `source, event, date, ts, scheduled, detail, url`.

    Fetched by `galata-fetch events` from federalreserve.gov and bls.gov.
    `ts` is the release instant in UTC (08:30 New York for CPI and jobs,
    14:00 for a scheduled FOMC statement), null for an unscheduled action,
    whose time the page does not state.
    """
    lo, hi = utils.window(start, end)
    path = root() / "events" / "events.parquet"
    if not path.is_file():
        raise Refused(f"the reference store holds no event calendar ({path}); fetch it with galata-fetch events")
    frame = pl.read_parquet(path)
    held = sorted(frame["source"].unique())
    wanted = _names(sources, held, "events source")
    midnight = pl.col("date").cast(pl.Datetime("us")).dt.epoch("us")
    return frame.filter(pl.col("source").is_in(wanted) & (midnight >= lo) & (midnight < hi)).sort("date", "source", "event")


def coverage() -> pl.DataFrame:
    """One row per (kind, venue, ticker): `first, last, days_ok, days_absent, days_mismatch, days_missing`.

    `first` and `last` are the first and last `ok` days; `days_missing` counts
    days between them with no manifest row at all, fetched never. A sampled
    fetch (`--sample weekly:wed`) shows six missing days in seven, as it should.
    """
    m = _manifest.read(root())
    key = ["kind", "venue", "ticker"]
    ok = pl.col("status") == "ok"
    series = m.group_by(key).agg(
        pl.col("date").filter(ok).min().alias("first"),
        pl.col("date").filter(ok).max().alias("last"),
        ok.sum().cast(pl.UInt32).alias("days_ok"),
        (pl.col("status") == "absent").sum().cast(pl.UInt32).alias("days_absent"),
        (pl.col("status") == "mismatch").sum().cast(pl.UInt32).alias("days_mismatch"),
    )
    # Days between the first and last ok day that have any manifest row.
    rowed = (
        m.join(series.select(*key, "first", "last"), on=key)
        .filter(pl.col("date").is_between(pl.col("first"), pl.col("last")))
        .group_by(key)
        .agg(pl.col("date").n_unique().alias("rowed"))
    )
    span = (pl.col("last") - pl.col("first")).dt.total_days() + 1
    return (
        series.join(rowed, on=key, how="left")
        .with_columns((span - pl.col("rowed")).fill_null(0).cast(pl.UInt32).alias("days_missing"))
        .drop("rowed")
        .sort(key)
    )


def funding(tickers, start, end, *, venues=None, as_of=None, engine: str = "polars"):
    """Binance's settled funding with `ts` in `[start, end)`: `rate` per settlement and its `interval_hours`.

    A long pays `rate` × notional at `ts` when it is positive, a short
    receives it. Known at `ts`: the settlement is the rate that was charged.
    The archive publishes a month only once it has ended, so the current
    month is absent until then.
    """
    return _finish(_load("funding", tickers, start, end, venues, as_of), "funding", engine)


def premium(tickers, start, end, *, venues=None, as_of=None, engine: str = "polars"):
    """Binance's premium index in 1m bars, known at `close_ts`: (the perp's impact price − its index) / its index.

    It is what Binance's funding rate is computed from, and the basis a
    position long the perp and short the index bears.
    """
    return _finish(_load("premium", tickers, start, end, venues, as_of), "premium", engine)


def funding_hours(settled: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    """Settlements as one row per hour, `ticker, ts, rate`, for `gr.backtest.returns(funding=...)`.

    Each settlement covers the `interval_hours` up to it: its hour carries
    the rate, and the hours before it within the interval carry 0, since
    nothing settles then. A bar is charged by `backtest.returns` only when
    every hour it spans is present, so a missing settlement still leaves the
    bars over its interval uncharged, never charged a partial sum. A
    settlement stamped a few milliseconds past its hour is that hour's.
    """
    utils.require(settled, ("ticker", "ts", "rate", "interval_hours"), "load funding with gr.reference.funding")
    hour = pl.col("ts").dt.truncate("1h")
    return (
        utils.lazy(settled)
        .select(
            "ticker",
            pl.datetime_ranges(
                hour - pl.duration(hours=pl.col("interval_hours") - 1), hour, interval="1h", closed="both"
            ).alias("ts"),
            pl.col("rate"),
            hour.alias("_settles"),
        )
        .explode("ts", empty_as_null=True)
        # When the interval changes, a window can reach back over an earlier settlement: its hour keeps its rate.
        .group_by("ticker", "ts")
        .agg(pl.col("rate").filter(pl.col("ts") == pl.col("_settles")).first().fill_null(0.0))
        .sort("ticker", "ts")
        .collect()
    )


def _names(asked, held: list[str], what: str) -> list[str]:
    if asked is None:
        return held
    asked = [asked] if isinstance(asked, str) else list(asked)
    unknown = [a for a in asked if a not in held]
    if unknown:
        raise Refused(
            f"the reference store holds no {what} for {', '.join(unknown)}; it holds {', '.join(held) or 'none'}"
        )
    return asked


def _load(kind, tickers, start, end, venues, as_of) -> pl.LazyFrame | None:
    lo, hi = utils.window(start, end)
    bound = None if as_of is None else utils.instant("as_of", as_of)
    store = root()
    m = _manifest.read(store).filter((pl.col("kind") == kind) & (pl.col("status") == "ok"))
    if m.is_empty():
        raise Refused(f"the reference store {store} holds no {kind}; fetch it with galata-fetch {kind} ...")
    wanted_venues = _names(venues, sorted(m["venue"].unique()), f"{kind} venue")
    m = m.filter(pl.col("venue").is_in(wanted_venues))
    wanted = _names(tickers, sorted(m["ticker"].unique()), kind)

    first, last = _scan._day(lo), _scan._day(hi - 1)
    if kind in MONTHLY:
        # A month's file is held under its first day.
        first = first.replace(day=1)
    # A venue whose archive day ends early (OKX's, at 16:00 UTC) holds the end of `last` in the next file.
    early = pl.col("venue").is_in(list(DAY_ENDS_EARLY))
    until = pl.when(early).then(pl.lit(last + timedelta(days=1))).otherwise(pl.lit(last))
    days = m.filter(pl.col("ticker").is_in(wanted) & (pl.col("date") >= first) & (pl.col("date") <= until))
    files = [
        p
        for r in days.iter_rows(named=True)
        if (p := _manifest.day_path(store, kind, r["venue"], r["ticker"], r["date"])).is_file()
    ]
    if not files:
        return None

    at = lambda micros: pl.lit(micros).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    lf = pl.scan_parquet(files, hive_partitioning=False).filter((pl.col("ts") >= at(lo)) & (pl.col("ts") < at(hi)))
    if bound is not None:
        # A bar is known at its close, a snapshot or an execution at its time.
        known = "close_ts" if kind in ("candles", "premium") else "ts"
        lf = lf.filter(pl.col(known) <= at(bound))
    return lf


def _finish(lf: pl.LazyFrame | None, kind: str, engine: str):
    if engine not in _scan.ENGINES:
        raise Refused(f"engine={engine!r} is not one of {', '.join(_scan.ENGINES)}")
    schema = _OUT[kind]
    if lf is None:
        return _scan.finish(pl.LazyFrame(schema=schema), engine)
    # Each file is one day in the venue's order; the sort is stable, so ties keep it.
    out = lf.select(list(schema)).sort(["venue", "ticker", "ts"], maintain_order=True)
    return _scan.finish(out, engine)
