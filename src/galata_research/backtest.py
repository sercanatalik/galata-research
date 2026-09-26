"""Positions to modelled returns: a position decided at a close earns the next bar, never its own.

A signal computed from a bar's close is known at `close_ts`, so the earliest
return it can earn is the next bar's. Earning the same bar is the backtest
that "was measuring the future" (v/D-029). Costs are charged on turnover at
Hyperliquid's base taker rate (0.045%), and every figure says it is
modelled. Funding is charged only when settled funding is passed in, and
only on a bar whose every settlement hour has a rate; every row says whether
it was.
"""

from datetime import timedelta

import polars as pl

from ._errors import Refused

TAKER_FEE = 0.00045

_HOUR = timedelta(hours=1)


def returns(
    bars: pl.LazyFrame | pl.DataFrame,
    position: pl.Expr,
    *,
    fee: float = TAKER_FEE,
    funding: pl.LazyFrame | pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Per-bar modelled returns of `position`, per ticker.

    `bars` are candles from `gr.market.candles` (closed, one row per bar).
    `position` is an expression over them, using only data up to each bar's
    close, and is evaluated per ticker. Columns: `ticker, ts, close_ts,
    position, bar_return, gross, cost, funding, net, modelled,
    funding_charged`; `bar_return` is the close-to-close return the held
    position earned.

    `funding`, when given, is settled funding from `gr.market.funding`
    (`ticker, ts, rate`). A bar is charged `held × Σ rate` over the
    settlement hours H with `ts < H ≤ close_ts`: the position is in place
    through the close, and a long pays a positive rate while a short
    receives it. A bar missing any of those hours is not charged at all, and
    says so: a partial charge presented as a full one is the flattering
    number this exists to remove. Without it, nothing is charged and every
    figure is what it was.
    """
    if fee < 0:
        raise Refused(f"fee={fee} is negative")
    lf = bars.lazy()
    names = lf.collect_schema().names()
    missing = [c for c in ("ticker", "ts", "close_ts", "close") if c not in names]
    if missing:
        raise Refused(f"bars need {', '.join(missing)}; load them with gr.market.candles")
    out = (
        lf.sort("ticker", "ts")
        .with_columns(position.over("ticker").cast(pl.Float64).alias("position"))
        .with_columns(
            # The position decided at the previous close is what this bar is held with.
            pl.col("position").shift(1).over("ticker").alias("_held"),
            (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1).alias("_ret"),
            # A bar whose previous row is not the bar just before it spans a hole.
            (pl.col("close_ts").shift(1).over("ticker") == pl.col("ts")).alias("_next"),
        )
        .with_columns(
            (pl.col("_held") - pl.col("_held").shift(1).over("ticker").fill_null(0.0)).abs().alias("_turn"),
        )
        .with_columns(
            pl.when(pl.col("_next")).then(pl.col("_held") * pl.col("_ret")).alias("gross"),
            pl.when(pl.col("_next")).then(pl.col("_turn") * fee).alias("cost"),
            pl.when(pl.col("_next")).then(pl.col("_ret")).alias("bar_return"),
        )
    )
    if funding is None:
        out = out.with_columns(pl.lit(None, pl.Float64).alias("_rate_sum"), pl.lit(False).alias("_covered"))
    else:
        out = out.join(_rates_per_bar(out, funding), on=["ticker", "ts"], how="left")
    return (
        out.with_columns(
            pl.when(pl.col("_next") & pl.col("_covered")).then(pl.col("_held") * pl.col("_rate_sum")).alias("funding"),
        )
        .with_columns(
            (pl.col("gross") - pl.col("cost") - pl.col("funding").fill_null(0.0)).alias("net"),
            pl.lit(True).alias("modelled"),
            pl.col("funding").is_not_null().alias("funding_charged"),
        )
        .select(
            "ticker",
            "ts",
            "close_ts",
            pl.col("_held").alias("position"),
            "bar_return",
            "gross",
            "cost",
            "funding",
            "net",
            "modelled",
            "funding_charged",
        )
        .sort("ticker", "ts")
        .collect()
    )


def _rates_per_bar(bars: pl.LazyFrame, funding: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """Per bar: the sum of the settled rates due in `(ts, close_ts]`, and whether every due hour had one."""
    f = funding.lazy()
    missing = [c for c in ("ticker", "ts", "rate") if c not in f.collect_schema().names()]
    if missing:
        raise Refused(f"funding needs {', '.join(missing)}; load it with gr.market.funding")
    # A settlement is stamped a few ms past its hour (06:00:00.121): it is that hour's.
    settled = (
        f.select("ticker", pl.col("ts").dt.truncate("1h").alias("_hour"), pl.col("rate").cast(pl.Float64))
        .unique(["ticker", "_hour"], keep="first")
    )
    due = (
        bars.select("ticker", "ts", "close_ts")
        .with_columns(
            # The hours H with ts < H <= close_ts: the first whole hour after the
            # open, through the last whole hour at or before the close.
            pl.datetime_ranges(
                pl.col("ts").dt.truncate("1h") + _HOUR,
                pl.col("close_ts").dt.truncate("1h"),
                interval="1h",
                closed="both",
            ).alias("_hour")
        )
        .explode("_hour", empty_as_null=True)
    )
    return (
        due.join(settled, on=["ticker", "_hour"], how="left")
        .group_by("ticker", "ts")
        .agg(
            pl.col("rate").sum().alias("_rate_sum"),
            (pl.col("_hour").count() == pl.col("rate").count()).alias("_covered"),
        )
    )
