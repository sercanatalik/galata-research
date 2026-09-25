"""Positions to modelled returns: a position decided at a close earns the next bar, never its own.

A signal computed from a bar's close is known at `close_ts`, so the earliest
return it can earn is the next bar's. Earning the same bar is the backtest
that "was measuring the future" (v/D-029). Costs are charged on turnover at
Hyperliquid's base taker rate (0.045%), and every figure says it is
modelled. Funding is not charged, because the record holds only days of
settled funding, and every row says that too.
"""

import polars as pl

from ._errors import Refused

TAKER_FEE = 0.00045


def returns(bars: pl.LazyFrame | pl.DataFrame, position: pl.Expr, *, fee: float = TAKER_FEE) -> pl.DataFrame:
    """Per-bar modelled returns of `position`, per ticker.

    `bars` are candles from `gr.market.candles` (closed, one row per bar).
    `position` is an expression over them, using only data up to each bar's
    close, and is evaluated per ticker. Columns: `ticker, ts, close_ts,
    position, bar_return, gross, cost, net, modelled, funding_charged`;
    `bar_return` is the close-to-close return the held position earned.
    """
    if fee < 0:
        raise Refused(f"fee={fee} is negative")
    lf = bars.lazy()
    names = lf.collect_schema().names()
    missing = [c for c in ("ticker", "ts", "close_ts", "close") if c not in names]
    if missing:
        raise Refused(f"bars need {', '.join(missing)}; load them with gr.market.candles")
    return (
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
        .with_columns(
            (pl.col("gross") - pl.col("cost")).alias("net"),
            pl.lit(True).alias("modelled"),
            pl.lit(False).alias("funding_charged"),
        )
        .select("ticker", "ts", "close_ts", pl.col("_held").alias("position"), "bar_return", "gross", "cost", "net", "modelled", "funding_charged")
        .collect()
    )
