"""Trial families that hand back every trial, so N is the true N.

A family is a parameter grid run over the same bars: moving-average
crossovers and time-series momentum. Each trial's modelled net returns are
kept, and `summary` scores them one row per `(trial, ticker)`, including the
trials too short to score, because the Deflated Sharpe Ratio needs the count
of everything that was tried.
"""

from collections.abc import Iterable

import polars as pl

from . import backtest

MIN_RETURNS = 30
SIDES = ("long_flat", "long_short")


def moving_average(
    bars: pl.LazyFrame | pl.DataFrame,
    fasts: Iterable[int],
    slows: Iterable[int],
    *,
    sides: Iterable[str] = SIDES,
    fee: float = backtest.TAKER_FEE,
) -> pl.DataFrame:
    """Every `fast < slow` crossover, per side: long while the fast mean is above the slow one.

    `long_flat` is flat otherwise; `long_short` is short. Both are null until
    the slow mean has `slow` bars, so a trial never trades on a partial mean.
    """
    frames = []
    for side in sides:
        if side not in SIDES:
            raise ValueError(f"side={side!r} is not one of {', '.join(SIDES)}")
        for fast in fasts:
            for slow in slows:
                if fast >= slow:
                    continue
                spread = pl.col("close").rolling_mean(fast) - pl.col("close").rolling_mean(slow)
                position = spread.sign()
                if side == "long_flat":
                    position = position.clip(lower_bound=0)
                frames.append(_trial(bars, position, f"ma {fast}/{slow} {side}", fee))
    return pl.concat(frames)


def momentum(
    bars: pl.LazyFrame | pl.DataFrame,
    lookbacks: Iterable[int],
    *,
    fee: float = backtest.TAKER_FEE,
) -> pl.DataFrame:
    """Long/short on the sign of the trailing return over `lookback` bars, rebalanced every bar."""
    frames = []
    for lookback in lookbacks:
        position = (pl.col("close") / pl.col("close").shift(lookback) - 1).sign()
        frames.append(_trial(bars, position, f"mom {lookback}", fee))
    return pl.concat(frames)


def summary(frame: pl.DataFrame, periods_per_year: float | None = None) -> pl.DataFrame:
    """One row per `(trial, ticker)`: per-period Sharpe, periods, skewness, raw kurtosis, turnover, total.

    A trial with fewer than 30 net returns keeps its row with a null Sharpe:
    it was run, so it counts toward N.
    """
    net = pl.col("net").drop_nulls()
    out = (
        frame.group_by("trial", "ticker", maintain_order=True)
        .agg(
            net.count().alias("periods"),
            (net.mean() / net.std(ddof=1)).alias("_sr"),
            net.skew(bias=True).alias("skew"),
            net.kurtosis(fisher=False, bias=True).alias("kurt"),
            ((net + 1).product() - 1).alias("total_net"),
            ((pl.col("gross").drop_nulls() + 1).product() - 1).alias("total_gross"),
        )
        .with_columns(
            pl.when((pl.col("periods") >= MIN_RETURNS) & pl.col("_sr").is_finite()).then(pl.col("_sr")).alias("sharpe")
        )
        .drop("_sr")
    )
    if periods_per_year is not None:
        out = out.with_columns((pl.col("sharpe") * periods_per_year**0.5).alias("sharpe_annual"))
    return out


def matrix(frame: pl.DataFrame) -> pl.DataFrame:
    """Every `(trial, ticker)` as a column, `"<trial> | <ticker>"`, on `ts`, where all have a net return.

    PBO compares strategies over the same periods, so a warm-up, or a day
    one trial has no return for, is dropped for all of them rather than
    filled.
    """
    return (
        frame.drop_nulls("net")
        .with_columns(pl.concat_str("trial", pl.lit(" | "), "ticker").alias("_column"))
        .pivot(on="_column", index="ts", values="net", aggregate_function=None)
        .sort("ts")
        .drop_nulls()
    )


def _trial(bars, position: pl.Expr, name: str, fee: float) -> pl.DataFrame:
    r = backtest.returns(bars, position, fee=fee)
    return r.select(pl.lit(name).alias("trial"), "ticker", "ts", "position", "gross", "net")
