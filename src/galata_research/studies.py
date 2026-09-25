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


DONCHIAN_LOOKBACKS = (5, 10, 20, 30, 60, 90, 150, 250, 360)


def donchian_ensemble(
    bars: pl.LazyFrame | pl.DataFrame,
    *,
    lookbacks: Iterable[int] = DONCHIAN_LOOKBACKS,
    target_vol: float = 0.25,
    vol_window: int = 90,
    max_leverage: float = 1.0,
    periods_per_year: int = 365,
    sized: bool = True,
    fee: float = backtest.TAKER_FEE,
) -> pl.DataFrame:
    """Zarattini, Pagani and Barbon's Donchian ensemble, as pre-registered.

    `planning/preregistered/donchian-ensemble.md` freezes these rules. Per
    lookback L, at each close t, with `high`, `low` and `mid` taken over the
    previous L closes (t is excluded):
    1. if open and the close is below the stop set by earlier closes, close;
    2. if flat and the close is above `high`, open, with the stop at `mid`;
    3. if open, ratchet the stop to `max(stop, mid)`.
    The signal is the fraction of lookbacks open. The position is the signal
    times `min(max_leverage, target_vol / σ)`, with σ the annualised
    deviation of the last `vol_window` returns, or the signal alone when
    `sized=False`. Long-only. The position acts from the next bar.
    """
    lookbacks = list(lookbacks)
    frame = bars.lazy().sort("ticker", "ts").collect()
    positions = []
    for (ticker,), group in frame.group_by("ticker", maintain_order=True):
        closes = group["close"].to_list()
        signal = _donchian_signal(closes, lookbacks)
        if sized:
            rets = [None] + [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
            scale = []
            for i in range(len(closes)):
                window = rets[max(1, i - vol_window + 1) : i + 1]
                if len(window) < vol_window:
                    scale.append(None)
                    continue
                mean = sum(window) / len(window)
                sd = (sum((r - mean) ** 2 for r in window) / (len(window) - 1)) ** 0.5 * periods_per_year**0.5
                scale.append(min(max_leverage, target_vol / sd) if sd > 0 else max_leverage)
            signal = [None if k is None else s * k for s, k in zip(signal, scale)]
        positions.append(group.select("ticker", "ts").with_columns(pl.Series("_position", signal, dtype=pl.Float64)))
    with_positions = frame.join(pl.concat(positions), on=["ticker", "ts"], how="left")
    name = f"donchian {'sized' if sized else 'unsized'}"
    return _trial(with_positions, pl.col("_position"), name, fee)


def _donchian_signal(closes: list[float], lookbacks: list[int]) -> list[float]:
    """The fraction of lookbacks open after each close, by the three registered steps."""
    open_ = {L: False for L in lookbacks}
    stop = {L: 0.0 for L in lookbacks}
    out = []
    for t, close in enumerate(closes):
        for L in lookbacks:
            if t < L:
                continue
            previous = closes[t - L : t]
            high, low = max(previous), min(previous)
            mid = (high + low) / 2
            if open_[L] and close < stop[L]:
                open_[L] = False
            elif not open_[L] and close > high:
                open_[L], stop[L] = True, mid
            if open_[L]:
                stop[L] = max(stop[L], mid)
        out.append(sum(open_.values()) / len(lookbacks))
    return out


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
