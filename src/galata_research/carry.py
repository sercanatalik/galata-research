"""Funding as a trade: hedged carry, the persistence it needs, and the unhedged fade of extreme funding.

    day = gr.carry.daily(gr.reference.premium(...), gr.reference.funding(...))
    rules = gr.carry.rules(day)                      # the 13 registered carry rules, every one
    gr.carry.persistence(day)                        # lag-one autocorrelation of daily funding
    fades = gr.carry.fades(bars, day, settled)       # the 8 registered fades, funding charged

**The hedged trade.** A position s held against the index earns the perp's
premium over the index as it moves, and pays the funding: per day held,
s × (Pₜ − Pₜ₋₁) / (1 + Pₜ₋₁) − s × Fₜ − fee × |Δs|, with P the premium index at
the close and F the day's settled funding. s = −1 (short the perp, long the
index) receives positive funding. The fee is both legs' per unit of
turnover. This is per unit of notional, the index stands in for a spot fill,
and a short spot leg's borrow is not charged (`planning/preregistered/trade-the-funding.md`).

**Days.** A day's premium is its last minute's close, kept only when all
1,440 minutes are held. Its funding is the sum of the settlements in
`(ts, close_ts]` (the one at midnight closes the day before), kept only when
their intervals cover its 24 hours. A day missing either is not in the frame:
it is a hole, nothing is earned across it, and the rules' windows restart
after it.

**No lookahead.** A position decided at a day's close reads that day's
funding, which settled at the close, and earlier days; it is held from the
next day. The same one shift as `gr.backtest.returns`.
"""

from collections.abc import Iterable
from math import atanh, sqrt
from statistics import NormalDist

import polars as pl

from . import indicators as ind
from . import studies, timeseries, utils
from ._errors import Refused
from .reference import funding_hours

# Binance's base taker fees, the perp's and spot's, paid on both legs of a hedged change.
PERP_FEE = 0.0005
SPOT_FEE = 0.0010
BOTH_LEGS = PERP_FEE + SPOT_FEE

_MINUTES = 1440
_COLUMNS = ["trial", "ticker", "ts", "close_ts", "position", "bar_return", "gross", "cost", "funding", "net"]


def daily(premium: pl.LazyFrame | pl.DataFrame, settled: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    """`ticker, ts, close_ts, premium, funding` per whole UTC day held in both, sorted.

    `premium` is 1m bars from `gr.reference.premium`, `settled` is
    `gr.reference.funding`.
    """
    utils.require(premium, ("ticker", "ts", "close"), "load the premium with gr.reference.premium")
    utils.require(settled, ("ticker", "ts", "rate", "interval_hours"), "load funding with gr.reference.funding")
    day = pl.col("ts").dt.truncate("1d")
    p = (
        utils.lazy(premium)
        .sort("ticker", "ts")
        .group_by("ticker", day.alias("day"))
        .agg(pl.col("close").last().alias("premium"), pl.col("ts").n_unique().alias("_minutes"))
        .filter(pl.col("_minutes") == _MINUTES)
        .drop("_minutes")
    )
    # A settlement at H closes the interval before it: the one at midnight is the previous day's.
    settles = pl.col("ts").dt.truncate("1h")
    f = (
        utils.lazy(settled)
        .unique(["ticker", "ts"])
        .group_by("ticker", (settles - pl.duration(microseconds=1)).dt.truncate("1d").alias("day"))
        .agg(pl.col("rate").sum().alias("funding"), pl.col("interval_hours").sum().alias("_hours"))
        .filter(pl.col("_hours") == 24)
        .drop("_hours")
    )
    return (
        p.join(f, on=["ticker", "day"], how="inner")
        .select("ticker", pl.col("day").alias("ts"), (pl.col("day") + pl.duration(days=1)).alias("close_ts"), "premium", "funding")
        .sort("ticker", "ts")
        .collect()
    )


def returns(day: pl.DataFrame, position: pl.Expr, *, fee: float = BOTH_LEGS) -> pl.DataFrame:
    """Per day, the hedged position's `bar_return` (the basis move), `gross`, `cost`, `funding` and `net`.

    `position` is evaluated per contiguous stretch of `day` (`gr.indicators.add`),
    so a window never spans a missing day. `position` in the result is what
    was held through the day: the one decided at the close before.
    """
    if fee < 0:
        raise Refused(f"fee={fee} is negative")
    utils.require(day, ("ticker", "ts", "close_ts", "premium", "funding"), "build the days with gr.carry.daily")
    framed = ind.add(day, _decided=position)
    prev = pl.col("premium").shift(1).over("ticker")
    return (
        framed.with_columns(
            pl.col("_decided").shift(1).over("ticker").alias("_held"),
            timeseries.contiguous().alias("_next"),
            ((pl.col("premium") - prev) / (1 + prev)).alias("_basis"),
        )
        .with_columns((pl.col("_held") - pl.col("_held").shift(1).over("ticker").fill_null(0.0)).abs().alias("_turn"))
        .select(
            "ticker",
            "ts",
            "close_ts",
            pl.col("_held").alias("position"),
            pl.when(pl.col("_next")).then(pl.col("_basis")).alias("bar_return"),
            pl.when(pl.col("_next")).then(pl.col("_held") * pl.col("_basis")).alias("gross"),
            pl.when(pl.col("_next")).then(pl.col("_turn") * fee).alias("cost"),
            pl.when(pl.col("_next")).then(pl.col("_held") * pl.col("funding")).alias("funding"),
        )
        .with_columns((pl.col("gross") - pl.col("cost") - pl.col("funding")).alias("net"))
    )


def trial(day: pl.DataFrame, position: pl.Expr, name: str, *, fee: float = BOTH_LEGS) -> pl.DataFrame:
    """One named hedged trial, in `gr.studies`' columns plus `funding`."""
    return returns(day, position, fee=fee).with_columns(pl.lit(name).alias("trial")).select(_COLUMNS)


def rules(
    day: pl.DataFrame,
    *,
    lookbacks: Iterable[int] = (1, 7, 30),
    hurdles: Iterable[float] = (0.0, 0.0003),
    sides: Iterable[str] = ("both", "short_only"),
    fee: float = BOTH_LEGS,
) -> pl.DataFrame:
    """The registered carry rules: `always` (short the perp every day), and per lookback, hurdle and side.

    `carry d h both` is −sign(F̄_d) when |F̄_d| > h, else flat; `short_only`
    is −1 when F̄_d > h, else flat. F̄_d is the mean daily funding over the
    last d days, null until d days exist.
    """
    frames = [trial(day, pl.lit(-1.0), "always", fee=fee)]
    for side in sides:
        if side not in ("both", "short_only"):
            raise Refused(f"side={side!r} is not one of both, short_only")
        for d in lookbacks:
            for h in hurdles:
                mean = ind.sma(d, "funding")
                if side == "both":
                    position = pl.when(mean.abs() > h).then(-mean.sign()).when(mean.is_not_null()).then(0.0)
                else:
                    position = pl.when(mean > h).then(-1.0).when(mean.is_not_null()).then(0.0)
                frames.append(trial(day, position, f"carry {d} {h:g} {side}", fee=fee))
    return pl.concat(frames)


def persistence(day: pl.DataFrame) -> pl.DataFrame:
    """Per ticker: daily funding's lag-one autocorrelation over consecutive days, `n, rho, z, p`.

    z is Fisher's, atanh(ρ)·√(n − 3); p is one-sided, against ρ ≤ 0.
    """
    utils.require(day, ("ticker", "ts", "close_ts", "funding"), "build the days with gr.carry.daily")
    pairs = (
        day.sort("ticker", "ts")
        .with_columns(pl.col("funding").shift(1).over("ticker").alias("_prev"), timeseries.contiguous().alias("_next"))
        .filter(pl.col("_next"))
    )
    rows = []
    for (ticker,), part in pairs.group_by("ticker", maintain_order=True):
        n = part.height
        rho = part.select(pl.corr("funding", "_prev")).item() if n > 3 else None
        z = atanh(rho) * sqrt(n - 3) if rho is not None and abs(rho) < 1 else None
        rows.append({"ticker": ticker, "n": n, "rho": rho, "z": z, "p": None if z is None else 1 - NormalDist().cdf(z)})
    return pl.DataFrame(rows, schema={"ticker": pl.String, "n": pl.Int64, "rho": pl.Float64, "z": pl.Float64, "p": pl.Float64})


def fades(
    bars: pl.LazyFrame | pl.DataFrame,
    day: pl.DataFrame,
    settled: pl.LazyFrame | pl.DataFrame,
    *,
    windows: Iterable[int] = (30, 90),
    thresholds: Iterable[float] = (1.5, 2.0),
    sides: Iterable[str] = ("long_short", "long_flat"),
    fee: float = PERP_FEE,
) -> pl.DataFrame:
    """The registered fades: an unhedged perp position against extreme daily funding, funding charged.

    z is daily funding's z-score over w days. −1 from z > k until z < 0.5,
    +1 from z < −k until z > −0.5; `long_flat` is flat for −1. `bars` are
    the perp's daily bars (`ticker, ts, close_ts, close`), joined to `day`'s
    funding on `ts`; a bar with no funding day has no z. Each trial's return
    is `gr.backtest.returns` with `settled` charged hour by hour.
    """
    utils.require(bars, ("ticker", "ts", "close_ts", "close"), "load daily bars of the perp")
    joined = utils.lazy(bars).join(day.lazy().select("ticker", "ts", "funding"), on=["ticker", "ts"], how="left").collect()
    hours = funding_hours(settled)
    frames = []
    for side in sides:
        for w in windows:
            for k in thresholds:
                z = ind.zscore(w, "funding")
                position = ind.hold(z < -k, z > -0.5, z > k, z < 0.5)
                if side == "long_flat":
                    position = position.clip(lower_bound=0)
                elif side != "long_short":
                    raise Refused(f"side={side!r} is not one of long_short, long_flat")
                framed = ind.add(joined, _position=position)
                frames.append(studies.trial(framed, pl.col("_position"), f"fade {w} {k:g} {side}", fee=fee, funding=hours))
    return pl.concat(frames)
