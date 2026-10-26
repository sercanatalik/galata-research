"""Short- and long-term momentum as volatility-scaled z's, known at each bar's close and never spanning a hole.

    f = gr.momentum.features(bars_1m, decision="5m")           # z_15m … z_4w, sigma, after_gap
    e = gr.momentum.encoded(f, "score")                         # e_15m … e_4w, interaction

A lookback L's z is its log return over the volatility a random walk would
give it: z_L = (log close_t − log close_{t−L}) / (σ_t·√L), with L in decision
bars. σ_t is per decision bar and deseasonalised: each bar's realized variance,
the sum of its fine bars' squared log returns, is divided by the square of its
`hour_x_weekday` factor, and σ is the square root of an EWMA of that, one day's
half-life by default. The factors are refit every Monday 00:00 UTC on all
returns closed before it, so every quantity on a row is known at its close; a
test appends bars and checks that no earlier row moves.

**The ladder.** Short: 15m, 1h, 4h. Long: 1d, 3d, 1w, 4w. Moskowitz, Ooi and
Pedersen (2012) is the sign of one lookback's return; Baz, Granger, Harvey, Le
Roux and Rattray (2015) is the same signal scaled by volatility, over several
spans, and squashed by z·exp(−z²/4)/0.89 so that the strongest trends are
not the largest positions.

**Holes.** As in `gr.indicators`, everything is evaluated per contiguous
stretch of decision bars: a lookback whose window holds a hole is null, the
EWMA starts again after one, and nothing is interpolated. `after_gap` is true
on a row of a later stretch while any of its features is still null.

**Encodings.** `binary` is sign(z) with sign(0) = +1, `ternary` is 0 inside
|z| < 0.5, `score` is the squashed z. Each encoding is a trial, and each
lookback set too: the ladder is fixed so that it is not searched.

Bars with neither volume nor trades (Hyperliquid's bars before 2023-02-26) are
prices nothing traded at: `features` drops them, and `untraded` counts them.
"""

import math
import re

import polars as pl

from . import timeseries, utils
from ._errors import Refused

SHORT = ("15m", "1h", "4h")
LONG = ("1d", "3d", "1w", "4w")
LADDER = (*SHORT, *LONG)
ENCODINGS = ("binary", "ternary", "score")
TERNARY_BAND = 0.5

_UNIT_US = {
    "m": 60_000_000,
    "h": 3_600_000_000,
    "d": 86_400_000_000,
    "w": 604_800_000_000,
}
_STRETCH = "_stretch"


def micros(interval: str) -> int:
    """An interval such as `5m`, `4h`, `1d` or `1w` in microseconds."""
    m = re.fullmatch(r"(\d+)([mhdw])", interval)
    if not m or int(m.group(1)) < 1:
        raise Refused(f"interval {interval!r} is not a whole number of minutes, hours, days or weeks (5m, 4h, 1d, 1w)")
    return int(m.group(1)) * _UNIT_US[m.group(2)]


def bars_in(interval: str, decision: str) -> int:
    """How many `decision` bars make `interval`, refused unless it is a whole number of them."""
    span, step = micros(interval), micros(decision)
    if span % step:
        raise Refused(f"{interval} is not a whole number of {decision} bars")
    return span // step


def _traded() -> pl.Expr:
    return ~((pl.col("volume") == 0) & (pl.col("trade_count") == 0))


def untraded(bars: pl.LazyFrame | pl.DataFrame) -> int:
    """The bars with zero volume and zero trades, which `features` drops before computing anything."""
    utils.require(
        bars,
        ("volume", "trade_count"),
        "load candles with their volume and trade_count",
    )
    return utils.lazy(bars).filter(~_traded()).select(pl.len()).collect().item()


def _width(bars: pl.LazyFrame) -> int:
    widths = bars.select((pl.col("close_ts") - pl.col("ts")).dt.total_microseconds().unique()).collect().to_series().to_list()
    if len(widths) != 1:
        raise Refused(f"the bars are of several widths ({sorted(widths)} µs); pass one interval")
    return widths[0]


def _decision_bars(fine: pl.LazyFrame, width: int, decision: str) -> pl.LazyFrame:
    """Decision bars with `rv`, the sum of their fine bars' squared log returns, null unless every one is there."""
    step = micros(decision)
    r = fine.sort("ticker", "ts").with_columns(
        pl.when(timeseries.contiguous()).then(timeseries.log_return()).alias("_r"),
        pl.col("ts").dt.truncate(decision).alias("_bucket"),
    )
    if step == width:
        return r.select("ticker", "ts", "close_ts", "close", pl.col("_r").pow(2).alias("rv"))
    if decision not in timeseries.RESAMPLE:
        raise Refused(f"decision bars of {decision} cannot be resampled; use one of {', '.join(timeseries.RESAMPLE)}")
    expected = step // width
    rv = r.group_by("ticker", "_bucket").agg(pl.when(pl.col("_r").count() == expected).then(pl.col("_r").pow(2).sum()).alias("rv"))
    coarse = timeseries.resample(fine, decision).lazy().select("ticker", "ts", "close_ts", "close")
    return coarse.join(rv, left_on=["ticker", "ts"], right_on=["ticker", "_bucket"], how="left")


def weekly_factors(returns: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    """`ticker, week, weekday, hour, factor`: each week's `hour_x_weekday` factors, fitted on every return closed before its Monday.

    The same table `gr.timeseries.seasonal_factors(returns, fit=(first, monday),
    by="hour_x_weekday", stat="mean_abs")` gives for each Monday, built from
    running sums so that six years of 5m bars are one pass. A week whose past
    holds no return for some hour or weekday has null factors.
    """
    utils.require(
        returns,
        ("ticker", "ts", "close_ts", "return"),
        "make returns with gr.timeseries.returns",
    )
    rows = (
        utils.lazy(returns)
        .drop_nulls("return")
        .with_columns(
            pl.col("ts").dt.weekday().alias("weekday"),
            pl.col("ts").dt.hour().alias("hour"),
            # A return belongs to the first Monday at or after its close: the fit that may read it.
            (pl.col("close_ts") - pl.duration(microseconds=1)).dt.truncate("1w").alias("_week"),
        )
        .with_columns((pl.col("_week") + pl.duration(weeks=1)).alias("_usable"))
    )
    weeks = utils.lazy(returns).select("ticker", pl.col("ts").dt.truncate("1w").alias("week")).unique()
    cells = rows.select("ticker", "weekday", "hour").unique()

    def running(key: str, out: str) -> pl.LazyFrame:
        sums = rows.group_by("ticker", "_usable", key).agg(pl.col("return").abs().sum().alias("_a"), pl.len().alias("_n"))
        grid = weeks.join(sums.select("ticker", key).unique(), on="ticker")
        return (
            grid.join(
                sums,
                left_on=["ticker", "week", key],
                right_on=["ticker", "_usable", key],
                how="left",
            )
            .sort("ticker", key, "week")
            .with_columns(
                pl.col("_a").fill_null(0).cum_sum().over("ticker", key),
                pl.col("_n").fill_null(0).cum_sum().over("ticker", key),
            )
            .with_columns(pl.when(pl.col("_n") > 0).then(pl.col("_a") / pl.col("_n")).alias("_s"))
            .with_columns(
                pl.when(pl.col("_s").is_null().any().over("ticker", "week").not_()).then(pl.col("_s") / pl.col("_s").mean().over("ticker", "week")).alias(out)
            )
            .select("ticker", "week", key, out)
        )

    table = (
        weeks.join(cells, on="ticker")
        .join(running("hour", "_h"), on=["ticker", "week", "hour"], how="left")
        .join(running("weekday", "_d"), on=["ticker", "week", "weekday"], how="left")
        .with_columns((pl.col("_h") * pl.col("_d")).alias("_s"))
        .with_columns(
            pl.when(pl.col("_s").is_null().any().over("ticker", "week").not_())
            .then(pl.col("_s") / pl.col("_s").pow(2).mean().over("ticker", "week").sqrt())
            .alias("factor")
        )
    )
    return table.select("ticker", "week", "weekday", "hour", "factor").sort("ticker", "week", "weekday", "hour").collect()


def features(
    bars: pl.LazyFrame | pl.DataFrame,
    *,
    decision: str = "5m",
    half_life: str = "1d",
    ladder: tuple[str, ...] = LADDER,
) -> pl.DataFrame:
    """`ticker, ts, close_ts, close, rv, factor, sigma, z_<L>…, after_gap` per decision bar, from bars of one finer or equal width.

    `bars` are candles (`gr.reference.candles` or `gr.market.candles`) with
    `volume` and `trade_count`; untraded bars are dropped first (`untraded`
    counts them). Refused when the decision bar or a lookback is not a whole
    number of the bars under it.
    """
    utils.require(
        bars,
        ("ticker", "ts", "close_ts", "close", "volume", "trade_count"),
        "load candles with gr.reference.candles",
    )
    fine = utils.lazy(bars).filter(_traded())
    width = _width(fine)
    step = micros(decision)
    if step < width or step % width:
        raise Refused(f"a {decision} decision bar is not a whole multiple of the {width // 60_000_000}m bars given")
    lookbacks = {name: bars_in(name, decision) for name in ladder}
    span = bars_in(half_life, decision)

    coarse = _decision_bars(fine, width, decision).sort("ticker", "ts")
    returns = coarse.with_columns(pl.when(timeseries.contiguous()).then(timeseries.log_return()).alias("return")).select("ticker", "ts", "close_ts", "return")
    factors = weekly_factors(returns).lazy()
    framed = (
        coarse.with_columns(
            pl.col("ts").dt.truncate("1w").alias("week"),
            pl.col("ts").dt.weekday().alias("weekday"),
            pl.col("ts").dt.hour().alias("hour"),
        )
        .join(factors, on=["ticker", "week", "weekday", "hour"], how="left")
        .sort("ticker", "ts")
        .with_columns((~timeseries.contiguous()).fill_null(True).cum_sum().over("ticker").alias(_STRETCH))
        .with_columns(
            (pl.col("rv") / pl.col("factor").pow(2)).ewm_mean(half_life=span, adjust=False, min_samples=span).sqrt().over("ticker", _STRETCH).alias("sigma")
        )
        .with_columns(
            *[(pl.col("close").log().diff(n).over("ticker", _STRETCH) / (pl.col("sigma") * math.sqrt(n))).alias(f"z_{name}") for name, n in lookbacks.items()]
        )
    )
    zs = [f"z_{name}" for name in lookbacks]
    return (
        framed.with_columns(((pl.col(_STRETCH) > 1) & pl.any_horizontal(pl.col(zs).is_null())).alias("after_gap"))
        .select(
            "ticker",
            "ts",
            "close_ts",
            "close",
            "rv",
            "factor",
            "sigma",
            *zs,
            "after_gap",
        )
        .collect()
    )


def encode(z: pl.Expr, how: str) -> pl.Expr:
    """One z encoded: `binary` sign(z) with sign(0) = +1, `ternary` 0 inside |z| < 0.5, `score` z·exp(−z²/4)/0.89 (Baz et al. 2015)."""
    if how == "binary":
        return pl.when(z.is_null()).then(None).when(z >= 0).then(1.0).otherwise(-1.0)
    if how == "ternary":
        return pl.when(z.is_null()).then(None).when(z.abs() < TERNARY_BAND).then(0.0).when(z > 0).then(1.0).otherwise(-1.0)
    if how == "score":
        return z * (-(z.pow(2)) / 4).exp() / 0.89
    raise Refused(f"encoding {how!r} is not one of {', '.join(ENCODINGS)}")


def encoded(frame: pl.DataFrame, how: str) -> pl.DataFrame:
    """The features with `e_<L>` per lookback, in one encoding, and `interaction`: the mean short `e` times the mean long `e`."""
    names = [c.removeprefix("z_") for c in frame.columns if c.startswith("z_")]
    short = [f"e_{n}" for n in names if n in SHORT]
    long = [f"e_{n}" for n in names if n in LONG]
    if not short or not long:
        raise Refused(f"the interaction needs a short ({', '.join(SHORT)}) and a long ({', '.join(LONG)}) lookback; the frame has {', '.join(names)}")
    out = frame.with_columns(*[encode(pl.col(f"z_{n}"), how).alias(f"e_{n}") for n in names])
    return out.with_columns((pl.mean_horizontal(short) * pl.mean_horizontal(long)).alias("interaction"))
