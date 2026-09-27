"""CARR on the walk-forward schedule: the bar's range forecast, and its variance by the Brownian range constant."""

import numpy as np
import polars as pl

from ... import timeseries, utils
from ..._errors import Refused
from .._arch import SCALE
from . import custom

_OUT = ("ticker", "ts", "close_ts", "h", "target_ts", "variance", "cum_variance", "fitted_through", "fit_from", "refit", "after_gap", "filtered", "nu")


def carr(
    bars: pl.LazyFrame | pl.DataFrame,
    *,
    split,
    window: int | str = "expanding",
    every: int = 1,
    horizons=(1,),
    min_obs: int = 500,
) -> pl.DataFrame:
    """CARR(1,1) (Chou 2005) walked forward: `walk_forward`'s columns, from one ticker's bars.

    λₜ = ω + αRₜ₋₁ + βλₜ₋₁ on Rₜ = ln(Hₜ/Lₜ), by exponential QMLE, refitted
    every `every` origins on an expanding or rolling window of bars. A bar's
    variance is (λ / √(8/π))², √(8/π) being the mean range of a driftless
    Brownian bar per unit σ (measured in the tests, a little lower under
    discrete sampling). Zero ranges (flat bars) are kept; a high below the low
    is refused. `after_gap` marks the first bar after a hole.
    """
    hs = sorted({int(h) for h in horizons})
    if not hs or hs[0] < 1:
        raise Refused(f"horizons={list(horizons)}: each must be a whole number of bars ≥ 1")
    utils.require(bars, ("ticker", "ts", "close_ts", "high", "low"), "load bars with gr.market.candles")
    frame = utils.lazy(bars).sort("ticker", "ts").with_columns((~timeseries.contiguous().fill_null(True)).alias("after_gap")).collect()
    tickers = frame["ticker"].unique().sort().to_list()
    if len(tickers) != 1:
        raise Refused(f"walk forward one ticker at a time; these bars hold {', '.join(tickers) or 'none'}")
    bad = frame.filter(pl.col("high") < pl.col("low"))
    if bad.height:
        raise Refused(f"{bad.height} bar(s) have a high below the low, first at {bad['ts'][0]}")
    r = (frame["high"] / frame["low"]).log().to_numpy() * SCALE
    schedule = timeseries.walk_forward_origins(frame.select("ticker", "ts", "close_ts"), split, window=window, every=every)
    n = r.size
    first = n - schedule.height
    refits = [first + i for i, flag in enumerate(schedule["refit"].to_list()) if flag]
    if (refits[0] + 1 if window == "expanding" else window) < min_obs:
        raise Refused(f"the first refit's window holds {refits[0] + 1 if window == 'expanding' else window} bars, under min_obs={min_obs}")
    H = hs[-1]
    blocks = []
    for k, at in enumerate(refits):
        stop = refits[k + 1] if k + 1 < len(refits) else n
        lo = 0 if window == "expanding" else at - window + 1
        fitted = r[lo : at + 1]
        p = custom.estimate("carr", fitted)["params"]
        blocks.append(custom.forecast("carr", p, r[lo:stop], start=at - lo, horizon=H, init=float(np.mean(fitted)), simulations=0, seed=0))
    variances = np.vstack(blocks) / SCALE**2
    cumulative = np.cumsum(variances, axis=1)
    width = frame["close_ts"][0] - frame["ts"][0]
    base = schedule.select("ticker", "ts", "close_ts", "fitted_through", "fit_from", "refit").with_columns(
        frame["after_gap"].slice(first).alias("after_gap")
    )
    rows = [
        base.with_columns(
            pl.lit(h, pl.Int64).alias("h"),
            (pl.col("close_ts") + width * (h - 1)).alias("target_ts"),
            pl.Series("variance", variances[:, h - 1]),
            pl.Series("cum_variance", cumulative[:, h - 1]),
            pl.lit(False).alias("filtered"),
            pl.lit(None, pl.Float64).alias("nu"),
        )
        for h in hs
    ]
    return pl.concat(rows).select(_OUT).sort("close_ts", "h")
