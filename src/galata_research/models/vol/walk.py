"""Rolling-origin variance forecasts: every row says which bars fitted it, and none after its origin did.

Rows are the non-null returns (holes bridged, roadmap D5); row i of the kept
frame is arch's index i. For each refit, one model is built on the rows up to
the block's last origin, fitted on the refit's window, and forecast from the
refit onward with its parameters fixed and the filter run over the observed
returns. arch's row i uses data through i (checked: a shock planted at index
600 first moves row 600's one-step forecast).
"""

import numpy as np
import polars as pl

from ... import timeseries, utils
from ..._errors import Refused
from .. import _arch
from .._arch import DISTS, SCALE, SIMULATED
from . import custom
from .garch import MODELS

_OUT = ("ticker", "ts", "close_ts", "h", "target_ts", "variance", "cum_variance", "fitted_through", "fit_from", "refit", "after_gap", "filtered")


def walk_forward(
    returns: pl.LazyFrame | pl.DataFrame,
    *,
    model: str,
    dist: str = "t",
    split,
    window: int | str = "expanding",
    every: int = 1,
    horizons=(1,),
    factors: pl.DataFrame | None = None,
    column: str = "return",
    simulations: int = 1000,
    seed: int = 0,
    min_obs: int = 500,
) -> pl.DataFrame:
    """Variance forecasts from every origin at or after `split`, one row per (origin, h in `horizons`).

    `ticker, ts, close_ts, h, target_ts, variance, cum_variance, fitted_through,
    fit_from, refit, after_gap, filtered`, in squared return units per bar
    (`filtered` is always false here; HAR's insanity filter sets it). `variance` is
    E[σ²] of the h-th bar after the origin; `cum_variance` sums bars 1…h, the
    figure realized variance over those bars is scored against (Andersen,
    Bollerslev, Christoffersen and Diebold 2006). `target_ts` is the `ts` of
    the bar opening (h − 1) bar-widths after the origin's close.

    Parameters are re-estimated every `every` origins on an `"expanding"` or
    rolling window of that many returns (`gr.timeseries.walk_forward_origins`)
    and held fixed between refits. EGARCH and APARCH have no analytic forecast
    beyond one step and are simulated there, with `simulations` paths from
    the fitted distribution seeded from `seed` and the refit.

    With `factors` (from `gr.timeseries.seasonal_factors`, fitted on a window
    ending no later than `split`), the model is fitted to returns divided by
    their cell's factor, and each target bar's variance is multiplied back by
    its cell's factor².
    """
    if model not in MODELS:
        raise Refused(f"model={model!r} is not one of {', '.join(MODELS)}")
    if dist not in DISTS:
        raise Refused(f"dist={dist!r} is not one of {', '.join(DISTS)}")
    custom.check_model(model, dist)
    hs = sorted({int(h) for h in horizons})
    if not hs or hs[0] < 1:
        raise Refused(f"horizons={list(horizons)}: each must be a whole number of bars ≥ 1")
    utils.require(returns, ("ticker", "ts", "close_ts", column), "make returns with gr.timeseries.returns")
    frame = utils.lazy(returns).sort("ts").collect()
    tickers = frame["ticker"].unique().sort().to_list()
    if len(tickers) != 1:
        raise Refused(f"walk forward one ticker at a time; these returns hold {', '.join(tickers) or 'none'}")
    at = utils.instant("split", split)
    if factors is not None:
        ends = factors.filter(pl.col("ticker") == tickers[0])["fit_end"]
        if ends.len() == 0:
            raise Refused(f"no factors for {tickers[0]}")
        if ends.max().timestamp() * 1_000_000 > at:
            raise Refused(f"the factors were fitted through {ends.max()}, after the split {split}: that is lookahead")
        if column != "return":
            raise Refused(f"factors deseasonalise raw returns; column={column!r} would divide twice")
        frame = timeseries.deseasonalize(frame, factors)
        column = "deseasonalized"
    kept = frame.with_columns(pl.col(column).is_null().shift(1).fill_null(False).alias("after_gap")).drop_nulls(column)
    kept = kept.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(False).otherwise(pl.col("after_gap")).alias("after_gap"))
    schedule = timeseries.walk_forward_origins(kept.select("ticker", "ts", "close_ts"), split, window=window, every=every)
    width = kept["close_ts"][0] - kept["ts"][0]
    first = kept.height - schedule.height
    refits = [first + i for i, r in enumerate(schedule["refit"].to_list()) if r]
    if (refits[0] + 1 if window == "expanding" else window) < min_obs:
        raise Refused(f"the first refit's window holds {refits[0] + 1 if window == 'expanding' else window} returns, under min_obs={min_obs}")
    y = _arch.values(kept, column)
    H = hs[-1]
    blocks = []
    for n, r in enumerate(refits):
        stop = refits[n + 1] if n + 1 < len(refits) else kept.height
        first_obs = 0 if window == "expanding" else r - window + 1
        if model in custom.MODELS:
            fitted = y[first_obs : r + 1]
            p = custom.estimate(model, fitted)["params"]
            blocks.append(
                custom.forecast(model, p, y[first_obs:stop], start=r - first_obs, horizon=H, init=float(np.var(fitted)), simulations=simulations, seed=seed + r)
            )
            continue
        res = _arch.model(y[:stop], model, dist, seed=seed + r).fit(disp="off", first_obs=first_obs, last_obs=r + 1)
        blocks.append(_arch.forecast(res, start=r, horizon=H, simulate=model in SIMULATED, simulations=simulations))
    variances = np.vstack(blocks) / SCALE**2
    if factors is not None:
        variances = variances * _target_factor2(schedule, factors, width, H)
    cumulative = np.cumsum(variances, axis=1)
    base = schedule.select("ticker", "ts", "close_ts", "fitted_through", "fit_from", "refit").with_columns(
        kept["after_gap"].slice(first).alias("after_gap")
    )
    rows = []
    for h in hs:
        rows.append(
            base.with_columns(
                pl.lit(h, pl.Int64).alias("h"),
                (pl.col("close_ts") + width * (h - 1)).alias("target_ts"),
                pl.Series("variance", variances[:, h - 1]),
                pl.Series("cum_variance", cumulative[:, h - 1]),
                pl.lit(False).alias("filtered"),
            )
        )
    return pl.concat(rows).select(_OUT).sort("close_ts", "h")


def _target_factor2(schedule: pl.DataFrame, factors: pl.DataFrame, width, H: int) -> np.ndarray:
    """factor² of each origin's target bars' cells, (origins × H)."""
    table = factors.select("ticker", "weekday", "hour", "factor")
    cols = []
    for h in range(1, H + 1):
        targets = schedule.select("ticker", (pl.col("close_ts") + width * (h - 1)).alias("ts"))
        joined = targets.with_columns(pl.col("ts").dt.weekday().alias("weekday"), pl.col("ts").dt.hour().alias("hour")).join(
            table, on=["ticker", "weekday", "hour"], how="left", maintain_order="left"
        )
        if joined["factor"].null_count():
            raise Refused(f"no factor for {joined['factor'].null_count()} target bar(s) at h={h}; fit factors on every cell the bars occupy")
        cols.append(joined["factor"].to_numpy() ** 2)
    return np.column_stack(cols)
