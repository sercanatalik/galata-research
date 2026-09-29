"""HAR, SHAR and HARQ: realized variance forecast from its own recent averages, on the walk-forward schedule.

Levels, one direct regression per horizon and target (Corsi 2009; Bollerslev,
Patton and Quaedvlieg 2016). At a refit at bucket r, bucket s trains the
horizon-h regressions only if s + h ≤ r: its whole target is known at r's
close. A forecast outside its training targets' range is replaced by their
mean (the insanity filter) and says so in `filtered`.
"""

from datetime import timedelta

import numpy as np
import polars as pl

from ... import timeseries, utils
from ..._errors import Refused

MODELS = ("har", "shar", "harq", "harj", "harcj", "hartcj")
_NEEDS = {
    "har": ("rv",),
    "shar": ("rv", "rs_plus", "rs_minus"),
    "harq": ("rv", "rq"),
    "harj": ("rv", "bv"),
    "harcj": ("rv", "c_bns", "j_bns"),
    "hartcj": ("rv", "c_tcj", "j_tcj"),
}
_MADE_BY = {"harj": "gr.models.vol.realized_jumps", "harcj": "gr.models.vol.realized_jumps", "hartcj": "gr.models.vol.realized_jumps"}
_LAGS = {timedelta(days=1): (1, 7, 30), timedelta(hours=4): (1, 6, 42)}
_OUT = ("ticker", "ts", "close_ts", "h", "target_ts", "variance", "cum_variance", "fitted_through", "fit_from", "refit", "after_gap", "filtered", "nu")


def har(
    measures: pl.LazyFrame | pl.DataFrame,
    *,
    model: str = "har",
    split,
    window: int | str = "expanding",
    every: int = 1,
    horizons=(1,),
    lags: tuple[int, int, int] | None = None,
    estimator: str = "ols",
    min_obs: int = 250,
) -> pl.DataFrame:
    """Realized-variance forecasts from every origin at or after `split`, in `walk_forward`'s columns plus `filtered`.

    `measures` is one ticker's `gr.timeseries.realized_from` output;
    incomplete buckets are dropped and the recursion bridged (roadmap D5).

    - `har`: 1, RVₛ, the mean of the last w RV and of the last m (Corsi 2009).
    - `shar`: RS⁺ₛ and RS⁻ₛ in place of RVₛ (Patton and Sheppard 2015).
    - `harq`: adds RVₛ·(√RQₛ − the training rows' mean √RQ), so a noisily
      measured RV gets less weight (Bollerslev, Patton and Quaedvlieg 2016;
      that they demean is reported second-hand).

    - `harj`: adds max(RVₛ − BVₛ, 0), the untested jump (Andersen, Bollerslev
      and Diebold 2007, HAR-RV-J).
    - `harcj`: 1, Cₛ, Cʷ, Cᵐ, Jₛ, Jʷ, Jᵐ from the bipower test's split (ABD
      2007, HAR-RV-CJ); `hartcj` the same from the threshold split (Corsi,
      Pirino and Renò 2010, HAR-TCJ). Both take `realized_jumps` output; a
      training window with no jump is rank-deficient, and least squares'
      minimum-norm solution is then HAR's fit.

    `lags` default to (1, 7, 30) for daily buckets (crypto trades every day; a
    convention reported second-hand) and (1, 6, 42) for 4h (bar, day, week).
    `estimator="wls"` refits with weights 1 / the OLS fit (Patton and
    Sheppard). `min_obs` counts the first refit's training rows at h = 1.
    """
    if model not in MODELS:
        raise Refused(f"model={model!r} is not one of {', '.join(MODELS)}")
    if estimator not in ("ols", "wls"):
        raise Refused(f"estimator={estimator!r} is not one of ols, wls")
    hs = sorted({int(h) for h in horizons})
    if not hs or hs[0] < 1:
        raise Refused(f"horizons={list(horizons)}: each must be a whole number of buckets ≥ 1")
    utils.require(measures, ("ticker", "ts", "close_ts", *_NEEDS[model]), f"make measures with {_MADE_BY.get(model, 'gr.timeseries.realized_from')}")
    frame = utils.lazy(measures).sort("ts").collect()
    tickers = frame["ticker"].unique().sort().to_list()
    if len(tickers) != 1:
        raise Refused(f"forecast one ticker at a time; these measures hold {', '.join(tickers) or 'none'}")
    width = frame["close_ts"][0] - frame["ts"][0]
    if lags is None:
        if width not in _LAGS:
            raise Refused(f"buckets of {width} have no default lags; pass lags=(1, w, m)")
        lags = _LAGS[width]
    _, w, m = lags
    kept = frame.with_columns(pl.col("rv").is_null().shift(1).fill_null(False).alias("after_gap")).drop_nulls("rv")
    kept = kept.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(False).otherwise(pl.col("after_gap")).alias("after_gap"))
    usable = kept.slice(m - 1)
    rv = kept["rv"].to_numpy()
    n = rv.size
    weekly = np.array([rv[max(0, i - w + 1) : i + 1].mean() for i in range(n)])
    monthly = np.array([rv[max(0, i - m + 1) : i + 1].mean() for i in range(n)])
    schedule = timeseries.walk_forward_origins(usable.select("ticker", "ts", "close_ts"), split, window=window, every=every)
    first = n - schedule.height
    refits = [first + i for i, r in enumerate(schedule["refit"].to_list()) if r]
    H = hs[-1]
    point = np.full((schedule.height, H), np.nan)
    cumul = np.full((schedule.height, H), np.nan)
    flags = np.zeros((schedule.height, H), dtype=bool)
    cum_rv = np.concatenate([[0.0], np.cumsum(rv)])
    sqrt_rq = np.sqrt(kept["rq"].to_numpy()) if "rq" in kept.columns else np.ones(n)
    columns = {"rv": rv, "weekly": weekly, "monthly": monthly}
    for name in _NEEDS[model][1:]:
        columns[name] = kept[name].to_numpy()
    if model == "harj":
        columns["jump"] = np.maximum(rv - columns["bv"], 0.0)
    for name in ("c_bns", "j_bns", "c_tcj", "j_tcj"):
        if name in columns:
            x = columns[name]
            columns[name + "_w"] = np.array([x[max(0, i - w + 1) : i + 1].mean() for i in range(n)])
            columns[name + "_m"] = np.array([x[max(0, i - m + 1) : i + 1].mean() for i in range(n)])
    for k, r in enumerate(refits):
        stop = refits[k + 1] if k + 1 < len(refits) else n
        lo = m - 1 if window == "expanding" else max(m - 1, r - window + 1)
        origins = np.arange(r, stop)
        for h in hs:
            train = np.arange(lo, r - h + 1)  # s + h ≤ r: the target is known at r's close
            if h == 1 and k == 0 and train.size < min_obs:
                raise Refused(f"the first refit trains on {train.size} buckets, under min_obs={min_obs}")
            if train.size < 5:
                raise Refused(f"the refit at bucket {r} has {train.size} training rows at h={h}")
            q_mean = sqrt_rq[train].mean()
            x_train = _design(model, columns, sqrt_rq, q_mean, train)
            x_new = _design(model, columns, sqrt_rq, q_mean, origins)
            for target, out in ((rv[train + h], point), (cum_rv[train + h + 1] - cum_rv[train + 1], cumul)):
                beta = _estimate(x_train, target, estimator)
                f = x_new @ beta
                bad = (f < target.min()) | (f > target.max())
                f = np.where(bad, target.mean(), f)
                out[origins - first, h - 1] = f
                if out is point:
                    flags[origins - first, h - 1] = bad
    base = schedule.select("ticker", "ts", "close_ts", "fitted_through", "fit_from", "refit").with_columns(
        usable["after_gap"].slice(usable.height - schedule.height).alias("after_gap")
    )
    rows = [
        base.with_columns(
            pl.lit(h, pl.Int64).alias("h"),
            (pl.col("close_ts") + width * (h - 1)).alias("target_ts"),
            pl.Series("variance", point[:, h - 1]),
            pl.Series("cum_variance", cumul[:, h - 1]),
            pl.Series("filtered", flags[:, h - 1]),
            pl.lit(None, pl.Float64).alias("nu"),
        )
        for h in hs
    ]
    return pl.concat(rows).select(_OUT).sort("close_ts", "h")


def _design(model: str, c: dict, sqrt_rq: np.ndarray, q_mean: float, rows: np.ndarray) -> np.ndarray:
    one = np.ones(rows.size)
    if model == "har":
        cols = [one, c["rv"][rows], c["weekly"][rows], c["monthly"][rows]]
    elif model == "harj":
        cols = [one, c["rv"][rows], c["weekly"][rows], c["monthly"][rows], c["jump"][rows]]
    elif model in ("harcj", "hartcj"):
        cc, jj = ("c_bns", "j_bns") if model == "harcj" else ("c_tcj", "j_tcj")
        cols = [one, *(c[k][rows] for k in (cc, cc + "_w", cc + "_m", jj, jj + "_w", jj + "_m"))]
    elif model == "shar":
        cols = [one, c["rs_plus"][rows], c["rs_minus"][rows], c["weekly"][rows], c["monthly"][rows]]
    else:
        cols = [one, c["rv"][rows], c["rv"][rows] * (sqrt_rq[rows] - q_mean), c["weekly"][rows], c["monthly"][rows]]
    return np.column_stack(cols)


def _estimate(x: np.ndarray, y: np.ndarray, estimator: str) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    if estimator == "wls":
        fitted = np.maximum(x @ beta, y[y > 0].min() if (y > 0).any() else 1e-12)
        weight = np.sqrt(1 / fitted)
        beta, *_ = np.linalg.lstsq(x * weight[:, None], y * weight, rcond=None)
    return beta
