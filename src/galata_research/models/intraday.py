"""Tomorrow's liquidity: one-hour-ahead forecasts of an hourly series, walked forward.

    f = gr.models.intraday.forecast(depth, split="2025-01-01T00:00Z", model="decomposition")
    gr.models.intraday.score(pl.concat([f, seasonal, persistence]), benchmark="seasonal")

The intraday-volume literature splits a series into a seasonal part and a
dynamic one: Bialkowski, Darolles and Le Fol (2008), JBF 32(9), with ARMA
dynamics, and Brownlees, Cipollini and Gallo (2011), JFEC 9(3), with a
component MEM. A comparison found the simpler decomposition more accurate and
far faster (Finance Research Letters 21, 2017). Here, in logs:
log y = s(hour of week) + e, and e follows a HAR on its last hour, day and week.
"""

import math
from datetime import datetime

import numpy as np
import polars as pl

from .. import utils
from .._errors import Refused
from . import evaluate

MODELS = ("seasonal", "persistence", "decomposition")
_UTC_US = pl.Datetime("us", "UTC")


def _grid(series: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    utils.require(series, ("ts", "value"), "hourly rows with ts and value")
    s = utils.lazy(series).select("ts", "value").collect().sort("ts")
    if s.filter(pl.col("value") <= 0).height:
        raise Refused("value must be positive: forecasts are of its log")
    full = pl.DataFrame({"ts": pl.datetime_range(s["ts"].min(), s["ts"].max(), "1h", eager=True, time_zone="UTC")})
    return full.join(s, on="ts", how="left").with_columns(
        pl.col("value").log().alias("y"),
        # Cast first: weekday and hour are Int8, and 6 × 24 + 23 overflows it.
        ((pl.col("ts").dt.weekday().cast(pl.Int64) - 1) * 24 + pl.col("ts").dt.hour().cast(pl.Int64)).alias("how"),
    )


def _rolling_mean(e: np.ndarray, n: int) -> np.ndarray:
    """mean(e[j−n … j−1]) at each j, NaN when any is missing or j < n."""
    ok = np.isfinite(e)
    v = np.where(ok, e, 0.0)
    cs = np.concatenate([[0.0], np.cumsum(v)])
    cn = np.concatenate([[0], np.cumsum(ok)])
    out = np.full(e.shape, np.nan)
    j = np.arange(n, e.size)
    sums, counts = cs[j] - cs[j - n], cn[j] - cn[j - n]
    out[j] = np.where(counts == n, sums / n, np.nan)
    return out


def forecast(series: pl.LazyFrame | pl.DataFrame, *, split, model: str, refit_every: int = 720) -> pl.DataFrame:
    """`ts, model, forecast, actual, fitted_through` for every hour from `split`, in log(value).

    The forecast for the hour starting at t uses only hours with close ≤ t;
    parameters are refitted every `refit_every` hours on all hours before.
    """
    if model not in MODELS:
        raise Refused(f"model={model!r} is not one of {', '.join(MODELS)}")
    g = _grid(series)
    y, how, ts = g["y"].to_numpy(), g["how"].to_numpy(), g["ts"].to_list()
    lo = utils.instant("split", split)
    start = next((i for i, t in enumerate(ts) if int(t.timestamp() * 1e6) >= lo), None)
    if start is None or start < 168 + 24:
        raise Refused("split leaves too little history: at least 192 hours are needed before it")
    n = y.size
    fc = np.full(n, np.nan)
    fitted = [None] * n
    for r in range(start, n, refit_every):
        end = min(r + refit_every, n)
        fit = slice(0, r)
        # The seasonal: hour-of-day mean plus weekday effect, in logs: 31 numbers, where 168
        # free cells scored no better on a simulated series (0.0101 against 0.0099).
        yf, hf = y[fit], how[fit]
        ok = np.isfinite(yf)
        grand = yf[ok].mean()
        hod = np.array([yf[ok & ((hf % 24) == h)].mean() if (ok & ((hf % 24) == h)).any() else grand for h in range(24)])
        wd = np.array([yf[ok & ((hf // 24) == d)].mean() - grand if (ok & ((hf // 24) == d)).any() else 0.0 for d in range(7)])
        seasonal = np.array([hod[c % 24] + wd[c // 24] for c in range(168)])
        s = seasonal[how]
        targets = np.arange(r, end)
        if model == "seasonal":
            fc[targets] = s[targets]
        elif model == "persistence":
            fc[targets] = y[targets - 1]
        else:
            e = y - s
            lag1 = np.concatenate([[np.nan], e[:-1]])
            m24, m168 = _rolling_mean(e, 24), _rolling_mean(e, 168)
            X = np.column_stack([np.ones(n), lag1, m24, m168])
            rows = np.arange(168, r)
            use = rows[np.isfinite(X[rows]).all(axis=1) & np.isfinite(e[rows])]
            if use.size < 10:
                raise Refused(f"{use.size} complete hours before {ts[r]} are too few to fit the dynamic part")
            beta, *_ = np.linalg.lstsq(X[use], e[use], rcond=None)
            fc[targets] = s[targets] + X[targets] @ beta
        for i in targets:
            fitted[i] = ts[r - 1]
    rows = slice(start, n)
    return pl.DataFrame(
        {"ts": ts[rows], "model": model, "forecast": fc[rows], "actual": y[rows], "fitted_through": fitted[rows]},
        schema={"ts": _UTC_US, "model": pl.String, "forecast": pl.Float64, "actual": pl.Float64, "fitted_through": _UTC_US},
    ).with_columns(pl.col("forecast").fill_nan(None), pl.col("actual").fill_nan(None))  # a missing hour is null, not NaN


def score(forecasts: pl.DataFrame, *, benchmark: str) -> pl.DataFrame:
    """`model, n, mse, r2_oos, dm_stat, dm_p` against `benchmark`, on hours where every model has a forecast."""
    utils.require(forecasts, ("ts", "model", "forecast", "actual"), "make forecasts with gr.models.intraday.forecast")
    wide = forecasts.drop_nulls(["forecast", "actual"]).pivot(on="model", index=["ts", "actual"], values="forecast").drop_nulls()
    models = [c for c in wide.columns if c not in ("ts", "actual")]
    if benchmark not in models:
        raise Refused(f"benchmark={benchmark!r} is not among {', '.join(models)}")
    actual = wide["actual"].to_numpy()
    base = (wide[benchmark].to_numpy() - actual) ** 2
    rows = []
    for m in models:
        loss = (wide[m].to_numpy() - actual) ** 2
        test = evaluate.dm(loss, base) if m != benchmark else {"statistic": math.nan, "p_value": math.nan}
        rows.append({"model": m, "n": int(loss.size), "mse": float(loss.mean()), "r2_oos": float(1 - loss.mean() / base.mean()),
                     "dm_stat": test["statistic"], "dm_p": test["p_value"]})  # fmt: skip
    return pl.DataFrame(rows).sort("mse")
