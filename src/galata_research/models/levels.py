"""The random level shift (RLS) model of volatility: Lu and Perron (2010), fitted and walked forward.

    gr.models.levels.fit(y)                                  # {"a", "p", "sigma_eta", "sigma_c", "loglik", ...}
    gr.models.levels.walk_forward(returns, split=t, every=30, horizons=(1, 7, 30))

- **The model.** On y_t = log |r_t|: y_t = a + τ_t + c_t with
  c_t ~ N(0, σ_c²), and τ_t = τ_{t−1} + δ_t, where δ_t is 0 with
  probability 1 − p and N(0, σ_η²) with probability p.
  - c_t is quasi-ML: log |z| is not normal.
  - a and the diffuse τ_0 are not separately identified; only a + τ_t
    enters a forecast.
- **The filter.** Each step's two branches (a shift or not) are updated by
  Kalman's equations and collapsed to one Gaussian by moment matching. A
  missing y (a zero return) is predicted through, not updated.
- **The forecast.** The level is a random walk, so its variance forecast is
  flat: κ · exp(2(a + τ_{t|t})) for every horizon. κ maps the level of
  log |r| to the variance of r, estimated on the fitting window from the
  one-step predicted levels.

The loop is plain Python floats: no numba here, and a refit starts from the
previous refit's parameters.
"""

import math

import numpy as np
import polars as pl
from scipy.optimize import minimize

from .. import timeseries, utils
from .._errors import Refused

_DIFFUSE = 1e4
_LOG_2PI = math.log(2 * math.pi)
_OUT = ("ticker", "ts", "close_ts", "h", "target_ts", "variance", "cum_variance", "fitted_through", "fit_from", "refit", "after_gap")


def _filter(y: list, a: float, p: float, s_eta2: float, s_c2: float, *, m0: float = 0.0, p0: float = _DIFFUSE):
    """The collapsed two-branch filter: loglik, and per step the predicted and filtered level."""
    m, var = m0, p0
    loglik = 0.0
    predicted, filtered = [], []
    q0, q1 = 1.0 - p, p
    for i, obs in enumerate(y):
        predicted.append(m)
        if obs is None:
            var = var + p * s_eta2
            filtered.append(m)
            continue
        v = obs - a - m
        v0, v1 = var, var + s_eta2  # the level's prior variance without and with a shift
        f0, f1 = v0 + s_c2, v1 + s_c2
        l0 = q0 * math.exp(-0.5 * v * v / f0) / math.sqrt(f0)
        l1 = q1 * math.exp(-0.5 * v * v / f1) / math.sqrt(f1)
        total = l0 + l1
        if total <= 0.0:
            return -math.inf, predicted, filtered
        if i > 0:  # the first observation only locates the diffuse level
            loglik += math.log(total) - 0.5 * _LOG_2PI
        w0, w1 = l0 / total, l1 / total
        k0, k1 = v0 / f0, v1 / f1
        m0_, m1_ = m + k0 * v, m + k1 * v
        p0_, p1_ = v0 * (1 - k0), v1 * (1 - k1)
        m = w0 * m0_ + w1 * m1_
        var = w0 * (p0_ + (m0_ - m) ** 2) + w1 * (p1_ + (m1_ - m) ** 2)
        filtered.append(m)
    return loglik, predicted, filtered


def _unpack(theta):
    a, logit_p, log_eta, log_c = theta
    p = 1e-4 + (0.5 - 1e-4) / (1 + math.exp(-logit_p))
    return a, p, math.exp(2 * log_eta), math.exp(2 * log_c)


def fit(y, *, start: dict | None = None) -> dict:
    """ML estimates of (a, p, σ_η, σ_c) on y (None or NaN for missing), p ∈ [1e-4, 0.5]; `start` warm-starts from a previous fit."""
    ys = [None if (v is None or not math.isfinite(v)) else float(v) for v in y]
    observed = [v for v in ys if v is not None]
    if len(observed) < 250:
        raise Refused(f"a random level shift fit needs at least 250 observations, not {len(observed)}")
    if start is None:
        sd = float(np.std(observed))
        theta0 = [float(np.mean(observed)), math.log(0.01 / 0.49), math.log(sd * 0.5), math.log(sd * 0.9)]
    else:
        theta0 = [start["a"], math.log((start["p"] - 1e-4) / (0.5 - start["p"])), math.log(start["sigma_eta"]), math.log(start["sigma_c"])]

    def negative(theta):
        ll = _filter(ys, *_unpack(theta))[0]
        return 1e12 if not math.isfinite(ll) else -ll

    res = minimize(negative, theta0, method="L-BFGS-B", options={"maxiter": 200})
    a, p, s_eta2, s_c2 = _unpack(res.x)
    return {"a": a, "p": p, "sigma_eta": math.sqrt(s_eta2), "sigma_c": math.sqrt(s_c2), "loglik": -float(res.fun), "converged": bool(res.success), "n": len(observed)}


def walk_forward(returns: pl.LazyFrame | pl.DataFrame, *, split, every: int, horizons=(1,), min_obs: int = 250) -> pl.DataFrame:
    """RLS variance forecasts from every origin at or after `split`, in `vol.walk_forward`'s columns.

    Refitted every `every` origins on an expanding window, each refit warm-started
    from the last; the level is filtered through every bar. κ is re-estimated at
    each refit from that window alone.
    """
    utils.require(returns, ("ticker", "ts", "close_ts", "return"), "make returns with gr.timeseries.returns")
    frame = utils.lazy(returns).sort("ts").collect()
    if frame["ticker"].n_unique() != 1:
        raise Refused("walk one ticker at a time")
    hs = tuple(sorted({int(h) for h in horizons}))
    kept = frame.with_columns(pl.col("return").is_null().shift(1).fill_null(False).alias("after_gap")).drop_nulls("return")
    kept = kept.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(False).otherwise(pl.col("after_gap")).alias("after_gap"))
    schedule = timeseries.walk_forward_origins(kept.select("ticker", "ts", "close_ts"), split, window="expanding", every=every)
    first = kept.height - schedule.height
    refits = [first + i for i, r in enumerate(schedule["refit"].to_list()) if r]
    if refits[0] + 1 < min_obs:
        raise Refused(f"the first refit's window holds {refits[0] + 1} returns, under min_obs={min_obs}")
    r = kept["return"].to_list()
    y = [math.log(abs(v)) if v != 0 else None for v in r]
    width = kept["close_ts"][0] - kept["ts"][0]
    level, kappa = [0.0] * kept.height, [0.0] * kept.height
    params = None
    for n, at in enumerate(refits):
        stop = refits[n + 1] if n + 1 < len(refits) else kept.height
        params = fit(y[: at + 1], start=params)
        pr = (params["a"], params["p"], params["sigma_eta"] ** 2, params["sigma_c"] ** 2)
        _, predicted, filtered = _filter(y[:stop], *pr)
        ratios = [r[s] ** 2 / math.exp(2 * (params["a"] + predicted[s])) for s in range(1, at + 1)]
        k = float(np.mean(ratios))
        for s in range(at, stop):
            level[s] = params["a"] + filtered[s]
            kappa[s] = k
    v = np.array([kappa[s] * math.exp(2 * level[s]) for s in range(first, kept.height)])
    base = schedule.select("ticker", "ts", "close_ts", "fitted_through", "fit_from", "refit").with_columns(kept["after_gap"].slice(first).alias("after_gap"))
    rows = [
        base.with_columns(
            pl.lit(h, pl.Int64).alias("h"),
            (pl.col("close_ts") + width * (h - 1)).alias("target_ts"),
            pl.Series("variance", v),
            pl.Series("cum_variance", v * h),
        )
        for h in hs
    ]
    return pl.concat(rows).select(_OUT).sort("close_ts", "h")
