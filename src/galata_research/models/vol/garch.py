"""In-sample fits of the GARCH family: persistence by each model's own formula, news impact, diagnostics.

Returns are fitted × 100 (unscaled, arch warns `DataScaleWarning` and
converges worse); parameters are reported on that scale and say so, and
every σ is converted back to return units.
"""

from dataclasses import dataclass, field
from math import log, sqrt

import numpy as np
import polars as pl
from scipy import integrate, stats

from ... import utils
from ..._errors import Refused
from .. import _arch
from .._arch import DISTS, MODELS, SCALE

_SHAPE = {"normal": 0, "t": 1, "skewt": 2, "ged": 1}
_EPS = 1e-9


@dataclass(frozen=True)
class Fit:
    """One in-sample fit. `params` and `std_err` are on returns × `scale`; `series` is in return units."""

    ticker: str
    model: str
    dist: str
    nobs: int
    scale: float
    params: dict
    std_err: dict
    loglik: float
    aic: float
    bic: float
    converged: bool
    persistence: float | None
    half_life: float | None
    sigma_bar: float | None
    series: pl.DataFrame = field(repr=False)


def fit(
    returns: pl.LazyFrame | pl.DataFrame,
    *,
    model: str = "garch",
    dist: str = "t",
    fit: tuple | None = None,
    column: str = "return",
    min_obs: int = 500,
) -> Fit:
    """One ticker's returns fitted in-sample: a constant mean and a (1,1) process.

    `model`: `ewma` (λ = 0.94, fixed), `rm2006`, `garch`, `gjr`, `egarch`,
    `aparch`, `figarch`. `dist`: `normal`, `t`, `skewt`, `ged`. `returns` as
    `gr.timeseries.returns` gives them, or `deseasonalize`'s with
    `column="deseasonalized"`. `fit=(start, end)` keeps returns with
    `ts ≥ start` and `close_ts ≤ end`. Null returns are dropped and the
    recursion runs across the hole (roadmap D5); the first return after each
    is `after_gap` in `series`. Fewer than `min_obs` returns is refused: ν is
    unstable on short samples.

    Persistence: GARCH α+β; GJR α+β+γκ, κ = E[z²·1(z<0)] under the fitted
    distribution (0.5 only when z is symmetric); EGARCH β; APARCH
    α·E[(|z|−γz)^δ]+β; EWMA 1; FIGARCH and RiskMetrics 2006 none (FIGARCH's
    summary is `d`). arch's EGARCH centres |z| at √(2/π), E|z| for a normal,
    whatever the distribution (arch/univariate/recursions_python.py:40); under
    t the intercept absorbs the difference and β is unaffected.
    """
    if model not in MODELS:
        raise Refused(f"model={model!r} is not one of {', '.join(MODELS)}")
    if dist not in DISTS:
        raise Refused(f"dist={dist!r} is not one of {', '.join(DISTS)}")
    utils.require(returns, ("ticker", "ts", "close_ts", column), "make returns with gr.timeseries.returns")
    frame = utils.lazy(returns).sort("ts").collect()
    tickers = frame["ticker"].unique().sort().to_list()
    if len(tickers) != 1:
        raise Refused(f"fit one ticker at a time; these returns hold {', '.join(tickers) or 'none'}")
    if fit is not None:
        lo, hi = utils.window(*fit)
        frame = frame.filter((pl.col("ts").dt.epoch("us") >= lo) & (pl.col("close_ts").dt.epoch("us") <= hi))
    marked = frame.with_columns(pl.col(column).is_null().shift(1).fill_null(False).alias("after_gap")).drop_nulls(column)
    marked = marked.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(False).otherwise(pl.col("after_gap")).alias("after_gap"))
    if marked.height < min_obs:
        raise Refused(f"{marked.height} returns, under min_obs={min_obs}: a fat-tailed fit is unstable on so few")
    res = _arch.fit(_arch.values(marked, column), model, dist)
    s = _arch.summary(res)
    p = s["params"]
    persistence = _persistence(model, dist, p)
    series = marked.select(
        "ts",
        "close_ts",
        pl.col(column).alias("return"),
        pl.Series("sigma", s["sigma"]) / SCALE,
        pl.Series("z", s["z"]),
        "after_gap",
    )
    return Fit(
        ticker=tickers[0],
        model=model,
        dist=dist,
        nobs=marked.height,
        scale=SCALE,
        params=p,
        std_err=s["std_err"],
        loglik=s["loglik"],
        aic=s["aic"],
        bic=s["bic"],
        converged=s["converged"],
        persistence=persistence,
        half_life=log(0.5) / log(persistence) if persistence is not None and 0 < persistence < 1 else None,
        sigma_bar=_sigma_bar(model, p, persistence),
        series=series,
    )


def expectation(dist: str, params: dict, g, *, split: bool = False) -> float:
    """E[g(z)] under the fitted, standardised distribution, integrating g(F⁻¹(u)) over u in (0, 1).

    `split=True` breaks the integral at F(0), where g may have a kink or a jump.
    """
    q = _arch.ppf(dist, params)
    at_zero = _cdf_zero(dist, params) if split else None
    points = [at_zero] if at_zero is not None else None
    value, _ = integrate.quad(lambda u: g(q(u)), _EPS, 1 - _EPS, points=points, limit=400)
    return float(value)


def kappa(dist: str, params: dict) -> float:
    """κ = E[z²·1(z<0)]: GJR's weight on γ in its persistence (0.5 for a symmetric z)."""
    return expectation(dist, params, lambda z: z * z if z < 0 else 0.0, split=True)


def _cdf_zero(dist: str, params: dict) -> float:
    lo, hi = _EPS, 1 - _EPS
    q = _arch.ppf(dist, params)
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if q(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def _persistence(model: str, dist: str, p: dict) -> float | None:
    if model == "garch":
        return p["alpha[1]"] + p["beta[1]"]
    if model == "gjr":
        return p["alpha[1]"] + p["beta[1]"] + p["gamma[1]"] * kappa(dist, p)
    if model == "egarch":
        return p["beta[1]"]
    if model == "aparch":
        a, g, d = p["alpha[1]"], p["gamma[1]"], p["delta"]
        return a * expectation(dist, p, lambda z: (abs(z) - g * z) ** d, split=True) + p["beta[1]"]
    if model == "ewma":
        return 1.0
    return None


def _sigma_bar(model: str, p: dict, persistence: float | None) -> float | None:
    if persistence is None or not 0 < persistence < 1 or model not in ("garch", "gjr", "aparch"):
        return None
    level = p["omega"] / (1 - persistence)
    variance = level if model != "aparch" else level ** (2 / p["delta"])
    return sqrt(variance) / SCALE


def table(fits) -> pl.DataFrame:
    """One row per fit: the summary columns, then every parameter as its own column (on the fitted scale)."""
    rows = []
    for f in fits:
        row = {
            "ticker": f.ticker,
            "model": f.model,
            "dist": f.dist,
            "nobs": f.nobs,
            "loglik": f.loglik,
            "aic": f.aic,
            "bic": f.bic,
            "persistence": f.persistence,
            "half_life": f.half_life,
            "sigma_bar": f.sigma_bar,
            "converged": f.converged,
        }
        row.update({k: float(v) for k, v in f.params.items()})
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def news_impact(f: Fit, z) -> pl.DataFrame:
    """`z, sigma2`: next-bar variance (return units²) after a shock ε = z·σ̄, holding σₜ₋₁ at σ̄.

    Engle and Ng (1993) hold σₜ₋₁ at the unconditional variance, which EGARCH
    and EWMA do not have; σ̄ here is the fitted σ's root mean square, for every
    model alike. FIGARCH and RiskMetrics 2006 depend on the whole past, not on
    εₜ₋₁ and σₜ₋₁ alone, and are refused.
    """
    if f.model in ("figarch", "rm2006"):
        raise Refused(f"{f.model}'s next variance depends on the whole past; it has no one-shock news-impact curve")
    p = f.params
    sbar2 = float((f.series["sigma"] * SCALE).pow(2).mean())
    zs = np.asarray(list(z), dtype=float)
    eps = zs * sqrt(sbar2)
    if f.model == "garch":
        s2 = p["omega"] + p["beta[1]"] * sbar2 + p["alpha[1]"] * eps**2
    elif f.model == "gjr":
        s2 = p["omega"] + p["beta[1]"] * sbar2 + (p["alpha[1]"] + p["gamma[1]"] * (eps < 0)) * eps**2
    elif f.model == "aparch":
        d = p["delta"]
        s2 = (p["omega"] + p["beta[1]"] * sbar2 ** (d / 2) + p["alpha[1]"] * (np.abs(eps) - p["gamma[1]"] * eps) ** d) ** (2 / d)
    elif f.model == "egarch":
        s2 = np.exp(p["omega"] + p["alpha[1]"] * (np.abs(zs) - sqrt(2 / np.pi)) + p["gamma[1]"] * zs + p["beta[1]"] * np.log(sbar2))
    else:  # ewma
        s2 = 0.94 * sbar2 + 0.06 * eps**2
    return pl.DataFrame({"z": zs, "sigma2": s2 / SCALE**2})


def ljung_box(x, lags: int) -> float:
    """Q(m) = n(n+2) Σₖ₌₁..ₘ ρ̂ₖ² / (n−k) (Ljung and Box 1978)."""
    v = np.asarray(x, dtype=float)
    n = v.size
    d = v - v.mean()
    denom = float(d @ d)
    return float(n * (n + 2) * sum((float(d[k:] @ d[:-k]) / denom) ** 2 / (n - k) for k in range(1, lags + 1)))


def arch_lm(z, lags: int) -> tuple[float, float]:
    """Engle's (1982) ARCH-LM on standardised residuals: n·R² of z² on a constant and `lags` lags; χ²(lags) p-value.

    n is the whole sample, not the n − lags rows of the regression: arch's
    `arch_lm_test` convention, which this agrees with.
    """
    e2 = np.asarray(z, dtype=float) ** 2
    y = e2[lags:]
    x = np.column_stack([np.ones(y.size), *[e2[lags - k : e2.size - k] for k in range(1, lags + 1)]])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    stat = e2.size * r2
    return stat, float(stats.chi2.sf(stat, lags))


def diagnose(f: Fit, lags=(10, 20)) -> pl.DataFrame:
    """`test, lags, statistic, df, p_value`: Ljung–Box on z and z², and ARCH-LM on z.

    Ljung–Box on z² takes df = lags minus the volatility parameters, the common
    McLeod–Li-style correction (not Li and Mak's 1994 statistic).
    """
    z = f.series["z"].to_numpy()
    k = len(f.params) - 1 - _SHAPE[f.dist]
    rows = []
    for m in lags:
        q = ljung_box(z, m)
        rows.append(("ljung_box z", m, q, m, float(stats.chi2.sf(q, m))))
        q2 = ljung_box(z**2, m)
        df2 = max(m - k, 1)
        rows.append(("ljung_box z² (McLeod–Li df)", m, q2, df2, float(stats.chi2.sf(q2, df2))))
        stat, pv = arch_lm(z, m)
        rows.append(("arch_lm z", m, stat, m, pv))
    return pl.DataFrame(rows, schema=["test", "lags", "statistic", "df", "p_value"], orient="row")
