"""Correlation fitted from the volatility model: two-step DCC over the GARCH family, and RiskMetrics' EWMA.

    f = gr.models.corr.fit(returns, model="gjr", dist="t")          # a, b, Q̄, R per bar
    gr.models.corr.ewma(returns, lam=0.94)                            # Σ and ρ from one λ, nothing estimated
    gr.models.corr.walk_forward(returns, model="gjr", split=t, every=6, horizons=[1, 6])

**One fit, so Σ is consistent** (roadmap D13). Step 1 is each ticker's
`gr.models.vol` model, through the same arch calls `vol.walk_forward` makes;
its σ standardises the returns, zᵢ = (rᵢ − μᵢ)/σᵢ. Step 2 is Engle's (2002)
DCC(1,1) on z,

    Qₜ = (1 − a − b)Q̄ + a·zₜ₋₁zₜ₋₁ᵀ + b·Qₜ₋₁,   Rₜ = diag(Qₜ)^−½ Qₜ diag(Qₜ)^−½,

by Gaussian quasi-likelihood, whatever step 1's tail: a multivariate t would
force one ν on every ticker, and the normal QML stays consistent for (a, b)
under heavier tails. Aielli's (2013) cDCC, which runs the recursion on
z*ₜ = diag(Qₜ)^½ zₜ so that Q̄ targeting is consistent, is `corr="cdcc"`.
Multi-step R follows Engle and Sheppard (2001):
Rₜ₊ₕ ≈ R̄ + (a + b)^(h−1)(Rₜ₊₁ − R̄). The unfitted fallback is RiskMetrics'
(1996) covariance, one λ for every variance and covariance, so Σ is PSD.

**The joint sample.** The recursion needs every zₜ in full, so a bar missing
for any ticker is dropped for all of them and bridged (roadmap D5); the
sample starts at the youngest ticker's first return. Rows are long pairs,
`ticker_i ≤ ticker_j` in string order, the diagonal included where a
variance is reported.

**n_eff** is Kish's (1965) effective sample size, (Σw)²/Σw²: what a weighted
figure rests on, which is not the rows in its window.
"""

from dataclasses import dataclass, field
from math import ceil, inf, log

import numpy as np
import polars as pl
from scipy import optimize, signal
from scipy.special import expit

from .. import timeseries, utils
from .._errors import Refused
from . import _arch
from ._arch import DISTS, SCALE, SIMULATED
from .vol import garch as _garch

MODELS = ("garch", "gjr", "egarch", "ewma")
CORRS = ("dcc", "cdcc", "ccc", "iewma")
_POLYNOMIAL_TAILS = ("t", "skewt")
_STARTS = ((0.02, 0.95), (0.05, 0.90))
_SMAX = 0.9999  # a + b stays below one
_OUT = (
    "ts", "close_ts", "h", "target_ts", "ticker_i", "ticker_j", "covariance", "cum_covariance", "correlation",
    "correlation_target", "fitted_through", "fit_from", "refit", "after_gap", "n_eff", "a", "b",
)


@dataclass(frozen=True)
class CorrFit:
    """One in-sample two-step fit. `qbar` is on z, in `tickers` order; `series` is R per bar in long pairs i < j."""

    tickers: list
    model: str
    dist: str
    corr: str
    nobs: int
    a: float
    b: float
    persistence: float
    half_life: float | None
    qbar: np.ndarray = field(repr=False)
    loglik: float
    converged: bool
    fits: dict = field(repr=False)
    series: pl.DataFrame = field(repr=False)


@dataclass(frozen=True)
class EwmaFit:
    """RiskMetrics' covariance: nothing fitted. `series` is Σ and ρ per bar in long pairs i ≤ j."""

    tickers: list
    lam: float
    warmup: int
    nobs: int
    fitted: bool
    series: pl.DataFrame = field(repr=False)


# ── n_eff ────────────────────────────────────────────────────────────────────


def n_eff(weights) -> float:
    """(Σw)²/Σw², Kish's effective sample size."""
    w = np.asarray(list(weights), dtype=float)
    return float(w.sum() ** 2 / (w * w).sum())


def ewma_n_eff(lam: float, n: float = inf) -> float:
    """n_eff of the weights λᵏ, k = 0…n−1: (1 − λⁿ)²(1 + λ)/((1 − λ)(1 − λ²ⁿ)); (1 + λ)/(1 − λ) as n → ∞."""
    if not 0 <= lam < 1:
        raise Refused(f"lam={lam} must be in [0, 1)")
    if n == inf:
        return (1 + lam) / (1 - lam)
    if n < 1:
        raise Refused(f"n={n}: an effective size needs at least one weight")
    ln, l2n = lam**n, lam ** (2 * n)
    return (1 - ln) ** 2 * (1 + lam) / ((1 - lam) * (1 - l2n))


# ── the joint sample ─────────────────────────────────────────────────────────


def joint(returns: pl.LazyFrame | pl.DataFrame, *, column: str = "return") -> tuple[list, pl.DataFrame, str]:
    """`(tickers, frame, youngest)`: `ts, close_ts, after_gap` and one column per ticker, only where every ticker has a return.

    `youngest` is the ticker whose first return starts the sample.
    """
    utils.require(returns, ("ticker", "ts", "close_ts", column), "make returns with gr.timeseries.returns")
    long = utils.lazy(returns).select("ticker", "ts", "close_ts", pl.col(column).cast(pl.Float64)).collect()
    tickers = long["ticker"].unique().sort().to_list()
    if len(tickers) < 2:
        raise Refused(f"a correlation needs two tickers or more; these returns hold {', '.join(tickers) or 'none'}")
    firsts = long.drop_nulls(column).group_by("ticker").agg(pl.col("ts").min()).sort("ts", "ticker")
    youngest = firsts["ticker"][-1] if firsts.height == len(tickers) else next(t for t in tickers if t not in firsts["ticker"].to_list())
    wide = long.pivot(on="ticker", index=["ts", "close_ts"], values=column).sort("ts")
    present = pl.all_horizontal(pl.col(t).is_not_null() for t in tickers) if all(t in wide.columns for t in tickers) else pl.lit(False)
    wide = wide.with_columns(present.alias("_present"))
    wide = wide.with_columns((~pl.col("_present")).shift(1).fill_null(False).alias("after_gap")).filter(pl.col("_present"))
    wide = wide.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(False).otherwise(pl.col("after_gap")).alias("after_gap"))
    return tickers, wide.select("ts", "close_ts", "after_gap", *tickers), youngest


def _enough(frame: pl.DataFrame, youngest: str, needed: int, what: str) -> None:
    if frame.height < needed:
        raise Refused(f"the joint sample holds {frame.height} returns, under {what}={needed}: it starts at {youngest}'s first return")


# ── the recursion and step 2 ─────────────────────────────────────────────────


def _normalise(q: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.einsum("...ii->...i", q))
    return q / (d[..., :, None] * d[..., None, :])


def filter(z: np.ndarray, a: float, b: float, qbar: np.ndarray, *, corr: str = "dcc") -> tuple[np.ndarray, np.ndarray]:
    """`(Q, R)`, each (T + 1) × N × N: Q₀ = Q̄, and row t is built from z through t − 1, so row T is the forecast after the last z.

    DCC's Q is linear in its input and runs as one first-order filter; cDCC
    rescales z by diag(Qₜ)^½, which is not, and runs bar by bar; its `qbar`
    is the unit-diagonal S of `cdcc_target`.
    """
    z = np.asarray(z, dtype=float)
    T, N = z.shape
    if corr in ("dcc", "ccc", "iewma"):
        x = np.empty((T + 1, N, N))
        x[0] = qbar
        x[1:] = (1 - a - b) * qbar + a * np.einsum("ti,tj->tij", z, z)
        q = signal.lfilter([1.0], [1.0, -b], x, axis=0)
        return q, _normalise(q)
    if corr != "cdcc":
        raise Refused(f"corr={corr!r} is not one of {', '.join(CORRS)}")
    q = np.empty((T + 1, N, N))
    q[0] = qbar
    base = (1 - a - b) * qbar
    for t in range(T):
        u = np.sqrt(np.diag(q[t])) * z[t]
        q[t + 1] = base + a * np.outer(u, u) + b * q[t]
    return q, _normalise(q)


def cdcc_target(z: np.ndarray, a: float, b: float) -> np.ndarray:
    """cDCC's S̃(a, b): the correlation matrix of T⁻¹Σz*ₜz*ₜᵀ, with z*ₜ = diag(Qₜ)^½ zₜ (Aielli 2013, Def. 3.3).

    With S unit-diagonal, qᵢᵢ,ₜ₊₁ = (1 − a − b) + a·z*²ᵢ,ₜ + b·qᵢᵢ,ₜ does not depend
    on S's off-diagonal, so the N scalar recursions run first, from qᵢᵢ,₀ = 1.
    """
    z = np.asarray(z, dtype=float)
    T, N = z.shape
    q = np.ones(N)
    star = np.empty_like(z)
    for t in range(T):
        star[t] = np.sqrt(q) * z[t]
        q = (1 - a - b) + a * star[t] ** 2 + b * q
    return _normalise(star.T @ star / T)


def loglik(z: np.ndarray, a: float, b: float, qbar: np.ndarray, *, corr: str = "dcc") -> float:
    """The correlation part of the Gaussian quasi-likelihood, −½Σₜ(ln|Rₜ| + zₜᵀRₜ⁻¹zₜ); −½zₜᵀzₜ is dropped (no a or b in it)."""
    _, r = filter(z, a, b, qbar, corr=corr)
    r = r[:-1]
    sign, logdet = np.linalg.slogdet(r)
    if np.any(sign <= 0):
        return -np.inf
    quad = np.einsum("ti,ti->t", z, np.linalg.solve(r, z[..., None])[..., 0])
    return float(-0.5 * (logdet.sum() + quad.sum()))


def _ab(theta) -> tuple[float, float]:
    s, u = _SMAX * expit(theta[0]), expit(theta[1])
    return s * u, s * (1 - u)


def _theta(a: float, b: float) -> np.ndarray:
    s, u = (a + b) / _SMAX, a / (a + b)
    return np.array([log(s / (1 - s)), log(u / (1 - u))])


def estimate(z: np.ndarray, *, corr: str = "dcc") -> dict:
    """(a, b) by Gaussian QML on z, a + b < 1 by construction; two starts, the better kept.

    DCC targets Q̄ at the uncentred second moment of z over n (rmgarch uses
    cov(z), centred over n − 1; the two differ by the mean of z, near zero).
    cDCC profiles its target: S̃(a, b) is recomputed at every trial (a, b).
    """
    z = np.asarray(z, dtype=float)
    if corr == "ccc":
        # Bollerslev's (1990) constant correlation: R = Q̄ normalised, nothing to estimate.
        moment = z.T @ z / len(z)
        return {"a": 0.0, "b": 0.0, "qbar": moment, "loglik": loglik(z, 0.0, 0.0, moment), "converged": True}
    best = None
    for a0, b0 in _STARTS:
        moment = z.T @ z / len(z)

        def objective(theta):
            a, b = _ab(theta)
            qbar = moment if corr == "dcc" else cdcc_target(z, a, b)
            value = loglik(z, a, b, qbar, corr=corr)
            return 1e10 if not np.isfinite(value) else -value / len(z)

        res = optimize.minimize(objective, _theta(a0, b0), method="L-BFGS-B")
        a, b = _ab(res.x)
        qbar = moment if corr == "dcc" else cdcc_target(z, a, b)
        ll = loglik(z, a, b, qbar, corr=corr)
        if best is None or ll > best["loglik"]:
            best = {"a": a, "b": b, "qbar": qbar, "loglik": ll, "converged": bool(res.success)}
    return best


# ── step 1 ───────────────────────────────────────────────────────────────────


def _check(model: str, dist: str, corr: str) -> None:
    if model not in MODELS:
        raise Refused(f"model={model!r} is not one of {', '.join(MODELS)}: step 1 needs a σ per bar, which HAR, HARQ and CARR do not give")
    if dist not in DISTS:
        raise Refused(f"dist={dist!r} is not one of {', '.join(DISTS)}")
    if corr not in CORRS:
        raise Refused(f"corr={corr!r} is not one of {', '.join(CORRS)}")


def _pairs(tickers: list, *, diagonal: bool) -> list[tuple[int, int]]:
    n = len(tickers)
    return [(i, j) for i in range(n) for j in range(i if diagonal else i + 1, n)]


def fit(
    returns: pl.LazyFrame | pl.DataFrame,
    *,
    model: str = "gjr",
    dist: str = "t",
    corr: str = "dcc",
    fit: tuple | None = None,
    column: str = "return",
    min_obs: int = 500,
) -> CorrFit:
    """Two-step DCC in sample on the joint sample of every ticker in `returns`.

    Step 1 is `gr.models.vol.fit(model, dist)` per ticker on the joint rows;
    step 2 is (a, b) on its z. `series`: `ts, close_ts, ticker_i, ticker_j,
    correlation, after_gap`, with Rₜ the correlation of bar t given the bars
    before it. `fit=(start, end)` keeps rows with `ts ≥ start` and
    `close_ts ≤ end`.
    """
    _check(model, dist, corr)
    tickers, frame, youngest = joint(returns, column=column)
    if fit is not None:
        lo, hi = utils.window(*fit)
        frame = frame.filter((pl.col("ts").dt.epoch("us") >= lo) & (pl.col("close_ts").dt.epoch("us") <= hi))
    _enough(frame, youngest, min_obs, "min_obs")
    fits = {}
    for t in tickers:
        one = frame.select(pl.lit(t).alias("ticker"), "ts", "close_ts", pl.col(t).alias("return"))
        fits[t] = _garch.fit(one, model=model, dist=dist, min_obs=min_obs)
    z = np.column_stack([fits[t].series["z"].to_numpy() for t in tickers])
    est = estimate(z, corr=corr)
    _, r = filter(z, est["a"], est["b"], est["qbar"], corr=corr)
    pairs = _pairs(tickers, diagonal=False)
    series = pl.concat(
        [
            frame.select(
                "ts", "close_ts", pl.lit(tickers[i]).alias("ticker_i"), pl.lit(tickers[j]).alias("ticker_j"),
                pl.Series("correlation", r[:-1, i, j]), "after_gap",
            )
            for i, j in pairs
        ]
    ).sort("ts", "ticker_i", "ticker_j")
    s = est["a"] + est["b"]
    return CorrFit(
        tickers=tickers,
        model=model,
        dist=dist,
        corr=corr,
        nobs=frame.height,
        a=est["a"],
        b=est["b"],
        persistence=s,
        half_life=log(0.5) / log(est["b"]) if 0 < est["b"] < 1 else None,
        qbar=est["qbar"],
        loglik=est["loglik"],
        converged=est["converged"],
        fits=fits,
        series=series,
    )


# ── EWMA ─────────────────────────────────────────────────────────────────────


def _warmup(lam: float, warmup: int | None) -> int:
    if not 0 < lam < 1:
        raise Refused(f"lam={lam} must be inside (0, 1)")
    return ceil((1 + lam) / (1 - lam)) if warmup is None else int(warmup)


def _ewma_path(y: np.ndarray, lam: float, w: int) -> np.ndarray:
    """Σ after each row, (T × N × N): row t is Σₜ₊₁, which includes rₜ; rows before w − 1 are NaN.

    Seeded at row w − 1 with the second moment of the first w returns, zero mean.
    """
    T, N = y.shape
    out = np.full((T, N, N), np.nan)
    s = y[:w].T @ y[:w] / w
    out[w - 1] = s
    x = np.empty((T - w + 1, N, N))
    x[0] = s
    x[1:] = (1 - lam) * np.einsum("ti,tj->tij", y[w:], y[w:])
    out[w - 1 :] = signal.lfilter([1.0], [1.0, -lam], x, axis=0)
    return out


def ewma(returns: pl.LazyFrame | pl.DataFrame, *, lam: float = 0.94, warmup: int | None = None, column: str = "return") -> EwmaFit:
    """RiskMetrics' Σₜ₊₁ = λΣₜ + (1 − λ)rₜrₜᵀ on the joint sample; nothing estimated.

    Seeded with the second moment of the first `warmup` returns, by default
    ⌈(1 + λ)/(1 − λ)⌉ (n_eff at λ: 33 at 0.94), with no row before it.
    `series`: `ts, close_ts, ticker_i, ticker_j, covariance, correlation,
    after_gap`; the row at close t includes rₜ and is the forecast for t + 1,
    as `gr.timeseries.ewma_vol` indexes σ.
    """
    w = _warmup(lam, warmup)
    tickers, frame, youngest = joint(returns, column=column)
    _enough(frame, youngest, w, "warmup")
    y = frame.select(tickers).to_numpy()
    sigma = _ewma_path(y, lam, w)[w - 1 :]
    rho = _normalise(sigma)
    kept = frame.slice(w - 1)
    series = pl.concat(
        [
            kept.select(
                "ts", "close_ts", pl.lit(tickers[i]).alias("ticker_i"), pl.lit(tickers[j]).alias("ticker_j"),
                pl.Series("covariance", sigma[:, i, j]), pl.Series("correlation", rho[:, i, j]), "after_gap",
            )
            for i, j in _pairs(tickers, diagonal=True)
        ]
    ).sort("ts", "ticker_i", "ticker_j")
    return EwmaFit(tickers=tickers, lam=lam, warmup=w, nobs=frame.height, fitted=False, series=series)


# ── the walk-forward ─────────────────────────────────────────────────────────


def walk_forward(
    returns: pl.LazyFrame | pl.DataFrame,
    *,
    model: str = "gjr",
    dist: str = "t",
    corr: str = "dcc",
    split,
    window: int | str = "expanding",
    every: int = 1,
    horizons=(1,),
    factors: pl.DataFrame | None = None,
    column: str = "return",
    lam: float = 0.94,
    sample_window: int = 180,
    simulations: int = 1000,
    seed: int = 0,
    min_obs: int = 500,
) -> pl.DataFrame:
    """Σ forecasts from every origin at or after `split`, one row per (origin, h, pair i ≤ j), one refit schedule for all tickers.

    `ts, close_ts, h, target_ts, ticker_i, ticker_j, covariance,
    cum_covariance, correlation, correlation_target, fitted_through, fit_from,
    refit, after_gap, n_eff, a, b`, in squared return units per bar. At each refit, step 1 (per
    ticker, arch) and step 2 (a, b) are fitted on the refit's window only;
    origins until the next refit are filtered with them fixed. D at h is each
    ticker's own h-step variance, as `vol.walk_forward` forecasts it; R at h is
    R̄ + (a + b)^(h−1)(Rₜ₊₁ − R̄); `cum_covariance` sums Σ over bars 1…h.
    `correlation_target` is that R̄, the correlation the fit reverts to: Q̄
    normalised (DCC, CCC) or the profiled S̃ (cDCC). It is null where nothing
    reverts (`ewma`, `sample`, `iewma`).

    `corr="ewma"` is RiskMetrics' covariance with `lam`: nothing is fitted
    (`refit` false, `fitted_through` the origin's close, `a` and `b` null), the
    forecast is flat, and `min_obs` does not apply, only the warm-up. `n_eff`
    is `ewma_n_eff(b)` over the rows since the refit window began (DCC), or
    `ewma_n_eff(lam)` over the rows so far (EWMA).

    `corr="iewma"` is RiskMetrics' EWMA applied to the same step 1's z, not to
    the returns: a = 1 − `lam`, b = `lam`, nothing estimated in step 2, and R
    never reverts. With the σ shared, it differs from DCC in R alone.
    `corr="ccc"` is Bollerslev's (1990) constant correlation over the same
    step 1: R = Q̄ normalised, a = b = 0, `n_eff` the window's rows.
    `corr="sample"` is the equal-weight second moment of the last
    `sample_window` joint returns, zero mean, nothing fitted, flat.

    With `factors` (`gr.timeseries.seasonal_factors`, fitted no later than
    `split`), each ticker's returns are divided by its cell's factor before
    anything is fitted, and each target bar's Σᵢⱼ is multiplied by fᵢ·fⱼ of
    its cell.
    """
    ewma_path = corr in ("ewma", "sample")
    if not ewma_path:
        _check(model, dist, corr)
    hs = sorted({int(h) for h in horizons})
    if not hs or hs[0] < 1:
        raise Refused(f"horizons={list(horizons)}: each must be a whole number of bars ≥ 1")
    if not ewma_path and model == "egarch" and dist in _POLYNOMIAL_TAILS and hs[-1] > 1:
        raise Refused(
            f"EGARCH with dist={dist!r} has no variance beyond one step: E_t[σ²_t+2] carries E[exp(α|z| + γz)], "
            "infinite under a Student-t tail; walk it with horizons=[1], or dist='normal' or 'ged'"
        )
    at = utils.instant("split", split)
    if factors is not None:
        if column != "return":
            raise Refused(f"factors deseasonalise raw returns; column={column!r} would divide twice")
        ends = factors["fit_end"]
        if ends.len() and ends.max().timestamp() * 1_000_000 > at:
            raise Refused(f"the factors were fitted through {ends.max()}, after the split {split}: that is lookahead")
        returns = timeseries.deseasonalize(utils.lazy(returns).collect(), factors).with_columns(pl.col("deseasonalized").alias("return"))
    tickers, frame, youngest = joint(returns, column=column)
    N, H = len(tickers), hs[-1]
    y = frame.select(tickers).to_numpy()
    schedule = timeseries.walk_forward_origins(frame.select(pl.lit("joint").alias("ticker"), "ts", "close_ts"), split, window=window, every=every)
    first = frame.height - schedule.height
    width = frame["close_ts"][0] - frame["ts"][0]
    origins = np.arange(first, frame.height)

    if corr == "sample":
        if first + 1 < sample_window:
            raise Refused(f"the first origin has {first + 1} joint returns, under sample_window={sample_window}: it starts at {youngest}'s first return")
        path = np.stack([y[t - sample_window + 1 : t + 1].T @ y[t - sample_window + 1 : t + 1] / sample_window for t in origins])
        sig = np.repeat(path[:, None], H, axis=1)  # flat, as EWMA's
        schedule = schedule.with_columns(pl.col("close_ts").alias("fitted_through"), pl.lit(False).alias("refit"))
        ne = np.full(len(origins), float(sample_window))
        a_col = b_col = np.full(len(origins), np.nan)
        tgt = np.full((len(origins), N, N), np.nan)
    elif ewma_path:
        w = _warmup(lam, None)
        if first + 1 < w:
            raise Refused(f"the first origin has {first + 1} joint returns, under the warm-up {w} at lam={lam}: it starts at {youngest}'s first return")
        path = _ewma_path(y, lam, w)[origins]
        sig = np.repeat(path[:, None], H, axis=1)  # flat: Σₜ₊ₕ = Σₜ₊₁
        schedule = schedule.with_columns(pl.col("close_ts").alias("fitted_through"), pl.lit(False).alias("refit"))
        ne = np.array([ewma_n_eff(lam, t + 1) for t in origins])
        a_col = b_col = np.full(len(origins), np.nan)
        tgt = np.full((len(origins), N, N), np.nan)
    else:
        refits = [first + i for i, r in enumerate(schedule["refit"].to_list()) if r]
        first_window = refits[0] + 1 if window == "expanding" else window
        if first_window < min_obs:
            raise Refused(f"the first refit's window holds {first_window} joint returns, under min_obs={min_obs}: the sample starts at {youngest}'s first return")
        sig, ne, a_col, b_col, tgt = [], [], [], [], []
        ys = y * SCALE
        for n, r in enumerate(refits):
            stop = refits[n + 1] if n + 1 < len(refits) else frame.height
            lo = 0 if window == "expanding" else r - window + 1
            z = np.empty((stop - lo, N))
            d = np.empty((stop - r, H, N))
            for k in range(N):
                m = _arch.model(ys[:stop, k], model, dist, seed=seed + r)
                res = m.fit(disp="off", first_obs=lo, last_obs=r + 1)
                fixed = m.fix(res.params, first_obs=lo, last_obs=stop)
                z[:, k] = np.asarray(fixed.std_resid, dtype=float)[lo:stop]
                d[:, :, k] = _arch.forecast(res, start=r, horizon=H, simulate=model in SIMULATED, simulations=simulations) / SCALE**2
            if corr == "iewma":
                # EWMA on z (integrated DCC): a = 1 − λ, b = λ, so (1 − a − b)Q̄ vanishes and R never reverts.
                zt = z[: r + 1 - lo]
                est = {"a": 1 - lam, "b": lam, "qbar": zt.T @ zt / len(zt)}
            else:
                est = estimate(z[: r + 1 - lo], corr=corr)
            a, b, qbar = est["a"], est["b"], est["qbar"]
            _, rr = filter(z, a, b, qbar, corr=corr)
            rbar = _normalise(qbar)
            r1 = rr[r + 1 - lo : stop + 1 - lo]  # R for the bar after each origin r … stop − 1
            decay = (a + b) ** np.arange(H)
            rh = rbar + decay[None, :, None, None] * (r1 - rbar)[:, None]
            sd = np.sqrt(d)
            sig.append(sd[..., :, None] * rh * sd[..., None, :])
            tgt.append(np.repeat((np.full((N, N), np.nan) if corr == "iewma" else rbar)[None], stop - r, axis=0))
            # CCC's R̄ rests on the whole window, equally weighted.
            ne.extend(float(r + 1 - lo) if corr == "ccc" else ewma_n_eff(b, t - lo + 1) for t in range(r, stop))
            a_col.extend([a] * (stop - r))
            b_col.extend([b] * (stop - r))
        sig, tgt = np.concatenate(sig), np.concatenate(tgt)
        ne, a_col, b_col = np.asarray(ne), np.asarray(a_col), np.asarray(b_col)

    if factors is not None:
        f = np.stack([_target_factors(schedule, factors, t, width, H) for t in tickers], axis=-1)  # origins × H × N
        sig = sig * f[..., :, None] * f[..., None, :]
    cum = np.cumsum(sig, axis=1)
    rho = _normalise(sig)
    base = schedule.select("ts", "close_ts", "fitted_through", "fit_from", "refit").with_columns(
        frame["after_gap"].slice(first).alias("after_gap"), pl.Series("n_eff", ne), pl.Series("a", a_col), pl.Series("b", b_col)
    )
    if ewma_path:
        base = base.with_columns(pl.lit(None, pl.Float64).alias("a"), pl.lit(None, pl.Float64).alias("b"))
    rows = []
    for h in hs:
        for i, j in _pairs(tickers, diagonal=True):
            rows.append(
                base.with_columns(
                    pl.lit(h, pl.Int64).alias("h"),
                    (pl.col("close_ts") + width * (h - 1)).alias("target_ts"),
                    pl.lit(tickers[i]).alias("ticker_i"),
                    pl.lit(tickers[j]).alias("ticker_j"),
                    pl.Series("covariance", sig[:, h - 1, i, j]),
                    pl.Series("cum_covariance", cum[:, h - 1, i, j]),
                    pl.Series("correlation", rho[:, h - 1, i, j]),
                    pl.Series("correlation_target", tgt[:, i, j], nan_to_null=True),
                )
            )
    return pl.concat(rows).select(_OUT).sort("close_ts", "h", "ticker_i", "ticker_j")


def _target_factors(schedule: pl.DataFrame, factors: pl.DataFrame, ticker: str, width, H: int) -> np.ndarray:
    """One ticker's factor at each origin's target bars, (origins × H)."""
    from .vol.walk import _target_factor2

    one = factors.filter(pl.col("ticker") == ticker)
    if one.height == 0:
        raise Refused(f"no factors for {ticker}")
    return np.sqrt(_target_factor2(schedule.with_columns(pl.lit(ticker).alias("ticker")), one, width, H))


# ── scoring covariance forecasts ─────────────────────────────────────────────

LOSSES = ("stein", "frobenius", "gmv")


def _matrices(walked: pl.DataFrame, h: int) -> dict:
    """`{close_ts: (tickers, Σ)}` of one walk at horizon h."""
    rows = walked.filter(pl.col("h") == h)
    tickers = sorted(set(rows["ticker_i"].to_list()) | set(rows["ticker_j"].to_list()))
    index = {t: k for k, t in enumerate(tickers)}
    out = {}
    for (close,), group in rows.group_by("close_ts", maintain_order=True):
        s = np.full((len(tickers), len(tickers)), np.nan)
        for i, j, v in zip(group["ticker_i"], group["ticker_j"], group["covariance"]):
            s[index[i], index[j]] = s[index[j], index[i]] = v
        out[close] = (tickers, s)
    return out


def losses(sigma, r) -> dict:
    """The three losses of one forecast Σ against the realised returns r of its bar, with r rᵀ as the proxy.

    - `stein`: ln|Σ| + rᵀΣ⁻¹r, the multivariate QLIKE (Stein) loss, robust to a
      noisy conditionally unbiased proxy (Patton and Sheppard 2009; Laurent,
      Rombouts and Violante 2013);
    - `frobenius`: ‖r rᵀ − Σ‖²_F, the multivariate MSE, also robust;
    - `gmv`: (wᵀr)², the realised variance of the global minimum-variance
      portfolio w = Σ⁻¹1 / 1ᵀΣ⁻¹1 (Engle and Colacito 2006).
    """
    s = (np.asarray(sigma, dtype=float) + np.asarray(sigma, dtype=float).T) / 2
    r = np.asarray(r, dtype=float)
    chol = np.linalg.cholesky(s)
    z = np.linalg.solve(chol, r)
    logdet = 2 * np.log(np.diag(chol)).sum()
    ones = np.linalg.solve(chol.T, np.linalg.solve(chol, np.ones(len(r))))
    w = ones / ones.sum()
    return {"stein": float(logdet + z @ z), "frobenius": float(((np.outer(r, r) - s) ** 2).sum()), "gmv": float((w @ r) ** 2)}


def score(forecasts: dict, returns: pl.LazyFrame | pl.DataFrame, *, h: int = 1, column: str = "return") -> pl.DataFrame:
    """`close_ts, model, stein, frobenius, gmv`: each model's h-step Σ scored against the realised joint returns of its target bar.

    Only origins every model forecast, whose target bar is in the joint sample,
    are scored, so every model is judged on the same bars.
    """
    tickers, frame, _ = joint(returns, column=column)
    width = frame["close_ts"][0] - frame["ts"][0]
    realised = {ts: row for ts, row in zip(frame["ts"].to_list(), frame.select(tickers).rows())}
    per = {name: _matrices(w, h) for name, w in forecasts.items()}
    common = set.intersection(*(set(m) for m in per.values()))
    rows = []
    for close in sorted(common):
        target = close + width * (h - 1)
        if target not in realised:
            continue
        for name, mats in per.items():
            names, s = mats[close]
            if names != tickers:
                raise Refused(f"{name} forecasts {names}; the returns hold {tickers}")
            rows.append({"close_ts": close, "model": name, **losses(s, realised[target])})
    if not rows:
        raise Refused("no origin was forecast by every model with its target bar in the sample")
    return pl.DataFrame(rows)


def compare(scores: pl.DataFrame, *, loss: str, benchmark: str, size: float = 0.1, reps: int = 1000, seed: int = 0) -> pl.DataFrame:
    """`model, mean_loss, dm, dm_p, mcs_p, included`: Diebold–Mariano against `benchmark` and the Model Confidence Set.

    DM on d = loss − benchmark's (negative favours the model), its long-run
    variance Newey–West (Bartlett) at ⌊4(T/100)^(2/9)⌋ lags, with the
    Harvey–Leybourne–Newbold factor at h = 1, √((T − 1)/T), and a two-sided
    p from t(T − 1). The MCS is Hansen, Lunde and Nason's (2011), range
    statistic, stationary bootstrap with the block chosen as `evaluate`
    chooses it, on the loss matrix of the scored bars.
    """
    from arch.bootstrap import MCS

    from . import evaluate

    if loss not in LOSSES:
        raise Refused(f"loss={loss!r} is not one of {', '.join(LOSSES)}")
    wide = scores.pivot(on="model", index="close_ts", values=loss).sort("close_ts").drop_nulls()
    names = [c for c in wide.columns if c != "close_ts"]
    if benchmark not in names:
        raise Refused(f"benchmark={benchmark!r} is not among {', '.join(names)}")
    x = wide.select(names).to_numpy()
    test = MCS(x, size, reps=reps, block_size=evaluate._block(x), method="R", bootstrap="stationary", seed=seed)
    test.compute()
    pv = {names[int(i)]: float(v) for i, v in zip(test.pvalues.index, np.asarray(test.pvalues).ravel())}
    included = {names[int(i)] for i in test.included}
    rows = []
    for k, name in enumerate(names):
        d = dm_hac(x[:, k] - x[:, names.index(benchmark)]) if name != benchmark else {"statistic": None, "p_value": None}
        rows.append({"model": name, "mean_loss": float(x[:, k].mean()), "dm": d.get("statistic"), "dm_p": d.get("p_value"), "mcs_p": pv[name], "included": name in included, "bars": x.shape[0]})
    return pl.DataFrame(rows, infer_schema_length=None).sort("mean_loss")


def dm_hac(d) -> dict:
    """Diebold–Mariano at h = 1 with a Newey–West variance at ⌊4(T/100)^(2/9)⌋ lags and the HLN factor."""
    from scipy import stats as sstats

    from . import evaluate

    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 3:
        raise Refused(f"{n} loss differentials are too few for a Diebold–Mariano test")
    lags = int(4 * (n / 100) ** (2 / 9))
    v = evaluate._lrv(d, lags, "bartlett")
    stat = d.mean() / np.sqrt(v / n) * np.sqrt((n - 1) / n)
    return {"statistic": float(stat), "p_value": float(2 * sstats.t.sf(abs(stat), n - 1)), "n": n, "lags": lags}


# ── constancy (test-the-constant-correlation) ───────────────────────────────


def constancy(z, rbar, *, lags: int = 5, robust: bool = True) -> dict:
    """Engle and Sheppard's (2001, NBER WP 8554, §4) test that the correlation is constant at `rbar`.

    `z` (T × N) are the margins' standardised returns and `rbar` the constant
    correlation under test (a Q̄ is normalised first). Whitened by the symmetric
    inverse square root, uₜ = R̄^(−½)zₜ, the off-diagonal products uᵢₜuⱼₜ have
    mean zero and no dynamics under the null. They are stacked over the
    N(N−1)/2 pairs into one regression on a constant and `lags` of themselves,
    with the coefficients shared across pairs, and δ̂ (constant and lags) is
    tested jointly: χ²(lags + 1). The constant catches a correlation that has
    moved away from `rbar`, the lags one that moves with its past.

    The paper's statistic is δ̂′X′Xδ̂/σ̂², homoskedastic. The products of
    Student-t returns at ν ≈ 3 have no finite variance to speak of, so
    `robust=True` (the default) uses White's heteroskedasticity-consistent
    covariance instead: δ̂′(X′X)(X′Ω̂X)⁻¹(X′X)δ̂. The test cannot tell dynamic
    correlation from a misspecified margin (the paper's footnote 8).
    """
    from scipy import stats as sstats

    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[1] < 2:
        raise Refused(f"a constancy test needs two series or more; z has shape {z.shape}")
    if lags < 1:
        raise Refused(f"lags={lags}: at least one")
    T, N = z.shape
    if T <= lags + 2:
        raise Refused(f"{T} returns are too few for {lags} lags")
    r = _normalise(np.asarray(rbar, dtype=float))
    w, v = np.linalg.eigh(r)
    if w.min() <= 0:
        raise Refused(f"rbar is not positive definite (smallest eigenvalue {w.min():.3g})")
    u = z @ (v @ np.diag(w**-0.5) @ v.T)
    i, j = np.triu_indices(N, k=1)
    y = u[:, i] * u[:, j]  # T × pairs
    rows = T - lags
    regressand = y[lags:].T.reshape(-1)  # pair-major
    columns = [np.ones(rows * len(i))] + [y[lags - k : T - k].T.reshape(-1) for k in range(1, lags + 1)]
    x = np.column_stack(columns)
    xtx = x.T @ x
    delta = np.linalg.solve(xtx, x.T @ regressand)
    e = regressand - x @ delta
    if robust:
        meat = (x * (e**2)[:, None]).T @ x
        stat = float(delta @ xtx @ np.linalg.solve(meat, xtx @ delta))
    else:
        stat = float(delta @ xtx @ delta / (e @ e / (len(e) - x.shape[1])))
    return {"statistic": stat, "p_value": float(sstats.chi2.sf(stat, lags + 1)), "lags": lags, "n": T, "robust": robust}
