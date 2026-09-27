"""Scoring variance forecasts: robust losses, pairwise and many-model comparison, stability, and tail backtests.

Every score starts from `align`, so no test sees a row another does not.
Sources: QLIKE (Patton 2011, eq. 6), MZ-GLS (Patton and Sheppard 2009),
Diebold–Mariano with Harvey, Leybourne and Newbold's (1997) correction, the
Model Confidence Set (Hansen, Lunde and Nason 2011) and SPA (Hansen 2005)
through arch.bootstrap, Giacomini and Rossi's (2010) fluctuation test, and
VaR/ES backtests: Kupiec (1995), Christoffersen (1998), Engle and Manganelli's
(2004) DQ, and FZ0 (Patton, Ziegel and Chen 2019).
"""

from math import log, sqrt

import numpy as np
import polars as pl
from arch.bootstrap import MCS, SPA
from scipy import stats

from .. import timeseries, utils
from .._errors import Refused

KINDS = ("r2", "parkinson", "rv")
# Giacomini and Rossi (2010), Table I, two-sided: μ = m/P → (5%, 10%). Taken from
# Rossi's own Stata command giacross.ado, which hardcodes the table; the
# journal's text was not reached.
FLUCTUATION = {
    0.1: (3.393, 3.170),
    0.2: (3.179, 2.948),
    0.3: (3.012, 2.766),
    0.4: (2.890, 2.626),
    0.5: (2.779, 2.500),
    0.6: (2.634, 2.356),
    0.7: (2.560, 2.252),
    0.8: (2.433, 2.130),
    0.9: (2.248, 1.950),
}


# ── Proxies and alignment ───────────────────────────────────────────────────


def proxies(frame: pl.LazyFrame | pl.DataFrame, kind: str = "r2") -> pl.DataFrame:
    """`ticker, ts, proxy`: one realized variance per bar, on the bar's `ts`.

    `r2`: the squared contiguous log return (unbiased, noisy). `parkinson`:
    ln(H/L)²/(4 ln 2), less noisy but biased low under discrete sampling.
    `rv`: `realized_from`'s `rv` for the bucket (pass that frame).
    """
    if kind not in KINDS:
        raise Refused(f"kind={kind!r} is not one of {', '.join(KINDS)}")
    if kind == "rv":
        utils.require(frame, ("ticker", "ts", "rv"), "pass gr.timeseries.realized_from output")
        return utils.lazy(frame).select("ticker", "ts", pl.col("rv").alias("proxy")).drop_nulls("proxy").sort("ticker", "ts").collect()
    if kind == "parkinson":
        utils.require(frame, ("ticker", "ts", "high", "low"), "load bars with gr.market.candles")
        return utils.lazy(frame).select("ticker", "ts", timeseries.parkinson_term().alias("proxy")).sort("ticker", "ts").collect()
    r = timeseries.returns(frame, kind="log")
    return r.select("ticker", "ts", (pl.col("return") ** 2).alias("proxy")).drop_nulls("proxy")


def align(forecasts: pl.DataFrame, proxy: pl.DataFrame, *, cumulative: bool = True, drop_after_gap: bool = True) -> pl.DataFrame:
    """The forecasts with `forecast` and `proxy`: point at `target_ts`, or cumulative over exactly the h bars.

    A cumulative proxy sums the bars from the origin's next bar through
    `target_ts`, and is null unless all h are there, so a hole leaves the
    target blank rather than short. Rows marked `after_gap` are dropped by
    default (roadmap D5).
    """
    utils.require(forecasts, ("ticker", "close_ts", "h", "target_ts", "variance", "cum_variance"), "forecast with gr.models.vol")
    f = forecasts.filter(~pl.col("after_gap")) if drop_after_gap and "after_gap" in forecasts.columns else forecasts
    p = proxy.sort("ticker", "ts").with_columns(
        pl.col("proxy").cum_sum().over("ticker").alias("_cs"),
        pl.int_range(pl.len()).over("ticker").alias("_ci"),
    )
    if not cumulative:
        return f.join(p.select("ticker", pl.col("ts").alias("target_ts"), "proxy"), on=["ticker", "target_ts"], how="left").with_columns(
            pl.col("variance").alias("forecast")
        )
    end = p.select("ticker", pl.col("ts").alias("target_ts"), pl.col("_cs").alias("_cs_end"), pl.col("_ci").alias("_ci_end"))
    start = p.select(
        "ticker", pl.col("ts").alias("close_ts"), (pl.col("_cs") - pl.col("proxy")).alias("_cs_before"), pl.col("_ci").alias("_ci_start")
    )
    out = f.join(end, on=["ticker", "target_ts"], how="left").join(start, on=["ticker", "close_ts"], how="left")
    complete = (pl.col("_ci_end") - pl.col("_ci_start") + 1) == pl.col("h")
    return out.with_columns(
        pl.when(complete).then(pl.col("_cs_end") - pl.col("_cs_before")).alias("proxy"),
        pl.col("cum_variance").alias("forecast"),
    ).drop("_cs_end", "_ci_end", "_cs_before", "_ci_start")


# ── Losses and pairwise tests ───────────────────────────────────────────────


def qlike(proxy, forecast):
    """proxy/f + ln f (Patton 2011, eq. 6): ranks as the textbook form does, and is finite at a zero proxy."""
    return proxy / forecast + np.log(forecast) if not isinstance(proxy, pl.Expr) else proxy / forecast + forecast.log()


def mse(proxy, forecast):
    """(proxy − f)²."""
    return (proxy - forecast) ** 2


def _lrv(d: np.ndarray, lags: int, kernel: str) -> float:
    e = d - d.mean()
    n = e.size
    v = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 if kernel == "rectangular" else 1 - k / (lags + 1)
        v += 2 * w * float(e[k:] @ e[:-k]) / n
    return v


def dm(loss, benchmark_loss, h: int = 1) -> dict:
    """Diebold–Mariano on d = loss − benchmark_loss (negative favours the model), HLN-corrected.

    Rectangular long-run variance over h − 1 lags (d is MA(h−1) under
    optimality); Bartlett when that is not positive, and `kernel` says which.
    DM* = DM·√((n+1−2h+h(h−1)/n)/n), two-sided p from t(n−1).
    """
    d = np.asarray(loss, dtype=float) - np.asarray(benchmark_loss, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 3:
        raise Refused(f"{n} loss differentials are too few for a Diebold–Mariano test")
    lags, kernel = max(h - 1, 0), "rectangular"
    v = _lrv(d, lags, kernel)
    if v <= 0:
        kernel = "bartlett"
        v = _lrv(d, max(lags, 1), kernel)
    stat = d.mean() / sqrt(v / n) if v > 0 else float("nan")
    factor = sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    adj = stat * factor
    return {"statistic": float(adj), "p_value": float(2 * stats.t.sf(abs(adj), n - 1)), "n": n, "kernel": kernel, "hln_factor": factor}


def mz_gls(proxy, forecast, h: int = 1) -> dict:
    """proxy/f = α·(1/f) + β + e by OLS; Wald of α = 0, β = 1 with a Newey–West covariance over max(h−1, 0) lags."""
    p, f = np.asarray(proxy, dtype=float), np.asarray(forecast, dtype=float)
    keep = np.isfinite(p) & np.isfinite(f) & (f > 0)
    p, f = p[keep], f[keep]
    y = p / f
    x = np.column_stack([1 / f, np.ones(f.size)])
    xtx_inv = np.linalg.inv(x.T @ x)
    b = xtx_inv @ x.T @ y
    u = y - x @ b
    s = np.zeros((2, 2))
    lags = max(h - 1, 0)
    for k in range(lags + 1):
        g = (x[k:] * u[k:, None]).T @ (x[: x.shape[0] - k] * u[: u.size - k, None])
        w = 1 - k / (lags + 1) if k else 1.0
        s += w * (g if k == 0 else g + g.T)
    cov = xtx_inv @ s @ xtx_inv
    r = b - np.array([0.0, 1.0])
    wald = float(r @ np.linalg.solve(cov, r))
    return {"alpha": float(b[0]), "beta": float(b[1]), "wald": wald, "p_value": float(stats.chi2.sf(wald, 2)), "n": int(f.size)}


# ── Many models ─────────────────────────────────────────────────────────────


def losses(aligned: pl.DataFrame, h: int) -> pl.DataFrame:
    """Wide QLIKE losses on the origins every model forecast at horizon h: `close_ts` then one column per model."""
    utils.require(aligned, ("model", "close_ts", "h", "forecast", "proxy"), "align forecasts that carry a model column")
    long = aligned.filter((pl.col("h") == h) & pl.col("proxy").is_not_null() & (pl.col("forecast") > 0)).with_columns(
        qlike(pl.col("proxy"), pl.col("forecast")).alias("_loss")
    )
    return long.pivot(on="model", index="close_ts", values="_loss", aggregate_function=None).drop_nulls().sort("close_ts")


def scorecard(aligned: pl.DataFrame, *, benchmark: str) -> pl.DataFrame:
    """Per model and h: `n, qlike, mse, qlike_ratio, dm, dm_p`, each on the rows the model and the benchmark share."""
    rows = []
    for h in sorted(aligned["h"].unique().to_list()):
        at = aligned.filter((pl.col("h") == h) & pl.col("proxy").is_not_null() & (pl.col("forecast") > 0))
        base = at.filter(pl.col("model") == benchmark).select("close_ts", pl.col("forecast").alias("_bf"), "proxy")
        if base.height == 0:
            raise Refused(f"the benchmark {benchmark!r} has no scored rows at h={h}")
        for model in sorted(at["model"].unique().to_list()):
            m = at.filter(pl.col("model") == model).select("close_ts", "forecast").join(base, on="close_ts")
            p, f, bf = m["proxy"].to_numpy(), m["forecast"].to_numpy(), m["_bf"].to_numpy()
            lm, lb = qlike(p, f), qlike(p, bf)
            test = dm(lm, lb, h) if model != benchmark and m.height >= 3 else {"statistic": None, "p_value": None}
            rows.append(
                {
                    "model": model,
                    "h": h,
                    "n": m.height,
                    "qlike": float(lm.mean()),
                    "mse": float(mse(p, f).mean()),
                    "qlike_ratio": _ratio(p, f, bf),
                    "dm": test["statistic"],
                    "dm_p": test["p_value"],
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None)


def _ratio(p: np.ndarray, f: np.ndarray, bf: np.ndarray) -> float | None:
    """Mean textbook QLIKE, p/f − ln(p/f) − 1 ≥ 0, of the model over the benchmark's, on rows with a positive proxy.

    proxy/f + ln f can be negative, so a ratio of its means is meaningless; the
    textbook form differs from it only by terms in the proxy, so the ranking
    is the same.
    """
    keep = p > 0
    if not keep.any():
        return None
    textbook = lambda g: p[keep] / g[keep] - np.log(p[keep] / g[keep]) - 1  # noqa: E731
    return float(textbook(f).mean() / textbook(bf).mean())


def _block(losses: np.ndarray) -> int:
    d = losses - losses.mean(axis=1, keepdims=True)
    return max(1, int(round(float(np.median([timeseries.optimal_block(pl.Series(d[:, j])) for j in range(d.shape[1])])))))


def mcs(aligned: pl.DataFrame, h: int, *, size: float = 0.1, reps: int = 1000, seed: int = 0) -> pl.DataFrame:
    """`model, pvalue, included`: the Model Confidence Set on QLIKE over the origins every model forecast at h."""
    wide = losses(aligned, h)
    names = [c for c in wide.columns if c != "close_ts"]
    if len(names) < 2:
        raise Refused(f"a confidence set needs two models or more; h={h} has {names}")
    x = wide.select(names).to_numpy()
    test = MCS(x, size, reps=reps, block_size=_block(x), method="R", bootstrap="stationary", seed=seed)
    test.compute()
    pv = {names[int(i)]: float(v) for i, v in zip(test.pvalues.index, np.asarray(test.pvalues).ravel())}
    included = {names[int(i)] for i in test.included}
    return pl.DataFrame({"model": names, "pvalue": [pv[n] for n in names], "included": [n in included for n in names], "rows": [x.shape[0]] * len(names)}).sort(
        "pvalue", descending=True
    )


def spa(aligned: pl.DataFrame, h: int, *, benchmark: str, reps: int = 1000, seed: int = 0) -> dict:
    """SPA of every other model against `benchmark` on QLIKE at h: `lower, consistent, upper` p-values."""
    wide = losses(aligned, h)
    if benchmark not in wide.columns:
        raise Refused(f"no {benchmark!r} losses at h={h}")
    others = [c for c in wide.columns if c not in ("close_ts", benchmark)]
    b, m = wide[benchmark].to_numpy(), wide.select(others).to_numpy()
    test = SPA(b, m, block_size=_block(np.column_stack([b, m])), reps=reps, bootstrap="stationary", seed=seed)
    test.compute()
    p = np.asarray(test.pvalues).ravel()
    return {"lower": float(p[0]), "consistent": float(p[1]), "upper": float(p[2]), "rows": int(b.size), "models": others}


def fluctuation(aligned: pl.DataFrame, h: int, *, model: str, benchmark: str, mu: float = 0.3) -> pl.DataFrame:
    """Giacomini and Rossi's (2010) fluctuation test: √m · rolling mean of d over m = round(μP) / the full-sample HAC σ̂."""
    if round(mu, 1) not in FLUCTUATION or abs(mu - round(mu, 1)) > 1e-9:
        raise Refused(f"mu={mu} has no tabulated critical value; use one of {', '.join(str(k) for k in FLUCTUATION)}")
    wide = losses(aligned, h)
    for name in (model, benchmark):
        if name not in wide.columns:
            raise Refused(f"no {name!r} losses at h={h}")
    d = wide[model].to_numpy() - wide[benchmark].to_numpy()
    P = d.size
    m = max(2, round(mu * P))
    sigma = sqrt(_lrv(d, max(h - 1, int(P ** (1 / 3))), "bartlett"))
    roll = np.convolve(d, np.ones(m) / m, mode="valid")
    c5, c10 = FLUCTUATION[round(mu, 1)]
    return pl.DataFrame(
        {"close_ts": wide["close_ts"][m - 1 :], "statistic": np.sqrt(m) * roll / sigma, "critical_5": c5, "critical_10": c10}
    )


# ── Value at risk ───────────────────────────────────────────────────────────


def value_at_risk(aligned: pl.DataFrame, z, alpha: float) -> pl.DataFrame:
    """VaR and ES by filtered historical simulation: σ̂·q_α(z) and σ̂·mean(z | z ≤ q_α(z)), from in-sample z; mean zero.

    Pass h = 1 rows (the next bar); z are a fit's in-sample standardised
    residuals, so the fitted distribution is not assumed right, which is what
    is being tested.
    """
    zz = np.sort(np.asarray(z, dtype=float)[np.isfinite(np.asarray(z, dtype=float))])
    if not 0 < alpha < 0.5:
        raise Refused(f"alpha={alpha} must be a lower-tail probability in (0, 0.5)")
    q = float(np.quantile(zz, alpha))
    es = float(zz[zz <= q].mean())
    return aligned.filter(pl.col("h") == 1).with_columns(
        (pl.col("variance").sqrt() * q).alias("var"), (pl.col("variance").sqrt() * es).alias("es")
    )


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * log(y)


def var_backtest(returns, var, es, alpha: float) -> dict:
    """Hits of r < VaR, and Kupiec, Christoffersen, DQ and FZ0 on them.

    - Kupiec LR_uc = 2[(T−x) ln((1−α̂)/(1−α)) + x ln(α̂/α)], χ²(1); null with no hits (undefined; Campbell 2005).
    - Christoffersen LR_ind from the transition counts with 0·ln 0 = 0, χ²(1); LR_cc = LR_uc + LR_ind, χ²(2).
    - DQ: Hit − α on a constant, four lagged hits and VaR; δ̂′X′Xδ̂/(α(1−α)), χ²(6).
    - FZ0: mean of −1{r≤v}(v−r)/(αe) + v/e + ln(−e) − 1, requiring e ≤ v < 0.
    """
    r, v, e = (np.asarray(a, dtype=float) for a in (returns, var, es))
    keep = np.isfinite(r) & np.isfinite(v) & np.isfinite(e)
    r, v, e = r[keep], v[keep], e[keep]
    if np.any(v >= 0) or np.any(e > v):
        raise Refused("VaR and ES must be negative returns with ES ≤ VaR")
    hit = (r < v).astype(float)
    T, x = hit.size, int(hit.sum())
    a_hat = x / T
    lr_uc = None if x == 0 else 2 * (_xlogy(T - x, (1 - a_hat) / (1 - alpha)) + x * log(a_hat / alpha))
    prev, cur = hit[:-1], hit[1:]
    n00 = float(((prev == 0) & (cur == 0)).sum())
    n01 = float(((prev == 0) & (cur == 1)).sum())
    n10 = float(((prev == 1) & (cur == 0)).sum())
    n11 = float(((prev == 1) & (cur == 1)).sum())
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    transitions = n00 + n01 + n10 + n11
    lr_ind = None
    if transitions:
        pi = (n01 + n11) / transitions
        ll0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
        ll1 = _xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11)
        lr_ind = -2 * (ll0 - ll1)
    lr_cc = None if lr_uc is None or lr_ind is None else lr_uc + lr_ind
    lags = 4
    dq = None
    if T > lags + 6:
        y = hit[lags:] - alpha
        X = np.column_stack([np.ones(y.size), *[hit[lags - k : T - k] for k in range(1, lags + 1)], v[lags:]])
        delta, *_ = np.linalg.lstsq(X, y, rcond=None)
        dq = float(delta @ X.T @ X @ delta / (alpha * (1 - alpha)))
    fz0 = float(np.mean(-(r <= v).astype(float) * (v - r) / (alpha * e) + v / e + np.log(-e) - 1))
    return {
        "n": T,
        "hits": x,
        "expected": alpha * T,
        "kupiec": lr_uc,
        "kupiec_p": None if lr_uc is None else float(stats.chi2.sf(lr_uc, 1)),
        "independence": lr_ind,
        "independence_p": None if lr_ind is None else float(stats.chi2.sf(lr_ind, 1)),
        "conditional_coverage": lr_cc,
        "conditional_coverage_p": None if lr_cc is None else float(stats.chi2.sf(lr_cc, 2)),
        "dq": dq,
        "dq_p": None if dq is None else float(stats.chi2.sf(dq, lags + 2)),
        "fz0": fz0,
    }


# ── Across horizons ─────────────────────────────────────────────────────────
#
# Quaedvlieg (2021), JBES 39(1):40-53: one verdict across the horizon path.
# Read in the accepted manuscript (EUR repository) and the authors' R package
# MultiHorizonSPA: the uniform test (better at every horizon) and the average
# test (better on average), QS HAC with bandwidth 1.3·T^(1/5), a moving-block
# bootstrap with block 3 and 999 reps, equal weights.


def horizon_losses(aligned: pl.DataFrame, *, model: str, benchmark: str) -> pl.DataFrame:
    """`close_ts` and one column per horizon of QLIKE(benchmark) − QLIKE(model): positive favours the model.

    Only origins where both models have every horizon scored are kept.
    """
    hs = sorted(aligned["h"].unique().to_list())
    frames = []
    for h in hs:
        wide = losses(aligned, h)
        for name in (model, benchmark):
            if name not in wide.columns:
                raise Refused(f"no {name!r} losses at h={h}")
        frames.append(wide.select("close_ts", (pl.col(benchmark) - pl.col(model)).alias(f"h{h}")))
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, on="close_ts", how="inner")
    return out.sort("close_ts")


def _qs_variance(d: np.ndarray) -> float:
    """The Quadratic Spectral HAC long-run variance, bandwidth 1.3·T^(1/5) (Andrews 1991; the paper's choice)."""
    t = d.size
    e = d - d.mean()
    band = 1.3 * t ** 0.2
    v = float(e @ e) / t
    for j in range(1, t):
        x = j / band
        a = 6 * np.pi * x / 5
        k = 25 / (12 * np.pi**2 * x**2) * (np.sin(a) / a - np.cos(a))
        if abs(k) < 1e-10 and j > 5 * band:
            break
        v += 2 * k * float(e[j:] @ e[:-j]) / t
    return max(v, 1e-300)


def _mbb_index(t: int, block: int, rng) -> np.ndarray:
    starts = rng.integers(0, t - block + 1, size=-(-t // block))
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:t]


def _block_variance(x: np.ndarray, block: int) -> np.ndarray:
    """The natural moving-block variance of each column: mean over blocks of (block sum)² / block."""
    t = x.shape[0] - x.shape[0] % block
    sums = x[:t].reshape(-1, block, x.shape[1]).sum(axis=1)
    return (sums**2).mean(axis=0) / block


def _horizon_test(d: np.ndarray, weights, block: int, reps: int, seed: int, uniform: bool) -> tuple[float, float]:
    t = d.shape[0]
    if t < 2 * block:
        raise Refused(f"{t} origins are too few for a block bootstrap with block {block}")
    if not uniform:
        d = (d @ weights)[:, None]
    mean = d.mean(axis=0)
    omega = np.array([_qs_variance(d[:, j]) for j in range(d.shape[1])])
    stat = float(np.min(np.sqrt(t) * mean / np.sqrt(omega)))
    rng = np.random.default_rng(seed)
    centred = d - mean
    exceed = 0
    for _ in range(reps):
        sample = centred[_mbb_index(t, block, rng)]
        var = _block_variance(sample - sample.mean(axis=0), block)
        boot = float(np.min(np.sqrt(t) * sample.mean(axis=0) / np.sqrt(np.maximum(var, 1e-300))))
        exceed += boot > stat
    return stat, exceed / reps


def uspa(aligned: pl.DataFrame, *, model: str, benchmark: str, block: int = 3, reps: int = 999, seed: int = 0) -> dict:
    """Quaedvlieg's uniform SPA: is `model` better than `benchmark` at every horizon? t = minₕ √T d̄ₕ/ω̂ₕ."""
    d = horizon_losses(aligned, model=model, benchmark=benchmark)
    x = d.drop("close_ts").to_numpy()
    stat, p = _horizon_test(x, None, block, reps, seed, uniform=True)
    return {"statistic": stat, "p_value": p, "rows": d.height, "horizons": [c for c in d.columns if c != "close_ts"]}


def aspa(aligned: pl.DataFrame, *, model: str, benchmark: str, weights=None, block: int = 3, reps: int = 999, seed: int = 0) -> dict:
    """Quaedvlieg's average SPA: is `model` better on the weighted average across horizons? t = √T w′d̄/ζ̂; w = 1/H by default."""
    d = horizon_losses(aligned, model=model, benchmark=benchmark)
    x = d.drop("close_ts").to_numpy()
    hcount = x.shape[1]
    w = np.full(hcount, 1 / hcount) if weights is None else np.asarray(weights, dtype=float)
    if w.size != hcount or np.any(w < 0) or abs(w.sum() - 1) > 1e-9:
        raise Refused(f"weights {list(np.round(w, 4))} must be {hcount} non-negative numbers summing to 1")
    stat, p = _horizon_test(x, w, block, reps, seed, uniform=False)
    return {"statistic": stat, "p_value": p, "rows": d.height, "weights": w.tolist()}


# ── When a model wins ───────────────────────────────────────────────────────

GW_THEORY = "Giacomini-White asymptotics assume a rolling estimation window; expanding-window forecasts are outside them (their Comment 2)."


def _gw_frame(aligned: pl.DataFrame, h: int, model: str, benchmark: str) -> pl.DataFrame:
    """`close_ts, d, lagged_diff, log_forecast` at horizon h: d = QLIKE(benchmark) − QLIKE(model); lag h origins."""
    wide = losses(aligned, h)
    for name in (model, benchmark):
        if name not in wide.columns:
            raise Refused(f"no {name!r} losses at h={h}")
    bench_f = aligned.filter((pl.col("model") == benchmark) & (pl.col("h") == h)).select("close_ts", pl.col("forecast").log().alias("log_forecast"))
    return (
        wide.select("close_ts", (pl.col(benchmark) - pl.col(model)).alias("d"))
        .join(bench_f, on="close_ts", how="left")
        .sort("close_ts")
        .with_columns(pl.col("d").shift(h).alias("lagged_diff"))
        .slice(h)
        .drop_nulls()
    )


def gw(aligned: pl.DataFrame, h: int, *, model: str, benchmark: str) -> dict:
    """Giacomini and White's (2006) conditional predictive ability test: can the origin's information predict who wins?

    d = QLIKE(benchmark) − QLIKE(model) (positive favours the model).
    Instruments at origin t: 1, d at t − h (the latest whose target was
    realized by t's close), and the benchmark's log forecast variance, the
    volatility state. Statistic n·Z̄′Ω̂⁻¹Z̄, Z = instruments × d (their eq. 4;
    beyond one step a Bartlett-weighted HAC over h − 1 lags, eq. 8), χ²(3).
    `coefficients` regress d on the instruments; `decision_share` is how often
    the fitted rule picks the model (§3.4). `theory` states the window
    assumption.
    """
    frame = _gw_frame(aligned, h, model, benchmark)
    n = frame.height
    if n < 20:
        raise Refused(f"{n} origins are too few for a conditional test")
    names = ("constant", "lagged_diff", "log_forecast")
    x = np.column_stack([np.ones(n), frame["lagged_diff"].to_numpy(), frame["log_forecast"].to_numpy()])
    d = frame["d"].to_numpy()
    z = x * d[:, None]
    zbar = z.mean(axis=0)
    omega = z.T @ z / n
    for j in range(1, h):
        g = z[j:].T @ z[:-j] / n
        omega += (1 - j / h) * (g + g.T)
    stat = float(n * zbar @ np.linalg.solve(omega, zbar))
    alpha, *_ = np.linalg.lstsq(x, d, rcond=None)
    return {
        "statistic": stat,
        "p_value": float(stats.chi2.sf(stat, len(names))),
        "coefficients": dict(zip(names, (float(a) for a in alpha))),
        "decision_share": float(np.mean(x @ alpha > 0)),
        "n": n,
        "theory": GW_THEORY,
    }


def _loss_tensor(aligned: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """QLIKE losses, T × H × M, on the origins where every model has every horizon; and the model names."""
    hs = sorted(aligned["h"].unique().to_list())
    wides = [losses(aligned, h) for h in hs]
    names = sorted(set.intersection(*[set(w.columns) - {"close_ts"} for w in wides]))
    joined = wides[0].select("close_ts", *[pl.col(n).alias(f"{n}@0") for n in names])
    for k, w in enumerate(wides[1:], start=1):
        joined = joined.join(w.select("close_ts", *[pl.col(n).alias(f"{n}@{k}") for n in names]), on="close_ts", how="inner")
    cube = np.stack([np.column_stack([joined[f"{n}@{k}"].to_numpy() for k in range(len(hs))]) for n in names], axis=2)
    return cube, names


def _pair_stats(d: np.ndarray, w, mean_ref: np.ndarray | None, block: int) -> np.ndarray:
    """Bootstrap-style pair statistics for a stack of differentials d (P × T × H): centred on `mean_ref`, block-variance studentised."""
    p, t, _ = d.shape
    x = d if w is None else (d @ w)[..., None]
    centre = x.mean(axis=1) - (0 if mean_ref is None else (mean_ref if w is None else (mean_ref @ w)[..., None]))
    dem = x - x.mean(axis=1, keepdims=True)
    tt = t - t % block
    sums = dem[:, :tt].reshape(p, -1, block, x.shape[2]).sum(axis=2)
    var = (sums**2).mean(axis=1) / block
    return np.min(np.sqrt(t) * centre / np.sqrt(np.maximum(var, 1e-300)), axis=1)


def mcs_horizons(
    aligned: pl.DataFrame,
    *,
    uniform: bool = True,
    weights=None,
    alpha_t: float = 0.05,
    alpha_mcs: float = 0.1,
    block: int = 3,
    outer: int = 199,
    inner: int = 99,
    seed: int = 0,
) -> pl.DataFrame:
    """Quaedvlieg's (2021, §2.2) multi-horizon Model Confidence Set: `model, pvalue, included, eliminated`.

    Uniform (uMCS, `uniform=True`): models best at every horizon; average
    (aMCS): best on the weighted average. The equivalence statistic is
    max over pairs of tᵢⱼ − cᵢⱼ (dᵢⱼ = Lᵢ − Lⱼ; cᵢⱼ the pair's (1 − α_t)
    bootstrap critical value), its distribution a double moving-block
    bootstrap; the model with the largest row-maximum is eliminated (the
    authors' R code); p-values never fall and the last model's is 1.
    `included` is p ≥ α_mcs, Hansen, Lunde and Nason's convention (the R code's
    last line keeps p ≥ 1 − α, which fits only a reversed p). `outer` and
    `inner` default to 199 and 99; the paper uses 999.
    """
    cube, names = _loss_tensor(aligned)
    t, hcount, m = cube.shape
    if m < 2:
        raise Refused(f"a confidence set needs two models or more with every horizon scored; found {names}")
    w = None if uniform else (np.full(hcount, 1 / hcount) if weights is None else np.asarray(weights, dtype=float))
    rng = np.random.default_rng(seed)
    alive = list(range(m))
    pvalues = {n: None for n in names}
    eliminated = {n: None for n in names}
    running = 0.0
    step = 0
    while len(alive) > 1:
        pairs = [(i, j) for i in alive for j in alive if i != j]
        d = np.stack([cube[:, :, i] - cube[:, :, j] for i, j in pairs])  # P × T × H
        dbar = d.mean(axis=1)
        x = d if w is None else (d @ w)[..., None]
        omega = np.array([[_qs_variance(x[p_, :, k]) for k in range(x.shape[2])] for p_ in range(len(pairs))])
        t_obs = np.min(np.sqrt(t) * x.mean(axis=1) / np.sqrt(omega), axis=1)

        def crit(sample: np.ndarray) -> np.ndarray:
            dem = sample - sample.mean(axis=1, keepdims=True)
            boots = np.empty((inner, len(pairs)))
            for r in range(inner):
                idx = _mbb_index(t, block, rng)
                boots[r] = _pair_stats(dem[:, idx], w, None, block)
            return np.quantile(boots, 1 - alpha_t, axis=0)

        c_obs = crit(d)
        t_max = np.max(t_obs - c_obs)
        exceed = 0
        for _ in range(outer):
            idx = _mbb_index(t, block, rng)
            sample = d[:, idx] - dbar[:, None, :]
            tb = _pair_stats(sample, w, None, block)
            exceed += np.max(tb - crit(sample)) > t_max
        p = exceed / outer
        running = max(running, float(p))
        rows = {}
        for (i, j), v in zip(pairs, t_obs - c_obs):
            rows[i] = max(rows.get(i, -np.inf), v)
        out = max(rows, key=rows.get)
        step += 1
        pvalues[names[out]] = running
        eliminated[names[out]] = step
        alive.remove(out)
    pvalues[names[alive[0]]] = 1.0
    return pl.DataFrame(
        {
            "model": names,
            "pvalue": [pvalues[n] for n in names],
            "included": [pvalues[n] >= alpha_mcs for n in names],
            "eliminated": [eliminated[n] for n in names],
        },
        schema_overrides={"eliminated": pl.Int64},
    ).sort("pvalue", descending=True)


# ── Two Sharpe ratios ───────────────────────────────────────────────────────
#
# Ledoit and Wolf (2008), "Robust performance hypothesis testing with the
# Sharpe ratio", J. Empirical Finance 15(5), 850–859, read in their working
# paper (IEW 320). Δ = f(μ_a, μ_b, γ_a, γ_b) with γ the uncentred second
# moment; its HAC error is the prewhitened QS kernel (Andrews and Monahan
# 1992), and the test inverts a symmetric studentized circular block
# bootstrap interval. Jobson–Korkie/Memmel assumes i.i.d. normal returns and is
# liberal under fat tails and clustering (their Table 1), which is what returns
# sized on σ̂ have.


def _sharpe_delta(mu: np.ndarray, gamma: np.ndarray):
    """Δ = f(v) and ∇f(v) for v = (μ_a, μ_b, γ_a, γ_b), over any leading axes (LW eq. 2 and the gradient under eq. 4)."""
    a, b, c, d = mu[..., 0], mu[..., 1], gamma[..., 0], gamma[..., 1]
    va, vb = c - a * a, d - b * b
    delta = a / np.sqrt(va) - b / np.sqrt(vb)
    grad = np.stack([c / va**1.5, -d / vb**1.5, -0.5 * a / va**1.5, 0.5 * b / vb**1.5], axis=-1)
    return delta, grad


def _moment_columns(x: np.ndarray) -> np.ndarray:
    """y_t = (r_a − μ̂_a, r_b − μ̂_b, r_a² − γ̂_a, r_b² − γ̂_b) over the last two axes (…, T, 2) → (…, T, 4)."""
    sq = x**2
    return np.concatenate([x - x.mean(axis=-2, keepdims=True), sq - sq.mean(axis=-2, keepdims=True)], axis=-1)


def _qs_kernel(x: np.ndarray) -> np.ndarray:
    a = 6 * np.pi * x / 5
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 25 / (12 * np.pi**2 * x**2) * (np.sin(a) / a - np.cos(a))
    return np.where(x == 0, 1.0, k)


def _prewhitened_qs(y: np.ndarray) -> np.ndarray:
    """Ψ̂ of LW §3.1: a VAR(1) prewhitening with singular values capped at 0.97 (Andrews and Monahan 1992), the QS
    kernel on its residuals at Andrews' (1991) AR(1) plug-in bandwidth 1.3221(α̂(2)T)^{1/5}, recoloured, times T/(T−4)."""
    t, k = y.shape
    coef, *_ = np.linalg.lstsq(y[:-1], y[1:], rcond=None)
    u, s, vt = np.linalg.svd(coef.T)
    a = u @ np.diag(np.minimum(s, 0.97)) @ vt
    e = y[1:] - y[:-1] @ a.T
    n = e.shape[0]
    rho = np.array([float(e[1:, i] @ e[:-1, i]) / float(e[:-1, i] @ e[:-1, i]) for i in range(k)])
    s2 = np.array([float(np.mean((e[1:, i] - rho[i] * e[:-1, i]) ** 2)) for i in range(k)])
    alpha2 = float(np.sum(4 * rho**2 * s2**2 / (1 - rho) ** 8) / np.sum(s2**2 / (1 - rho) ** 4))
    band = 1.3221 * (alpha2 * n) ** 0.2
    psi = e.T @ e / n
    if band > 0:
        weights = _qs_kernel(np.arange(1, n) / band)
        for j, w in enumerate(weights, start=1):
            g = e[j:].T @ e[:-j] / n
            psi += w * (g + g.T)
    back = np.linalg.inv(np.eye(k) - a)
    return back @ psi @ back.T * t / (t - 4)


def _block_psi(y: np.ndarray, block: int) -> np.ndarray:
    """Ψ̂* of LW §3.2.2 (Götze and Künsch 1996): (1/l) Σ ζ_j ζ_j′, ζ_j the j-th block sum of y over √b; over leading axes."""
    l = y.shape[-2] // block
    z = y[..., : l * block, :].reshape(*y.shape[:-2], l, block, y.shape[-1]).sum(axis=-2) / sqrt(block)
    return np.einsum("...li,...lj->...ij", z, z) / l


def sharpe_difference(a, b, *, block: int | str = "auto", reps: int = 4999, seed: int = 0, level: float = 0.95) -> dict:
    """Ledoit and Wolf's (2008) test of H0: SR_a = SR_b on paired per-period returns.

    Δ̂ = μ̂_a/σ̂_a − μ̂_b/σ̂_b (σ̂ with divisor T, as f(v̂) gives). `se_hac` is
    s(Δ̂) = √(∇f′Ψ̂∇f/T) from the prewhitened QS kernel; `p_hac` = 2Φ(−|Δ̂|/s).
    The bootstrap resamples circular blocks of pairs, studentizes each replicate
    by its own block Ψ̂*, and gives `p_boot` = (#{|Δ̂*−Δ̂|/s(Δ̂*) ≥ |Δ̂|/s(Δ̂)} + 1)/(M + 1)
    (LW eq. 9) and the symmetric `level` interval Δ̂ ± z*·s(Δ̂) (eq. 7).
    `block="auto"`: the circular-block Politis–White length, 1.5^{1/3} × the
    median stationary length over y's four columns (D_CB = (4/3)ĝ² against
    D_SB = 2ĝ²; Patton, Politis and White 2009), rounded up. A null in either
    series drops the pair. Not annualised: multiply Δ̂ by √(periods per year).
    """
    from math import ceil

    pair = pl.DataFrame({"a": pl.Series(a, dtype=pl.Float64), "b": pl.Series(b, dtype=pl.Float64)}).drop_nulls().drop_nans()
    x = pair.to_numpy()
    t = x.shape[0]
    if t < 10:
        raise Refused(f"{t} paired returns are too few to compare two Sharpe ratios")
    if (x.std(axis=0) == 0).any():
        raise Refused("a constant series has no Sharpe ratio")
    y = _moment_columns(x)
    if block == "auto":
        stationary = float(np.median([timeseries.optimal_block(pl.Series(y[:, j])) for j in range(4)]))
        block = max(1, ceil(1.5 ** (1 / 3) * stationary))
    if t < 2 * block:
        raise Refused(f"{t} paired returns are too few for blocks of {block}")
    delta, grad = _sharpe_delta(x.mean(axis=0), (x**2).mean(axis=0))
    se = sqrt(float(grad @ _prewhitened_qs(y) @ grad) / t)
    d = abs(delta) / se

    rng = np.random.default_rng(seed)
    count = -(-t // block)
    stats_ = []
    for start in range(0, reps, 500):
        m = min(500, reps - start)
        idx = ((rng.integers(0, t, size=(m, count))[:, :, None] + np.arange(block)) % t).reshape(m, -1)[:, :t]
        xs = x[idx]
        ds, gs = _sharpe_delta(xs.mean(axis=1), (xs**2).mean(axis=1))
        psi = _block_psi(_moment_columns(xs), block)
        ses = np.sqrt(np.einsum("mi,mij,mj->m", gs, psi, gs) / t)
        stats_.append(np.abs(ds - delta) / ses)
    boot = np.concatenate(stats_)
    z = float(np.quantile(boot, level))
    return {
        "delta": float(delta),
        "sharpe_a": float(x[:, 0].mean() / x[:, 0].std()),
        "sharpe_b": float(x[:, 1].mean() / x[:, 1].std()),
        "se_hac": se,
        "p_hac": float(2 * stats.norm.sf(d)),
        "p_boot": (int(np.sum(boot >= d)) + 1) / (reps + 1),
        "ci": (float(delta - z * se), float(delta + z * se)),
        "block": int(block),
        "reps": reps,
        "seed": seed,
        "rows": t,
    }
