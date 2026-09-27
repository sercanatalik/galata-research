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
