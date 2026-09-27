"""Likelihoods arch does not have: component GARCH, Beta-t-EGARCH and CARR, each checkable without trusting it.

- Component GARCH (Engle and Lee 1999): a short-run variance around a
  long-run level that itself moves. With φ = 0 the level is constant and the
  model is GARCH(1,1) (a test asserts the recursions agree). Katsiampa (2017)
  found it the best fit for daily BTC: ρ 0.9999, φ 0.0549, α 0.1825, β 0.7855.
- Beta-t-EGARCH (Harvey and Chakravarty 2008): log scale λ driven by the
  Student-t score u ∈ [−1, ν], so one extreme return moves λ a bounded amount.
  λ here is the log scale (Harvey 2013; betategarch); the 2008 paper writes the
  scale as exp(λ/2), the same model.
- CARR (Chou 2005): the conditional mean of the bar's range ln(H/L), fitted by
  exponential QMLE; a bar's σ is λ/√(8/π), the mean range of a Brownian bar.

Every series is on the ×100 scale arch uses; callers convert back.
"""

from math import exp, lgamma, log, pi, sqrt

import numpy as np
from scipy import optimize

from ..._errors import Refused

MODELS = ("cgarch", "betat", "rgarch", "msgarch")
NEEDS_MEASURES = ("rgarch",)
NAMES = {
    "cgarch": ("mu", "omega", "alpha", "beta", "rho", "phi", "nu"),
    "betat": ("mu", "omega", "phi", "kappa", "kappa_star", "nu"),
    "carr": ("omega", "alpha", "beta"),
    "rgarch": ("mu", "omega", "beta", "gamma", "xi", "phi", "tau1", "tau2", "sigma_u"),
    "msgarch": ("mu", "omega1", "alpha1", "beta1", "omega2", "alpha2", "beta2", "p11", "p22", "nu"),
}
RANGE_MEAN = sqrt(8 / pi)  # E[ln(H/L)] / σ for driftless Brownian motion over the bar


# ── Component GARCH ─────────────────────────────────────────────────────────


def cgarch_filter(p: dict, y: np.ndarray, init: float) -> tuple[np.ndarray, np.ndarray]:
    """σ² and q, length n + 1: entry t is the variance of y[t] given y[:t]; entry n is the next bar's."""
    e = y - p["mu"]
    n = e.size
    s, q = np.empty(n + 1), np.empty(n + 1)
    s[0] = q[0] = init
    w, a, b, rho, phi = p["omega"], p["alpha"], p["beta"], p["rho"], p["phi"]
    for t in range(n):
        e2 = e[t] * e[t]
        q[t + 1] = w + rho * (q[t] - w) + phi * (e2 - s[t])
        s[t + 1] = q[t + 1] + a * (e2 - q[t]) + b * (s[t] - q[t])
    return s, q


def _t_standardised_ll(e: np.ndarray, var: np.ndarray, nu: float) -> float:
    c = lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(pi * (nu - 2))
    return float(np.sum(c - 0.5 * np.log(var) - (nu + 1) / 2 * np.log1p(e * e / (var * (nu - 2)))))


def _cgarch_nll(x: np.ndarray, y: np.ndarray, init: float) -> float:
    p = dict(zip(NAMES["cgarch"], x))
    if p["alpha"] + p["beta"] >= p["rho"]:
        return np.inf
    s, q = cgarch_filter(p, y, init)
    if not np.all(np.isfinite(s[:-1])) or np.any(s[:-1] <= 0) or np.any(q[:-1] <= 0):
        return np.inf
    return -_t_standardised_ll(y - p["mu"], s[:-1], p["nu"])


# ── Beta-t-EGARCH ───────────────────────────────────────────────────────────


def betat_filter(p: dict, y: np.ndarray) -> np.ndarray:
    """λ, length n + 1: entry t is the log scale of y[t] given y[:t]; entry n is the next bar's."""
    e = y - p["mu"]
    n = e.size
    lam = np.empty(n + 1)
    lam[0] = p["omega"] / (1 - p["phi"])
    w, phi, k, ks, nu = p["omega"], p["phi"], p["kappa"], p["kappa_star"], p["nu"]
    for t in range(n):
        e2 = e[t] * e[t]
        u = (nu + 1) * e2 / (nu * np.exp(2 * lam[t]) + e2) - 1
        lam[t + 1] = w + phi * lam[t] + k * u + ks * np.sign(-e[t]) * (u + 1)
    return lam


def betat_variance(lam: np.ndarray, nu: float) -> np.ndarray:
    return np.exp(2 * lam) * nu / (nu - 2)


def _betat_nll(x: np.ndarray, y: np.ndarray) -> float:
    p = dict(zip(NAMES["betat"], x))
    lam = betat_filter(p, y)[:-1]
    if not np.all(np.isfinite(lam)):
        return np.inf
    nu, e = p["nu"], y - p["mu"]
    c = lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(nu * pi)
    return -float(np.sum(c - lam - (nu + 1) / 2 * np.log1p(e * e / (nu * np.exp(2 * lam)))))


# ── Realized GARCH ──────────────────────────────────────────────────────────
#
# Hansen, Huang and Shek (2012), log-linear RG(1,1): the variance is driven by
# a realized measure x, and a measurement equation ties x back to the variance.
# SPY (their Table II): β 0.55, γ 0.41, φ 1.04, τ₁ −0.07, τ₂ 0.07, σᵤ 0.38.


def rgarch_filter(p: dict, y: np.ndarray, x: np.ndarray, init: float) -> np.ndarray:
    """log h, length n + 1: entry t is log h of y[t] given y[:t], x[:t]; entry n is the next bar's."""
    n = y.size
    lh = np.empty(n + 1)
    lh[0] = init
    lx = np.log(x)
    for t in range(n):
        lh[t + 1] = p["omega"] + p["beta"] * lh[t] + p["gamma"] * lx[t]
    return lh


def _rgarch_parts(p: dict, y: np.ndarray, x: np.ndarray, init: float):
    lh = rgarch_filter(p, y, x, init)[:-1]
    h = np.exp(lh)
    z = (y - p["mu"]) / np.sqrt(h)
    u = np.log(x) - p["xi"] - p["phi"] * lh - p["tau1"] * z - p["tau2"] * (z * z - 1)
    return lh, h, z, u


def _rgarch_ll(p: dict, y: np.ndarray, x: np.ndarray, init: float) -> tuple[float, float]:
    """(ℓ(r), ℓ(x | r)), each Gaussian: the paper's §5.2 factorisation."""
    lh, h, z, u = _rgarch_parts(p, y, x, init)
    lr = -0.5 * float(np.sum(np.log(2 * pi) + lh + z * z))
    s2 = p["sigma_u"] ** 2
    lx = -0.5 * float(np.sum(np.log(2 * pi) + np.log(s2) + u * u / s2))
    return lr, lx


def _rgarch_nll(xs: np.ndarray, y: np.ndarray, x: np.ndarray, init: float) -> float:
    p = dict(zip(NAMES["rgarch"], xs))
    if p["beta"] + p["phi"] * p["gamma"] >= 1 or p["sigma_u"] <= 0:
        return np.inf
    lr, lx = _rgarch_ll(p, y, x, init)
    total = lr + lx
    return -total if np.isfinite(total) else np.inf


# ── Markov-switching GARCH ──────────────────────────────────────────────────
#
# Haas, Mittnik and Paolella (2004), the form R's MSGARCH implements: two GARCH
# variances updated in parallel on the observed shock, so the likelihood is
# exact with no path dependence; a two-state Markov chain picks the regime.


def _std_t_pdf(e: np.ndarray, var: np.ndarray, nu: float) -> np.ndarray:
    c = np.exp(lgamma((nu + 1) / 2) - lgamma(nu / 2)) / np.sqrt(pi * (nu - 2) * var)
    return c * (1 + e * e / (var * (nu - 2))) ** (-(nu + 1) / 2)


def msgarch_filter(p: dict, y: np.ndarray, init: float):
    """(σ² per regime, n+1 × 2; ξ_{t|t−1}, n+1 × 2; ξ_{t|t}, n × 2; log-likelihood per step, n).

    Plain floats in the loop: numpy on two-element arrays cost 70 s a fit.
    """
    e = (y - p["mu"]).tolist()
    n = len(e)
    w1, a1, b1 = p["omega1"], p["alpha1"], p["beta1"]
    w2, a2, b2 = p["omega2"], p["alpha2"], p["beta2"]
    p11, p22, nu = p["p11"], p["p22"], p["nu"]
    c = exp(lgamma((nu + 1) / 2) - lgamma(nu / 2)) / sqrt(pi * (nu - 2))
    k = -(nu + 1) / 2
    s2 = np.empty((n + 1, 2))
    pred = np.empty((n + 1, 2))
    filt = np.empty((n, 2))
    ll = np.empty(n)
    v1 = v2 = init
    q1 = (1 - p22) / (2 - p11 - p22)
    s2[0] = (v1, v2)
    pred[0] = (q1, 1 - q1)
    for t in range(n):
        et2 = e[t] * e[t]
        f1 = c / sqrt(v1) * (1 + et2 / (v1 * (nu - 2))) ** k
        f2 = c / sqrt(v2) * (1 + et2 / (v2 * (nu - 2))) ** k
        j1, j2 = q1 * f1, (1 - q1) * f2
        tot = j1 + j2
        if tot > 0:
            ll[t] = log(tot)
            r1 = j1 / tot
        else:
            ll[t] = -np.inf
            r1 = q1
        filt[t] = (r1, 1 - r1)
        q1 = r1 * p11 + (1 - r1) * (1 - p22)
        pred[t + 1] = (q1, 1 - q1)
        v1 = w1 + a1 * et2 + b1 * v1
        v2 = w2 + a2 * et2 + b2 * v2
        s2[t + 1] = (v1, v2)
    return s2, pred, filt, ll


def _msgarch_nll(xs: np.ndarray, y: np.ndarray, init: float) -> float:
    p = dict(zip(NAMES["msgarch"], xs))
    if p["alpha1"] + p["beta1"] >= 1 or p["alpha2"] + p["beta2"] >= 1:
        return np.inf
    ll = msgarch_filter(p, y, init)[3]
    total = float(ll.sum())
    return -total if np.isfinite(total) else np.inf


def _unconditional(p: dict, k: int) -> float:
    return p[f"omega{k}"] / (1 - p[f"alpha{k}"] - p[f"beta{k}"])


def order_regimes(p: dict) -> dict:
    """Regime 1 is the one with the lower unconditional variance (MSGARCH's identification); the likelihood is unchanged."""
    if _unconditional(p, 1) <= _unconditional(p, 2):
        return dict(p)
    q = dict(p)
    for name in ("omega", "alpha", "beta"):
        q[f"{name}1"], q[f"{name}2"] = p[f"{name}2"], p[f"{name}1"]
    q["p11"], q["p22"] = p["p22"], p["p11"]
    return q


# ── CARR ────────────────────────────────────────────────────────────────────


def carr_filter(p: dict, r: np.ndarray, init: float) -> np.ndarray:
    """λ, length n + 1: entry t is the expected range of bar t given ranges before it."""
    n = r.size
    lam = np.empty(n + 1)
    lam[0] = init
    for t in range(n):
        lam[t + 1] = p["omega"] + p["alpha"] * r[t] + p["beta"] * lam[t]
    return lam


def _carr_nll(x: np.ndarray, r: np.ndarray, init: float) -> float:
    p = dict(zip(NAMES["carr"], x))
    if p["alpha"] + p["beta"] >= 1:
        return np.inf
    lam = carr_filter(p, r, init)[:-1]
    if np.any(lam <= 0):
        return np.inf
    return float(np.sum(np.log(lam) + r / lam))


# ── Fitting ─────────────────────────────────────────────────────────────────


def _objective(model: str, y: np.ndarray, x: np.ndarray | None = None):
    if model == "msgarch":
        init = float(np.var(y))
        return (lambda v: _msgarch_nll(v, y, init)), init
    if model == "rgarch":
        init = float(np.log(np.var(y)))
        return (lambda v: _rgarch_nll(v, y, x, init)), init
    if model == "cgarch":
        init = float(np.var(y))
        return (lambda x: _cgarch_nll(x, y, init)), init
    if model == "betat":
        return (lambda x: _betat_nll(x, y)), None
    init = float(np.mean(y))
    return (lambda x: _carr_nll(x, y, init)), init


def _start(model: str, y: np.ndarray, x: np.ndarray | None = None) -> tuple[np.ndarray, list]:
    m, v = float(np.mean(y)), float(np.var(y))
    sd = sqrt(v)
    if model == "msgarch":
        return np.array([m, 0.02 * v, 0.05, 0.9, 0.1 * v, 0.1, 0.8, 0.97, 0.97, 6.0]), [
            (m - 10 * sd, m + 10 * sd), (1e-6 * v, 10 * v), (0.0, 1.0), (0.0, 1.0), (1e-6 * v, 10 * v), (0.0, 1.0), (0.0, 1.0), (0.01, 0.9999), (0.01, 0.9999), (2.1, 100.0)
        ]
    if model == "rgarch":
        # The paper's SPY estimates, ω set so log h starts at the log sample variance.
        beta, gamma, phi, xi = 0.55, 0.41, 1.04, float(np.mean(np.log(x)) - 1.04 * log(v))
        omega = (1 - beta) * log(v) - gamma * (xi + phi * log(v))
        return np.array([m, omega, beta, gamma, xi, phi, -0.07, 0.07, 0.38]), [
            (m - 10 * sd, m + 10 * sd), (-20.0, 20.0), (-0.999, 0.999), (0.0, 2.0), (-20.0, 20.0), (0.0, 3.0), (-1.0, 1.0), (-1.0, 1.0), (1e-3, 5.0)
        ]
    if model == "cgarch":
        return np.array([m, v, 0.05, 0.85, 0.98, 0.02, 6.0]), [
            (m - 10 * sd, m + 10 * sd), (1e-6 * v, 100 * v), (0.0, 0.5), (0.0, 0.999), (0.5, 0.99999), (0.0, 0.5), (2.1, 100.0)
        ]
    if model == "betat":
        phi, nu = 0.95, 6.0
        return np.array([m, (1 - phi) * log(sd * sqrt((nu - 2) / nu)), phi, 0.05, 0.0, nu]), [
            (m - 10 * sd, m + 10 * sd), (-5.0, 5.0), (0.0, 0.9999), (0.0, 1.0), (-0.5, 0.5), (2.1, 100.0)
        ]
    return np.array([0.1 * m, 0.15, 0.75]), [(1e-8, 10 * m + 1e-8), (0.0, 1.0), (0.0, 1.0)]


def _hessian(f, x: np.ndarray) -> np.ndarray:
    k = x.size
    h = 1e-4 * np.maximum(np.abs(x), 1e-2)
    out = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = h[i], h[j]
            val = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
            out[i, j] = out[j, i] = val
    return out


def estimate(model: str, y: np.ndarray, x: np.ndarray | None = None) -> dict:
    """Fit by bounded Nelder–Mead from stated start values; standard errors from a numerical Hessian.

    Derivative-free on purpose: outside a joint constraint (α+β < ρ, α+β < 1)
    the likelihood is +∞, and L-BFGS-B's finite-difference gradient stepped
    into it and stopped at the start values (measured on BTC daily: 2902.8 at
    the start against Nelder–Mead's 2886.5).
    """
    nll, _ = _objective(model, y, x)
    x0, bounds = _start(model, y, x)
    res = optimize.minimize(nll, x0, method="Nelder-Mead", bounds=bounds, options={"maxiter": 6000, "maxfev": 12000, "xatol": 1e-7, "fatol": 1e-7})
    if model == "msgarch":
        # Ten parameters: one restart from the first result. On BTC daily the first pass
        # stopped at its evaluation cap (nll 2875.4); the restart converged at 2874.3.
        res = optimize.minimize(nll, res.x, method="Nelder-Mead", bounds=bounds, options={"maxiter": 20000, "maxfev": 40000, "xatol": 1e-7, "fatol": 1e-7})
    x = res.x
    names = NAMES[model]
    se = {n: None for n in names}
    try:
        cov = np.linalg.inv(_hessian(nll, x))
        diag = np.diag(cov)
        if np.all(np.isfinite(diag)) and np.all(diag > 0):
            se = {n: float(sqrt(d)) for n, d in zip(names, diag)}
    except np.linalg.LinAlgError:
        pass
    params = {n: float(v) for n, v in zip(names, x)}
    if model == "msgarch":
        swapped = order_regimes(params)
        if swapped != params:
            se = order_regimes(se) if all(v is not None for v in se.values()) else {n: None for n in names}
        params = swapped
    return {"params": params, "std_err": se, "nll": float(res.fun), "converged": bool(res.success)}


def summary(model: str, y: np.ndarray, x: np.ndarray | None = None) -> dict:
    """The shape `_arch.summary` returns, for a hand-written model on the ×100 scale.

    For `rgarch`, `loglik` is the returns part ℓ(r) of the joint likelihood (the
    paper's comparison with GARCH); AIC and BIC count every parameter.
    """
    est = estimate(model, y, x)
    p, n = est["params"], y.size
    extra = {}
    if model == "msgarch":
        s2, pred, filt, _ = msgarch_filter(p, y, float(np.var(y)))
        var = (pred[:-1] * s2[:-1]).sum(axis=1)
        extra["p_high"] = filt[:, 1].tolist()
    elif model == "rgarch":
        init = float(np.log(np.var(y)))
        lr, _ = _rgarch_ll(p, y, x, init)
        est["nll"] = -lr
        var = np.exp(rgarch_filter(p, y, x, init)[:-1])
    elif model == "cgarch":
        var = cgarch_filter(p, y, float(np.var(y)))[0][:-1]
    else:
        var = betat_variance(betat_filter(p, y)[:-1], p["nu"])
    sigma = np.sqrt(var)
    k = len(p)
    loglik = -est["nll"]
    return {
        "params": p,
        "std_err": est["std_err"],
        "loglik": loglik,
        "aic": 2 * k - 2 * loglik,
        "bic": k * log(n) - 2 * loglik,
        "converged": est["converged"],
        "sigma": sigma.tolist(),
        "z": ((y - p["mu"]) / sigma).tolist(),
        **extra,
    }


# ── Forecasts ───────────────────────────────────────────────────────────────


def forecast(
    model: str, p: dict, y: np.ndarray, *, start: int, horizon: int, init: float | None, simulations: int, seed: int, x: np.ndarray | None = None
) -> np.ndarray:
    """Variance forecasts (×100 scale) from origins start…len(y)−1: row i uses y[:i+1], parameters fixed."""
    if model == "msgarch":
        s2, pred, _, _ = msgarch_filter(p, y, init)
        s2, pred = s2[start + 1 :], pred[start + 1 :]
        out = np.empty((s2.shape[0], horizon))
        out[:, 0] = (pred * s2).sum(axis=1)
        if horizon > 1:
            rng = np.random.default_rng(seed)
            nu = p["nu"]
            w = np.array([p["omega1"], p["omega2"]])
            a = np.array([p["alpha1"], p["alpha2"]])
            b = np.array([p["beta1"], p["beta2"]])
            stay = np.array([p["p11"], p["p22"]])
            m, sims = s2.shape[0], simulations
            var = np.repeat(s2[:, None, :], sims, axis=1)  # origins × sims × regimes
            regime = (rng.random((m, sims)) >= pred[:, 0:1]).astype(int)
            for h in range(horizon):
                current = np.take_along_axis(var, regime[..., None], axis=2)[..., 0]
                if h > 0:
                    out[:, h] = current.mean(axis=1)
                z = rng.standard_t(nu, size=(m, sims)) * sqrt((nu - 2) / nu)
                e2 = current * z * z
                var = w + a * e2[..., None] + b * var
                keep = rng.random((m, sims)) < stay[regime]
                regime = np.where(keep, regime, 1 - regime)
        return out
    if model == "rgarch":
        lh_all = rgarch_filter(p, y, x, init)
        lh = lh_all[start + 1 :]
        out = np.empty((lh.size, horizon))
        out[:, 0] = np.exp(lh)
        if horizon > 1:
            # In-sample (z, u) pairs, drawn together so their dependence is kept (the paper's §6.2 advice).
            _, _, z, u = _rgarch_parts(p, y[: start + 1], x[: start + 1], init)
            rng = np.random.default_rng(seed)
            paths = np.repeat(lh[:, None], simulations, axis=1)
            for h in range(1, horizon):
                pick = rng.integers(0, z.size, size=paths.shape)
                zz, uu = z[pick], u[pick]
                lx = p["xi"] + p["phi"] * paths + p["tau1"] * zz + p["tau2"] * (zz * zz - 1) + uu
                paths = p["omega"] + p["beta"] * paths + p["gamma"] * lx
                out[:, h] = np.exp(paths).mean(axis=1)
        return out
    if model == "cgarch":
        s, q = cgarch_filter(p, y, init)
        s1, q1 = s[start + 1 :], q[start + 1 :]
        hs = np.arange(horizon)
        long = p["omega"] + np.outer(q1 - p["omega"], p["rho"] ** hs)
        short = np.outer(s1 - q1, (p["alpha"] + p["beta"]) ** hs)
        return long + short
    if model == "betat":
        lam = betat_filter(p, y)[start + 1 :]
        out = np.empty((lam.size, horizon))
        out[:, 0] = betat_variance(lam, p["nu"])
        if horizon > 1:
            rng = np.random.default_rng(seed)
            nu, w, phi, k, ks = p["nu"], p["omega"], p["phi"], p["kappa"], p["kappa_star"]
            paths = np.repeat(lam[:, None], simulations, axis=1)
            for h in range(1, horizon):
                eps = rng.standard_t(nu, size=paths.shape) * np.exp(paths)
                e2 = eps * eps
                u = (nu + 1) * e2 / (nu * np.exp(2 * paths) + e2) - 1
                paths = w + phi * paths + k * u + ks * np.sign(-eps) * (u + 1)
                out[:, h] = betat_variance(paths, nu).mean(axis=1)
        return out
    lam = carr_filter(p, y, init)[start + 1 :]
    persistence = p["alpha"] + p["beta"]
    level = p["omega"] / (1 - persistence)
    path = level + np.outer(lam - level, persistence ** np.arange(horizon))
    return (path / RANGE_MEAN) ** 2


def check_model(model: str, dist: str, measures=None) -> None:
    want = "normal" if model == "rgarch" else "t"
    if model in MODELS and dist != want:
        raise Refused(f"{model} is written for {'Gaussian' if want == 'normal' else 'Student-t'} innovations; use dist={want!r}, not {dist!r}")
    if model in NEEDS_MEASURES and measures is None:
        raise Refused(f"{model} needs measures: pass gr.timeseries.realized_from output as measures=")
