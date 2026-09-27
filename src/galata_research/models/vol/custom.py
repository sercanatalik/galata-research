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

from math import lgamma, log, pi, sqrt

import numpy as np
from scipy import optimize

from ..._errors import Refused

MODELS = ("cgarch", "betat")
NAMES = {
    "cgarch": ("mu", "omega", "alpha", "beta", "rho", "phi", "nu"),
    "betat": ("mu", "omega", "phi", "kappa", "kappa_star", "nu"),
    "carr": ("omega", "alpha", "beta"),
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


def _objective(model: str, y: np.ndarray):
    if model == "cgarch":
        init = float(np.var(y))
        return (lambda x: _cgarch_nll(x, y, init)), init
    if model == "betat":
        return (lambda x: _betat_nll(x, y)), None
    init = float(np.mean(y))
    return (lambda x: _carr_nll(x, y, init)), init


def _start(model: str, y: np.ndarray) -> tuple[np.ndarray, list]:
    m, v = float(np.mean(y)), float(np.var(y))
    sd = sqrt(v)
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


def estimate(model: str, y: np.ndarray) -> dict:
    """Fit by bounded Nelder–Mead from stated start values; standard errors from a numerical Hessian.

    Derivative-free on purpose: outside a joint constraint (α+β < ρ, α+β < 1)
    the likelihood is +∞, and L-BFGS-B's finite-difference gradient stepped
    into it and stopped at the start values (measured on BTC daily: 2902.8 at
    the start against Nelder–Mead's 2886.5).
    """
    nll, _ = _objective(model, y)
    x0, bounds = _start(model, y)
    res = optimize.minimize(nll, x0, method="Nelder-Mead", bounds=bounds, options={"maxiter": 6000, "maxfev": 12000, "xatol": 1e-7, "fatol": 1e-7})
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
    return {"params": {n: float(v) for n, v in zip(names, x)}, "std_err": se, "nll": float(res.fun), "converged": bool(res.success)}


def summary(model: str, y: np.ndarray) -> dict:
    """The shape `_arch.summary` returns, for a hand-written model on the ×100 scale."""
    est = estimate(model, y)
    p, n = est["params"], y.size
    if model == "cgarch":
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
    }


# ── Forecasts ───────────────────────────────────────────────────────────────


def forecast(model: str, p: dict, y: np.ndarray, *, start: int, horizon: int, init: float | None, simulations: int, seed: int) -> np.ndarray:
    """Variance forecasts (×100 scale) from origins start…len(y)−1: row i uses y[:i+1], parameters fixed."""
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


def check_model(model: str, dist: str) -> None:
    if model in MODELS and dist != "t":
        raise Refused(f"{model} is written for Student-t innovations; dist={dist!r} is not available for it")
