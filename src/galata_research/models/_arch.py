"""The only module that touches arch's pandas: numpy in, floats, dicts and lists out.

arch returns `params` and forecasts as pandas objects. They are taken apart
here, so no caller of `gr.models` ever holds one.
"""

import numpy as np
import polars as pl
from arch.univariate import APARCH, EGARCH, FIGARCH, GARCH, ConstantMean, EWMAVariance, GeneralizedError, Normal, RiskMetrics2006, SkewStudent, StudentsT

from .._errors import Refused

SCALE = 100.0
MODELS = ("ewma", "rm2006", "garch", "gjr", "egarch", "aparch", "figarch")
DISTS = ("normal", "t", "skewt", "ged")
_DIST = {"normal": Normal, "t": StudentsT, "skewt": SkewStudent, "ged": GeneralizedError}
_VOL = {
    "ewma": lambda: EWMAVariance(0.94),
    "rm2006": lambda: RiskMetrics2006(),
    "garch": lambda: GARCH(p=1, o=0, q=1),
    "gjr": lambda: GARCH(p=1, o=1, q=1),
    "egarch": lambda: EGARCH(p=1, o=1, q=1),
    "aparch": lambda: APARCH(p=1, o=1, q=1),
    "figarch": lambda: FIGARCH(p=1, q=1),
}
# arch refuses an analytic forecast beyond one step for these; they are simulated.
SIMULATED = ("egarch", "aparch")


def values(frame: pl.DataFrame, column: str) -> np.ndarray:
    """A column as float64 × SCALE; a null is refused by name (drop or bridge before calling)."""
    s = frame[column]
    if s.null_count():
        raise Refused(f"{column} holds {s.null_count()} null(s); arch cannot recurse through one")
    return s.cast(pl.Float64).to_numpy() * SCALE


def model(y: np.ndarray, name: str, dist: str, *, seed: int = 0):
    """A constant-mean arch model; the distribution is seeded, so a simulated forecast reproduces."""
    return ConstantMean(y, volatility=_VOL[name](), distribution=_DIST[dist](seed=np.random.default_rng(seed)))


def fit(y: np.ndarray, name: str, dist: str, *, first_obs: int | None = None, last_obs: int | None = None, seed: int = 0):
    """An arch result for a constant mean and a (1,1) process on `y[first_obs:last_obs]` (`last_obs` exclusive)."""
    return model(y, name, dist, seed=seed).fit(disp="off", first_obs=first_obs, last_obs=last_obs)


def forecast(res, *, start: int, horizon: int, simulate: bool, simulations: int) -> np.ndarray:
    """Variance forecasts from origins `start`… to the end of the model's data: an (origins × horizon) array.

    Row i uses the data through origin i (align="origin"), with the fitted
    parameters fixed and the filter run over the observed returns.
    """
    method = "simulation" if simulate and horizon > 1 else "analytic"
    f = res.forecast(horizon=horizon, start=start, method=method, simulations=simulations, reindex=False)
    return np.asarray(f.variance, dtype=float)


def summary(res) -> dict:
    """Plain values from an arch result: params and std_err as dicts, the series as lists."""
    return {
        "params": {k: float(v) for k, v in res.params.items()},
        "std_err": {k: float(v) for k, v in res.std_err.items()},
        "loglik": float(res.loglikelihood),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "converged": int(res.convergence_flag) == 0,
        "sigma": np.asarray(res.conditional_volatility, dtype=float).tolist(),
        "z": np.asarray(res.std_resid, dtype=float).tolist(),
    }


def ppf(dist: str, params: dict):
    """The fitted distribution's quantile function, u ↦ z, for integrating E[g(z)]."""
    d = _DIST[dist]()
    shape = {"normal": [], "t": ["nu"], "skewt": ["eta", "lambda"], "ged": ["nu"]}[dist]
    p = np.array([params[k] for k in shape], dtype=float)

    def q(u: float) -> float:
        return float(d.ppf(np.array([u]), p)[0])

    return q
