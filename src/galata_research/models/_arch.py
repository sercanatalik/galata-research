"""The only module that touches arch's pandas: numpy in, floats, dicts and lists out.

arch returns `params` and forecasts as pandas objects. They are taken apart
here, so no caller of `gr.models` ever holds one.
"""

import numpy as np
import polars as pl
from arch import arch_model
from arch.univariate import ConstantMean, EWMAVariance, GeneralizedError, Normal, RiskMetrics2006, SkewStudent, StudentsT

from .._errors import Refused

SCALE = 100.0
MODELS = ("ewma", "rm2006", "garch", "gjr", "egarch", "aparch", "figarch")
DISTS = ("normal", "t", "skewt", "ged")
_DIST = {"normal": Normal, "t": StudentsT, "skewt": SkewStudent, "ged": GeneralizedError}
_VOL = {
    "garch": dict(vol="GARCH", p=1, o=0, q=1),
    "gjr": dict(vol="GARCH", p=1, o=1, q=1),
    "egarch": dict(vol="EGARCH", p=1, o=1, q=1),
    "aparch": dict(vol="APARCH", p=1, o=1, q=1),
    "figarch": dict(vol="FIGARCH", p=1, q=1),
}


def values(frame: pl.DataFrame, column: str) -> np.ndarray:
    """A column as float64 × SCALE; a null is refused by name (drop or bridge before calling)."""
    s = frame[column]
    if s.null_count():
        raise Refused(f"{column} holds {s.null_count()} null(s); arch cannot recurse through one")
    return s.cast(pl.Float64).to_numpy() * SCALE


def fit(y: np.ndarray, model: str, dist: str, *, last_obs: int | None = None):
    """An arch result for a constant mean and a (1,1) process; `last_obs` ends the estimation sample."""
    if model == "ewma":
        m = ConstantMean(y, volatility=EWMAVariance(0.94), distribution=_DIST[dist]())
    elif model == "rm2006":
        m = ConstantMean(y, volatility=RiskMetrics2006(), distribution=_DIST[dist]())
    else:
        m = arch_model(y, mean="Constant", dist={"normal": "normal", "t": "t", "skewt": "skewt", "ged": "ged"}[dist], **_VOL[model])
    return m.fit(disp="off", last_obs=last_obs)


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
