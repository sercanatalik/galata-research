"""Sharpe, and how much of it survives selection: PSR, the expected maximum, and the Deflated Sharpe Ratio.

Bailey and López de Prado, "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting and Non-Normality", *Journal of
Portfolio Management*, 2014. Pinned to the paper's own example: N=100,
V[SR]=0.5/250, T=1250, skewness −3, kurtosis 10, SR 2.5/√250 per day, which
give SR₀ 0.1132 per day and DSR 0.9004 (0.9505 at N=46).

Everything is per period. Annualise separately, with `annualize`.
"""

from math import e, sqrt
from statistics import NormalDist

import polars as pl

from ._errors import Refused

EULER_MASCHERONI = 0.5772156649015329
_N = NormalDist()


def sharpe(returns) -> float | None:
    """Mean over sample standard deviation (ddof 1), per period. Null when it is undefined."""
    r = _series(returns)
    if r.len() < 2:
        return None
    sd = r.std(ddof=1)
    return None if not sd else float(r.mean() / sd)


def moments(returns) -> tuple[float | None, float | None]:
    """Skewness and **raw** kurtosis (3 for a normal), population moments."""
    r = _series(returns)
    if r.len() < 2:
        return None, None
    skew = r.skew(bias=True)
    kurt = r.kurtosis(fisher=False, bias=True)
    return (None if skew is None else float(skew)), (None if kurt is None else float(kurt))


def annualize(sr: float | None, periods_per_year: float) -> float | None:
    """A per-period Sharpe scaled by √periods: 365 for daily crypto, 2190 for 4h."""
    return None if sr is None else sr * sqrt(periods_per_year)


def psr(sr: float, periods: int, skew: float, kurt: float, benchmark: float = 0.0) -> float:
    """The probability the true Sharpe exceeds `benchmark`, given T, skewness and raw kurtosis."""
    if periods < 2:
        raise Refused(f"periods={periods}: a Probabilistic Sharpe needs at least 2 returns")
    spread = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if spread <= 0:
        raise Refused(f"skew={skew}, kurt={kurt} at sr={sr} give a non-positive variance term; the moments are inconsistent")
    return _N.cdf((sr - benchmark) * sqrt(periods - 1) / sqrt(spread))


def expected_max_sharpe(trials: int, variance: float) -> float:
    """The Sharpe the best of `trials` independent trials reaches when every true Sharpe is 0.

    `variance` is the variance across the trials' estimated Sharpe ratios, per period.
    """
    if trials < 2:
        raise Refused(f"trials={trials}: with fewer than 2 trials there is no selection to correct")
    if variance < 0:
        raise Refused(f"variance={variance} is negative")
    g = EULER_MASCHERONI
    return sqrt(variance) * ((1 - g) * _N.inv_cdf(1 - 1 / trials) + g * _N.inv_cdf(1 - 1 / (trials * e)))


def dsr(sr: float, periods: int, skew: float, kurt: float, trials: int, variance: float) -> float:
    """The Deflated Sharpe Ratio: PSR against the expected maximum of `trials` trials."""
    return psr(sr, periods, skew, kurt, benchmark=expected_max_sharpe(trials, variance))


def deflate(summary: pl.DataFrame) -> dict:
    """The best trial of `summary` (one row per trial: sharpe, periods, skew, kurt), deflated by all of them.

    Every row counts toward N, a trial with a null Sharpe included, because
    it was run. V[SR] is taken across the non-null Sharpe ratios.
    """
    needed = {"sharpe", "periods", "skew", "kurt"}
    missing = needed - set(summary.columns)
    if missing:
        raise Refused(f"deflate needs {', '.join(sorted(missing))} per trial")
    trials = summary.height
    scored = summary.filter(pl.col("sharpe").is_not_null())
    if scored.height < 2:
        raise Refused(f"{scored.height} of {trials} trials have a Sharpe; deflating needs at least 2")
    variance = float(scored["sharpe"].var(ddof=1))
    best = scored.sort("sharpe", descending=True).row(0, named=True)
    benchmark = expected_max_sharpe(trials, variance)
    return {
        **best,
        "benchmark": benchmark,
        "dsr": psr(best["sharpe"], best["periods"], best["skew"], best["kurt"], benchmark=benchmark),
        "trials": trials,
        "variance": variance,
    }


def _series(returns) -> pl.Series:
    s = returns if isinstance(returns, pl.Series) else pl.Series(list(returns), dtype=pl.Float64)
    return s.drop_nulls().cast(pl.Float64)
