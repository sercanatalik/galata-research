"""Long memory, and a test of it against level shifts: Robinson's local Whittle and Qu's W.

    gr.models.memory.local_whittle(x, m)       # {"d", "se", "m"}
    gr.models.memory.qu_test(x, m, epsilon=0.02)  # {"W", "d", "m", "critical", "reject_5pct"}

Both follow `LongMemoryTS` (`local.W`, `Qu.test`) line for line:
- **The periodogram.** I_j = |Σ x_t e^{−iλ_j t}|² / (2πT) at λ_j = 2πj/T,
  j ≥ 1. Its scale cancels in both.
- **The estimator.** Robinson's (1995) concentrated likelihood,
  R(d) = log mean(λ_j^{2d} I_j) − 2d mean(log λ_j), minimised over
  [−0.5, 2.5].
- **Qu's (2011) W.** W = sup_{r∈[ε,1]} |Σ_{j≤⌊mr⌋} ν_j (I_j/(Ĝ λ_j^{−2d̂}) − 1)| / √(Σ_{j≤m} ν_j²),
  with ν_j the demeaned log λ_j. It is large when the spectrum near zero is
  steeper than long memory allows, as level shifts make it.
"""

import numpy as np
from scipy.optimize import minimize_scalar

from .._errors import Refused

# Qu (2011), Table 1: the null of true long memory is rejected above these.
CRITICAL = {0.02: {0.10: 1.118, 0.05: 1.252, 0.025: 1.374, 0.01: 1.517}, 0.05: {0.10: 1.022, 0.05: 1.155, 0.025: 1.277, 0.01: 1.426}}


def _periodogram(x) -> tuple[np.ndarray, np.ndarray, int]:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise Refused("a long-memory estimate needs one finite series, with no nulls")
    n = x.size
    half = n // 2
    periodogram = np.abs(np.fft.fft(x)[1 : half + 1]) ** 2 / (2 * np.pi * n)
    return 2 * np.pi * np.arange(1, half + 1) / n, periodogram, n


def _bandwidth(n: int, m: int | None) -> int:
    m = int(np.floor(1 + n**0.7)) if m is None else int(m)
    if m < 10:
        raise Refused(f"m = {m} Fourier frequencies is too few for a local Whittle estimate")
    return min(m, n // 2)


def _whittle(lam: np.ndarray, periodogram: np.ndarray, m: int) -> float:
    lam, per = lam[:m], periodogram[:m]
    log_lam = np.log(lam)

    def objective(d: float) -> float:
        return float(np.log(np.mean(lam ** (2 * d) * per)) - 2 * d * log_lam.mean())

    return float(minimize_scalar(objective, bounds=(-0.5, 2.5), method="bounded", options={"xatol": 1e-8}).x)


def local_whittle(x, m: int | None = None) -> dict:
    """Robinson's local Whittle estimate of d from the first m Fourier frequencies (default ⌊1 + T^0.7⌋), with se 1/(2√m)."""
    lam, per, n = _periodogram(x)
    m = _bandwidth(n, m)
    return {"d": _whittle(lam, per, m), "se": 1 / (2 * np.sqrt(m)), "m": m}


def qu_test(x, m: int | None = None, *, epsilon: float = 0.02) -> dict:
    """Qu's W against the null of true long memory; `reject_5pct` when W exceeds the 5% critical value."""
    if epsilon not in CRITICAL:
        raise Refused(f"epsilon={epsilon} has no published critical values; use one of {sorted(CRITICAL)}")
    lam, per, n = _periodogram(x)
    m = _bandwidth(n, m)
    d = _whittle(lam, per, m)
    lam_m, per_m = lam[:m], per[:m]
    g = np.mean(lam_m ** (2 * d) * per_m)
    nu = np.log(lam_m) - np.log(lam_m).mean()
    terms = nu * (per_m / (g * lam_m ** (-2 * d)) - 1)
    partial = np.abs(np.cumsum(terms)) / np.sqrt(np.sum(nu**2))
    first = max(int(np.floor(m * epsilon)), 1)
    w = float(partial[first - 1 :].max())
    critical = CRITICAL[epsilon]
    return {"W": w, "d": d, "m": m, "critical": critical, "reject_5pct": w > critical[0.05]}
