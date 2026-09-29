"""Lead-lag between two asynchronously quoted prices (the `[models]` extra).

    gr.models.leadlag.contrast(t_x, x, t_y, y, lags)   # ρ(θ) per lag
    gr.models.leadlag.estimate(t_x, x, t_y, y)          # θ̂, ρ(θ̂), ρ(0), LLR

Hoffmann, Rosenbaum and Yoshida's (2013) shifted Hayashi–Yoshida contrast:
U(θ) = Σᵢⱼ ΔXᵢ·ΔYⱼ·1{Iᵢ ∩ (Jⱼ − θ) ≠ ∅}, every pair of changes whose
intervals overlap once Y's clock is moved back by θ. **No resampling**: a
previous-tick grid makes the more active price look like the leader whatever
the truth (Huth and Abergel 2014's simulations); overlap does not. θ > 0 means
X leads: Y moves θ after X. ρ(θ) = U(θ)/√(ΣΔX²·ΣΔY²), θ̂ = argmax|U(θ)|
(the smallest |θ| on a tie), and the lead/lag ratio
LLR = Σ_{θ>0} ρ(θ)² / Σ_{θ<0} ρ(−θ)² over the grid (Huth and Abergel): above 1,
X leads.

Pass prices in **tick time**: only changes of the price (repeats are dropped
here), with times as numbers in any one unit.
"""

from __future__ import annotations

import numpy as np

from .._errors import Refused

#: Huth and Abergel's grid to 30 s, in seconds: 0, 0.1…1, 2…10, 15, 20, 30.
GRID_S = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 30.0)


def _ticks(t, p) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(t, kind="stable")
    t, p = t[order], p[order]
    keep = np.concatenate([[True], np.diff(p) != 0])
    return t[keep], p[keep]


def _u(tx: np.ndarray, x: np.ndarray, ty: np.ndarray, y: np.ndarray) -> float:
    """Σᵢ ΔXᵢ · (the sum of Y's changes whose intervals overlap (tx[i−1], tx[i]])."""
    a, b = tx[:-1], tx[1:]
    dx = np.diff(x)
    lo = np.searchsorted(ty, a, side="right")  # first Y time after a: its interval ends inside (a, …]
    hi = np.searchsorted(ty, b, side="left")  # first Y time at or after b: its interval starts before b
    lo = np.clip(lo, 1, len(ty) - 1)
    hi = np.clip(hi, 1, len(ty) - 1)
    overlap = np.where(hi >= lo, y[hi] - y[lo - 1], 0.0)
    return float((dx * overlap).sum())


def contrast(t_x, x, t_y, y, lags) -> dict[float, float]:
    """ρ(θ) for each θ in `lags`, on log prices in tick time."""
    tx, px = _ticks(t_x, x)
    ty, py = _ticks(t_y, y)
    if len(tx) < 3 or len(ty) < 3:
        raise Refused(f"{len(tx)} and {len(ty)} price changes: a lead-lag needs more than two in each")
    lx, ly = np.log(px), np.log(py)
    norm = np.sqrt((np.diff(lx) ** 2).sum() * (np.diff(ly) ** 2).sum())
    if norm == 0:
        raise Refused("a price did not move")
    return {float(th): _u(tx, lx, ty - th, ly) / norm for th in lags}


def estimate(t_x, x, t_y, y, *, grid=GRID_S, min_changes: int = 300) -> dict:
    """θ̂, ρ(θ̂), ρ(0) and LLR over ± `grid` (seconds when times are seconds). Refuses under `min_changes` in either price."""
    for name, (t, p) in (("x", (t_x, x)), ("y", (t_y, y))):
        n = len(_ticks(t, p)[0]) - 1
        if n < min_changes:
            raise Refused(f"{name} changed {n} times, under {min_changes}")
    lags = sorted({g for g in grid} | {-g for g in grid})
    rho = contrast(t_x, x, t_y, y, lags)
    best = max(lags, key=lambda th: (abs(rho[th]), -abs(th)))
    ahead = sum(rho[g] ** 2 for g in grid if g > 0)
    behind = sum(rho[-g] ** 2 for g in grid if g > 0)
    return {
        "lag": best,
        "rho": rho[best],
        "rho0": rho[0.0],
        "llr": ahead / behind if behind > 0 else float("inf"),
        "edge": abs(best) == max(grid),
        "changes": (len(_ticks(t_x, x)[0]) - 1, len(_ticks(t_y, y)[0]) - 1),
    }
