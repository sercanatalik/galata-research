"""Whose price is the price: two venues' shares of price discovery, from a VECM on one clock.

    p = gr.models.discovery.grid(binance, bybit, "100ms")   # ts, p1, p2 (log, carried forward)
    fit = gr.models.discovery.vecm(p, lags=20)               # alpha, omega, n
    gr.models.discovery.shares(fit)                          # cs, is_low/is_mid/is_high, ils per venue

- Gonzalo and Granger (1995): the component share, from the error-correction
  speeds, CS = α⊥ normalised, α⊥ = (α₂, −α₁) for two prices with β = (1, −1).
- Hasbrouck (1995): the information share, the part of the efficient price's
  innovation variance each venue carries. With ψ ∝ α⊥ and F a Cholesky factor
  of Ω, IS_j = ([ψF]_j)² / ψΩψ′; the ordering matters, so each venue gets a
  lower and an upper bound (Baillie, Booth, Tse and Zabotina 2002).
- Putniņš (2013): IS and CS both mix leadership with noise; their ratio does
  not. IL₁ = |IS₁/IS₂ · CS₂/CS₁| and ILS₁ = IL₁/(IL₁ + IL₂).
"""

import numpy as np
import polars as pl

from .. import utils
from .._errors import Refused


def grid(x: pl.LazyFrame | pl.DataFrame, y: pl.LazyFrame | pl.DataFrame, every: str) -> pl.DataFrame:
    """`ts, p1, p2`: each venue's log last trade price at or before each step's end, carried forward.

    A step is labelled by its end. Steps before either venue's first trade are dropped.
    """
    frames = []
    for name, f in (("p1", x), ("p2", y)):
        utils.require(f, ("ts", "price"), "trades with ts and price")
        frames.append(
            utils.lazy(f)
            .group_by(pl.col("ts").dt.truncate(every).dt.offset_by(every).alias("ts"), maintain_order=True)
            .agg(pl.col("price").last().log().alias(name))
            .collect()
        )
    lo = max(frames[0]["ts"].min(), frames[1]["ts"].min())
    hi = max(frames[0]["ts"].max(), frames[1]["ts"].max())
    steps = pl.DataFrame({"ts": pl.datetime_range(lo, hi, every, eager=True, time_zone="UTC")})
    out = steps.join(frames[0], on="ts", how="left").join(frames[1], on="ts", how="left").sort("ts")
    return out.with_columns(pl.col("p1").forward_fill(), pl.col("p2").forward_fill()).drop_nulls()


def vecm(prices: pl.DataFrame, lags: int = 20) -> dict:
    """OLS per equation of Δp_t = α(p1 − p2)_{t−1} + Σ_k Γ_k Δp_{t−k} + c + ε: `alpha`, `omega`, `n`."""
    utils.require(prices, ("p1", "p2"), "make prices with gr.models.discovery.grid")
    p = prices.select("p1", "p2").to_numpy()
    if p.shape[0] < lags + 10:
        raise Refused(f"{p.shape[0]} steps is too few for a VECM with {lags} lags")
    dp = np.diff(p, axis=0)
    z = (p[:-1, 0] - p[:-1, 1])
    t0 = lags
    rows = dp.shape[0] - t0
    cols = [z[t0 : t0 + rows]]
    for k in range(1, lags + 1):
        cols.append(dp[t0 - k : t0 - k + rows, 0])
        cols.append(dp[t0 - k : t0 - k + rows, 1])
    cols.append(np.ones(rows))
    X = np.column_stack(cols)
    Y = dp[t0:]
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ coef
    omega = resid.T @ resid / (rows - X.shape[1])
    return {"alpha": coef[0].copy(), "omega": omega, "n": rows, "lags": lags}


def shares(fit: dict) -> pl.DataFrame:
    """Per venue (1, 2): `cs`, `is_low`, `is_mid`, `is_high`, `ils`; null when there is no error correction."""
    a1, a2 = (float(v) for v in fit["alpha"])
    omega = np.asarray(fit["omega"], dtype=float)
    empty = {"venue": [1, 2], **{k: [None, None] for k in ("cs", "is_low", "is_mid", "is_high", "ils")}}
    if abs(a2 - a1) < 1e-12:
        return pl.DataFrame(empty, schema={"venue": pl.Int64, **{k: pl.Float64 for k in empty if k != "venue"}})
    perp = np.array([a2, -a1]) / (a2 - a1)
    psi = perp
    variance = float(psi @ omega @ psi)
    bounds = []
    for order in ((0, 1), (1, 0)):
        o = list(order)
        f = np.linalg.cholesky(omega[np.ix_(o, o)])
        contribution = (psi[o] @ f) ** 2 / variance
        share = np.empty(2)
        share[o] = contribution
        bounds.append(share)
    lo, hi = np.minimum(*bounds), np.maximum(*bounds)
    mid = (lo + hi) / 2
    il1 = abs(mid[0] / mid[1] * perp[1] / perp[0])
    il2 = abs(mid[1] / mid[0] * perp[0] / perp[1])
    ils = np.array([il1, il2]) / (il1 + il2)
    return pl.DataFrame({"venue": [1, 2], "cs": perp, "is_low": lo, "is_mid": mid, "is_high": hi, "ils": ils})
