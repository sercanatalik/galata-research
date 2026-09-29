"""Jump-robust realized measures per bucket, two ratio tests, and the continuous/jump split.

From the fine log returns `gr.timeseries.realized_from` uses for the same
bucket, and only those: a bucket's split is known at its close, and no
later return moves it.

- Bipower BV and tripower quarticity TQ (Barndorff-Nielsen and Shephard 2004,
  2006), with the small-sample factors M/(M−1) and M/(M−2).
- MedRV (Andersen, Dobrev and Schaumburg 2012, eq. 4).
- Corrected threshold bipower C-TBV and tripower C-TTQ (Corsi, Pirino and
  Renò 2010): a return above c_θ²·V̂ counts as its expectation beyond that
  threshold under a normal, k_γ(c_θ)·ϑ^(γ/2), 1.094·ϑ^½ at γ = 1, c_θ = 3.
  V̂ is CPR's iterated Gaussian-kernel local variance, run here inside the
  bucket (CPR run it over the whole series). The kernel form, L = 25,
  c_V = 3 and the excluded neighbours are from secondary descriptions of
  CPR; the correction constants are verified.
- Huang and Tauchen's (2005) ratio statistic with the max adjustment
  (ABD 2007 eq. 19–20), on BV and TQ (`z_bns`) and on C-TBV and C-TTQ
  (`z_ctz`, CPR's C-Tz).
- The split: J = 1{z > Φ⁻¹(α)}·max(RV − X, 0), C = RV − J (ABD 2007
  eq. 21–22), with α = 0.999 as ABD, CPR and Shen, Urquhart and Wang (2020).
"""

import math

import numpy as np
import polars as pl
from scipy.special import gamma, gammaincc
from scipy.stats import norm

from ... import timeseries, utils
from ..._errors import Refused
from ...market import INTERVALS

THETA = math.pi**2 / 4 + math.pi - 5  # ≈ 0.609
MU1 = math.sqrt(2 / math.pi)
MU43 = 2 ** (2 / 3) * math.gamma(7 / 6) / math.gamma(0.5)
MEDRV = math.pi / (6 - 4 * math.sqrt(3) + math.pi)
BANDWIDTH = 25
C_V = 3.0
_PASSES = 10
_OUT = ("ticker", "ts", "close_ts", "n", "expected", "rv", "bv", "tq", "medrv", "ctbv", "cttq", "z_bns", "z_ctz", "c_bns", "j_bns", "c_tcj", "j_tcj")
_MEASURES = _OUT[5:]


def k_gamma(g: float, c: float) -> float:
    """E[|X|^γ | X² > c²σ²] / (c²σ²)^(γ/2) for X ~ N(0, σ²): the corrected threshold's multiplier."""
    upper = gammaincc((g + 1) / 2, c * c / 2) * gamma((g + 1) / 2)
    return 2 ** (g / 2) * upper / (math.sqrt(math.pi) * 2 * norm.sf(c)) / c**g


def local_variance(r: np.ndarray, *, bandwidth: int = BANDWIDTH, c_v: float = C_V) -> np.ndarray:
    """CPR's local variance of each return from the others in `r`: kernel-weighted r², large ones removed, iterated.

    The return itself and its two neighbours are excluded, and the window is
    cut at the ends of `r`, so nothing outside the array is read.
    """
    m = r.size
    lags = np.arange(-bandwidth, bandwidth + 1)
    weight = np.exp(-0.5 * (lags / bandwidth) ** 2)
    weight[np.abs(lags) <= 1] = 0.0
    idx = np.arange(m)[:, None] + lags[None, :]
    inside = (idx >= 0) & (idx < m)
    idx = np.clip(idx, 0, m - 1)
    r2 = r * r
    v = np.full(m, np.inf)
    kept = np.ones(m, dtype=bool)
    for _ in range(_PASSES):
        w = weight[None, :] * inside * kept[idx]
        den = w.sum(axis=1)
        v = np.where(den > 0, (w * r2[idx]).sum(axis=1) / np.where(den > 0, den, 1.0), np.inf)
        now = r2 <= c_v**2 * v
        if (now == kept).all():
            break
        kept = now
    return v


def _measure(r: np.ndarray, c_theta: float, q: float) -> dict:
    m = r.size
    a = np.abs(r)
    rv = float(r @ r)
    bv = (1 / MU1**2) * (m / (m - 1)) * float(a[1:] @ a[:-1])
    tq = m * MU43**-3 * (m / (m - 2)) * float(((a[2:] * a[1:-1] * a[:-2]) ** (4 / 3)).sum())
    med = np.median(np.stack([a[:-2], a[1:-1], a[2:]]), axis=0)
    medrv = MEDRV * (m / (m - 2)) * float(med @ med)
    theta = c_theta**2 * local_variance(r)
    above = r * r > theta
    z1 = np.where(above, k_gamma(1.0, c_theta) * np.sqrt(theta), a)
    z43 = np.where(above, k_gamma(4 / 3, c_theta) * theta ** (2 / 3), a ** (4 / 3))
    ctbv = (1 / MU1**2) * (m / (m - 1)) * float(z1[1:] @ z1[:-1])
    cttq = m * MU43**-3 * (m / (m - 2)) * float((z43[2:] * z43[1:-1] * z43[:-2]).sum())

    def z(x, quart):
        if rv <= 0 or x <= 0:
            return None
        return math.sqrt(m) * (1 - x / rv) / math.sqrt(THETA * max(1.0, quart / x**2))

    z_bns, z_ctz = z(bv, tq), z(ctbv, cttq)
    j_bns = max(rv - bv, 0.0) if z_bns is not None and z_bns > q else 0.0
    j_tcj = max(rv - ctbv, 0.0) if z_ctz is not None and z_ctz > q else 0.0
    return {"rv": rv, "bv": bv, "tq": tq, "medrv": medrv, "ctbv": ctbv, "cttq": cttq, "z_bns": z_bns, "z_ctz": z_ctz,
            "c_bns": rv - j_bns, "j_bns": j_bns, "c_tcj": rv - j_tcj, "j_tcj": j_tcj}  # fmt: skip


def realized_jumps(fine: pl.LazyFrame | pl.DataFrame, interval: str, *, alpha: float = 0.999, c_theta: float = 3.0) -> pl.DataFrame:
    """Per bucket of the coarser `interval`: RV, the jump-robust measures, both ratio statistics and both C/J splits.

    `ticker, ts, close_ts, n, expected, rv, bv, tq, medrv, ctbv, cttq, z_bns, z_ctz, c_bns, j_bns, c_tcj, j_tcj`.
    The returns are `realized_from`'s: contiguous fine log returns whose bars
    open in the bucket. Every measure is null unless `n = expected`.
    """
    if interval not in INTERVALS:
        raise Refused(f"interval={interval!r} is not one of {', '.join(INTERVALS)}")
    if not 0 < alpha < 1:
        raise Refused(f"alpha={alpha}: a test level is in (0, 1)")
    utils.require(fine, ("ticker", "ts", "close_ts", "close"), "load fine bars with gr.market.candles")
    lf = utils.lazy(fine).sort("ticker", "ts")
    widths = lf.select((pl.col("close_ts") - pl.col("ts")).dt.total_microseconds().unique()).collect().to_series().to_list()
    coarse = INTERVALS[interval]
    if len(widths) != 1 or widths[0] >= coarse or coarse % widths[0]:
        raise Refused(f"the fine bars are {widths} µs wide; {interval} must be a coarser whole multiple of one width")
    expected = coarse // widths[0]
    if expected < 3:
        raise Refused(f"{expected} returns a bucket cannot make tripower variation")
    r = pl.when(timeseries.contiguous()).then(timeseries.log_return())
    grouped = (
        lf.with_columns(r.alias("_r"), pl.col("ts").dt.truncate(interval).alias("_bucket"))
        .group_by("ticker", "_bucket")
        .agg(pl.col("_r").sort_by("ts").drop_nulls().alias("_rs"), pl.col("_r").count().alias("n"))
        .sort("ticker", "_bucket")
        .collect()
    )
    q = float(norm.ppf(alpha))
    empty = dict.fromkeys(_MEASURES)
    rows = [_measure(np.asarray(rs, dtype=float), c_theta, q) if n == expected else empty for rs, n in zip(grouped["_rs"].to_list(), grouped["n"].to_list(), strict=True)]
    measures = pl.DataFrame(rows, schema=dict.fromkeys(_MEASURES, pl.Float64))
    return (
        grouped.select(
            "ticker",
            pl.col("_bucket").alias("ts"),
            (pl.col("_bucket") + pl.duration(microseconds=coarse)).alias("close_ts"),
            pl.col("n").cast(pl.Int64),
            pl.lit(expected, pl.Int64).alias("expected"),
        )
        .hstack(measures)
        .select(_OUT)
    )
