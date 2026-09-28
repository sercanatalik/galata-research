"""Who moves first: the shifted Hayashi–Yoshida correlation of two asynchronously traded prices.

    gr.leadlag.hayashi_yoshida(binance, bybit, LAGS)     # lag_ms, hy, rho
    gr.leadlag.lead_lag(binance, bybit, LAGS, every="1h") # lead_ms, rho_lead, rho_0, llr per hour

Resampling two tick series to a grid pulls their correlation toward zero (the
Epps effect) and loses the milliseconds a lead lives in. Hayashi and Yoshida
(2005) sum the products of price changes over every pair of intervals that
overlap, on each series' own clock. Hoffmann, Rosenbaum and Yoshida (2013)
shift one clock by θ and take the θ that maximises |covariance| as the lead.
Huth and Abergel (2014) summarise the lag grid by LLR = Σ_{θ>0} ρ² / Σ_{θ<0} ρ².

**Sign.** y's clock is shifted by θ, so if y copies x with a delay d, ρ peaks
at θ = d > 0: a positive lead, or LLR above 1, means **x leads y**.

**Clocks.** Each venue stamps its own trades; legacy measured one venue 35 ms
off the local clock (`legacy/galata-legacy/design/measured.md:249-256`). A lead
within a few tens of milliseconds is within what the clocks can tell apart.
"""

from collections.abc import Sequence
from datetime import timedelta

import polars as pl

from . import utils
from ._errors import Refused

LAGS_MS = (-5000, -2000, -1000, -500, -200, -100, -50, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)


def _ticks(frame: pl.LazyFrame | pl.DataFrame, name: str) -> tuple[pl.Series, pl.Series]:
    """Distinct timestamps in micros and the log of the last price at each."""
    utils.require(frame, ("ts", "price"), f"{name} needs ts and price, as trades have")
    ticks = (
        utils.lazy(frame)
        .select(pl.col("ts").dt.epoch("us").alias("t"), pl.col("price").log().alias("p"))
        .drop_nulls()
        .group_by("t", maintain_order=True)
        .agg(pl.col("p").last())
        .sort("t")
        .collect()
    )
    if ticks.height < 2:
        raise Refused(f"{name} has {ticks.height} distinct timestamp(s); a price change needs two")
    return ticks["t"], ticks["p"]


def _micros(lag) -> int:
    if isinstance(lag, timedelta):
        return (lag.days * 86_400 + lag.seconds) * 1_000_000 + lag.microseconds
    return int(lag) * 1000


def _hy(t: pl.Series, x: pl.Series, s: pl.Series, y: pl.Series, shift: int) -> float:
    """Σ_j Δy_j · Σ_{k overlaps J_j − shift} Δx_k, the inner sum telescoped to x[k₂] − x[k₁ − 1]."""
    n = t.len() - 1
    a, b = s.slice(0, s.len() - 1) - shift, s.slice(1) - shift
    dy = y.diff().slice(1)
    k1 = t.search_sorted(a, side="right").clip(1, n)
    k2 = t.search_sorted(b, side="left").clip(1, n)
    frame = pl.DataFrame({"k1": k1, "k2": k2, "dy": dy})
    # An x interval (t_{k-1}, t_k] overlaps (a, b] iff t_k > a and t_{k-1} < b; beyond the ends there is none.
    frame = frame.with_columns(
        ((pl.col("k1") <= pl.col("k2")) & (a < t[n]) & (b > t[0])).alias("hit")
    ).filter("hit")
    if frame.is_empty():
        return 0.0
    upper = x.gather(frame["k2"])
    lower = x.gather(frame["k1"] - 1)
    return float(((upper - lower) * frame["dy"]).sum())


def hayashi_yoshida(
    x: pl.LazyFrame | pl.DataFrame, y: pl.LazyFrame | pl.DataFrame, lags: Sequence = LAGS_MS
) -> pl.DataFrame:
    """`lag_ms, hy, rho` for each lag: y's clock shifted by the lag, log price changes, ties collapsed to the last."""
    t, px = _ticks(x, "x")
    s, py = _ticks(y, "y")
    return _curve(t, px, s, py, lags)


def _curve(t, px, s, py, lags) -> pl.DataFrame:
    scale = (float((px.diff().slice(1) ** 2).sum()) * float((py.diff().slice(1) ** 2).sum())) ** 0.5
    rows = []
    for lag in lags:
        shift = _micros(lag)
        hy = _hy(t, px, s, py, shift)
        rows.append({"lag_ms": shift / 1000, "hy": hy, "rho": hy / scale if scale else None})
    return pl.DataFrame(rows, schema={"lag_ms": pl.Float64, "hy": pl.Float64, "rho": pl.Float64})


def _summary(curve: pl.DataFrame) -> dict:
    best = curve.drop_nulls("rho").with_columns(pl.col("rho").abs().alias("_a")).sort("_a", descending=True)
    rho2 = curve.with_columns((pl.col("rho") ** 2).alias("r2"))
    ahead = rho2.filter(pl.col("lag_ms") > 0)["r2"].sum()
    behind = rho2.filter(pl.col("lag_ms") < 0)["r2"].sum()
    zero = curve.filter(pl.col("lag_ms") == 0)["rho"]
    return {
        "lead_ms": None if best.is_empty() else best["lag_ms"][0],
        "rho_lead": None if best.is_empty() else best["rho"][0],
        "rho_0": zero[0] if zero.len() else None,
        "llr": ahead / behind if behind else None,
    }


def lead_lag(
    x: pl.LazyFrame | pl.DataFrame, y: pl.LazyFrame | pl.DataFrame, lags: Sequence = LAGS_MS, *, every: str | None = None
) -> pl.DataFrame:
    """`ts, lead_ms, rho_lead, rho_0, llr, x_ticks, y_ticks`, per bucket of width `every` or once.

    `lead_ms` maximises |ρ|; `llr` = Σ_{lag>0} ρ² / Σ_{lag<0} ρ². Positive, or
    above 1, means x leads y. A bucket keeps only the ticks inside it, on both
    sides, and is skipped when either side has fewer than two.
    """
    t, px = _ticks(x, "x")
    s, py = _ticks(y, "y")
    schema = {"ts": pl.Datetime("us", "UTC"), "lead_ms": pl.Float64, "rho_lead": pl.Float64, "rho_0": pl.Float64,
              "llr": pl.Float64, "x_ticks": pl.Int64, "y_ticks": pl.Int64}  # fmt: skip
    xs = pl.DataFrame({"t": t, "p": px})
    ys = pl.DataFrame({"t": s, "p": py})
    if every is None:
        spans = [(t[0], None)]
    else:
        bucket = pl.col("t").cast(pl.Datetime("us", "UTC")).dt.truncate(every).dt.epoch("us")
        xs, ys = xs.with_columns(bucket.alias("b")), ys.with_columns(bucket.alias("b"))
        spans = [(b, b) for b in sorted(set(xs["b"].unique()) & set(ys["b"].unique()))]
    rows = []
    for start, b in spans:
        xb = xs if b is None else xs.filter(pl.col("b") == b)
        yb = ys if b is None else ys.filter(pl.col("b") == b)
        if xb.height < 2 or yb.height < 2:
            continue
        summary = _summary(_curve(xb["t"], xb["p"], yb["t"], yb["p"], lags))
        rows.append({"ts": pl.Series([start]).cast(pl.Datetime("us", "UTC"))[0], **summary, "x_ticks": xb.height, "y_ticks": yb.height})
    return pl.DataFrame(rows, schema=schema)
