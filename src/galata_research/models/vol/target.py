"""Volatility targeting from forecasts: a position at each close, every trial counted, and the economics beside Sharpe.

The position sits on the bar whose close is the forecast's origin;
`gr.backtest.returns` holds it through the next bar, so there is no second
shift. The target is ex-ante: the realized volatility of the bars closed by
the split, or a declared number, never a full-sample scale (Liu, Tang and
Zhou 2019 on Moreira and Muir 2017).
"""

import numpy as np
import polars as pl

from ... import backtest, stats, studies, timeseries, utils
from ..._errors import Refused

RULES = ("inverse_vol", "inverse_variance", "conditional")
_MIN_HISTORY = 20


def estimation_target(bars: pl.DataFrame, split, periods_per_year: int) -> float:
    """Annualised close-to-close volatility of the bars with `close_ts ≤ split`: known at the split."""
    at = utils.instant("split", split)
    before = bars.filter(pl.col("close_ts").dt.epoch("us") <= at)
    r = timeseries.returns(before, kind="log")["return"].drop_nulls()
    if r.len() < 30:
        raise Refused(f"{r.len()} returns before the split are too few to set a target")
    return float(r.std(ddof=1)) * periods_per_year**0.5


def target(
    forecasts: pl.DataFrame,
    bars: pl.LazyFrame | pl.DataFrame,
    *,
    split,
    target: float | str = "estimation",
    rule: str = "inverse_vol",
    cap: float = 2.0,
    band: float = 0.0,
) -> pl.DataFrame:
    """One ticker's bars with `position`, set on each origin bar from the forecast made at its close.

    σ̂ = √(h = 1 variance × periods per year). `inverse_vol` is min(τ/σ̂, cap);
    `inverse_variance` min((τ/σ̂)², cap), Moreira and Muir's shape with c = 1
    fixed in advance, doubling the leverage swings; `conditional` (Bongaerts,
    Kang and van Dijk 2020) is min(τ/σ̂, cap) when σ̂ is in the top or bottom
    quintile of the earlier origins' σ̂ (expanding; medium until 20 exist)
    and 1 otherwise. Long only. `cap` 2 is QuantPedia's; Harvey et al. (2018)
    use none. `band` is a no-trade region: the position moves only when the
    rule's weight differs from it by more than `band` × the position. 0.25 is
    the static-portfolio 5/25 heuristic, not a vol-targeting result. A bar
    without a forecast has a null position.
    """
    if rule not in RULES:
        raise Refused(f"rule={rule!r} is not one of {', '.join(RULES)}")
    if cap <= 0 or band < 0:
        raise Refused(f"cap={cap} must be positive and band={band} non-negative")
    utils.require(bars, ("ticker", "ts", "close_ts", "close"), "load bars with gr.market.candles")
    frame = utils.lazy(bars).sort("ticker", "ts").collect()
    tickers = frame["ticker"].unique().to_list()
    if len(tickers) != 1:
        raise Refused(f"target one ticker at a time; these bars hold {', '.join(sorted(tickers))}")
    width = frame["close_ts"][0] - frame["ts"][0]
    per_year = timeseries.periods_per_year(_interval(width))
    if target == "estimation":
        tau = estimation_target(frame, split, per_year)
    elif isinstance(target, (int, float)) and not isinstance(target, bool) and target > 0:
        tau = float(target)
    else:
        raise Refused(f"target={target!r} must be 'estimation' or a positive annualised volatility declared in advance")
    one = forecasts.filter(pl.col("h") == 1).select("close_ts", "variance").sort("close_ts")
    sigma = (one["variance"] * per_year).sqrt().to_numpy()
    raw = np.minimum(tau / sigma, cap) if rule != "inverse_variance" else np.minimum((tau / sigma) ** 2, cap)
    if rule == "conditional":
        weights = np.ones(sigma.size)
        for i in range(sigma.size):
            past = sigma[:i]
            past = past[np.isfinite(past)]
            if past.size >= _MIN_HISTORY:
                lo, hi = np.quantile(past, [0.2, 0.8])
                if sigma[i] <= lo or sigma[i] >= hi:
                    weights[i] = raw[i]
        raw = weights
    held = np.empty(raw.size)
    current = None
    for i, w in enumerate(raw):
        if not np.isfinite(w):
            held[i] = np.nan
            continue
        if current is None or abs(w - current) > band * current:
            current = float(w)
        held[i] = current
    positions = pl.DataFrame({"close_ts": one["close_ts"], "position": held}).with_columns(pl.col("position").fill_nan(None))
    return frame.join(positions, on="close_ts", how="left")


def _interval(width) -> str:
    micros = int(width.total_seconds() * 1_000_000)
    for name, us in timeseries.INTERVALS.items():
        if us == micros:
            return name
    raise Refused(f"bars {width} wide are not a record interval")


def trials(
    bars: pl.LazyFrame | pl.DataFrame,
    forecasts: pl.DataFrame,
    *,
    split,
    target: float | str = "estimation",
    rules=("inverse_vol",),
    bands=(0.0,),
    cap: float = 2.0,
    fee: float = backtest.TAKER_FEE,
    funding: pl.LazyFrame | pl.DataFrame | None = None,
) -> pl.DataFrame:
    """`hold` and every (model, rule, band), as `studies.trial` frames from the first origin on, so every one counts.

    `forecasts` carries a `model` column. Trial names are
    `"{model} {rule} band {band}"`.
    """
    utils.require(forecasts, ("model", "close_ts", "h", "variance"), "concatenate gr.models.vol forecasts with a model column")
    frame = utils.lazy(bars).sort("ticker", "ts").collect()
    start = forecasts["close_ts"].min()
    out = []
    hold = frame.with_columns(pl.when(pl.col("close_ts") >= start).then(1.0).alias("position"))
    out.append(_trial(hold, "hold", fee, funding))
    for model in sorted(forecasts["model"].unique().to_list()):
        f = forecasts.filter(pl.col("model") == model)
        for rule in rules:
            for band in bands:
                positioned = globals()["target"](f, frame, split=split, target=target, rule=rule, cap=cap, band=band)
                out.append(_trial(positioned, f"{model} {rule} band {band:g}", fee, funding))
    return pl.concat(out)


def _trial(positioned: pl.DataFrame, name: str, fee: float, funding) -> pl.DataFrame:
    r = backtest.returns(positioned, pl.col("position"), fee=fee, funding=funding)
    return r.select(pl.lit(name).alias("trial"), "ticker", "ts", "position", "bar_return", "gross", "cost", "funding", "net")


def economics(frame: pl.DataFrame, *, benchmark: str = "hold", periods_per_year: int) -> tuple[pl.DataFrame, dict]:
    """Per trial the economics beside Sharpe, and the Deflated Sharpe Ratio over every trial.

    `sharpe_annual, max_drawdown, drawdown_per_vol` (max drawdown ÷ the
    trial's own annualised volatility: the Bloomberg 2021 check that a smaller
    drawdown is not just a smaller position), `turnover_per_year, fees,
    funding, fee_bp_g1, fee_bp_g10` (Fleming, Kirby and Ostdiek's fee against
    `benchmark` on the same bars).
    """
    scored = frame.drop_nulls("net")
    names = scored["trial"].unique(maintain_order=True).to_list()
    if benchmark not in names:
        raise Refused(f"no {benchmark!r} trial to price the others against")
    base = scored.filter(pl.col("trial") == benchmark).select("ts", pl.col("net").alias("_b"))
    rows = []
    for name in names:
        t = scored.filter(pl.col("trial") == name)
        joined = t.join(base, on="ts")
        net = t["net"]
        vol = float(net.std(ddof=1)) * periods_per_year**0.5 if t.height > 1 else None
        dd = stats.max_drawdown(net)
        sr = stats.sharpe(net)
        turnover = float((t["position"].fill_null(0).diff().abs().fill_null(t["position"].fill_null(0)[0])).sum())
        rows.append(
            {
                "trial": name,
                "periods": t.height,
                "sharpe_annual": None if sr is None else sr * periods_per_year**0.5,
                "max_drawdown": dd,
                "drawdown_per_vol": dd / vol if vol else None,
                "turnover_per_year": turnover / t.height * periods_per_year,
                "fees": float(t["cost"].fill_null(0).sum()),
                "funding": float(t["funding"].fill_null(0).sum()),
                "fee_bp_g1": stats.performance_fee(joined["net"], joined["_b"], 1.0, periods_per_year),
                "fee_bp_g10": stats.performance_fee(joined["net"], joined["_b"], 10.0, periods_per_year),
            }
        )
    table = pl.DataFrame(rows, infer_schema_length=None)
    summary = studies.summary(frame.with_columns(pl.col("net")), periods_per_year)
    return table, stats.deflate(summary)
