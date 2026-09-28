"""Jumps: Lee and Mykland's test, after Boudt, Croux and Laurent's robust periodicity.

    f = gr.jumps.periodicity(returns, slot="5m")                  # a factor per 5-minute slot of the week
    gr.jumps.lee_mykland(returns, periodicity=f)                  # L, sigma_local, threshold, jump

A jump test that ignores the day's shape over-detects in the busy hours and
misses in the quiet ones, so a map of its jumps by hour redraws the
volatility map. Boudt, Croux and Laurent (2011) estimate that shape robustly,
so that the jumps themselves do not inflate it. Lee and Mykland (2008) then
test each return against a local bipower scale that only earlier returns
make.

Lee and Mykland: L(i) = r_i / σ̂_i, σ̂_i² = (1/(K−2)) Σ_{j=i−K+2}^{i−1} |r_j||r_{j−1}|;
(max|L| − C_n)/S_n is Gumbel under no jump, C_n = (2 log n)^½/c − (log π +
log log n)/(2c(2 log n)^½), S_n = 1/(c(2 log n)^½), c = √(2/π). Their window
table (K = 78 at 1h, 156 at 15m, 270 at 5m) is K = ⌈√(252·n)⌉ with n the
returns in a 24-hour day.
"""

import math

import polars as pl

from . import utils
from ._errors import Refused

LAYOUTS = ("time_of_week", "time_of_day")
RULES = ("gumbel", "bh")
# Boudt, Croux and Laurent's weights: keep z² ≤ χ²₁(0.99), and correct the truncated variance back to 1.
CHI2_99 = 6.635
WSD_CONSISTENCY = 1.081
SHORTH_CONSISTENCY = 0.741
_C = math.sqrt(2 / math.pi)


def _minutes(slot: str) -> int:
    unit, n = slot[-1], slot[:-1]
    if not n.isdigit() or unit not in "mh" or int(n) < 1:
        raise Refused(f"slot={slot!r} is not a width like 5m or 1h")
    m = int(n) * (60 if unit == "h" else 1)
    if 1440 % m:
        raise Refused(f"slot={slot!r} does not divide the day")
    return m


def _keys(by: str, width: int) -> list[pl.Expr]:
    slot = ((pl.col("ts").dt.hour().cast(pl.Int32) * 60 + pl.col("ts").dt.minute().cast(pl.Int32)) // width).alias("slot")
    return [pl.col("ts").dt.weekday().alias("weekday"), slot] if by == "time_of_week" else [slot]


def _names(by: str) -> list[str]:
    return ["weekday", "slot"] if by == "time_of_week" else ["slot"]


def _shorth(values: list[float]) -> float | None:
    """0.741 × the shortest range holding half the values plus one (Rousseeuw and Leroy's shorth)."""
    x = sorted(values)
    n = len(x)
    if n < 2:
        return None
    h = n // 2 + 1
    return SHORTH_CONSISTENCY * min(x[i + h - 1] - x[i] for i in range(n - h + 1))


def periodicity(
    returns: pl.LazyFrame | pl.DataFrame, *, slot: str = "5m", by: str = "time_of_week", fit: tuple | None = None
) -> pl.DataFrame:
    """Boudt, Croux and Laurent's (2011) robust periodicity: `ticker`, slot keys, `n`, `factor`.

    1. Each return over its UTC day's bipower scale √((π/2) Σ|r_i||r_{i−1}| / (M−1)),
       pairs across a hole skipped. A day with fewer than half its returns has no scale.
    2. Per slot, the ShortH scale of the standardised returns, normalised to mean f² = 1.
    3. Per slot, the weighted standard deviation √(1.081 Σ w r̄² / Σ w), w = 1 where
       (r̄ / f^ShortH)² ≤ 6.635 (χ²₁ at 99%), normalised to mean f² = 1.

    A slot is a `slot`-wide piece of the UTC day, of each weekday for
    `time_of_week` (Monday = 1) or of every day for `time_of_day`.
    """
    if by not in LAYOUTS:
        raise Refused(f"by={by!r} is not one of {', '.join(LAYOUTS)}")
    width = _minutes(slot)
    utils.require(returns, ("ticker", "ts", "close_ts", "return"), "make returns with gr.timeseries.returns")
    lf = utils.lazy(returns).sort("ticker", "ts")
    if fit is not None:
        lo, hi = utils.window(*fit)
        lf = lf.filter((pl.col("ts").dt.epoch("us") >= lo) & (pl.col("close_ts").dt.epoch("us") <= hi))
    bar = lf.select((pl.col("close_ts") - pl.col("ts")).dt.total_minutes().median()).collect().item()
    per_day = 1440 / bar if bar else None
    frame = (
        lf.with_columns(
            (pl.col("return").abs() * pl.col("return").shift(1).over("ticker").abs()).alias("_bp"),
            pl.col("ts").dt.date().alias("_day"),
            *_keys(by, width),
        )
        .with_columns(
            pl.col("return").is_not_null().sum().over("ticker", "_day").alias("_m"),
            pl.col("_bp").sum().over("ticker", "_day").alias("_sbp"),
        )
        .with_columns(
            pl.when((pl.col("_m") >= (per_day or 0) / 2) & (pl.col("_m") > 1) & (pl.col("_sbp") > 0))
            .then(((math.pi / 2) * pl.col("_sbp") / (pl.col("_m") - 1)).sqrt())
            .alias("_scale")
        )
        .with_columns((pl.col("return") / pl.col("_scale")).alias("_z"))
        .filter(pl.col("_z").is_not_null())
        .collect()
    )
    keys = ["ticker", *_names(by)]
    groups = frame.group_by(keys).agg(pl.col("_z"), pl.len().alias("n"))
    shorth = groups.with_columns(pl.col("_z").map_elements(_shorth, return_dtype=pl.Float64).alias("_sh")).drop("_z")
    shorth = shorth.with_columns((pl.col("_sh") / (pl.col("_sh") ** 2).mean().over("ticker").sqrt()).alias("_fsh"))
    weighted = (
        frame.join(shorth.select(*keys, "_fsh"), on=keys)
        .with_columns(((pl.col("_z") / pl.col("_fsh")) ** 2 <= CHI2_99).cast(pl.Float64).alias("_w"))
        .group_by(keys)
        .agg((WSD_CONSISTENCY * (pl.col("_w") * pl.col("_z") ** 2).sum() / pl.col("_w").sum()).sqrt().alias("_wsd"))
    )
    return (
        shorth.join(weighted, on=keys)
        .with_columns((pl.col("_wsd") / (pl.col("_wsd") ** 2).mean().over("ticker").sqrt()).alias("factor"))
        .select(*keys, pl.col("n").cast(pl.Int64), "factor")
        .sort(keys)
    )


def gumbel_threshold(n: int, alpha: float) -> float:
    """C_n + S_n·β*, β* = −log(−log(1 − α)): the |L| above which a return is a jump among n."""
    if n < 3 or not 0 < alpha < 1:
        raise Refused(f"n={n}, alpha={alpha}: the Gumbel threshold needs n ≥ 3 and 0 < alpha < 1")
    root = math.sqrt(2 * math.log(n))
    c_n = root / _C - (math.log(math.pi) + math.log(math.log(n))) / (2 * _C * root)
    s_n = 1 / (_C * root)
    return c_n + s_n * -math.log(-math.log(1 - alpha))


def lee_mykland(
    returns: pl.LazyFrame | pl.DataFrame,
    *,
    window: int | None = None,
    alpha: float = 0.01,
    per_day: int | None = None,
    periodicity: pl.DataFrame | None = None,
    rule: str = "gumbel",
    q: float = 0.05,
) -> pl.DataFrame:
    """The returns with `sigma_local`, `L`, `threshold` and `jump`, per Lee and Mykland (2008).

    - `sigma_local` at a return is √ of the mean of |r_j||r_{j−1}| over the
      K − 2 pairs **strictly before** it, and null until that many exist
      after a hole: the scale never sees the return it judges.
    - K defaults to ⌈√(252·n)⌉, n = `per_day`, which defaults to the returns
      in 24 hours at the bars' width (270 at 5m).
    - `rule="gumbel"`: a jump is |L| > the threshold for n = `per_day`
      returns at `alpha`, which bounds the chance of any false jump in a day.
      `rule="bh"`: Benjamini–Hochberg at `q` over every return passed in, on
      two-sided normal p-values of L.
    - With `periodicity` (from `periodicity()`), each return is divided by its
      slot's factor first, and `factor` is kept.
    """
    if rule not in RULES:
        raise Refused(f"rule={rule!r} is not one of {', '.join(RULES)}")
    utils.require(returns, ("ticker", "ts", "close_ts", "return"), "make returns with gr.timeseries.returns")
    lf = utils.lazy(returns).sort("ticker", "ts")
    if per_day is None:
        bar = lf.select((pl.col("close_ts") - pl.col("ts")).dt.total_minutes().median()).collect().item()
        if not bar:
            raise Refused("cannot infer the bars' width; pass per_day")
        per_day = round(1440 / bar)
    k = window if window is not None else math.ceil(math.sqrt(252 * per_day))
    if k < 3:
        raise Refused(f"window={k}: the bipower scale needs K ≥ 3")
    r = pl.col("return")
    if periodicity is not None:
        names = [c for c in ("weekday", "slot") if c in periodicity.columns]
        width = _infer_width(periodicity)
        by = "time_of_week" if "weekday" in names else "time_of_day"
        lf = lf.with_columns(*_keys(by, width)).join(periodicity.lazy().select("ticker", *names, "factor"), on=["ticker", *names], how="left")
        r = pl.col("return") / pl.col("factor")
    frame = (
        lf.with_columns(r.alias("_r"))
        .with_columns((pl.col("_r").abs() * pl.col("_r").shift(1).over("ticker").abs()).alias("_bp"))
        .with_columns(
            pl.col("_bp").fill_null(0).rolling_sum(k - 2).over("ticker").alias("_sum"),
            pl.col("_bp").is_not_null().cast(pl.Int32).rolling_sum(k - 2).over("ticker").alias("_cnt"),
        )
        # The window ending at the previous return: never the return being judged.
        .with_columns(
            pl.when(pl.col("_cnt").shift(1).over("ticker") == k - 2)
            .then((pl.col("_sum").shift(1).over("ticker") / (k - 2)).sqrt())
            .alias("sigma_local")
        )
        .with_columns((pl.col("_r") / pl.col("sigma_local")).alias("L"))
        .collect()
    )
    if rule == "gumbel":
        threshold = gumbel_threshold(per_day, alpha)
        frame = frame.with_columns(pl.lit(threshold).alias("threshold"), (pl.col("L").abs() > threshold).alias("jump"))
    else:
        p = frame["L"].map_elements(lambda x: math.erfc(abs(x) / math.sqrt(2)), return_dtype=pl.Float64)
        tested = sorted(v for v in p.to_list() if v is not None)
        m = len(tested)
        cut = max((v for i, v in enumerate(tested, 1) if v <= i * q / m), default=-1.0)
        frame = frame.with_columns(p.alias("p_value")).with_columns(
            pl.lit(None, pl.Float64).alias("threshold"), (pl.col("p_value") <= cut).alias("jump")
        )
    drop = [c for c in ("_r", "_bp", "_sum", "_cnt") if c in frame.columns]
    return frame.drop(drop).with_columns(pl.col("jump").fill_null(False))


def _infer_width(table: pl.DataFrame) -> int:
    slots = table["slot"].max() + 1
    width = 1440 // slots
    if width * slots != 1440:
        raise Refused(f"a periodicity with {slots} slots does not tile the day")
    return width
