"""Bars to series: annualisation, contiguous returns, and resampling a series.

Polars only. A return never spans a hole: it exists only where the previous
row is the bar immediately before (its `close_ts` is this bar's `ts`), the
rule `backtest.returns` has always applied. Annualisation is calendar time,
because the venue never closes.
"""

from math import sqrt

import polars as pl

from . import utils
from ._errors import Refused
from .market import INTERVALS

_YEAR_US = 365 * 86_400_000_000
KINDS = ("simple", "log")
_BARS = ("ticker", "ts", "close_ts", "close")


def periods_per_year(interval: str) -> int:
    """Bars of `interval` in a 365-day year of continuous trading: 525,600 (1m), 8,760 (1h), 2,190 (4h), 365 (1d).

    A convention, stated: Hyperliquid perps trade every hour of every day, so
    the year is calendar time, not 252 sessions. No standard fixes 365 against
    365.25. vectorbt derives the same factor from a bar's frequency.
    """
    if interval not in INTERVALS:
        raise Refused(f"interval={interval!r} is not one of {', '.join(INTERVALS)}")
    return _YEAR_US // INTERVALS[interval]


def contiguous() -> pl.Expr:
    """True where the previous row, per ticker, is the bar immediately before this one. Rows sorted by ticker, ts."""
    return pl.col("close_ts").shift(1).over("ticker") == pl.col("ts")


def simple_return() -> pl.Expr:
    """Close over the previous close, minus one, per ticker, unmasked: combine with `contiguous()`."""
    return pl.col("close") / pl.col("close").shift(1).over("ticker") - 1


def log_return() -> pl.Expr:
    """The log of close over the previous close, per ticker, unmasked: combine with `contiguous()`."""
    return (pl.col("close") / pl.col("close").shift(1).over("ticker")).log()


def returns(bars: pl.LazyFrame | pl.DataFrame, *, kind: str = "simple") -> pl.DataFrame:
    """`ticker, ts, close_ts, return` per bar: close to close, null on a ticker's first bar and after a hole.

    `kind` is `"simple"` (close / previous − 1) or `"log"` (ln of the ratio).
    `bars` are candles from `gr.market.candles`.
    """
    if kind not in KINDS:
        raise Refused(f"kind={kind!r} is not one of {', '.join(KINDS)}")
    utils.require(bars, _BARS, "load bars with gr.market.candles")
    value = simple_return() if kind == "simple" else log_return()
    return (
        utils.lazy(bars)
        .sort("ticker", "ts")
        .select("ticker", "ts", "close_ts", pl.when(contiguous()).then(value).alias("return"))
        .collect()
    )


def _series(values) -> pl.Series:
    s = values if isinstance(values, pl.Series) else pl.Series(list(values), dtype=pl.Float64)
    return s.drop_nulls().cast(pl.Float64)


def stationary_bootstrap_indices(n: int, block: float, rng) -> list[int]:
    """Politis and Romano's (1994) stationary bootstrap: blocks of geometric length, mean `block`.

    Each index continues the last one (wrapping at `n`) with probability
    `1 − 1/block`, and otherwise starts a new block at a uniform position.
    """
    if n < 1 or block < 1:
        raise Refused(f"n={n}, block={block}: a stationary bootstrap needs n ≥ 1 and block ≥ 1")
    p = 1.0 / block
    out = [rng.randrange(n)]
    for _ in range(n - 1):
        out.append(rng.randrange(n) if rng.random() < p else (out[-1] + 1) % n)
    return out



def optimal_block(series) -> float:
    """Politis and White's (2004) optimal mean block for the stationary bootstrap, as corrected in 2009.

    Patton, Politis and White (2009) corrected the constants after Nordman
    (2008). This follows arch 8.0.0's `optimal_block_length` (its
    `stationary` column) step for step:
    - find m̂, the first run of `max(5, ⌊log₁₀ n⌋)` autocorrelations under
      `2√(log₁₀ n / n)`, and set `M = 2m̂`;
    - with a flat-top lag window, take `Ĝ = Σ 2λ(k/M)·k·γ̂(k)` and the
      long-run variance `ĝ = γ̂(0) + Σ 2λ(k/M)·γ̂(k)`;
    - `b = (2Ĝ² / (2ĝ²))^{1/3} n^{1/3}`, capped at `min(3√n, n/3)`.
    """
    from math import ceil, log10

    x = _series(series).to_list()
    n = len(x)
    if n < 10:
        raise Refused(f"{n} points are too few to choose a block")
    mean = sum(x) / n
    eps = [v - mean for v in x]
    b_max = ceil(min(3 * sqrt(n), n / 3))
    kn = max(5, int(log10(n)))
    m_max = int(ceil(sqrt(n))) + kn
    cv = 2 * sqrt(log10(n) / n)

    def dot(a, b):
        return sum(p * q for p, q in zip(a, b))

    acv, acorr, opt_m = [], [], None
    for i in range(m_max + 1):
        v1, v2 = dot(eps[i + 1 :], eps[i + 1 :]), dot(eps[: n - (i + 1)], eps[: n - (i + 1)])
        cross = dot(eps[i:], eps[: n - i])
        acv.append(cross / n)
        acorr.append(abs(cross) / sqrt(v1 * v2) if v1 * v2 > 0 else 0.0)
        if i >= kn and opt_m is None and all(a < cv for a in acorr[i - kn : i]):
            opt_m = i - kn
    m = min(2 * max(opt_m, 1) if opt_m is not None else m_max, m_max)

    g, long_run = 0.0, acv[0]
    for k in range(1, m + 1):
        lam = 1.0 if k / m <= 0.5 else 2 * (1 - k / m)
        g += 2 * lam * k * acv[k]
        long_run += 2 * lam * acv[k]
    if long_run <= 0:
        return 1.0
    b = ((2 * g**2) / (2 * long_run**2)) ** (1 / 3) * n ** (1 / 3)
    return min(b, b_max)


# ── Realized volatility ─────────────────────────────────────────────────────
#
# Per-period σ over a window, per ticker. Every estimator counts the same bars:
# one with a contiguous return and, when the frame carries `in_gap`, not in a
# gap (legacy crates/statistics/src/lib.rs:614-620, one count and one absence
# rule for every estimator). Nothing is clamped: a negative Garman-Klass or
# Rogers-Satchell term is a malformed bar, and a clamp would hide it.

ESTIMATORS = ("close_to_close", "parkinson", "garman_klass", "rogers_satchell", "yang_zhang")
_OHLC = ("ticker", "ts", "close_ts", "open", "high", "low", "close")
_LN2 = 0.6931471805599453


def parkinson_term() -> pl.Expr:
    """ln(H/L)² / (4 ln 2): a bar's variance from its range (Parkinson 1980; ~5.2× close-to-close's efficiency)."""
    return (pl.col("high") / pl.col("low")).log().pow(2) / (4 * _LN2)


def garman_klass_term() -> pl.Expr:
    """½ ln(H/L)² − (2 ln 2 − 1) ln(C/O)² (Garman and Klass 1980, the practical form; ~7.4×). Assumes no opening jump."""
    return 0.5 * (pl.col("high") / pl.col("low")).log().pow(2) - (2 * _LN2 - 1) * (pl.col("close") / pl.col("open")).log().pow(2)


def rogers_satchell_term() -> pl.Expr:
    """ln(H/C) ln(H/O) + ln(L/C) ln(L/O) (Rogers and Satchell 1991): unbiased under drift."""
    h, low, o, c = pl.col("high"), pl.col("low"), pl.col("open"), pl.col("close")
    return (h / c).log() * (h / o).log() + (low / c).log() * (low / o).log()


def _counted(names: list[str]) -> pl.Expr:
    ok = contiguous().fill_null(False)
    return ok & ~pl.col("in_gap").fill_null(False) if "in_gap" in names else ok


def realized(
    bars: pl.LazyFrame | pl.DataFrame,
    estimator: str,
    window: int,
    *,
    min_periods: int | None = None,
) -> pl.DataFrame:
    """`ticker, ts, close_ts, n, sigma`: per-period σ over the last `window` bars, known at the last bar's close.

    - `close_to_close`: the sample standard deviation (ddof 1) of log returns.
    - `parkinson`, `garman_klass`, `rogers_satchell`: √ of the window mean of
      the per-bar term. Range estimators read low under discrete sampling (the
      bar's extremes are sampled, not continuous); not corrected here.
    - `yang_zhang`: √(var(ln Oₜ/Cₜ₋₁) + k·var(ln Cₜ/Oₜ) + (1−k)·mean(RS)),
      k = 0.34/(1.34 + (n+1)/(n−1)) (Yang and Zhang 2000; up to ~14×). On a
      venue that never closes the open is the previous close and the first term
      is ~0; it matters on markets whose underlying closes.

    The efficiencies are the commonly cited ones, from secondary sources.
    `n` counts the window's bars with a contiguous return and not `in_gap`;
    `sigma` is null while `n < min_periods`, which defaults to `window`.
    """
    if estimator not in ESTIMATORS:
        raise Refused(f"estimator={estimator!r} is not one of {', '.join(ESTIMATORS)}")
    if window < 2:
        raise Refused(f"window={window}: a volatility needs at least 2 bars")
    floor = window if min_periods is None else min_periods
    if not 1 <= floor <= window:
        raise Refused(f"min_periods={min_periods} must be between 1 and window={window}")
    utils.require(bars, _OHLC if estimator != "close_to_close" else _BARS, "load bars with gr.market.candles")
    lf = utils.lazy(bars).sort("ticker", "ts")
    counted = _counted(lf.collect_schema().names())

    def roll(term: pl.Expr, how: str) -> pl.Expr:
        masked = pl.when(pl.col("_counted")).then(term)
        if how == "mean":
            return masked.rolling_mean(window, min_samples=1).over("ticker")
        if how == "var":
            return masked.rolling_var(window, min_samples=2, ddof=1).over("ticker")
        return masked.rolling_std(window, min_samples=2, ddof=1).over("ticker")

    if estimator == "close_to_close":
        variance = roll(log_return(), "std").pow(2)
    elif estimator == "yang_zhang":
        n = pl.col("n").cast(pl.Float64)
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        overnight = (pl.col("open") / pl.col("close").shift(1).over("ticker")).log()
        intraday = (pl.col("close") / pl.col("open")).log()
        variance = roll(overnight, "var") + k * roll(intraday, "var") + (1 - k) * roll(rogers_satchell_term(), "mean")
    else:
        term = {"parkinson": parkinson_term, "garman_klass": garman_klass_term, "rogers_satchell": rogers_satchell_term}[estimator]()
        variance = roll(term, "mean")
    return (
        lf.with_columns(counted.alias("_counted"))
        .with_columns(pl.col("_counted").cast(pl.Int64).rolling_sum(window, min_samples=1).over("ticker").alias("n"))
        .select(
            "ticker",
            "ts",
            "close_ts",
            pl.col("n").cast(pl.Int64),
            pl.when(pl.col("n") >= floor).then(variance.sqrt()).alias("sigma"),
        )
        .collect()
    )


def realized_from(fine: pl.LazyFrame | pl.DataFrame, interval: str) -> pl.DataFrame:
    """Per bucket of the coarser `interval`: realized variance and realized range from the fine bars inside it.

    `ticker, ts, close_ts, n, expected, rv, rr`, where `rv = Σ r²` over the
    contiguous fine log returns whose bars open in the bucket (the first spans
    from the previous bucket's last close), and `rr = Σ ln(H/L)² / (4 ln 2)`
    over the same bars (the realized range; Christensen and Podolskij 2007,
    Martens and van Dijk 2007), biased low by discrete sampling. Both are null
    unless the bucket holds every return it should (`n = expected`): a sum over
    a partial bucket understates.
    """
    if interval not in INTERVALS:
        raise Refused(f"interval={interval!r} is not one of {', '.join(INTERVALS)}")
    utils.require(fine, _OHLC, "load fine bars with gr.market.candles")
    lf = utils.lazy(fine).sort("ticker", "ts")
    widths = lf.select((pl.col("close_ts") - pl.col("ts")).dt.total_microseconds().unique()).collect().to_series().to_list()
    coarse = INTERVALS[interval]
    if len(widths) != 1 or widths[0] >= coarse or coarse % widths[0]:
        raise Refused(f"the fine bars are {widths} µs wide; {interval} must be a coarser whole multiple of one width")
    expected = coarse // widths[0]
    r = pl.when(contiguous()).then(log_return())
    return (
        lf.with_columns(r.alias("_r"), pl.col("ts").dt.truncate(interval).alias("_bucket"))
        .group_by("ticker", "_bucket")
        .agg(
            pl.col("_r").count().alias("n"),
            pl.col("_r").pow(2).sum().alias("_rv"),
            pl.when(pl.col("_r").is_not_null()).then(parkinson_term()).sum().alias("_rr"),
        )
        .select(
            "ticker",
            pl.col("_bucket").alias("ts"),
            (pl.col("_bucket") + pl.duration(microseconds=coarse)).alias("close_ts"),
            pl.col("n").cast(pl.Int64),
            pl.lit(expected, pl.Int64).alias("expected"),
            pl.when(pl.col("n") == expected).then(pl.col("_rv")).alias("rv"),
            pl.when(pl.col("n") == expected).then(pl.col("_rr")).alias("rr"),
        )
        .sort("ticker", "ts")
        .collect()
    )


def _warmup(lam: float) -> int:
    from math import ceil, log

    return ceil(log(0.01) / log(lam))


def ewma_vol(bars: pl.LazyFrame | pl.DataFrame, *, lam: float = 0.94, warmup: int | None = None) -> pl.DataFrame:
    """`ticker, ts, close_ts, n, sigma`: RiskMetrics' EWMA, σ²ₜ = λσ²ₜ₋₁ + (1−λ)rₜ², mean zero, on log returns.

    `sigma` at close t includes rₜ, so it is the forecast for t+1 (RiskMetrics
    indexes the same number by the day it forecasts). Seeded with the first
    squared return; null for the first `warmup` returns, by default until the
    seed weighs under 1% (⌈ln 0.01 / ln λ⌉: 75 at 0.94). A hole is bridged: the
    recursion skips the missing return without decaying (roadmap D5). `n`
    counts the returns so far.
    """
    if not 0 < lam < 1:
        raise Refused(f"lam={lam} must be inside (0, 1)")
    skip = _warmup(lam) if warmup is None else warmup
    utils.require(bars, _BARS, "load bars with gr.market.candles")
    r2 = pl.when(contiguous()).then(log_return()).pow(2)
    return (
        utils.lazy(bars)
        .sort("ticker", "ts")
        .with_columns(r2.alias("_r2"))
        .with_columns(
            pl.col("_r2").is_not_null().cast(pl.Int64).cum_sum().over("ticker").alias("n"),
            pl.col("_r2").ewm_mean(alpha=1 - lam, adjust=False, ignore_nulls=True).over("ticker").alias("_s2"),
        )
        .select("ticker", "ts", "close_ts", "n", pl.when(pl.col("n") > skip).then(pl.col("_s2").sqrt()).alias("sigma"))
        .collect()
    )


def ewma_max(bars: pl.LazyFrame | pl.DataFrame, *, fast: float = 0.94, slow: float = 0.97) -> pl.DataFrame:
    """`ticker, ts, close_ts, sigma`: the larger of a fast and a slow EWMA σ, which rises fast and falls slowly.

    The defaults are RiskMetrics' daily and monthly decays (half-lives ~11 and
    ~23 bars). The Bloomberg crypto note (2021) that uses the rule could not be
    read for its own legs, so these are not its numbers.
    """
    f, s = ewma_vol(bars, lam=fast), ewma_vol(bars, lam=slow)
    return f.select("ticker", "ts", "close_ts", pl.max_horizontal(pl.col("sigma"), s["sigma"]).alias("sigma")).with_columns(
        pl.when(pl.col("sigma").is_not_null() & s["sigma"].is_not_null()).then(pl.col("sigma")).alias("sigma")
    )


def signature(bars: pl.LazyFrame | pl.DataFrame, minutes) -> pl.DataFrame:
    """`ticker, minutes, days, mean_rv`: mean daily realized variance at each sampling interval, from 1m bars.

    For k minutes: closes on the k-minute grid, log returns between consecutive
    grid closes exactly k minutes apart, squares summed per UTC day of each
    return's start, averaged over days holding all 1440/k returns. Finer
    sampling inflates RV where microstructure noise dominates; where the curve
    flattens is the finest usable sampling (Andersen, Bollerslev, Diebold and
    Labys 2000).
    """
    utils.require(bars, _BARS, "load 1m bars with gr.market.candles")
    lf = utils.lazy(bars).sort("ticker", "ts")
    widths = lf.select((pl.col("close_ts") - pl.col("ts")).dt.total_microseconds().unique()).collect().to_series().to_list()
    if widths != [INTERVALS["1m"]]:
        raise Refused("signature needs 1m bars")
    frames = []
    for k in minutes:
        if k < 1 or 1440 % k:
            raise Refused(f"minutes={k} must divide a day")
        step = k * INTERVALS["1m"]
        grid = lf.filter(pl.col("close_ts").dt.epoch("us") % step == 0)
        frames.append(
            grid.with_columns(
                pl.when((pl.col("close_ts") - pl.col("close_ts").shift(1).over("ticker")).dt.total_microseconds() == step)
                .then((pl.col("close") / pl.col("close").shift(1).over("ticker")).log())
                .alias("_r"),
                (pl.col("close_ts") - pl.duration(microseconds=step)).dt.date().alias("_day"),
            )
            .group_by("ticker", "_day")
            .agg(pl.col("_r").count().alias("_n"), pl.col("_r").pow(2).sum().alias("_rv"))
            .filter(pl.col("_n") == 1440 // k)
            .group_by("ticker")
            .agg(pl.len().cast(pl.Int64).alias("days"), pl.col("_rv").mean().alias("mean_rv"))
            .with_columns(pl.lit(k, pl.Int64).alias("minutes"))
            .select("ticker", "minutes", "days", "mean_rv")
            .collect()
        )
    return pl.concat(frames).sort("ticker", "minutes")
