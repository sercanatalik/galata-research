"""Bars to series: annualisation, contiguous returns, and resampling a series.

Polars only. A return never spans a hole: it exists only where the previous
row is the bar immediately before (its `close_ts` is this bar's `ts`), the
rule `backtest.returns` has always applied. Annualisation is calendar time,
because the venue never closes.
"""

import math
from collections.abc import Sequence
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

    `ticker, ts, close_ts, n, expected, rv, rr, rs_plus, rs_minus, rq`, where `rv = Σ r²` over the
    contiguous fine log returns whose bars open in the bucket (the first spans
    from the previous bucket's last close), and `rr = Σ ln(H/L)² / (4 ln 2)`
    over the same bars (the realized range; Christensen and Podolskij 2007,
    Martens and van Dijk 2007), biased low by discrete sampling. `rs_plus` and
    `rs_minus` split `rv` by the sign of each return (the realized
    semivariances), and `rq = (n/3)·Σr⁴` is the realized quarticity. All are null
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
            pl.when(pl.col("_r") > 0).then(pl.col("_r").pow(2)).sum().alias("_rs_plus"),
            pl.when(pl.col("_r") < 0).then(pl.col("_r").pow(2)).sum().alias("_rs_minus"),
            pl.col("_r").pow(4).sum().alias("_q"),
        )
        .select(
            "ticker",
            pl.col("_bucket").alias("ts"),
            (pl.col("_bucket") + pl.duration(microseconds=coarse)).alias("close_ts"),
            pl.col("n").cast(pl.Int64),
            pl.lit(expected, pl.Int64).alias("expected"),
            pl.when(pl.col("n") == expected).then(pl.col("_rv")).alias("rv"),
            pl.when(pl.col("n") == expected).then(pl.col("_rr")).alias("rr"),
            pl.when(pl.col("n") == expected).then(pl.col("_rs_plus")).alias("rs_plus"),
            pl.when(pl.col("n") == expected).then(pl.col("_rs_minus")).alias("rs_minus"),
            pl.when(pl.col("n") == expected).then(pl.col("n").cast(pl.Float64) / 3 * pl.col("_q")).alias("rq"),
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


# ── Periodicity and the walk-forward ────────────────────────────────────────
#
# Hourly crypto volatility has a calendar: on BTC's 1h bars it peaks at 13:00
# UTC (1.58× the mean) and is 0.54× on Saturdays. GARCH fitted through that
# reads it as persistence: α+β 1.0000 raw, 0.9948 once divided by an hour ×
# weekday factor fitted on the first half (measured 2026-09-27; Andersen and
# Bollerslev 1997 found the same).

LAYOUTS = ("hour_x_weekday", "hour_of_week", "hour_of_day")
STATS = ("mean_abs", "median_abs")
_SLOT = ("weekday", "hour")


def _slots() -> list[pl.Expr]:
    return [pl.col("ts").dt.weekday().alias("weekday"), pl.col("ts").dt.hour().alias("hour")]


def seasonal_factors(
    returns: pl.LazyFrame | pl.DataFrame,
    *,
    fit: tuple,
    by: str = "hour_x_weekday",
    stat: str = "mean_abs",
) -> pl.DataFrame:
    """`ticker, weekday, hour, n, factor, fit_end`: a periodic volatility factor per calendar cell, fitted on `fit` only.

    A return belongs to the weekday (Monday = 1) and hour of its bar's `ts`, in
    UTC. Only returns with `ts ≥ start` and `close_ts ≤ end` are read. The
    table holds the cells the returns occupy (168 at 1h, 42 at 4h, 7 at 1d),
    scaled so the mean of factor² over them is 1: the average variance is kept.

    - `hour_x_weekday` (default): the hour's statistic over the mean of the
      24, times the weekday's over the mean of the 7. 31 numbers from about
      210 and 720 returns each at 1h, and on BTC and ETH the same GARCH
      persistence as 168 free cells.
    - `hour_of_week`: each cell's own statistic, for a market whose weekday
      pattern differs by hour (GOLD, whose underlying has sessions).
    - `hour_of_day`: the hour's statistic, on every weekday.

    `stat` is `mean_abs`, or `median_abs`, which one jump cannot move (the
    motivation of Boudt, Croux and Laurent 2011; this is not their estimator).
    `n` is the count of fitted returns in the cell.
    """
    if by not in LAYOUTS:
        raise Refused(f"by={by!r} is not one of {', '.join(LAYOUTS)}")
    if stat not in STATS:
        raise Refused(f"stat={stat!r} is not one of {', '.join(STATS)}")
    utils.require(returns, ("ticker", "ts", "close_ts", "return"), "make returns with gr.timeseries.returns")
    lo, hi = utils.window(*fit)
    rows = utils.lazy(returns).drop_nulls("return").with_columns(*_slots())
    cells = rows.select("ticker", *_SLOT).unique()
    fitted = rows.filter((pl.col("ts").dt.epoch("us") >= lo) & (pl.col("close_ts").dt.epoch("us") <= hi))
    scale = pl.col("return").abs().mean() if stat == "mean_abs" else pl.col("return").abs().median()

    def by_keys(keys):
        return fitted.group_by("ticker", *keys).agg(scale.alias("_s"))

    counts = fitted.group_by("ticker", *_SLOT).agg(pl.len().cast(pl.Int64).alias("n"))
    if by == "hour_of_week":
        raw = cells.join(by_keys(_SLOT), on=["ticker", *_SLOT], how="left")
    elif by == "hour_of_day":
        raw = cells.join(by_keys(["hour"]), on=["ticker", "hour"], how="left")
    else:
        hours = by_keys(["hour"]).with_columns((pl.col("_s") / pl.col("_s").mean().over("ticker")).alias("_h")).drop("_s")
        days = by_keys(["weekday"]).with_columns((pl.col("_s") / pl.col("_s").mean().over("ticker")).alias("_d")).drop("_s")
        raw = (
            cells.join(hours, on=["ticker", "hour"], how="left")
            .join(days, on=["ticker", "weekday"], how="left")
            .with_columns((pl.col("_h") * pl.col("_d")).alias("_s"))
        )
    table = raw.join(counts, on=["ticker", *_SLOT], how="left").with_columns(pl.col("n").fill_null(0)).collect()
    empty = table.filter(pl.col("_s").is_null() | (pl.col("_s") <= 0)).sort("ticker", *_SLOT)
    if empty.height:
        named = ", ".join(f"{r['ticker']} weekday {r['weekday']} hour {r['hour']}" for r in empty.head(5).iter_rows(named=True))
        raise Refused(f"no return in the fit window to estimate {empty.height} cell(s) from: {named}")
    return (
        table.with_columns((pl.col("_s") / (pl.col("_s").pow(2).mean().over("ticker")).sqrt()).alias("factor"))
        .with_columns(pl.from_epoch(pl.lit(hi), time_unit="us").dt.replace_time_zone("UTC").alias("fit_end"))
        .select("ticker", "weekday", "hour", "n", "factor", "fit_end")
        .sort("ticker", "weekday", "hour")
    )


def deseasonalize(returns: pl.LazyFrame | pl.DataFrame, factors: pl.DataFrame) -> pl.DataFrame:
    """The returns with `factor`, their bar's calendar cell's, and `deseasonalized = return / factor`.

    The factor depends only on the calendar, so the same join re-seasonalises a
    forecast for any future bar: σ̂ = factor(cell of the target bar) × σ̂ of the
    deseasonalized model, with no lookahead.
    """
    utils.require(returns, ("ticker", "ts", "return"), "make returns with gr.timeseries.returns")
    out = (
        utils.lazy(returns)
        .with_columns(*_slots())
        .join(factors.lazy().select("ticker", *_SLOT, "factor"), on=["ticker", *_SLOT], how="left")
        .with_columns((pl.col("return") / pl.col("factor")).alias("deseasonalized"))
        .drop(*_SLOT)
        .collect()
    )
    missing = out.filter(pl.col("factor").is_null())
    if missing.height:
        raise Refused(f"no factor for {missing.height} row(s), first {missing['ticker'][0]} at {missing['ts'][0]}; fit factors on these tickers and cells")
    return out


def walk_forward_origins(
    bars: pl.LazyFrame | pl.DataFrame,
    split,
    *,
    window: int | str = "expanding",
    every: int = 1,
) -> pl.DataFrame:
    """The rolling-origin schedule: `ticker, ts, close_ts, origin, refit, fit_from, fitted_through`.

    One row per bar closing at or after `split`: its `close_ts` is an origin,
    from which the next bar onward is forecast (Tashman 2000; Hyndman's
    evaluation on a rolling forecasting origin). Parameters are re-estimated at
    origins 0, k, 2k, …; a refit's fit spans `fit_from` through its own close,
    every bar from the ticker's first (`"expanding"`) or the last `window` bars.
    Between refits a row carries the last refit's span: parameters fixed, the
    filter run forward. `fitted_through ≤ close_ts` on every row. Windows count
    bars, not time, so a hole is bridged (roadmap D5).
    """
    if every < 1:
        raise Refused(f"every={every}: parameters must be re-estimated at least at the first origin")
    rolling = window != "expanding"
    if rolling and (not isinstance(window, int) or window < 2):
        raise Refused(f"window={window!r} must be 'expanding' or a number of bars ≥ 2")
    utils.require(bars, ("ticker", "ts", "close_ts"), "load bars with gr.market.candles")
    at = utils.instant("split", split)
    frame = utils.lazy(bars).select("ticker", "ts", "close_ts").sort("ticker", "ts").collect()
    out = []
    for (ticker,), group in frame.group_by("ticker", maintain_order=True):
        ts, close = group["ts"].to_list(), group["close_ts"].to_list()
        stamps = group["close_ts"].dt.epoch("us").to_list()
        first = next((i for i, s in enumerate(stamps) if s >= at), None)
        if first is None:
            raise Refused(f"{ticker}: the split {split} is after the last bar's close")
        if rolling and window > first + 1:
            raise Refused(f"{ticker}: window={window} bars, but only {first + 1} close by the first origin")
        for i in range(first, len(ts)):
            k = i - first
            r = first + (k // every) * every
            out.append((ticker, ts[i], close[i], k, k % every == 0, ts[r - window + 1] if rolling else ts[0], close[r]))
    schema = {
        "ticker": pl.String,
        "ts": frame.schema["ts"],
        "close_ts": frame.schema["close_ts"],
        "origin": pl.Int64,
        "refit": pl.Boolean,
        "fit_from": frame.schema["ts"],
        "fitted_through": frame.schema["close_ts"],
    }
    return pl.DataFrame(out, schema=schema, orient="row")


# ── Variance breaks ─────────────────────────────────────────────────────────
#
# Unmodelled shifts in the unconditional variance push GARCH persistence
# toward one (Lamoureux and Lastrapes 1990). Breaks are found by cumulative
# sums of squares: Inclán and Tiao's (1994) statistic assumes independent
# returns and over-detects under fat tails and clustering; Sansó, Aragó and
# Carrion's (2004) κ₂ studentises the same sum by a long-run variance of the
# squares. Neither paper could be read here (the CRAN ICSS manual could, and
# implements only the original); both statistics are checked by simulation.

BREAK_STATISTICS = ("kappa2", "inclan_tiao")
_CRITICAL = 1.358  # 5% point of the supremum of a Brownian bridge (Kolmogorov)


def _break_statistic(e2: list[float], statistic: str) -> tuple[float, int]:
    """(statistic, argmax k as a count of observations before the break) on squared demeaned returns."""
    t = len(e2)
    total = sum(e2)
    running, best, where = 0.0, -1.0, 0
    for k in range(1, t):
        running += e2[k - 1]
        gap = abs(running - k / t * total)
        if gap > best:
            best, where = gap, k
    if statistic == "inclan_tiao":
        return (sqrt(t / 2) * best / total if total > 0 else 0.0), where
    mean = total / t
    d = [x - mean for x in e2]

    def acov(j: int) -> float:
        return sum(d[i] * d[i - j] for i in range(j, t)) / t

    # Newey and West's (1994) data-driven Bartlett bandwidth: the fixed
    # ⌊4(T/100)^(2/9)⌋ (6 lags at T = 1,000) understates the long-run variance of
    # squares from a persistent GARCH, and κ₂ then rejected 34% of break-free
    # GARCH(0.1, 0.85)-t₅ series at a nominal 5%.
    pre = int(4 * (t / 100) ** (2 / 9))
    g = [acov(j) for j in range(pre + 1)]
    s0 = g[0] + 2 * sum(g[1:])
    s1 = 2 * sum(j * g[j] for j in range(1, pre + 1))
    band = min(t - 1, int(1.1447 * ((s1 / s0) ** 2) ** (1 / 3) * t ** (1 / 3))) if s0 > 0 else pre
    lrv = g[0]
    for j in range(1, band + 1):
        lrv += 2 * (1 - j / (band + 1)) * (g[j] if j <= pre else acov(j))
    return (best / sqrt(t) / sqrt(lrv) if lrv > 0 else 0.0), where


def _segment(e: list[float], lo: int, hi: int, statistic: str, min_segment: int, found: list):
    seg = e[lo:hi]
    if len(seg) < 2 * min_segment:
        return
    m = sum(seg) / len(seg)
    stat, k = _break_statistic([(x - m) ** 2 for x in seg], statistic)
    if stat > _CRITICAL and min_segment <= k <= len(seg) - min_segment:
        found.append(lo + k)
        _segment(e, lo, lo + k, statistic, min_segment, found)
        _segment(e, lo + k, hi, statistic, min_segment, found)


def variance_breaks(returns: pl.LazyFrame | pl.DataFrame, *, statistic: str = "kappa2", min_segment: int = 60) -> pl.DataFrame:
    """Breaks in one ticker's unconditional variance: `ts, close_ts, statistic`, one row per break.

    Binary segmentation on the cumulative sum of squared demeaned returns
    (nulls dropped), splitting where the statistic exceeds 1.358, then a
    pruning pass that re-tests each break between its neighbours until none
    is dropped. `kappa2` (default) is robust to fat tails and clustering;
    `inclan_tiao` is the original, for comparison. A break's `ts` is the first
    bar of the new regime.
    """
    if statistic not in BREAK_STATISTICS:
        raise Refused(f"statistic={statistic!r} is not one of {', '.join(BREAK_STATISTICS)}")
    utils.require(returns, ("ticker", "ts", "close_ts", "return"), "make returns with gr.timeseries.returns")
    frame = utils.lazy(returns).drop_nulls("return").sort("ts").collect()
    if frame["ticker"].n_unique() != 1:
        raise Refused("find breaks one ticker at a time")
    e = frame["return"].to_list()
    found: list[int] = []
    _segment(e, 0, len(e), statistic, min_segment, found)
    breaks = sorted(found)
    changed = True
    while changed and breaks:
        changed = False
        for i, b in enumerate(breaks):
            lo = breaks[i - 1] if i else 0
            hi = breaks[i + 1] if i + 1 < len(breaks) else len(e)
            seg = e[lo:hi]
            m = sum(seg) / len(seg)
            stat, k = _break_statistic([(x - m) ** 2 for x in seg], statistic)
            if stat <= _CRITICAL:
                breaks.pop(i)
                changed = True
                break
            breaks[i] = lo + k
    rows = []
    for i, b in enumerate(breaks):
        lo = breaks[i - 1] if i else 0
        hi = breaks[i + 1] if i + 1 < len(breaks) else len(e)
        seg = e[lo:hi]
        m = sum(seg) / len(seg)
        rows.append((frame["ts"][b], frame["close_ts"][b], _break_statistic([(x - m) ** 2 for x in seg], statistic)[0]))
    return pl.DataFrame(rows, schema={"ts": frame.schema["ts"], "close_ts": frame.schema["close_ts"], "statistic": pl.Float64}, orient="row")


def segments(returns: pl.LazyFrame | pl.DataFrame, breaks: pl.DataFrame) -> pl.DataFrame:
    """`from, to, n, variance` per segment between breaks (`from` a bar's ts, `to` a bar's close_ts)."""
    frame = utils.lazy(returns).drop_nulls("return").sort("ts").collect()
    cuts = [frame["ts"][0], *breaks["ts"].to_list()]
    rows = []
    for i, start in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else None
        seg = frame.filter((pl.col("ts") >= start) & ((pl.col("ts") < end) if end is not None else pl.lit(True)))
        rows.append((start, seg["close_ts"][-1], seg.height, float(seg["return"].var())))
    return pl.DataFrame(rows, schema={"from": frame.schema["ts"], "to": frame.schema["close_ts"], "n": pl.Int64, "variance": pl.Float64}, orient="row")


def local_clock(frame: pl.LazyFrame | pl.DataFrame, zone: str, prefix: str):
    """The frame with `<prefix>_hour`, `_weekday` (1 = Monday), `_date` and `_dst` on the clock of `zone`.

    Derived from `ts`, never stored in its place (settled point 3). polars
    carries its own time-zone database, so the rules are pinned with polars.
    `_dst` is true while the zone's daylight-saving offset is non-zero: New
    York moves 13:30 → 12:30 UTC for its 09:30 on the second Sunday of March,
    London on the last Sunday, so for two or three weeks each spring and
    autumn the two are four hours apart, not five.
    """
    utils.require(frame, ("ts",), "a frame with a UTC `ts`")
    try:
        pl.Series([0], dtype=pl.Datetime("us", "UTC")).dt.convert_time_zone(zone)
    except Exception:  # noqa: BLE001 -- polars raises its own error types for an unknown zone
        raise Refused(f"zone={zone!r} is not an IANA time zone polars knows, e.g. America/New_York") from None
    local = pl.col("ts").dt.convert_time_zone(zone)
    return frame.with_columns(
        local.dt.hour().alias(f"{prefix}_hour"),
        local.dt.weekday().alias(f"{prefix}_weekday"),
        local.dt.date().alias(f"{prefix}_date"),
        (local.dt.dst_offset() != pl.duration()).alias(f"{prefix}_dst"),
    )


def extremes(bars: pl.LazyFrame | pl.DataFrame, k: int, *, by: str = "range", spacing: str = "3d") -> pl.DataFrame:
    """The k most extreme bars per ticker, largest first, no two within `spacing` of each other.

    `by="range"` scores ln(high/low); `by="return"` scores |ln(close/previous
    close)| over contiguous bars. A bar in a gap is never chosen. Episodes
    are picked by the data, never typed: the largest, then the largest not
    within `spacing` of any already picked, and so on.
    """
    if by not in ("range", "return"):
        raise Refused(f"by={by!r} is not one of range, return")
    utils.require(bars, ("ticker", "ts", "close_ts", "high", "low", "close"), "load bars with gr.market.candles or gr.reference.candles")
    lf = utils.lazy(bars).sort("ticker", "ts")
    names = lf.collect_schema().names()
    if by == "range":
        score = (pl.col("high") / pl.col("low")).log()
    else:
        joined = pl.col("ts") == pl.col("close_ts").shift(1).over("ticker")
        score = pl.when(joined).then((pl.col("close") / pl.col("close").shift(1).over("ticker")).log().abs())
    frame = lf.with_columns(score.alias("score"))
    if "in_gap" in names:
        frame = frame.filter(~pl.col("in_gap").fill_null(False))
    frame = frame.drop_nulls("score").sort(["ticker", "score", "ts"], descending=[False, True, False]).collect()
    width = pl.select(pl.lit("2000-01-01").str.to_datetime().dt.offset_by(spacing) - pl.lit("2000-01-01").str.to_datetime()).item()
    out = []
    for (ticker,), g in frame.group_by("ticker", maintain_order=True):
        picked: list = []
        for row in g.iter_rows(named=True):
            if all(abs(row["ts"] - p["ts"]) >= width for p in picked):
                picked.append(row)
                if len(picked) == k:
                    break
        out.extend(picked)
    return pl.DataFrame(out, schema=frame.schema) if out else frame.head(0)


def _log_adjusted(frame: pl.DataFrame, cols: Sequence[str], seasonal: bool) -> pl.DataFrame:
    """Each column's log, less its hour-of-day mean and weekday effect when `seasonal`, on a complete hourly grid."""
    bad = frame.filter(pl.any_horizontal([pl.col(c) <= 0 for c in cols]))
    if bad.height:
        raise Refused(f"{', '.join(cols)} must be positive: the elasticity is of logs")
    grid = pl.DataFrame({"ts": pl.datetime_range(frame["ts"].min(), frame["ts"].max(), "1h", eager=True, time_zone="UTC")})
    g = grid.join(frame.select("ts", *cols), on="ts", how="left").with_columns(
        *[pl.col(c).log() for c in cols], pl.col("ts").dt.hour().alias("_h"), pl.col("ts").dt.weekday().alias("_d")
    )
    if seasonal:
        g = g.with_columns(
            *[(pl.col(c) - pl.col(c).mean().over("_h") - (pl.col(c).mean().over("_d") - pl.col(c).mean())).alias(c) for c in cols]
        )
    return g.drop("_h", "_d")


def log_elasticity(frame: pl.LazyFrame | pl.DataFrame, y: str, x: str, *, seasonal: bool = True) -> dict:
    """The OLS slope of log y on log x: `beta, se, t, r2, n`.

    With `seasonal`, each log first loses its hour-of-day mean and weekday
    effect, so the slope is about deviations from the day's shape, not the
    shape two series share (volatility up at 14 UTC while depth dips). The
    iid `t` ignores autocorrelation.
    """
    utils.require(frame, ("ts", y, x), "hourly rows with ts and both series")
    g = _log_adjusted(utils.lazy(frame).collect(), [y, x], seasonal).drop_nulls([y, x])
    n = g.height
    if n < 3:
        raise Refused(f"{n} hours with both series are too few")
    vx, vy = g[x].var(), g[y].var()
    c = g.select(pl.cov(x, y)).item()
    beta = c / vx
    r2 = c * c / (vx * vy)
    se = math.sqrt((1 - r2) * vy * (n - 1) / (n - 2) / (vx * (n - 1)))
    return {"beta": beta, "se": se, "t": beta / se if se else None, "r2": r2, "n": n}


def lagged_correlation(frame: pl.LazyFrame | pl.DataFrame, a: str, b: str, lags: Sequence[int], *, seasonal: bool = True) -> pl.DataFrame:
    """`lag, corr`: log a at t against log b at t + lag hours. A positive lag with the larger corr means a moves first."""
    utils.require(frame, ("ts", a, b), "hourly rows with ts and both series")
    g = _log_adjusted(utils.lazy(frame).collect(), [a, b], seasonal)
    rows = [{"lag": lag, "corr": g.select(pl.corr(pl.col(a), pl.col(b).shift(-lag))).item()} for lag in lags]
    return pl.DataFrame(rows, schema={"lag": pl.Int64, "corr": pl.Float64})
