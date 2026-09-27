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
