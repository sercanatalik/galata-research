"""Technical indicators as polars expressions, evaluated per contiguous stretch of bars so none spans a hole.

    framed = gr.indicators.add(bars, fast=gr.indicators.ema(12), slow=gr.indicators.ema(26), rsi=gr.indicators.rsi(14))
    studies.trial(framed, (pl.col("fast") - pl.col("slow")).sign(), "ema 12/26")

Each indicator is written for **one contiguous series** and is evaluated by
`add`, which numbers the stretches between holes per ticker and runs every
expression over `(ticker, stretch)`. After a hole an indicator starts again
and warms up again: an average that reached across the hole would mix bars
the venue never showed in sequence, and a backtest that trades on it reads a
value no bar ever produced. Before a full window every figure is null, never
a shorter window.

Every value at a bar uses that bar and earlier ones only, so it is known at
the bar's `close_ts`; a test truncates the bars and checks that no earlier
value moves. The smoothed indicators follow their authors' seeding (Wilder
1978 for RSI and ATR; the EMA seeded with the simple mean of its first n
values, as StockCharts does), and RSI is pinned to StockCharts'
worked example. They are polars-native and need no numpy.

Indicators are the trading side's standard vocabulary. A family of them is a
family of trials: each parameter set counts toward N for the Deflated Sharpe
Ratio.
"""

import polars as pl

from . import timeseries, utils
from ._errors import Refused

_STRETCH = "_stretch"


def add(bars: pl.LazyFrame | pl.DataFrame, **indicators: pl.Expr) -> pl.DataFrame:
    """`bars` sorted by ticker and time, plus one column per named indicator, each evaluated per contiguous stretch.

    `bars` are candles from `gr.market.candles` (or any frame with `ticker,
    ts, close_ts` and the columns the indicators read). A row whose previous
    row is not the bar just before it starts a new stretch.
    """
    utils.require(bars, ("ticker", "ts", "close_ts"), "load bars with gr.market.candles")
    if not indicators:
        return utils.lazy(bars).sort("ticker", "ts").collect()
    clash = sorted(set(indicators) & set(utils.lazy(bars).collect_schema().names()))
    if clash:
        raise Refused(f"the bars already have {', '.join(clash)}; name the indicators otherwise")
    return (
        utils.lazy(bars)
        .sort("ticker", "ts")
        .with_columns((~timeseries.contiguous()).fill_null(True).cum_sum().over("ticker").alias(_STRETCH))
        .with_columns(*[expr.over("ticker", _STRETCH).alias(name) for name, expr in indicators.items()])
        .drop(_STRETCH)
        .collect()
    )


def _positive(n: int, name: str = "n") -> None:
    if not isinstance(n, int) or n < 1:
        raise Refused(f"{name}={n!r} must be a positive whole number of bars")


def _seeded(x: pl.Expr, n: int, alpha: float) -> pl.Expr:
    """An exponential average with smoothing `alpha`, seeded with the simple mean of x's first n values.

    Null until x has n values. The recursion is y = alpha·x + (1 − alpha)·y₋₁.
    """
    count = x.is_not_null().cum_sum()
    start = pl.when((count == n) & x.is_not_null()).then(x.rolling_mean(n)).when(count > n).then(x)
    return start.ewm_mean(alpha=alpha, adjust=False, min_samples=1)


def sma(n: int, column: str = "close") -> pl.Expr:
    """The simple mean of the last `n` values of `column`."""
    _positive(n)
    return pl.col(column).rolling_mean(n)


def ema(n: int, column: str = "close") -> pl.Expr:
    """The exponential mean of `column` with span `n` (alpha 2/(n+1)), seeded with the mean of the first n."""
    _positive(n)
    return _seeded(pl.col(column), n, 2 / (n + 1))


def rsi(n: int = 14, column: str = "close") -> pl.Expr:
    """Wilder's Relative Strength Index, 0–100: average gain over average loss, each Wilder-smoothed (alpha 1/n).

    The first average is the simple mean of the first n changes, so the
    first figure is at the (n+1)-th bar. A stretch with no loss reads 100;
    with neither gain nor loss, null.
    """
    _positive(n)
    change = pl.col(column).diff()
    gain = _seeded(change.clip(lower_bound=0.0), n, 1 / n)
    loss = _seeded((-change).clip(lower_bound=0.0), n, 1 / n)
    return pl.when(loss > 0).then(100 - 100 / (1 + gain / loss)).when(gain > 0).then(100.0)


def macd(fast: int = 12, slow: int = 26, column: str = "close") -> pl.Expr:
    """Appel's MACD line: the fast EMA less the slow one."""
    if fast >= slow:
        raise Refused(f"fast={fast} must be shorter than slow={slow}")
    return ema(fast, column) - ema(slow, column)


def macd_signal(fast: int = 12, slow: int = 26, signal: int = 9, column: str = "close") -> pl.Expr:
    """The `signal`-span EMA of the MACD line, seeded with the mean of its first `signal` values."""
    _positive(signal, "signal")
    return _seeded(macd(fast, slow, column), signal, 2 / (signal + 1))


def zscore(n: int = 20, column: str = "close") -> pl.Expr:
    """(x − mean) / standard deviation over the last `n` values, population (ddof 0), as Bollinger's bands use.

    A close above the upper Bollinger band of width k is a z-score above k.
    A window with no variance is null.
    """
    _positive(n)
    x = pl.col(column)
    sd = x.rolling_std(n, ddof=0)
    return pl.when(sd > 0).then((x - x.rolling_mean(n)) / sd)


def bollinger(n: int = 20, k: float = 2.0, band: str = "upper", column: str = "close") -> pl.Expr:
    """Bollinger's band: the n-bar mean, plus (`upper`) or minus (`lower`) k population standard deviations, or the `mid`."""
    _positive(n)
    mid = pl.col(column).rolling_mean(n)
    width = k * pl.col(column).rolling_std(n, ddof=0)
    bands = {"upper": mid + width, "mid": mid, "lower": mid - width}
    if band not in bands:
        raise Refused(f"band={band!r} is not one of {', '.join(bands)}")
    return bands[band]


def true_range() -> pl.Expr:
    """The largest of high − low and the distances from the previous close; high − low on a stretch's first bar."""
    prev = pl.col("close").shift(1)
    return pl.max_horizontal(pl.col("high") - pl.col("low"), (pl.col("high") - prev).abs(), (pl.col("low") - prev).abs())


def atr(n: int = 14) -> pl.Expr:
    """Wilder's Average True Range: the true range Wilder-smoothed (alpha 1/n), seeded with the mean of the first n."""
    _positive(n)
    return _seeded(true_range(), n, 1 / n)


def stochastic(k: int = 14, d: int = 3) -> pl.Expr:
    """Lane's slow %D: the `d`-bar mean of %K = 100·(close − lowest low) / (highest high − lowest low) over `k` bars."""
    _positive(k, "k")
    _positive(d, "d")
    low, high = pl.col("low").rolling_min(k), pl.col("high").rolling_max(k)
    percent_k = pl.when(high > low).then(100 * (pl.col("close") - low) / (high - low))
    return percent_k.rolling_mean(d)


def supertrend(n: int = 10, multiplier: float = 3.0) -> pl.Expr:
    """The Supertrend's direction, +1 up or −1 down, on ATR(n) bands `multiplier` wide about (high + low) / 2.

    The bands ratchet: the lower band only rises while the close stays above
    it, the upper only falls while the close stays below it. The direction
    turns up when the close crosses the upper band and down when it crosses
    the lower. On the first bar with an ATR the direction is up if the close
    is at or above the midpoint. Null before the ATR exists.
    """
    _positive(n)
    if multiplier <= 0:
        raise Refused(f"multiplier={multiplier} must be positive")
    fields = pl.struct(pl.col("high"), pl.col("low"), pl.col("close"), atr(n).alias("atr"))
    return fields.map_batches(lambda s: _supertrend(s, multiplier), return_dtype=pl.Float64)


def _supertrend(s: pl.Series, multiplier: float) -> pl.Series:
    out: list[float | None] = []
    upper = lower = direction = prev_close = None
    for row in s.to_list():
        high, low, close, a = row["high"], row["low"], row["close"], row["atr"]
        if a is None or None in (high, low, close):
            out.append(None)
            prev_close = close
            continue
        mid = (high + low) / 2
        up_band, low_band = mid + multiplier * a, mid - multiplier * a
        if direction is None:
            upper, lower = up_band, low_band
            direction = 1.0 if close >= mid else -1.0
        else:
            upper = up_band if up_band < upper or prev_close > upper else upper
            lower = low_band if low_band > lower or prev_close < lower else lower
            if direction < 0 and close > upper:
                direction = 1.0
            elif direction > 0 and close < lower:
                direction = -1.0
        out.append(direction)
        prev_close = close
    return pl.Series(out, dtype=pl.Float64)


def hold(enter_long: pl.Expr, exit_long: pl.Expr, enter_short: pl.Expr | None = None, exit_short: pl.Expr | None = None) -> pl.Expr:
    """A position that holds between events: +1 from `enter_long` until `exit_long`, −1 from `enter_short` until `exit_short`.

    Each condition is a boolean expression read at a bar's close. An exit
    applies only to the side held, and an entry wins over an exit on the
    same bar, so an opposite entry flips. Null until any condition has a
    value (a warm-up), so a trial never trades before its indicator exists;
    flat from then until the first entry. Evaluate it through `add`, so a
    position never carries across a hole.
    """
    no = pl.lit(None, pl.Boolean)
    fields = pl.struct(
        enter_long.alias("el"),
        exit_long.alias("xl"),
        (no if enter_short is None else enter_short).alias("es"),
        (no if exit_short is None else exit_short).alias("xs"),
    )
    return fields.map_batches(_hold, return_dtype=pl.Float64)


def _hold(s: pl.Series) -> pl.Series:
    out: list[float | None] = []
    position: float | None = None
    for row in s.to_list():
        if position is None:
            if all(v is None for v in row.values()):
                out.append(None)
                continue
            position = 0.0
        if row["el"]:
            position = 1.0
        elif row["es"]:
            position = -1.0
        elif position > 0 and row["xl"] or position < 0 and row["xs"]:
            position = 0.0
        out.append(position)
    return pl.Series(out, dtype=pl.Float64)
