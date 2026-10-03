"""A trial's positions as trades: each one's return, costs, length and excursions, and what they add up to.

    t = gr.trades.table(frame, bars)            # one row per trade
    gr.trades.summary(frame, bars, periods_per_year=365)   # one row per (trial, ticker)

A **trade** is a maximal stretch of bars held on one side, read from the
`position` column of a trial frame (`gr.studies.trial`, `gr.backtest.returns`):
that column is the position each bar was held with. A resize keeps the trade
(a scale-in is not a new trade), a flip from long to short ends one and
starts another, and flat ends it. Its first bar is the one after the close
that decided it, so `entry_ts` is that close and `entry_price` its price.

**Costs land on the trade that caused them.** `backtest.returns` charges
turnover on the bar it happens, so the cost of leaving a trade sits on the
next row: a flat bar, or the first bar of the opposite trade. Here a flat
bar's cost goes wholly to the trade it closed, and a flip's is split in
proportion, |old| to the old trade and |new| to the new one, so the costs
of all closed trades sum to the frame's.

A trade still held on the last row is `open`: it has no exit cost and its
return is unrealized, so the win rates count closed trades only. A trade
that spans a hole (a held bar with no return) is `spans_gap`: the hole earned
nothing, as in the backtest, and is not filled.

MAE and MFE (Sweeney 1997) are the price's worst and best excursion from
`entry_price` over the held bars' lows and highs, per unit of price, not
scaled by the position: MAE ≤ 0 ≤ MFE. They need the bars.

These describe trades. Whether a trial's edge survives selection is the
Deflated Sharpe Ratio's question, asked of every trial (`gr.stats`).
"""

from statistics import median

import polars as pl

from . import utils

_KEYS = ("trial", "ticker")
_SIDES = {1: "long", -1: "short"}

TABLE = {
    "trial": pl.String,
    "ticker": pl.String,
    "side": pl.String,
    "entry_ts": pl.Datetime("us", "UTC"),
    "exit_ts": pl.Datetime("us", "UTC"),
    "bars": pl.Int64,
    "gross": pl.Float64,
    "cost": pl.Float64,
    "funding": pl.Float64,
    "net": pl.Float64,
    "open": pl.Boolean,
    "spans_gap": pl.Boolean,
}
_EXCURSIONS = {"entry_price": pl.Float64, "mae": pl.Float64, "mfe": pl.Float64}


def table(frame: pl.DataFrame, bars: pl.LazyFrame | pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per trade: `trial, ticker, side, entry_ts, exit_ts, bars, gross, cost, funding, net, open, spans_gap`.

    `gross` is ∏(1 + gross) − 1 over the held bars. `cost` is the turnover
    the trade caused, its exit included, and `funding` the funding it paid
    (null where none was charged). `net` compounds each bar's gross less its
    share of cost and funding. With `bars` (`ticker, ts, close_ts, high, low,
    close`, from `gr.market.candles`) it adds `entry_price`, `mae` and `mfe`.
    """
    utils.require(frame, (*_KEYS, "ts", "close_ts", "position", "bar_return", "gross", "cost"), "pass a trial frame from gr.studies")
    schema = dict(TABLE) | (_EXCURSIONS if bars is not None else {})
    lf = frame.lazy().sort(*_KEYS, "ts")
    if "funding" not in frame.columns:
        lf = lf.with_columns(pl.lit(None, pl.Float64).alias("funding"))
    side = pl.col("position").fill_null(0.0).sign().cast(pl.Int8)
    rows = (
        lf.with_columns(side.alias("side"))
        .with_columns(
            pl.col("side").shift(1).over(_KEYS).fill_null(0).alias("_prev_side"),
            pl.col("position").shift(1).over(_KEYS).fill_null(0.0).abs().alias("_prev_size"),
            pl.col("position").fill_null(0.0).abs().alias("_size"),
        )
        .with_columns(
            # The share of this row's cost that closed the trade before it.
            pl.when(pl.col("_prev_side") == 0)
            .then(0.0)
            .when(pl.col("side") == 0)
            .then(1.0)
            .when(pl.col("side") != pl.col("_prev_side"))
            .then(pl.col("_prev_size") / (pl.col("_prev_size") + pl.col("_size")))
            .otherwise(0.0)
            .alias("_exit_share"),
            (pl.col("side") != pl.col("_prev_side")).cum_sum().over(_KEYS).alias("_run"),
            (pl.int_range(pl.len()) == pl.len() - 1).over(_KEYS).alias("_last"),
        )
        .with_columns((pl.col("cost").fill_null(0.0) * pl.col("_exit_share")).alias("_exit_cost"))
        .with_columns(
            (pl.col("cost").fill_null(0.0) - pl.col("_exit_cost")).alias("_own_cost"),
            pl.col("_exit_cost").shift(-1).over(_KEYS).fill_null(0.0).alias("_next_exit_cost"),
        )
        .with_columns(
            (
                pl.col("gross").fill_null(0.0)
                - pl.col("_own_cost")
                - pl.col("_next_exit_cost")
                - pl.col("funding").fill_null(0.0)
            ).alias("_net")
        )
        .filter(pl.col("side") != 0)
    )
    out = (
        rows.group_by(*_KEYS, "_run", maintain_order=True)
        .agg(
            pl.col("side").first(),
            pl.col("ts").first().alias("entry_ts"),
            pl.col("close_ts").last().alias("exit_ts"),
            pl.len().cast(pl.Int64).alias("bars"),
            ((pl.col("gross").fill_null(0.0) + 1).product() - 1).alias("gross"),
            (pl.col("_own_cost").sum() + pl.col("_next_exit_cost").sum()).alias("cost"),
            pl.when(pl.col("funding").is_not_null().any()).then(pl.col("funding").sum()).alias("funding"),
            ((pl.col("_net") + 1).product() - 1).alias("net"),
            pl.col("_last").last().alias("open"),
            pl.col("bar_return").is_null().any().alias("spans_gap"),
        )
        .with_columns(pl.col("side").replace_strict(_SIDES, return_dtype=pl.String))
    )
    if bars is not None:
        out = _excursions(out, rows, bars)
    return out.select(list(schema)).collect().cast(schema)


def _excursions(trades: pl.LazyFrame, rows: pl.LazyFrame, bars) -> pl.LazyFrame:
    """`entry_price` (the deciding bar's close), and MAE and MFE over the held bars' lows and highs."""
    utils.require(bars, ("ticker", "ts", "close_ts", "high", "low", "close"), "load bars with gr.market.candles")
    b = utils.lazy(bars).select("ticker", "ts", "close_ts", pl.col("high", "low", "close").cast(pl.Float64))
    reach = (
        rows.select(*_KEYS, "_run", "ts")
        .join(b.select("ticker", "ts", "high", "low"), on=["ticker", "ts"], how="left")
        .group_by(*_KEYS, "_run")
        .agg(pl.col("high").max().alias("_high"), pl.col("low").min().alias("_low"))
    )
    entry = b.select("ticker", pl.col("close_ts").alias("entry_ts"), pl.col("close").alias("entry_price"))
    long = pl.col("side") == "long"
    up = pl.col("_high") / pl.col("entry_price") - 1
    down = pl.col("_low") / pl.col("entry_price") - 1
    return (
        trades.join(reach, on=[*_KEYS, "_run"], how="left")
        .join(entry, on=["ticker", "entry_ts"], how="left")
        .with_columns(
            pl.min_horizontal(pl.when(long).then(down).otherwise(-up), 0.0).alias("mae"),
            pl.max_horizontal(pl.when(long).then(up).otherwise(-down), 0.0).alias("mfe"),
        )
        .sort(*_KEYS, "entry_ts")
    )


SUMMARY = {
    "trial": pl.String,
    "ticker": pl.String,
    "trades": pl.Int64,
    "closed": pl.Int64,
    "long": pl.Int64,
    "short": pl.Int64,
    "win_rate": pl.Float64,
    "long_win_rate": pl.Float64,
    "short_win_rate": pl.Float64,
    "profit_factor": pl.Float64,
    "win_loss": pl.Float64,
    "mean": pl.Float64,
    "median": pl.Float64,
    "best": pl.Float64,
    "worst": pl.Float64,
    "mean_bars": pl.Float64,
    "mean_win_bars": pl.Float64,
    "mean_loss_bars": pl.Float64,
    "max_wins": pl.Int64,
    "max_losses": pl.Int64,
    "exposure": pl.Float64,
    "trades_per_year": pl.Float64,
    "mean_mae": pl.Float64,
    "mean_mfe": pl.Float64,
}


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _streaks(wins: list[bool]) -> tuple[int, int]:
    """The longest run of wins and of losses, in order."""
    best = {True: 0, False: 0}
    run, last = 0, None
    for w in wins:
        run = run + 1 if w == last else 1
        last = w
        best[w] = max(best[w], run)
    return best[True], best[False]


def summary(
    frame: pl.DataFrame,
    bars: pl.LazyFrame | pl.DataFrame | None = None,
    *,
    periods_per_year: float | None = None,
) -> pl.DataFrame:
    """One row per `(trial, ticker)` of `frame`, a trial with no trade included.

    Over closed trades: `win_rate` (net > 0; a zero is not a win),
    `profit_factor` (Σ winning nets over |Σ losing nets|), `win_loss` (mean
    win over |mean loss|), mean and median net, best and worst, mean length
    in bars (all, wins, losses), and the longest streaks of wins and
    losses. Over every row with a return: `exposure`, the share of them held,
    and `trades_per_year` when `periods_per_year` is given. With `bars`, the
    mean MAE and MFE. A ratio with no denominator is null, not infinite.
    """
    utils.require(frame, (*_KEYS, "position", "bar_return"), "pass a trial frame from gr.studies")
    trades = table(frame, bars)
    by = trades.partition_by(*_KEYS, as_dict=True, maintain_order=True)
    rows = []
    for (trial, ticker), part in frame.partition_by(*_KEYS, as_dict=True, maintain_order=True).items():
        t = by.get((trial, ticker), trades.clear())
        closed = t.filter(~pl.col("open")).sort("entry_ts")
        nets = closed["net"].to_list()
        lengths = closed["bars"].to_list()
        wins = [x for x in nets if x > 0]
        losses = [x for x in nets if x < 0]
        max_wins, max_losses = _streaks([x > 0 for x in nets])
        live = part.filter(pl.col("bar_return").is_not_null())
        held = live.filter(pl.col("position").fill_null(0.0) != 0).height
        side_rate = {}
        for s in _SIDES.values():
            n = closed.filter(pl.col("side") == s)["net"].to_list()
            side_rate[s] = sum(x > 0 for x in n) / len(n) if n else None
        rows.append(
            {
                "trial": trial,
                "ticker": ticker,
                "trades": t.height,
                "closed": len(nets),
                "long": t.filter(pl.col("side") == "long").height,
                "short": t.filter(pl.col("side") == "short").height,
                "win_rate": len(wins) / len(nets) if nets else None,
                "long_win_rate": side_rate["long"],
                "short_win_rate": side_rate["short"],
                "profit_factor": sum(wins) / -sum(losses) if losses else None,
                "win_loss": _mean(wins) / -_mean(losses) if wins and losses else None,
                "mean": _mean(nets),
                "median": median(nets) if nets else None,
                "best": max(nets) if nets else None,
                "worst": min(nets) if nets else None,
                "mean_bars": _mean(lengths),
                "mean_win_bars": _mean([n for n, x in zip(lengths, nets) if x > 0]),
                "mean_loss_bars": _mean([n for n, x in zip(lengths, nets) if x < 0]),
                "max_wins": max_wins,
                "max_losses": max_losses,
                "exposure": held / live.height if live.height else None,
                "trades_per_year": t.height / (live.height / periods_per_year) if periods_per_year and live.height else None,
                "mean_mae": _mean(closed["mae"].drop_nulls().to_list()) if bars is not None else None,
                "mean_mfe": _mean(closed["mfe"].drop_nulls().to_list()) if bars is not None else None,
            }
        )
    return pl.DataFrame(rows, schema=SUMMARY)
