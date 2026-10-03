"""Stop-loss, take-profit and cooldown laid over a trial's positions, filled inside the bar.

    stopped = gr.overlays.apply(bars, studies.trial(bars, rule, "rule"), stop=0.10, trailing=True)
    every = gr.overlays.grid(bars, base, stops=[0.05, 0.1], takes=[None, 0.25])

A trial frame is close to close: a position decided at a close earns the next
bar's close. A stop is not. It fills **inside** a bar, at its level, or at
the open when the bar opens beyond it. So an overlay is a re-run of the
trial against the bars' open, high and low, and its frame says on which bar
and why each exit happened (`exit`, `exit_price`).

**No level uses the bar it is checked on.** A trade's entry price is the
close that decided it, a trailing stop trails the highest high (lowest low,
for a short) from the entry through the *previous* bar, and the bar that
decided the position is never checked. A bar's path between its high and its
low is unknown, so when one bar touches both the stop and the take-profit,
the stop is taken. That is the costlier assumption, stated.

After an exit the position is flat for `cooldown` bars, then follows the
base trial's again: re-entry is charged like any entry. A base that is
null (a warm-up) stays null. A bar after a hole earns nothing and is not
checked, as in `gr.backtest.returns`; the trade's levels carry over it.

Each overlay variant is a trial. A stop chosen from a grid is chosen, and
counts toward N for the Deflated Sharpe Ratio (Kaminski and Lo 2014 on when
a stop can help at all: only when returns have momentum).
"""

from collections.abc import Iterable

import polars as pl

from . import backtest, utils
from ._errors import Refused

COLUMNS = ["trial", "ticker", "ts", "close_ts", "position", "bar_return", "gross", "cost", "net"]


def _pct(x: float) -> str:
    return f"{x * 100:g}%"


def label(stop: float | None, take: float | None, trailing: bool, cooldown: int) -> str:
    """The overlay's part of a trial name: `stop 10% trail, take 25%, cool 10`."""
    parts = []
    if stop is not None:
        parts.append(f"stop {_pct(stop)}{' trail' if trailing else ''}")
    if take is not None:
        parts.append(f"take {_pct(take)}")
    parts.append(f"cool {cooldown}")
    return ", ".join(parts)


def apply(
    bars: pl.LazyFrame | pl.DataFrame,
    trial: pl.DataFrame,
    *,
    stop: float | None = None,
    take: float | None = None,
    trailing: bool = False,
    cooldown: int = 0,
    fee: float | None = None,
    name: str | None = None,
) -> pl.DataFrame:
    """`trial` re-run with a stop (a fraction of the entry, or trailing) and a take-profit, filled inside the bar.

    `bars` are the candles the trial ran on (`ticker, ts, open, high, low,
    close`). Returns the trial's columns plus `exit` (`stop`, `take` or null)
    and `exit_price`. Without a stop or a take-profit it reproduces the trial.
    Its trials are renamed `"<trial> + <label>"` unless `name` is given.
    `fee` defaults to the one the trial was charged, read from its costs, so
    an overlay is never cheaper or dearer to trade than its base.
    """
    for value, what in ((stop, "stop"), (take, "take")):
        if value is not None and not 0 < value < 1:
            raise Refused(f"{what}={value} must be a fraction between 0 and 1")
    if trailing and stop is None:
        raise Refused("trailing=True needs a stop")
    if cooldown < 0:
        raise Refused(f"cooldown={cooldown} must not be negative")
    utils.require(trial, ("trial", "ticker", "ts", "close_ts", "position", "bar_return", "cost"), "pass a trial frame from gr.studies")
    if fee is None:
        fee = charged_fee(trial)
    if fee < 0:
        raise Refused(f"fee={fee} is negative")
    utils.require(bars, ("ticker", "ts", "open", "high", "low", "close"), "load bars with gr.market.candles")
    prices = utils.lazy(bars).select("ticker", "ts", pl.col("open", "high", "low", "close").cast(pl.Float64)).collect()
    joined = (
        trial.select("trial", "ticker", "ts", "close_ts", "position", "bar_return")
        .join(prices, on=["ticker", "ts"], how="left")
        .sort("trial", "ticker", "ts")
    )
    tag = label(stop, take, trailing, cooldown)
    out = []
    for (base, _), part in joined.partition_by("trial", "ticker", as_dict=True, maintain_order=True).items():
        rows = _run(part, stop, take, trailing, cooldown, fee)
        out.append(part.select("ticker", "ts", "close_ts", "bar_return").with_columns(
            pl.lit(name or f"{base} + {tag}").alias("trial"),
            *(pl.Series(k, v, dtype=pl.String if k == "exit" else pl.Float64) for k, v in rows.items()),
        ))  # fmt: skip
    if not out:
        return pl.DataFrame(schema={c: trial.schema.get(c, pl.Float64) for c in COLUMNS} | {"exit": pl.String, "exit_price": pl.Float64})
    return pl.concat(out).select(*COLUMNS, "exit", "exit_price")


def charged_fee(trial: pl.DataFrame) -> float:
    """The fee per unit of turnover `trial` was charged: its cost over its turnover, the same on every row that traded.

    A trial that never traded was charged nothing to read, and gets the taker fee.
    """
    keys = ("trial", "ticker")
    turn = (pl.col("position").fill_null(0.0) - pl.col("position").shift(1).over(keys).fill_null(0.0)).abs()
    rates = (
        trial.sort(*keys, "ts")
        .with_columns(turn.alias("_turn"))
        .filter(pl.col("cost").is_not_null() & (pl.col("_turn") > 1e-12))
        .select((pl.col("cost") / pl.col("_turn")).alias("rate"))["rate"]
    )
    if rates.is_empty():
        return backtest.TAKER_FEE
    lo, hi = rates.min(), rates.max()
    if hi - lo > 1e-12 * max(1.0, abs(hi)):
        raise Refused(f"the trial was charged between {lo:g} and {hi:g} per unit of turnover; pass fee= explicitly")
    return float(hi)


def _run(part: pl.DataFrame, stop, take, trailing, cooldown, fee) -> dict[str, list]:
    """The overlay over one (trial, ticker), bar by bar. `position` is the held position the base wanted."""
    cols = {k: [] for k in ("position", "gross", "cost", "net", "exit", "exit_price")}
    held = 0.0  # what was held into this bar, after any exit in the bar before
    entry = extreme = None
    blocked = 0
    prev_close = None
    for r in part.iter_rows(named=True):
        want, ret = r["position"], r["bar_return"]
        if want is None:
            for k in cols:
                cols[k].append(None)
            held, entry, extreme, prev_close = 0.0, None, None, r["close"]
            continue
        if blocked:
            want, blocked = 0.0, blocked - 1
        side = (want > 0) - (want < 0)
        if side and (side != (held > 0) - (held < 0)):
            entry = extreme = prev_close
        live = ret is not None
        cost = abs(want - held) * fee if live else None
        gross, why, fill = (want * ret if live else None), None, None
        if live and side and entry is not None and (stop is not None or take is not None):
            why, fill = _touch(side, entry, extreme, r, stop, take, trailing)
            if why:
                gross = want * (fill / prev_close - 1)
                cost += abs(want) * fee
        cols["position"].append(want)
        cols["gross"].append(gross)
        cols["cost"].append(cost)
        cols["net"].append(None if gross is None else gross - cost)
        cols["exit"].append(why)
        cols["exit_price"].append(fill)
        if why:
            held, entry, extreme, blocked = 0.0, None, None, cooldown
        else:
            held = want
            if side and extreme is not None and r["high"] is not None:
                extreme = max(extreme, r["high"]) if side > 0 else min(extreme, r["low"])
        prev_close = r["close"]
    return cols


def _touch(side: int, entry: float, extreme: float, bar: dict, stop, take, trailing) -> tuple[str | None, float | None]:
    """Which level this bar reached first, by the costlier rule, and at what price."""
    o, h, low = bar["open"], bar["high"], bar["low"]
    if None in (o, h, low):
        return None, None
    anchor = extreme if trailing else entry
    if side > 0:
        s = None if stop is None else anchor * (1 - stop)
        t = None if take is None else entry * (1 + take)
        if s is not None and o <= s:
            return "stop", o
        if s is not None and low <= s:
            return "stop", s
        if t is not None and o >= t:
            return "take", o
        if t is not None and h >= t:
            return "take", t
    else:
        s = None if stop is None else anchor * (1 + stop)
        t = None if take is None else entry * (1 - take)
        if s is not None and o >= s:
            return "stop", o
        if s is not None and h >= s:
            return "stop", s
        if t is not None and o <= t:
            return "take", o
        if t is not None and low <= t:
            return "take", t
    return None, None


def grid(
    bars: pl.LazyFrame | pl.DataFrame,
    trial: pl.DataFrame,
    *,
    stops: Iterable[float | None] = (0.05, 0.10, 0.20),
    trailing: Iterable[bool] = (False, True),
    takes: Iterable[float | None] = (None, 0.25),
    cooldown: int = 10,
    fee: float | None = None,
) -> pl.DataFrame:
    """Every overlay in the grid over `trial`, each its own trial: the registered 12 by default.

    A `None` stop pairs with `trailing=False` only. The base itself is not
    included: concatenate it to compare against it.
    """
    frames = []
    for s in stops:
        for tr in trailing:
            if s is None and tr:
                continue
            for t in takes:
                if s is None and t is None:
                    continue
                frames.append(apply(bars, trial, stop=s, take=t, trailing=tr, cooldown=cooldown, fee=fee))
    if not frames:
        raise Refused("the grid holds no overlay: give a stop or a take-profit")
    return pl.concat(frames)
