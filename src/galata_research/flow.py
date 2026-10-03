"""Order flow as a signal: does it predict, and does a taker who trades on it get paid?

    grid = gr.flow.seconds(book, leader)              # Bybit's touch and Binance's last price, per second
    gr.flow.slope(x, y, days)                         # OLS slope, Newey–West t, within days
    gr.flow.diebold_mariano(d, days, lags)            # does one forecast beat another
    gr.flow.taker(grid, signals, horizon)             # each signal traded at the touch, net of fees

Statistical predictability and a taker's profit are different questions, and
the second is the one that pays (`planning/preregistered/trade-the-flow.md`).
A taker buys at the ask and sells at the bid, so the spread is paid twice,
and pays the fee both ways. Here the decision at second t fills at second t's
own touch. That flatters every rule, since a real order arrives later, so a
rule that loses here loses in practice.

**Days are stretches.** No window, lag or trade spans midnight: the archive
replays each day's book from its own snapshot, so the seconds either side of
it come from different replays.

**The leader's price** at a second is its last trade at or before that second,
and is missing when that trade is more than `stale` old. A book row at second
s is the book after every message stamped at or before s (`gr.reference.book`).
Both are what was known at s.
"""

from math import exp, lgamma, log, sqrt
from statistics import NormalDist

import polars as pl

from . import utils
from ._errors import Refused

BYBIT_TAKER = 0.00055
_N = NormalDist()


def seconds(
    book: pl.LazyFrame | pl.DataFrame,
    leader: pl.LazyFrame | pl.DataFrame | None = None,
    *,
    stale: str = "5s",
) -> pl.DataFrame:
    """One row per (ticker, second) of `book`: the touch, sizes, mid, weighted mid, `day`, and the leader's last price.

    `book` is `gr.reference.book` (one row per second); `leader` is trades
    from another venue (`ticker, ts, price`), such as Binance's. A crossed
    or locked second is dropped. `wmid` is the size-weighted mid,
    (bid·ask_sz + ask·bid_sz)/(bid_sz + ask_sz): toward the side with less
    size, where the price is likelier to go.
    """
    utils.require(book, ("ticker", "ts", "bid_px", "ask_px", "bid_sz", "ask_sz"), "load the book with gr.reference.book")
    grid = (
        utils.lazy(book)
        .select("ticker", "ts", "bid_px", "ask_px", "bid_sz", "ask_sz")
        .filter(pl.col("ask_px") > pl.col("bid_px"))
        .with_columns(
            pl.col("ts").dt.truncate("1d").alias("day"),
            ((pl.col("bid_px") + pl.col("ask_px")) / 2).alias("mid"),
            (
                (pl.col("bid_px") * pl.col("ask_sz") + pl.col("ask_px") * pl.col("bid_sz"))
                / (pl.col("bid_sz") + pl.col("ask_sz"))
            ).alias("wmid"),
        )
        .sort("ticker", "ts")
        .collect()
    )
    if leader is None:
        return grid
    utils.require(leader, ("ticker", "ts", "price"), "load the leader's trades with gr.reference.trades")
    last = (
        utils.lazy(leader)
        .select("ticker", pl.col("ts").alias("_trade_ts"), pl.col("price").cast(pl.Float64).alias("leader_px"))
        .sort("ticker", "_trade_ts")
        .unique(["ticker", "_trade_ts"], keep="last", maintain_order=True)
        .collect()
    )
    return (
        grid.join_asof(last, left_on="ts", right_on="_trade_ts", by="ticker", strategy="backward", tolerance=stale, check_sortedness=False)
        .drop("_trade_ts")
        .sort("ticker", "ts")
    )


def _nw_sum(z: pl.DataFrame, lags: int) -> float:
    """Newey–West's long-run sum of `z` (already demeaned) over rows within each `day`: Σz² + 2Σ_l (1 − l/(L+1)) Σ zᵢzᵢ₋ₗ."""
    total = float((z["z"] ** 2).sum())
    for lag in range(1, lags + 1):
        cross = z.select((pl.col("z") * pl.col("z").shift(lag).over("day")).sum()).item() or 0.0
        total += 2 * (1 - lag / (lags + 1)) * float(cross)
    return total


def slope(frame: pl.DataFrame, x: str, y: str, *, lags: int = 6) -> dict:
    """The OLS slope of `y` on `x` (with an intercept) over rows with both, Newey–West t within each `day`.

    `frame` holds `day`, `x`, `y` in time order within each day. Returns
    `n, slope, t, p` (p one-sided, against a slope ≤ 0), and `r2`.
    """
    utils.require(frame, ("day", x, y), "give each row its day")
    f = frame.select("day", pl.col(x).alias("x"), pl.col(y).alias("y")).drop_nulls().filter(pl.col("x").is_finite() & pl.col("y").is_finite())
    n = f.height
    if n < 30:
        raise Refused(f"{n} rows with both {x} and {y}; a slope needs at least 30")
    xm, ym = f["x"].mean(), f["y"].mean()
    f = f.with_columns((pl.col("x") - xm).alias("xc"), (pl.col("y") - ym).alias("yc"))
    sxx = float((f["xc"] ** 2).sum())
    if sxx == 0:
        raise Refused(f"{x} has no variation")
    b = float((f["xc"] * f["yc"]).sum()) / sxx
    f = f.with_columns((pl.col("yc") - b * pl.col("xc")).alias("u"))
    se = sqrt(max(_nw_sum(f.select("day", (pl.col("xc") * pl.col("u")).alias("z")), lags), 0.0)) / sxx
    t = b / se if se > 0 else None
    r2 = 1 - float((f["u"] ** 2).sum()) / float((f["yc"] ** 2).sum())
    return {"n": n, "slope": b, "t": t, "p": None if t is None else 1 - _N.cdf(t), "r2": r2}


def diebold_mariano(frame: pl.DataFrame, d: str, *, lags: int) -> dict:
    """The mean of a loss differential `d` and its Newey–West t within each `day`: `n, mean, t, p` (one-sided, mean ≤ 0)."""
    utils.require(frame, ("day", d), "give each row its day")
    f = frame.select("day", pl.col(d).alias("d")).drop_nulls().filter(pl.col("d").is_finite())
    n = f.height
    if n < 30:
        raise Refused(f"{n} loss differentials; a test needs at least 30")
    mean = float(f["d"].mean())
    se = sqrt(max(_nw_sum(f.select("day", (pl.col("d") - mean).alias("z")), lags), 0.0)) / n
    t = mean / se if se > 0 else None
    return {"n": n, "mean": mean, "t": t, "p": None if t is None else 1 - _N.cdf(t)}


def _betacf(a: float, b: float, x: float) -> float:
    """The continued fraction of the incomplete beta (Numerical Recipes §6.4, modified Lentz)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c if abs(1 + aa / c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c if abs(1 + aa / c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return h


def t_sf(t: float, df: float) -> float:
    """P(T > t) for Student's t with `df` degrees of freedom, from the regularized incomplete beta."""
    if df <= 0:
        raise Refused(f"df={df} must be positive")
    x = df / (df + t * t)
    a, b = df / 2, 0.5
    front = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1 - x)) if 0 < x < 1 else 0.0
    ib = front * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - front * _betacf(b, a, 1 - x) / b
    tail = ib / 2
    return tail if t >= 0 else 1 - tail


def taker(grid: pl.DataFrame, signals: pl.DataFrame, horizon: int, *, fee: float = BYBIT_TAKER) -> pl.DataFrame:
    """Each signal traded at the touch: in at second t, out at second t + `horizon`, both crossing the spread.

    `signals` holds `ticker, ts, side` (+1 buys at the ask, −1 sells at the
    bid). Returns one row per trade that could close the same day, with
    `gross_bps` (the touch-to-touch return for the side, so the spread paid)
    and `net_bps` (less the fee both ways). A signal whose exit second is
    missing (a crossed book, a gap) or past midnight makes no trade.
    """
    if horizon < 1:
        raise Refused(f"horizon={horizon} must be at least one second")
    utils.require(signals, ("ticker", "ts", "side"), "give each signal its ticker, second and side")
    touch = grid.select("ticker", "ts", "day", "bid_px", "ask_px")
    out = touch.rename({"ts": "_exit_ts", "bid_px": "_exit_bid", "ask_px": "_exit_ask", "day": "_exit_day"})
    trades = (
        signals.select("ticker", "ts", pl.col("side").cast(pl.Float64))
        .filter(pl.col("side") != 0)
        .join(touch, on=["ticker", "ts"], how="inner")
        .with_columns((pl.col("ts") + pl.duration(seconds=horizon)).alias("_exit_ts"))
        .join(out, on=["ticker", "_exit_ts"], how="inner")
        .filter(pl.col("_exit_day") == pl.col("day"))
    )
    long = pl.col("side") > 0
    entry = pl.when(long).then(pl.col("ask_px")).otherwise(pl.col("bid_px"))
    exit_ = pl.when(long).then(pl.col("_exit_bid")).otherwise(pl.col("_exit_ask"))
    gross = 1e4 * pl.col("side") * (exit_ / entry - 1)
    return trades.select(
        "ticker", "day", "ts", "side", gross.alias("gross_bps"), (gross - 2e4 * fee).alias("net_bps")
    ).sort("ticker", "ts")


def score(trades: pl.DataFrame, *, min_trades: int = 30) -> pl.DataFrame:
    """Per rule and ticker: `trades, days, mean_bps` (net), and the one-sided t and p of the days' mean net returns.

    The day is the unit: overlapping trades within a day are not independent,
    so the t is over the daily means, with days − 1 degrees of freedom. A rule with fewer than `min_trades`
    trades keeps its row with no figure: it was run, and it counts.
    """
    utils.require(trades, ("rule", "ticker", "day", "net_bps"), "label each trade with its rule")
    daily = trades.group_by("rule", "ticker", "day").agg(pl.col("net_bps").mean().alias("_m"), pl.len().alias("_n"))
    rows = []
    for (rule, ticker), part in daily.group_by("rule", "ticker", maintain_order=True):
        n = int(part["_n"].sum())
        mean = float(trades.filter((pl.col("rule") == rule) & (pl.col("ticker") == ticker))["net_bps"].mean()) if n else None
        days = part.height
        t = p = None
        if n >= min_trades and days >= 2:
            sd = part["_m"].std()
            dm = float(part["_m"].mean())
            if sd:
                t = dm / (sd / sqrt(days))
                p = t_sf(t, days - 1)
        rows.append({"rule": rule, "ticker": ticker, "trades": n, "days": days, "mean_bps": mean if n >= min_trades else None, "t": t, "p": p})
    schema = {"rule": pl.String, "ticker": pl.String, "trades": pl.Int64, "days": pl.Int64, "mean_bps": pl.Float64, "t": pl.Float64, "p": pl.Float64}
    return pl.DataFrame(rows, schema=schema).sort("rule", "ticker")


def ofi_frame(book: pl.LazyFrame | pl.DataFrame, every: int = 10) -> pl.DataFrame:
    """H1's rows: per (ticker, bucket of `every` seconds), `x` = OFI over mean depth, `y` = the next bucket's mid return (bps).

    `decide_ts` is the bucket's last second, when its OFI is known. The next
    bucket must follow it directly on the same day. Its return runs from
    `decide_ts` to the next bucket's last second.
    """
    from . import liquidity

    buckets = liquidity.ofi(utils.lazy(book).filter(pl.col("ask_px") > pl.col("bid_px")), f"{every}s")
    step = pl.duration(seconds=every)
    keys = ["ticker", "day"]
    return (
        buckets.with_columns(pl.col("ts").dt.truncate("1d").alias("day"))
        .sort("ticker", "ts")
        .with_columns(
            pl.when(pl.col("depth") > 0).then(pl.col("ofi") / pl.col("depth")).alias("x"),
            pl.when(pl.col("ts").shift(-1).over(keys) == pl.col("ts") + step).then(pl.col("return_bps").shift(-1).over(keys)).alias("y"),
            (pl.col("ts") + step - pl.duration(seconds=1)).alias("decide_ts"),
        )
        .select("ticker", "day", "ts", "decide_ts", "x", "y")
    )


def ofi_rules(frame: pl.DataFrame, grid: pl.DataFrame, *, thresholds=(1.0, 2.0, 3.0), horizons=(10, 60), window: int = 360, fee: float = BYBIT_TAKER) -> pl.DataFrame:
    """H1's taker rules: buy when x's z-score over the day's previous `window` buckets is above k, sell below −k."""
    keys = ["ticker", "day"]
    prior = pl.col("x").shift(1)
    z = frame.sort("ticker", "ts").with_columns(
        ((pl.col("x") - prior.rolling_mean(window).over(keys)) / prior.rolling_std(window).over(keys)).alias("z")
    )
    out = []
    for k in thresholds:
        side = pl.when(pl.col("z") > k).then(1).when(pl.col("z") < -k).then(-1).otherwise(0)
        signals = z.select("ticker", pl.col("decide_ts").alias("ts"), side.alias("side")).filter(pl.col("side") != 0)
        for h in horizons:
            out.append(taker(grid, signals, h, fee=fee).with_columns(pl.lit(f"ofi z>{k:g} {h}s").alias("rule")))
    return pl.concat(out)


def microprice_frame(grid: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """H2's rows: per second, the squared errors (bps²) of the mid and the weighted mid as forecasts of the mid `horizon` seconds on, and `d` = mid's − weighted's."""
    later = grid.select("ticker", (pl.col("ts") - pl.duration(seconds=horizon)).alias("ts"), pl.col("mid").alias("_later"), pl.col("day").alias("_later_day"))
    bps = lambda forecast: 1e4 * (pl.col("_later") - forecast) / pl.col("mid")  # noqa: E731
    return (
        grid.join(later, on=["ticker", "ts"], how="inner")
        .filter(pl.col("_later_day") == pl.col("day"))
        .select("ticker", "day", "ts", (bps(pl.col("mid")) ** 2).alias("e_mid"), (bps(pl.col("wmid")) ** 2).alias("e_wmid"))
        .with_columns((pl.col("e_mid") - pl.col("e_wmid")).alias("d"))
        .sort("ticker", "ts")
    )


def lead_frame(grid: pl.DataFrame) -> pl.DataFrame:
    """H3's rows: per second t, `x` = the leader's log return over second t (bps), `y` = Bybit's mid log return over second t + 1 (bps)."""
    keys = ["ticker", "day"]
    one = pl.duration(seconds=1)
    follows = pl.col("ts").shift(1).over(keys) == pl.col("ts") - one
    leads = pl.col("ts").shift(-1).over(keys) == pl.col("ts") + one
    return (
        grid.sort("ticker", "ts")
        .with_columns(
            pl.when(follows).then(1e4 * (pl.col("leader_px") / pl.col("leader_px").shift(1).over(keys)).log()).alias("x"),
            pl.when(leads).then(1e4 * (pl.col("mid").shift(-1).over(keys) / pl.col("mid")).log()).alias("y"),
        )
        .select("ticker", "day", "ts", "x", "y")
    )


def lead_rules(frame: pl.DataFrame, grid: pl.DataFrame, *, thresholds=(2.0, 5.0, 10.0), horizons=(1, 5, 30), fee: float = BYBIT_TAKER) -> pl.DataFrame:
    """H3's taker rules: buy Bybit at the ask when the leader rose more than k bps in the last second, sell when it fell."""
    out = []
    for k in thresholds:
        side = pl.when(pl.col("x") > k).then(1).when(pl.col("x") < -k).then(-1).otherwise(0)
        signals = frame.select("ticker", "ts", side.alias("side")).filter(pl.col("side") != 0)
        for h in horizons:
            out.append(taker(grid, signals, h, fee=fee).with_columns(pl.lit(f"lead >{k:g}bps {h}s").alias("rule")))
    return pl.concat(out)
