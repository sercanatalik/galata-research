"""What trading costs: the spread measured from quotes and trades, and estimated from bars.

    gr.liquidity.quoted(gr.market.quotes(["BTC"], start, end))          # mid, spread_bps
    gr.liquidity.effective(trades, quotes)                              # effective, realized, impact
    gr.liquidity.from_bars(bars, "edge", "1h")                          # an estimate per hour
    gr.liquidity.bars(gr.reference.trades(["BTC"], s, e), "1m")        # bars on the venue measured

**Measured.** The effective spread compares a trade with the mid of the last
quote **strictly before** its millisecond. 98.7% of Hyperliquid trades share a
millisecond with a quote row, the book the trade itself changed (BTC and HYPE,
week of 2026-09-21). Against the strictly earlier quote, 84.6% of buys printed
at or above the ask; allowing the same millisecond, 64.7%.

**Estimated.** Four estimators read the spread from bars alone: Roll (1984),
Corwin and Schultz (2012), Abdi and Ranaldo (2017), and Ardia, Guidotti and
Kroencke's EDGE (2024). A pair is a bar and the one before it, contiguous
(`ts == previous close_ts`) and neither `in_gap`. On Hyperliquid's 1m bars, EDGE
over a week read BTC's spread at 0.13 bps against 0.12 quoted, and HYPE's at
0.94 against 0.11. So an estimate is calibrated against a measured spread on
the same venue and days before it is read (`notebooks/liquidity.py` ⑤).

The core stays free of numpy: everything here is polars, or plain Python.
"""

import math
from collections.abc import Sequence

import polars as pl

from . import utils
from ._errors import Refused

ESTIMATORS = ("roll", "corwin_schultz", "abdi_ranaldo", "edge")
_K = 3 - 2 * math.sqrt(2)


def quoted(frame: pl.LazyFrame | pl.DataFrame):
    """The frame with `mid` and `spread_bps = (ask − bid)/mid × 10⁴`: tape quotes or Bybit's book."""
    utils.require(frame, ("bid_px", "ask_px"), "load quotes with gr.market.quotes or a book with gr.reference.book")
    mid = (pl.col("bid_px") + pl.col("ask_px")) / 2
    return frame.with_columns(mid.alias("mid"), ((pl.col("ask_px") - pl.col("bid_px")) / mid * 1e4).alias("spread_bps"))


def effective(
    trades: pl.LazyFrame | pl.DataFrame,
    quotes: pl.LazyFrame | pl.DataFrame,
    *,
    tolerance: str = "2s",
    horizons: Sequence[str] = ("1s", "5s", "60s"),
) -> pl.DataFrame:
    """Each trade's effective spread, and its realized spread and price impact at each horizon, in bps.

    - `mid` is of the last quote of the trade's venue and ticker with `ts`
      **strictly before** the trade's, and no older than `tolerance`
      (Bybit's book is a 1 s grid; the tape's bbo pushes at a median 57 ms).
      Otherwise the row's spreads are null.
    - q = +1 for aggressor `bid` (a buy crossing), −1 for `ask`.
    - `effective_bps = 2q(p − mid)/mid`; `realized_bps_<Δ> = 2q(p − mid_Δ)/mid`,
      with mid_Δ the latest mid at or before `ts + Δ`; `impact_bps_<Δ>` is
      their difference, the part of the spread the market kept.
    - A realized spread is known only at `known_ts_<Δ> = ts + Δ`. It measures a
      cost afterwards; it is never a signal at `ts`.
    """
    utils.require(trades, ("venue", "ticker", "ts", "price", "aggressor"), "load trades with gr.market.trades or gr.reference.trades")
    utils.require(quotes, ("venue", "ticker", "ts", "bid_px", "ask_px"), "load quotes with gr.market.quotes or gr.reference.book")
    by = ["venue", "ticker"]
    mids = (
        utils.lazy(quotes)
        .select(*by, "ts", ((pl.col("bid_px") + pl.col("ask_px")) / 2).alias("mid"))
        .collect()
        .sort("ts")
    )
    t = utils.lazy(trades).collect().sort("ts")
    unknown = t.filter(~pl.col("aggressor").is_in(["bid", "ask"]))
    if unknown.height:
        raise Refused(f"aggressor {unknown['aggressor'][0]!r} is neither bid nor ask, the tape's words")
    out = t.join_asof(
        mids, on="ts", by=by, strategy="backward", allow_exact_matches=False, tolerance=tolerance, check_sortedness=False
    )
    q = pl.when(pl.col("aggressor") == "bid").then(1.0).otherwise(-1.0)
    out = out.with_columns((2 * q * (pl.col("price") - pl.col("mid")) / pl.col("mid") * 1e4).alias("effective_bps"))
    for h in horizons:
        later = out.select(*by, "ts").with_columns(pl.col("ts").dt.offset_by(h).alias(f"known_ts_{h}"))
        # Row order is restored by position: `later` and `out` share it before the sort.
        ahead = later.with_row_index("_i").sort(f"known_ts_{h}").join_asof(
            mids.rename({"ts": f"known_ts_{h}", "mid": f"_mid_{h}"}),
            on=f"known_ts_{h}",
            by=by,
            strategy="backward",
            tolerance=tolerance,
            check_sortedness=False,
        ).sort("_i")
        out = out.with_columns(ahead[f"known_ts_{h}"], ahead[f"_mid_{h}"])
        realized = 2 * q * (pl.col("price") - pl.col(f"_mid_{h}")) / pl.col("mid") * 1e4
        out = out.with_columns(realized.alias(f"realized_bps_{h}")).with_columns(
            (pl.col("effective_bps") - pl.col(f"realized_bps_{h}")).alias(f"impact_bps_{h}")
        ).drop(f"_mid_{h}")
    return out


def bars(trades: pl.LazyFrame | pl.DataFrame, interval: str) -> pl.DataFrame:
    """OHLCV bars from trades, one per venue, ticker and interval that held a trade.

    Open and close follow the trades' order (the venue's). An interval with no
    trade has **no bar**, so the pair across it is not contiguous, and an
    estimator counts nothing for it.
    """
    utils.require(trades, ("venue", "ticker", "ts", "price", "size"), "load trades with gr.market.trades or gr.reference.trades")
    return (
        utils.lazy(trades)
        .group_by("venue", "ticker", pl.col("ts").dt.truncate(interval).alias("bar"), maintain_order=True)
        .agg(
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("size").sum().alias("volume"),
            pl.len().cast(pl.UInt32).alias("trade_count"),
        )
        .rename({"bar": "ts"})
        .with_columns(pl.col("ts").dt.offset_by(interval).alias("close_ts"))
        .select("venue", "ticker", "ts", "close_ts", "open", "high", "low", "close", "volume", "trade_count")
        .sort("venue", "ticker", "ts")
        .collect()
    )


def edge(open: Sequence[float], high: Sequence[float], low: Sequence[float], close: Sequence[float], *, sign: bool = False) -> float | None:
    """EDGE, line for line with the authors' reference code (github.com/eguidotti/bidask, `edge.py`).

    Ardia, Guidotti and Kroencke (2024), "Efficient estimation of bid–ask
    spreads from open, high, low, and close prices", JFE 161, 103916. A
    fraction: 0.01 is a 1% spread. None where the reference returns missing.
    Pinned to the authors' test values in `tests/liquidity.py`.
    """
    n = len(open)
    if len(high) != n or len(low) != n or len(close) != n:
        raise Refused("open, high, low and close must have the same length")
    if n < 3:
        return None
    nan = math.nan

    def ln(v):
        return nan if v is None or (isinstance(v, float) and math.isnan(v)) else math.log(v)

    o, h, l, c = ([ln(v) for v in series] for series in (open, high, low, close))  # noqa: E741
    m = [(a + b) / 2 for a, b in zip(h, l, strict=True)]
    h1, l1, c1, m1 = h[:-1], l[:-1], c[:-1], m[:-1]
    o, h, l, c, m = o[1:], h[1:], l[1:], c[1:], m[1:]  # noqa: E741

    def isnan(*vs):
        return any(math.isnan(v) for v in vs)

    r1 = [a - b for a, b in zip(m, o, strict=True)]
    r2 = [a - b for a, b in zip(o, m1, strict=True)]
    r3 = [a - b for a, b in zip(m, c1, strict=True)]
    r4 = [a - b for a, b in zip(c1, m1, strict=True)]
    r5 = [a - b for a, b in zip(o, c1, strict=True)]
    tau = [nan if isnan(a, b, d) else float(a != b or b != d) for a, b, d in zip(h, l, c1, strict=True)]
    po1 = [t * (nan if isnan(a, b) else float(a != b)) for t, a, b in zip(tau, o, h, strict=True)]
    po2 = [t * (nan if isnan(a, b) else float(a != b)) for t, a, b in zip(tau, o, l, strict=True)]
    pc1 = [t * (nan if isnan(a, b) else float(a != b)) for t, a, b in zip(tau, c1, h1, strict=True)]
    pc2 = [t * (nan if isnan(a, b) else float(a != b)) for t, a, b in zip(tau, c1, l1, strict=True)]

    def mean(xs):
        kept = [x for x in xs if not math.isnan(x)]
        return sum(kept) / len(kept) if kept else nan

    pt = mean(tau)
    po = mean(po1) + mean(po2)
    pc = mean(pc1) + mean(pc2)
    if sum(t for t in tau if not math.isnan(t)) < 2 or po == 0 or pc == 0 or math.isnan(po) or math.isnan(pc):
        return None
    d1 = [r - mean(r1) / pt * t for r, t in zip(r1, tau, strict=True)]
    d3 = [r - mean(r3) / pt * t for r, t in zip(r3, tau, strict=True)]
    d5 = [r - mean(r5) / pt * t for r, t in zip(r5, tau, strict=True)]
    x1 = [-4.0 / po * a * b + -4.0 / pc * d * e for a, b, d, e in zip(d1, r2, d3, r4, strict=True)]
    x2 = [-4.0 / po * a * b + -4.0 / pc * d * e for a, b, d, e in zip(d1, r5, d5, r4, strict=True)]
    e1, e2 = mean(x1), mean(x2)
    v1 = mean([x * x for x in x1]) - e1 * e1
    v2 = mean([x * x for x in x2]) - e2 * e2
    vt = v1 + v2
    s2 = (v2 * e1 + v1 * e2) / vt if vt > 0 else (e1 + e2) / 2.0
    s = math.sqrt(abs(s2))
    return s * math.copysign(1.0, s2) if sign else s


def from_bars(bars: pl.LazyFrame | pl.DataFrame, estimator: str, every: str) -> pl.DataFrame:
    """A spread estimate per ticker and bucket of width `every`, from contiguous pairs of bars.

    `ticker, ts, pairs, spread, spread_bps`, with `venue` too when the bars
    have it. `ts` is the bucket's start; a pair belongs to the bucket of its
    later bar. On log prices o, h, l, c and m = (h + l)/2:

    - `roll`: 2√(−cov(Δc_t, Δc_{t−1})), null where the covariance is not negative.
    - `corwin_schultz`: per pair, β = ln(H_t/L_t)² + ln(H_{t−1}/L_{t−1})²,
      γ = ln(max H / min L)², α = (√(2β) − √β)/(3 − 2√2) − √(γ/(3 − 2√2)),
      S = 2(e^α − 1)/(1 + e^α), negative S set to 0, then the mean. No
      overnight adjustment: these markets do not close.
    - `abdi_ranaldo`: √max(mean of 4(c_{t−1} − m_{t−1})(c_{t−1} − m_t), 0).
    - `edge`: the reference code's estimator on the bucket's pairs; it agrees
      with `edge()` on the bucket's bars and the bar before its first.
    """
    if estimator not in ESTIMATORS:
        raise Refused(f"estimator={estimator!r} is not one of {', '.join(ESTIMATORS)}")
    utils.require(bars, ("ticker", "ts", "close_ts", "open", "high", "low", "close"), "load bars with gr.market.candles, gr.reference.candles or gr.liquidity.bars")
    lf = utils.lazy(bars)
    names = lf.collect_schema().names()
    keys = ["venue", "ticker"] if "venue" in names else ["ticker"]
    gap = pl.col("in_gap").fill_null(False) if "in_gap" in names else pl.lit(False)

    frame = (
        lf.sort(*keys, "ts")
        .with_columns(gap.alias("_gap"))
        # A bar in a gap says nothing: its prices are null, and so is any pair it is in.
        .with_columns(*[pl.when(pl.col("_gap")).then(None).otherwise(pl.col(c).log()).alias(c) for c in ("open", "high", "low", "close")])
        .with_columns(((pl.col("high") + pl.col("low")) / 2).alias("mid"))
        .with_columns((pl.col("ts") == pl.col("close_ts").shift(1).over(keys)).fill_null(False).alias("_joined"))
        .with_columns(
            *[pl.when(pl.col("_joined")).then(pl.col(c).shift(1).over(keys)).alias(f"{c}1") for c in ("high", "low", "close", "mid")],
            pl.col("ts").dt.truncate(every).alias("_bucket"),
        )
        .collect()
    )
    group = [*keys, "_bucket"]
    pair = pl.col("close1").is_not_null() & pl.col("close").is_not_null()
    frame = frame.with_columns(pair.alias("_pair"))

    if estimator == "roll":
        dc = pl.when(pl.col("_pair")).then(pl.col("close") - pl.col("close1"))
        frame = frame.with_columns(dc.alias("_dc")).with_columns(pl.col("_dc").shift(1).over(keys).alias("_dc1"))
        # The previous change must be the previous pair's, in the same run.
        frame = frame.with_columns(pl.when(pl.col("_joined")).then(pl.col("_dc1")).alias("_dc1"))
        both = pl.col("_dc").is_not_null() & pl.col("_dc1").is_not_null()
        out = frame.group_by(group).agg(
            both.sum().alias("pairs"),
            pl.cov(pl.col("_dc").filter(both), pl.col("_dc1").filter(both)).alias("_cov"),
        )
        spread = pl.when(pl.col("_cov") < 0).then(2 * (-pl.col("_cov")).sqrt())
    elif estimator == "corwin_schultz":
        beta = (pl.col("high") - pl.col("low")) ** 2 + (pl.col("high1") - pl.col("low1")) ** 2
        gamma = (pl.max_horizontal("high", "high1") - pl.min_horizontal("low", "low1")) ** 2
        alpha = ((2 * beta).sqrt() - beta.sqrt()) / _K - (gamma / _K).sqrt()
        s = (2 * (alpha.exp() - 1) / (1 + alpha.exp())).clip(lower_bound=0)
        out = frame.group_by(group).agg(pl.col("_pair").sum().alias("pairs"), s.filter(pl.col("_pair")).mean().alias("_s"))
        spread = pl.col("_s")
    elif estimator == "abdi_ranaldo":
        term = 4 * (pl.col("close1") - pl.col("mid1")) * (pl.col("close1") - pl.col("mid"))
        out = frame.group_by(group).agg(pl.col("_pair").sum().alias("pairs"), term.filter(pl.col("_pair")).mean().alias("_s2"))
        spread = pl.when(pl.col("_s2").is_not_null()).then(pl.max_horizontal(pl.col("_s2"), pl.lit(0.0)).sqrt())
    else:
        out, spread = _edge_buckets(frame, group), pl.col("_s")

    return (
        out.with_columns(spread.alias("spread"))
        .with_columns((pl.col("spread") * 1e4).alias("spread_bps"), pl.col("pairs").cast(pl.UInt32))
        .rename({"_bucket": "ts"})
        .select(*keys, "ts", "pairs", "spread", "spread_bps")
        .sort(*keys, "ts")
    )


def _edge_buckets(frame: pl.DataFrame, group: list[str]) -> pl.DataFrame:
    """EDGE per bucket, as the reference code computes it over the bucket's rows (a row is a bar and its predecessor)."""
    o, h, l, c, m = (pl.col(x) for x in ("open", "high", "low", "close", "mid"))  # noqa: E741
    h1, l1, c1, m1 = (pl.col(x) for x in ("high1", "low1", "close1", "mid1"))

    def ne(a, b):
        return pl.when(a.is_null() | b.is_null()).then(None).otherwise((a != b).cast(pl.Float64))

    tau = pl.when(h.is_null() | l.is_null() | c1.is_null()).then(None).otherwise(((h != l) | (l != c1)).cast(pl.Float64))
    # A bar that starts a run has no predecessor, as the reference's first bar
    # has none: it lends its prices to the next row and is not a row itself.
    rows = frame.filter(pl.col("_joined")).with_columns(
        (m - o).alias("r1"), (o - m1).alias("r2"), (m - c1).alias("r3"), (c1 - m1).alias("r4"), (o - c1).alias("r5"),
        tau.alias("tau"),
    ).with_columns(
        (pl.col("tau") * ne(o, h)).alias("po1"), (pl.col("tau") * ne(o, l)).alias("po2"),
        (pl.col("tau") * ne(c1, h1)).alias("pc1"), (pl.col("tau") * ne(c1, l1)).alias("pc2"),
    )  # fmt: skip
    first = rows.group_by(group).agg(
        pl.col("_pair").sum().alias("pairs"),
        pl.col("tau").mean().alias("pt"),
        pl.col("tau").sum().alias("st"),
        (pl.col("po1").mean() + pl.col("po2").mean()).alias("po"),
        (pl.col("pc1").mean() + pl.col("pc2").mean()).alias("pc"),
        pl.col("r1").mean().alias("mr1"), pl.col("r3").mean().alias("mr3"), pl.col("r5").mean().alias("mr5"),
    )  # fmt: skip
    rows = rows.join(first, on=group, how="left").with_columns(
        (pl.col("r1") - pl.col("mr1") / pl.col("pt") * pl.col("tau")).alias("d1"),
        (pl.col("r3") - pl.col("mr3") / pl.col("pt") * pl.col("tau")).alias("d3"),
        (pl.col("r5") - pl.col("mr5") / pl.col("pt") * pl.col("tau")).alias("d5"),
    ).with_columns(
        (-4.0 / pl.col("po") * pl.col("d1") * pl.col("r2") - 4.0 / pl.col("pc") * pl.col("d3") * pl.col("r4")).alias("x1"),
        (-4.0 / pl.col("po") * pl.col("d1") * pl.col("r5") - 4.0 / pl.col("pc") * pl.col("d5") * pl.col("r4")).alias("x2"),
    )
    second = rows.group_by(group).agg(
        pl.col("pairs").first(), pl.col("st").first(), pl.col("po").first(), pl.col("pc").first(),
        pl.col("x1").mean().alias("e1"), pl.col("x2").mean().alias("e2"),
        (pl.col("x1") ** 2).mean().alias("q1"), (pl.col("x2") ** 2).mean().alias("q2"),
    )  # fmt: skip
    v1 = pl.col("q1") - pl.col("e1") ** 2
    v2 = pl.col("q2") - pl.col("e2") ** 2
    s2 = pl.when(v1 + v2 > 0).then((v2 * pl.col("e1") + v1 * pl.col("e2")) / (v1 + v2)).otherwise((pl.col("e1") + pl.col("e2")) / 2)
    missing = (pl.col("st").fill_null(0) < 2) | (pl.col("po").fill_null(0) == 0) | (pl.col("pc").fill_null(0) == 0)
    return second.with_columns(pl.when(missing).then(None).otherwise(s2.abs().sqrt()).alias("_s"))


def _walk(points: list[tuple[float, float]], size: float) -> tuple[float | None, float | None]:
    """(cost_bps, reach_bps) of `size` against cumulative (distance_bps, notional) points, linear between them."""
    curve = [(0.0, 0.0), *sorted(points)]
    filled, weighted = 0.0, 0.0
    for (d0, n0), (d1, n1) in zip(curve, curve[1:], strict=False):
        if n1 <= n0:
            continue
        take = min(size, n1) - n0
        if take <= 0:
            continue
        reach = d0 + take / (n1 - n0) * (d1 - d0)
        weighted += take * (d0 + reach) / 2
        filled += take
        if n1 >= size:
            return weighted / size, reach
    return None, None


def cost_of_size(depth: pl.LazyFrame | pl.DataFrame, sizes: Sequence[float]) -> pl.DataFrame:
    """The cost of market orders of each notional size, per snapshot and side, from a cumulative depth curve.

    `depth` has rows `ts, band_pct, notional` (as `gr.reference.depth` or
    `book_points`); a positive band is the ask side (`buy`), a negative one
    the bid (`sell`). Each side's curve is anchored at (0, 0) and linear in
    notional between bands, so notional is spread evenly in price within a
    band. A real book is denser near the touch, so inside a wide first band
    this **overstates** the cost. `cost_bps` is the notional-weighted mean
    distance filled, `reach_bps` the distance the order reaches. A size past
    the deepest band is `beyond`, with no cost: nothing is extrapolated.
    """
    utils.require(depth, ("ts", "band_pct", "notional"), "load depth with gr.reference.depth or gr.liquidity.book_points")
    lf = utils.lazy(depth)
    keys = [k for k in ("venue", "ticker") if k in lf.collect_schema().names()]
    sides = (
        lf.with_columns(
            pl.when(pl.col("band_pct") > 0).then(pl.lit("buy")).otherwise(pl.lit("sell")).alias("side"),
            (pl.col("band_pct").abs() * 100).alias("d"),
        )
        .group_by(*keys, "ts", "side")
        .agg(pl.col("d"), pl.col("notional"))
        .sort(*keys, "ts", "side")
        .collect()
    )
    rows = []
    for r in sides.iter_rows(named=True):
        points = list(zip(r["d"], r["notional"], strict=True))
        for q in sizes:
            cost, reach = _walk(points, q)
            rows.append({**{k: r[k] for k in keys}, "ts": r["ts"], "side": r["side"], "size": float(q),
                         "cost_bps": cost, "reach_bps": reach, "beyond": cost is None})  # fmt: skip
    schema = {**{k: pl.String for k in keys}, "ts": sides.schema["ts"], "side": pl.String, "size": pl.Float64,
              "cost_bps": pl.Float64, "reach_bps": pl.Float64, "beyond": pl.Boolean}  # fmt: skip
    return pl.DataFrame(rows, schema=schema)


def book_points(book: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    """Bybit's rebuilt book as depth points `ts, band_pct, notional` (and `venue, ticker`), per side:

    the top size at the half spread, the depth within 2 bps, and within 10 bps
    where the archive reached it (null past the reach, and then left out).
    """
    utils.require(book, ("ts", "bid_px", "ask_px", "bid_sz", "ask_sz", "bid_depth_2bps", "ask_depth_2bps",
                         "bid_depth_10bps", "ask_depth_10bps"), "load a book with gr.reference.book")  # fmt: skip
    lf = utils.lazy(book)
    keys = [k for k in ("venue", "ticker") if k in lf.collect_schema().names()]
    mid = (pl.col("bid_px") + pl.col("ask_px")) / 2
    half = (pl.col("ask_px") - pl.col("bid_px")) / 2 / mid * 100
    parts = []
    for sign, side in ((1, "ask"), (-1, "bid")):
        for pct, notional in (
            (half, pl.col(f"{side}_sz") * mid),
            (pl.lit(0.02), pl.col(f"{side}_depth_2bps") * mid),
            (pl.lit(0.10), pl.col(f"{side}_depth_10bps") * mid),
        ):
            parts.append(lf.select(*keys, "ts", (sign * pct).alias("band_pct"), notional.alias("notional")))
    return pl.concat(parts).drop_nulls("notional").sort(*keys, "ts", "band_pct").collect()


def flow(trades: pl.LazyFrame | pl.DataFrame, every: str = "1m") -> pl.DataFrame:
    """Signed flow per venue, ticker and bucket that held a trade.

    `buy_notional` and `sell_notional` sum price × size by aggressor (`bid`
    is a buy crossing), `signed_notional` is their difference and
    `imbalance` it over their sum. `last_price` is the bucket's last trade
    in the venue's order, and `return_bps` the log change from the previous
    bucket's: null when the previous bucket held no trade.
    """
    utils.require(trades, ("venue", "ticker", "ts", "price", "size", "aggressor"), "load trades with gr.market.trades or gr.reference.trades")
    lf = utils.lazy(trades)
    unknown = lf.filter(~pl.col("aggressor").is_in(["bid", "ask"])).select("aggressor").head(1).collect()
    if unknown.height:
        raise Refused(f"aggressor {unknown['aggressor'][0]!r} is neither bid nor ask, the tape's words")
    notional = pl.col("price") * pl.col("size")
    out = (
        lf.group_by("venue", "ticker", pl.col("ts").dt.truncate(every).alias("bucket"), maintain_order=True)
        .agg(
            notional.filter(pl.col("aggressor") == "bid").sum().alias("buy_notional"),
            notional.filter(pl.col("aggressor") == "ask").sum().alias("sell_notional"),
            pl.len().cast(pl.UInt32).alias("trade_count"),
            pl.col("price").last().alias("last_price"),
        )
        .rename({"bucket": "ts"})
        .sort("venue", "ticker", "ts")
        .with_columns(
            (pl.col("buy_notional") - pl.col("sell_notional")).alias("signed_notional"),
            ((pl.col("buy_notional") - pl.col("sell_notional")) / (pl.col("buy_notional") + pl.col("sell_notional"))).alias("imbalance"),
            pl.when(pl.col("ts").dt.offset_by(f"-{every}") == pl.col("ts").shift(1).over("venue", "ticker"))
            .then(1e4 * (pl.col("last_price") / pl.col("last_price").shift(1).over("venue", "ticker")).log())
            .alias("return_bps"),
        )
        .collect()
    )
    return out.select("venue", "ticker", "ts", "buy_notional", "sell_notional", "signed_notional", "imbalance", "trade_count", "last_price", "return_bps")


def kyle_lambda(flows: pl.LazyFrame | pl.DataFrame, *, by: Sequence[str] | str | None = None) -> pl.DataFrame:
    """Kyle's λ per group: the OLS slope of `return_bps` on `signed_notional` in $M, in bps per $1M.

    With `se`, `t`, `r2` and `n` over buckets with a return. Flow and return
    in one bucket are jointly determined, so λ is an association, as it is
    usually estimated; the iid `t` ignores the flow's autocorrelation.
    """
    utils.require(flows, ("signed_notional", "return_bps"), "make flow with gr.liquidity.flow")
    keys = [by] if isinstance(by, str) else list(by or [])
    q, r = pl.col("signed_notional") / 1e6, pl.col("return_bps")
    lf = utils.lazy(flows).drop_nulls("return_bps").with_columns(q.alias("_q"))
    g = lf.group_by(keys) if keys else lf.group_by(pl.lit(1).alias("_all"))
    out = g.agg(
        pl.len().alias("n"),
        pl.cov("_q", "return_bps").alias("_c"),
        pl.col("_q").var().alias("_v"),
        pl.col("return_bps").var().alias("_vr"),
        ((pl.col("_q") - pl.col("_q").mean()) ** 2).sum().alias("_sxx"),
    ).with_columns((pl.col("_c") / pl.col("_v")).alias("lambda_bps_per_m"))
    out = out.with_columns(
        (pl.col("_c") ** 2 / (pl.col("_v") * pl.col("_vr"))).alias("r2"),
    ).with_columns(
        # Residual variance from R²: σ̂²_ε = (1 − R²)·var(r)·(n − 1)/(n − 2).
        (((1 - pl.col("r2")) * pl.col("_vr") * (pl.col("n") - 1) / (pl.col("n") - 2)) / pl.col("_sxx")).sqrt().alias("se")
    ).with_columns((pl.col("lambda_bps_per_m") / pl.col("se")).alias("t"))
    cols = [*keys, "lambda_bps_per_m", "se", "t", "r2", "n"]
    return out.select(cols).sort(keys).collect() if keys else out.select(cols).collect()


def amihud(flows: pl.LazyFrame | pl.DataFrame, *, by: Sequence[str] | str | None = None) -> pl.DataFrame:
    """The Amihud ratio per group: the mean of |return_bps| per $1M traded, over buckets with a return and notional."""
    utils.require(flows, ("buy_notional", "sell_notional", "return_bps"), "make flow with gr.liquidity.flow")
    keys = [by] if isinstance(by, str) else list(by or [])
    traded = (pl.col("buy_notional") + pl.col("sell_notional")) / 1e6
    lf = utils.lazy(flows).drop_nulls("return_bps").filter(traded > 0).with_columns((pl.col("return_bps").abs() / traded).alias("_a"))
    g = lf.group_by(keys) if keys else lf.group_by(pl.lit(1).alias("_all"))
    out = g.agg(pl.col("_a").mean().alias("amihud_bps_per_m"), pl.len().alias("n"))
    cols = [*keys, "amihud_bps_per_m", "n"]
    return out.select(cols).sort(keys).collect() if keys else out.select(cols).collect()


def shocks(flows: pl.LazyFrame | pl.DataFrame, *, share: float = 0.001) -> pl.DataFrame:
    """The buckets whose |signed_notional| is in the top `share` of their venue, ticker and UTC day.

    `side` is the book side the flow took: buying takes the `ask`, selling the `bid`.
    """
    utils.require(flows, ("venue", "ticker", "ts", "signed_notional"), "make flow with gr.liquidity.flow(trades, '1s')")
    if not 0 < share < 1:
        raise Refused(f"share={share} must be in (0, 1)")
    day = ["venue", "ticker", pl.col("ts").dt.date()]
    size = pl.col("signed_notional").abs()
    return (
        utils.lazy(flows)
        .filter(size >= size.quantile(1 - share, interpolation="higher").over(day))
        .with_columns(pl.when(pl.col("signed_notional") > 0).then(pl.lit("ask")).otherwise(pl.lit("bid")).alias("side"))
        .select("venue", "ticker", "ts", "signed_notional", "side")
        .sort("venue", "ticker", "ts")
        .collect()
    )


def _windows(book: pl.LazyFrame | pl.DataFrame, events: pl.DataFrame, horizon: int) -> pl.DataFrame:
    keys = [k for k in ("venue", "ticker") if k in events.columns and k in utils.lazy(book).collect_schema().names()]
    b = utils.lazy(book).select(
        *keys, "ts",
        pl.col("ask_depth_2bps"), pl.col("bid_depth_2bps"),
        ((pl.col("ask_px") - pl.col("bid_px")) / ((pl.col("ask_px") + pl.col("bid_px")) / 2) * 1e4).alias("spread_bps"),
    ).collect()  # fmt: skip
    grid = events.with_row_index("event").join(pl.DataFrame({"k": list(range(horizon + 1))}), how="cross")
    grid = grid.with_columns((pl.col("ts") + pl.duration(seconds=1) * pl.col("k")).alias("_t"))
    out = grid.join(b.rename({"ts": "_t"}), on=[*keys, "_t"], how="left")
    return out.with_columns(pl.when(pl.col("side") == "ask").then(pl.col("ask_depth_2bps")).otherwise(pl.col("bid_depth_2bps")).alias("depth"))


def resilience(book: pl.LazyFrame | pl.DataFrame, events: pl.DataFrame, *, horizon: int = 60) -> pl.DataFrame:
    """Per shock, how far the book side it hit fell and how fast its 2 bps depth came back.

    The book row at the shock second is the state before the second's trades
    (the reduction emits the state after every message at or before each
    whole second). `recovery_s` is the first second at or after the lowest
    depth where it is back to 90% of `depth_pre`; null if not within `horizon`.
    A 1 s book cannot see a refill inside the second: 1 means by the next one.
    """
    utils.require(events, ("ts", "side"), "make shocks with gr.liquidity.shocks")
    w = _windows(book, events, horizon)
    pre = w.filter(pl.col("k") == 0).select("event", pl.col("depth").alias("depth_pre"), pl.col("spread_bps").alias("spread_pre"))
    after = w.filter(pl.col("k") > 0).join(pre, on="event").filter(pl.col("depth_pre") > 0)
    after = after.with_columns((pl.col("depth") / pl.col("depth_pre")).alias("ratio"))
    lowest = after.group_by("event").agg(
        pl.col("ratio").min().alias("depth_min_ratio"),
        pl.col("k").sort_by("ratio").first().alias("_kmin"),
        (pl.col("spread_bps").max()).alias("_smax"),
    )
    back = (
        after.join(lowest.select("event", "_kmin"), on="event")
        .filter((pl.col("k") >= pl.col("_kmin")) & (pl.col("ratio") >= 0.9))
        .group_by("event")
        .agg(pl.col("k").min().alias("recovery_s"))
    )
    keys = [c for c in events.columns if c != "event"]
    return (
        events.with_row_index("event")
        .join(pre, on="event", how="inner")
        .filter(pl.col("depth_pre") > 0)
        .join(lowest, on="event", how="left")
        .join(back, on="event", how="left")
        .with_columns((pl.col("_smax") / pl.col("spread_pre")).alias("spread_peak_ratio"))
        .select(*keys, "depth_pre", "spread_pre", "depth_min_ratio", "recovery_s", "spread_peak_ratio")
    )


def resilience_curve(book: pl.LazyFrame | pl.DataFrame, events: pl.DataFrame, *, horizon: int = 60) -> pl.DataFrame:
    """The median depth over its pre-shock level at each second after a shock: `k, ratio, n`.

    The median, not the mean: a shock that starts from a near-empty 2 bps
    depth divides by almost nothing, and a mean of such ratios measured
    4× at one second on Bybit's BTC book (2026-09-28), a depletion read as a surge.
    """
    w = _windows(book, events, horizon)
    pre = w.filter(pl.col("k") == 0).select("event", pl.col("depth").alias("_pre"))
    return (
        w.join(pre, on="event")
        .filter(pl.col("_pre") > 0)
        .group_by("k")
        .agg((pl.col("depth") / pl.col("_pre")).median().alias("ratio"), pl.col("depth").is_not_null().sum().alias("n"))
        .sort("k")
    )


def placebo(flows: pl.LazyFrame | pl.DataFrame, shocks_: pl.DataFrame, *, seed: int = 20260928) -> pl.DataFrame:
    """As many quiet seconds per venue, ticker and day as there are shocks, drawn at random, each with a random side.

    Quiet means |signed_notional| below the day's median. A resilience figure
    is read only against this: the depth of any second dips and refills
    within a minute, and ~32 s is the placebo's own median "recovery" on
    Bybit's book (2026-09-28), about the same as a shock's.
    """
    import random

    utils.require(flows, ("venue", "ticker", "ts", "signed_notional"), "make flow with gr.liquidity.flow(trades, '1s')")
    rng = random.Random(seed)
    day = pl.col("ts").dt.date().alias("_day")
    counts = shocks_.group_by("venue", "ticker", day).len()
    quiet = (
        utils.lazy(flows)
        .with_columns(day)
        .filter(pl.col("signed_notional").abs() < pl.col("signed_notional").abs().median().over("venue", "ticker", "_day"))
        .collect()
    )
    out = []
    for r in counts.sort("venue", "ticker", "_day").iter_rows(named=True):
        pool = quiet.filter((pl.col("venue") == r["venue"]) & (pl.col("ticker") == r["ticker"]) & (pl.col("_day") == r["_day"]))
        n = min(r["len"], pool.height)
        if n == 0:
            continue
        picked = pool.sample(n=n, seed=rng.randrange(1 << 31)).with_columns(pl.Series("side", [rng.choice(("ask", "bid")) for _ in range(n)]))
        out.append(picked.select("venue", "ticker", "ts", "signed_notional", "side"))
    schema = {"venue": pl.String, "ticker": pl.String, "ts": quiet.schema["ts"], "signed_notional": pl.Float64, "side": pl.String}
    return pl.concat(out).sort("venue", "ticker", "ts") if out else pl.DataFrame(schema=schema)


def allocate(total: float, weights: Sequence[float]) -> list[float]:
    """`total` split in proportion to non-negative `weights`."""
    w = [float(x) for x in weights]
    if any(x < 0 for x in w) or sum(w) <= 0:
        raise Refused("weights must be non-negative and not all zero")
    s = sum(w)
    return [total * x / s for x in w]


def schedule_cost(plan: pl.DataFrame, depth: pl.DataFrame, *, slices: int = 60) -> dict:
    """The cost of a plan in bps of its total, against each hour's ±1% notional on the side taken.

    Each hour's notional is cut into `slices` equal children; a child c costs
    c/depth·50 bps (⑩'s linear book: half of the 100 bps band on average).
    The model overstates near-touch costs on a book as dense as BTC's (~16×
    in ⑩), so compare plans within it; do not read its level. `over_depth`
    counts hours whose child exceeds the depth, past the model's range.
    """
    utils.require(plan, ("ts", "notional"), "a plan with ts and notional")
    utils.require(depth, ("ts", "depth"), "depth with ts and the ±1% notional on the side taken")
    j = plan.join(depth, on="ts", how="left")
    missing = j.filter(pl.col("depth").is_null() | (pl.col("depth") <= 0))
    if missing.height:
        raise Refused(f"no depth for the plan hour {missing['ts'][0]}")
    child = pl.col("notional") / slices
    dollars = (slices * child * (child / pl.col("depth") * 50) / 1e4).sum()
    out = j.select(dollars.alias("dollars"), pl.col("notional").sum().alias("total"), (child > pl.col("depth")).sum().alias("over")).row(0, named=True)
    return {"cost_bps": out["dollars"] / out["total"] * 1e4 if out["total"] else None, "over_depth": int(out["over"])}


# ── order-flow imbalance (measure-the-order-flow) ────────────────────────────


def ofi(quotes: pl.LazyFrame | pl.DataFrame, every: str = "10s", *, max_gap_us: int = 5_000_000) -> pl.DataFrame:
    """Cont, Kukanov and Stoikov's (2014) order-flow imbalance per venue, ticker and bucket, from the best bid and ask.

    Each quote update n contributes
    eₙ = 1{Pbₙ ≥ Pbₙ₋₁}·qbₙ − 1{Pbₙ ≤ Pbₙ₋₁}·qbₙ₋₁ − 1{Paₙ ≤ Paₙ₋₁}·qaₙ + 1{Paₙ ≥ Paₙ₋₁}·qaₙ₋₁:
    size arriving at the bid or leaving the ask pushes the price up. `ofi` is
    their sum over the bucket (base units), `depth` the bucket's mean of
    (qb + qa)/2 over its updates (their AD), and `return_bps` the log change
    of the mid from the last update at or before the previous bucket's end to
    the last at or before this one's: null when the previous bucket held no
    update. **No term is computed across a crossed or locked state, or across
    a gap longer than `max_gap_us`** (a reconnect, a stale feed): one spurious
    term would dominate the bucket.
    """
    utils.require(quotes, ("venue", "ticker", "ts", "bid_px", "ask_px", "bid_sz", "ask_sz"), "load quotes with gr.market.quotes")
    keys = ["venue", "ticker"]
    gap = pl.duration(microseconds=max_gap_us)
    lf = utils.lazy(quotes).sort(*keys, "ts")
    valid = pl.col("ask_px") > pl.col("bid_px")
    prev = lambda c: pl.col(c).shift(1).over(keys)  # noqa: E731
    pb, pa, qb, qa = pl.col("bid_px"), pl.col("ask_px"), pl.col("bid_sz"), pl.col("ask_sz")
    e = (
        pl.when(pb >= prev("bid_px")).then(qb).otherwise(0.0)
        - pl.when(pb <= prev("bid_px")).then(prev("bid_sz")).otherwise(0.0)
        - pl.when(pa <= prev("ask_px")).then(qa).otherwise(0.0)
        + pl.when(pa >= prev("ask_px")).then(prev("ask_sz")).otherwise(0.0)
    )
    joined = (pl.col("ts") - prev("ts") <= gap) & prev("_valid")
    frame = lf.with_columns(valid.alias("_valid")).with_columns(
        pl.when(pl.col("_valid") & joined).then(e).otherwise(0.0).alias("_e"),
        ((pb + pa) / 2).alias("_mid"),
    )
    out = (
        frame.filter(pl.col("_valid"))
        .group_by(*keys, pl.col("ts").dt.truncate(every).alias("bucket"), maintain_order=True)
        .agg(
            pl.col("_e").sum().alias("ofi"),
            ((qb + qa) / 2).mean().alias("depth"),
            pl.col("_mid").last().alias("_last_mid"),
            pl.len().cast(pl.UInt32).alias("events"),
        )
        .rename({"bucket": "ts"})
        .sort(*keys, "ts")
        .with_columns(
            pl.when(pl.col("ts").dt.offset_by(f"-{every}") == pl.col("ts").shift(1).over(keys))
            .then(1e4 * (pl.col("_last_mid") / pl.col("_last_mid").shift(1).over(keys)).log())
            .alias("return_bps")
        )
        .drop("_last_mid")
        .collect()
    )
    return out


def impact(buckets: pl.DataFrame, x: str = "ofi_norm", y: str = "return_bps", *, winsor: float = 0.999, min_n: int = 100) -> dict:
    """OLS of `y` on `x` with an intercept, as Cont, Kukanov and Stoikov fit ΔP on OFI: β, R², White t, n.

    `x` is winsorised at its `winsor` and 1 − `winsor` quantiles first (heavy
    tails). Rows with either side null are dropped. Under `min_n` rows, or
    with no variation in `x`, it refuses. R² is contemporaneous explanatory
    power, not a forecast: Cont, Cucuringu and Zhang find next-minute
    out-of-sample R² of −0.37% for the same regressor.
    """
    frame = buckets.select(pl.col(x).cast(pl.Float64).alias("x"), pl.col(y).cast(pl.Float64).alias("y")).drop_nulls()
    if frame.height < min_n:
        raise Refused(f"{frame.height} buckets with both {x} and {y}, under {min_n}")
    lo, hi = frame["x"].quantile(1 - winsor), frame["x"].quantile(winsor)
    frame = frame.with_columns(pl.col("x").clip(lo, hi))
    n = frame.height
    mx, my = frame["x"].mean(), frame["y"].mean()
    dx, dy = frame["x"] - mx, frame["y"] - my
    sxx = float((dx * dx).sum())
    if sxx == 0:
        raise Refused(f"{x} does not vary over the {n} buckets")
    beta = float((dx * dy).sum()) / sxx
    alpha = my - beta * mx
    resid = frame["y"] - alpha - beta * frame["x"]
    syy = float((dy * dy).sum())
    r2 = 1 - float((resid * resid).sum()) / syy if syy > 0 else float("nan")
    # White's heteroskedasticity-consistent variance of the slope.
    var = float((dx * dx * resid * resid).sum()) / sxx**2
    return {"beta": beta, "alpha": float(alpha), "r2": r2, "t": beta / math.sqrt(var) if var > 0 else float("nan"), "n": n}
