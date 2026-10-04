import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import random
    from datetime import UTC, datetime

    import altair as alt
    import marimo as mo
    import polars as pl

    import galata_research as gr

    return UTC, alt, datetime, gr, mo, pl, random


@app.cell
def _(gr):
    # Cached results are keyed on the days the store holds, so a fetch recomputes them.
    try:
        HELD = int(gr.reference.coverage()["days_ok"].sum())
    except gr.Refused:
        HELD = 0
    return (HELD,)


@app.cell
def _(mo):
    mo.md(r"""
    # The liquidity study: What trading costs: the spread, a size, the flow, the refill, and depth against volatility (⑤ ⑩ ⑪ ⑭ ⑱)

    Part of the intraday-liquidity study; `liquidity.py` is where it begins (① to ④). The sections keep their numbers from the study, which the README and the archived designs cite. Data: the reference store (`galata-fetch`) beside the record.
    """)
    return

@app.cell
def _(gr, mo):
    EVER = ("2019-01-01T00:00Z", "2100-01-01T00:00Z")
    _held = sorted(set(gr.reference.coverage()["ticker"]) | {"BTC"})
    ticker = mo.ui.dropdown(_held, value="BTC", label="ticker")
    ticker
    return EVER, ticker

@app.cell
def _(mo):
    mo.md(r"""
    ## ⑤ What trading costs, by the hour

    **Measured.** The quoted spread is (ask − bid)/mid, averaged over the hour
    with each quote weighted by how long it stood; its median is shown too,
    and on these markets it sits at one tick in every hour. The effective spread,
    2q(p − m)/m, compares each trade with the mid of the last quote **strictly
    before** its millisecond, with q = +1 for a buy crossing. The realized
    spread uses the mid 5 s later, and the impact is their difference: the
    part of the spread the market kept (`gr.liquidity.effective`).

    **Estimated.** Roll, Corwin–Schultz, Abdi–Ranaldo and EDGE read a spread
    from bars alone, here from the same venue's 1m bars on the same days, per
    hour (`gr.liquidity.from_bars`).

    **The test, stated before the table.** An estimator *calibrates* for a
    ticker on a venue when the Spearman ρ between its 24-hour profile and the
    measured quoted spread's is **at least 0.6**. A measured profile that
    barely moves (max ÷ min < 1.1, a spread pinned at one tick) leaves
    **nothing to calibrate against**, and is flagged, not failed.

    **Two measurements, and where each is weak.**

    - *Bybit*: its book rebuilt to one row per second on the first Wednesday
      of each month, and that day's trades. A 1 s book is up to a second
      stale, so on a one-tick market the effective spread against it mostly
      measures how far the price moved inside the second.
    - *The tape*: Hyperliquid's event-driven top of book and trades, to the
      millisecond, over the days it holds.
    """)
    return


@app.cell
def _(EVER, gr, pl, ticker):
    def _days(lf):
        return sorted(lf.select(pl.col("ts").dt.date().unique()).collect().to_series().to_list())

    def _measure(trades, quotes, bars, _venue, day):
        # Per hour of one day: the quoted and effective spread, and each estimator on that venue's bars.
        hour = pl.col("ts").dt.hour().alias("hod")
        # Time-weighted: each quote counts for as long as it stood (to the next, at most 2 s). The
        # median sits at one tick in every hour on these markets; the mean sees how long the book is wider.
        stood = (pl.col("ts").shift(-1) - pl.col("ts")).dt.total_microseconds().clip(upper_bound=2_000_000).fill_null(0)
        q = (
            gr.liquidity.quoted(quotes.sort("ts"))
            .with_columns(stood.alias("w"))
            .group_by(hour)
            .agg(
                ((pl.col("spread_bps") * pl.col("w")).sum() / pl.col("w").sum()).alias("quoted"),
                pl.col("spread_bps").median().alias("quoted median"),
            )
        )
        e = gr.liquidity.effective(trades, quotes, horizons=("5s",)).group_by(hour).agg(
            pl.col("effective_bps").median().alias("effective"),
            pl.col("realized_bps_5s").mean().alias("realized 5s"),
            pl.col("impact_bps_5s").mean().alias("impact 5s"),
        )
        out = q.join(e, on="hod", how="full", coalesce=True)
        for name in gr.liquidity.ESTIMATORS:
            est = gr.liquidity.from_bars(bars, name, "1h").select(hour, pl.col("spread_bps").alias(name))
            out = out.join(est, on="hod", how="left")
        return out.with_columns(pl.lit(_venue).alias("venue"), pl.lit(day).alias("day"))

    def _bybit():
        try:
            days = _days(gr.reference.book(ticker.value, *EVER))
        except gr.Refused:
            return []
        frames = []
        for d in days:
            w = (f"{d}T00:00Z", f"{d}T23:59:59.999999Z")
            try:
                trades = gr.reference.trades(ticker.value, *w, venues="bybit-linear", rpi=False).collect()
            except gr.Refused:
                continue
            book = gr.reference.book(ticker.value, *w).collect()
            if trades.is_empty() or book.is_empty():
                continue
            frames.append(_measure(trades, book, gr.liquidity.bars(trades, "1m"), "bybit (1 s book)", d))
        return frames

    def _tape():
        try:
            days = _days(gr.market.quotes(ticker.value, *EVER))
        except gr.Refused:
            return []
        frames = []
        for d in days:
            w = (f"{d}T00:00Z", f"{d}T23:59:59.999999Z")
            trades, quotes = gr.market.trades(ticker.value, *w).collect(), gr.market.quotes(ticker.value, *w).collect()
            bars = gr.market.candles(ticker.value, "1m", *w).collect()
            # A partial capture day is not an hour profile.
            if trades.height < 1000 or quotes.is_empty() or bars.height < 1380:
                continue
            frames.append(_measure(trades, quotes, bars, "hyperliquid (tape)", d))
        return frames

    _frames = _bybit() + _tape()
    measured = pl.concat(_frames, how="diagonal_relaxed") if _frames else None
    return (measured,)


@app.cell
def _(alt, gr, measured, mo, pl, ticker):
    mo.stop(measured is None or measured.is_empty(), mo.md("No book or top of book held for this ticker."))
    cols = ["quoted", "quoted median", "effective", "realized 5s", "impact 5s", *gr.liquidity.ESTIMATORS]
    spread_profile = measured.group_by("venue", "hod").agg(*[pl.col(c).median() for c in cols], pl.col("day").n_unique().alias("days")).sort("venue", "hod")
    long = spread_profile.unpivot(index=["venue", "hod", "days"], on=["quoted", "effective", *gr.liquidity.ESTIMATORS], variable_name="measure", value_name="bps")
    _chart = (
        alt.Chart(long.filter(pl.col("bps") > 0))
        .mark_line(point=True)
        .encode(
            x=alt.X("hod:O", title="hour, UTC"),
            y=alt.Y("bps:Q", title="bps (log)", scale=alt.Scale(type="log")),
            color=alt.Color("measure:N", sort=["quoted", "effective", *gr.liquidity.ESTIMATORS]),
            strokeDash=alt.condition(alt.FieldOneOfPredicate("measure", ["quoted", "effective"]), alt.value([1, 0]), alt.value([4, 3])),
            tooltip=["venue:N", "hod:O", "measure:N", alt.Tooltip("bps:Q", format=".4f"), "days:Q"],
        )
        .properties(height=220, width=320)
        .facet(facet=alt.Facet("venue:N", title=None), columns=2)
    )
    costs = measured.group_by("venue").agg(
        pl.col("day").n_unique().alias("days"),
        *[pl.col(c).median().round(4).alias(f"{c} bps") for c in ["quoted", "quoted median", "effective", "realized 5s", "impact 5s"]],
    ).sort("venue")
    mo.vstack([mo.md(f"### {ticker.value}: measured (solid) and estimated (dashed), median by hour"), _chart, costs])
    return (spread_profile,)


@app.cell
def _(gr, mo, pl, spread_profile, ticker):
    _rows = []
    for _venue in spread_profile["venue"].unique().sort():
        p = spread_profile.filter(pl.col("venue") == _venue)
        _span = p["quoted"].max() / p["quoted"].min() if p["quoted"].min() else None
        for name in gr.liquidity.ESTIMATORS:
            both = p.select("quoted", "effective", name).drop_nulls()
            _rho = both.select(pl.corr(name, "quoted", method="spearman")).item() if both.height >= 12 else None
            flat = _span is not None and _span < 1.1
            verdict = "nothing to calibrate against" if flat else ("calibrates" if _rho is not None and _rho >= 0.6 else "does not calibrate")
            _rows.append({
                "venue": _venue, "estimator": name,
                "estimate ÷ quoted": round((both[name] / both["quoted"]).median(), 2) if both.height else None,
                "estimate ÷ effective": round((both[name] / both["effective"]).median(), 2) if both.height else None,
                "spearman ρ vs quoted": None if _rho is None else round(_rho, 2),
                "quoted max ÷ min": None if _span is None else round(_span, 2),
                "null hours": int(p[name].null_count()),
                "verdict": verdict,
            })  # fmt: skip
    calibration = pl.DataFrame(_rows)
    good = sorted(set(calibration.filter(pl.col("verdict") == "calibrates")["estimator"]))
    mo.vstack([
        mo.md(f"### {ticker.value}: does an estimate follow the measured spread through the day?"),
        calibration,
        mo.md(
            f"Calibrating for {ticker.value}: **{', '.join(good)}**." if good
            else f"**No estimator calibrates for {ticker.value}.** A level read from bars alone is not the spread here, and neither is its hour profile."
        ),
    ])  # fmt: skip
    return (good,)


@app.cell
def _(EVER, alt, gr, good, mo, pl, ticker):
    mo.stop(not good, mo.md("With no estimator calibrating, the spread's history over the years is not drawn: it would be a picture of something else."))

    def _years():
        try:
            klines = gr.reference.candles(ticker.value, *EVER).collect()
        except gr.Refused:
            return None
        out = []
        for name in good:
            est = gr.liquidity.from_bars(klines, name, "1h").drop_nulls("spread_bps")
            out.append(
                est.with_columns(pl.col("ts").dt.date().alias("day"))
                .with_columns((pl.col("spread_bps") / pl.col("spread_bps").mean().over("day")).alias("x"))
                .group_by(pl.col("ts").dt.year().alias("year"), pl.col("ts").dt.hour().alias("hod"))
                .agg(pl.col("x").median())
                .with_columns(pl.lit(name).alias("estimator"))
            )
        return pl.concat(out) if out else None

    _yr = _years()
    mo.stop(_yr is None, mo.md("No Binance 1m bars held for this ticker."))
    _chart = (
        alt.Chart(_yr)
        .mark_line(point=True)
        .encode(
            x=alt.X("hod:O", title="hour, UTC"),
            y=alt.Y("x:Q", title="÷ day mean", scale=alt.Scale(zero=False)),
            color=alt.Color("year:O", scale=alt.Scale(scheme="viridis")),
            tooltip=["estimator:N", "year:O", "hod:O", alt.Tooltip("x:Q", format=".2f")],
        )
        .properties(height=200, width=320)
        .facet(facet=alt.Facet("estimator:N", title=None), columns=2)
    )
    mo.vstack([
        mo.md(
            f"### {ticker.value} on Binance: the estimated spread's shape through the day, per year\n\n"
            "Read with three cautions. The calibration above was on **another venue** and on the days the "
            "tape holds, which are few. It is a **shape**: the levels read several times the quoted spread. "
            "And a range-based estimator rises with volatility inside the hour, so part of what it shows is "
            "the volatility's own day, which ② already drew."
        ),
        _chart,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## ⑩ What a size costs, by the hour

    The spread is one tick all day on these markets (⑤), so it cannot say
    when size is cheap; depth can. `gr.liquidity.cost_of_size` walks each
    side's cumulative depth:
    - the curve is anchored at zero and linear between Binance's bands;
    - the cost of an order is the notional-weighted mean distance it fills at;
    - nothing is extrapolated past ±5%.

    **What the assumption does.** Notional is spread evenly in price inside a
    band. A real book is denser near the touch, so for an order inside the
    first band (±0.2% since 2026-01-15, ±1% before) this **overstates** the
    cost. It also makes cost exactly proportional to size there: a $1M order
    costs ten times a $100k one, and the hour profile of cost is depth's hour
    profile turned upside down. Bybit's rebuilt book, whose points sit at the
    touch and at 2 bps, shows the near-touch cost on the days it is held.

    Each hour's **median book** (the median of each band over its 120
    snapshots) is walked: the typical book of the hour.
    """)
    return


@app.cell
def _(EVER, alt, gr, mo, pl, ticker):
    def _costs():
        try:
            d = (
                gr.reference.depth(ticker.value, *EVER)
                .group_by(pl.col("ts").dt.truncate("1h"), "band_pct")
                .agg(pl.col("notional").median(), pl.len().alias("n"))
                .filter(pl.col("n") >= 100)
                .drop("n")
                .collect()
            )
        except gr.Refused:
            return None
        return gr.liquidity.cost_of_size(d, [1e4, 1e5, 1e6]).with_columns(
            pl.col("ts").dt.hour().alias("hod"),
            pl.col("ts").dt.year().alias("year"),
            (pl.col("ts").dt.weekday() >= 6).alias("weekend"),
            pl.when(pl.col("ts") >= pl.datetime(2026, 1, 15, time_zone="UTC")).then(pl.lit("±0.2% band held")).otherwise(pl.lit("first band ±1%")).alias("era"),
            pl.format("${}k", (pl.col("size") / 1000).cast(pl.Int64)).alias("order"),
        )

    size_costs = _costs()
    mo.stop(size_costs is None, mo.md("No Binance depth held for this ticker."))
    _by_hour = size_costs.group_by("era", "order", "side", "hod").agg(pl.col("cost_bps").median()).sort("hod")
    _chart = (
        alt.Chart(_by_hour)
        .mark_line(point=True)
        .encode(
            x=alt.X("hod:O", title="hour, UTC"),
            y=alt.Y("cost_bps:Q", title="cost, bps (log)", scale=alt.Scale(type="log")),
            color=alt.Color("order:N", sort=["$10k", "$100k", "$1000k"]),
            strokeDash="side:N",
            tooltip=["era:N", "order:N", "side:N", "hod:O", alt.Tooltip("cost_bps:Q", format=".3f")],
        )
        .properties(height=200, width=320)
        .facet(facet=alt.Facet("era:N", title=None), columns=2)
    )
    _summary = (
        size_costs.group_by("era", "order", "size")
        .agg(
            pl.col("cost_bps").median().round(3).alias("median bps"),
            pl.col("beyond").mean().round(3).alias("share past ±5%"),
        )
        .sort("era", "size")
        .drop("size")
    )
    _hours = size_costs.filter(pl.col("size") == 1e6).group_by("hod").agg(pl.col("cost_bps").median()).sort("cost_bps")
    _weekend = size_costs.filter(pl.col("size") == 1e6).group_by("weekend").agg(pl.col("cost_bps").median().round(3).alias("$1M median bps")).sort("weekend")
    mo.vstack([
        mo.md(f"### {ticker.value} on Binance: the cost of a market order, by hour (solid: buy; dashed: sell)"),
        _chart, _summary,
        mo.md(
            f"$1M is cheapest at **{', '.join(str(h) for h in _hours.head(3)['hod'].to_list())} UTC** and dearest at "
            f"**{', '.join(str(h) for h in _hours.tail(3)['hod'].to_list())} UTC**, "
            f"{_hours['cost_bps'].max() / _hours['cost_bps'].min():.2f}× apart. Weekend against weekday:"
        ),
        _weekend,
    ])  # fmt: skip
    return (size_costs,)


@app.cell
def _(EVER, gr, mo, pl, size_costs, ticker):
    def _bybit():
        try:
            days = sorted(gr.reference.book(ticker.value, *EVER).select(pl.col("ts").dt.date().unique()).collect().to_series().to_list())
        except gr.Refused:
            return None
        out = []
        for d in days:
            book = gr.reference.book(ticker.value, f"{d}T00:00Z", f"{d}T23:59:59.999999Z").collect()
            if book.is_empty():
                continue
            hourly = (
                gr.liquidity.book_points(book)
                .with_columns(pl.col("ts").dt.truncate("1h"), pl.col("band_pct").round(4))
                .group_by("ts", "band_pct")
                .agg(pl.col("notional").median())
            )
            out.append(gr.liquidity.cost_of_size(hourly, [1e4, 1e5]))
        return pl.concat(out) if out else None

    near = _bybit()
    mo.stop(near is None, mo.md("No Bybit book held for this ticker."))
    _days = near["ts"].dt.date().unique().implode()
    _binance = size_costs.filter(pl.col("ts").dt.date().is_in(_days) & pl.col("size").is_in([1e4, 1e5]))
    compare = pl.concat([
        near.group_by("size").agg(pl.col("cost_bps").median().alias("cost bps"), pl.col("beyond").mean().alias("share beyond")).with_columns(pl.lit("bybit, near-touch points").alias("book")),
        _binance.group_by("size").agg(pl.col("cost_bps").median().alias("cost bps"), pl.col("beyond").mean().alias("share beyond")).with_columns(pl.lit("binance, bands").alias("book")),
    ]).sort("size", "book")  # fmt: skip
    mo.vstack([
        mo.md(
            f"### {ticker.value}: near the touch, on Bybit's book days ({near['ts'].dt.date().n_unique()} days)\n"
            "Bybit's points are the top size and the depth within 2 and 10 bps. Where a size fits there, its cost is read "
            "close to the touch; where it does not, it is `beyond`. Different venues, same days."
        ),
        compare,
    ])  # fmt: skip
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## ⑪ What the flow moves

    Depth (⑩) is what the book offers; the flow is what is taken. Kyle's λ is
    the price move per dollar of net aggressive buying. Here it is the OLS
    slope of each minute's return (bps) on that minute's signed notional
    ($M), from signed trades (`gr.liquidity.flow`, `kyle_lambda`). Amihud's
    ratio, |return| per $1M traded, is its cousin that needs no sign.

    **Read it as an association.** Flow and return in the same minute are
    jointly determined: λ says how they move together, as it is usually
    estimated, not what an order of mine would do. The iid t ignores the
    flow's autocorrelation, which is shown beside it.

    **Data.** Binance aggTrades and Bybit trades on the days both are held,
    OKX trades on those days when both of its files are (its day runs 16:00
    to 16:00 UTC), and the tape on its complete days. Each is in 1-minute buckets; a minute
    after an empty one has no return.
    """)
    return


@app.cell
def _(EVER, HELD, gr, mo, pl, ticker):
    def _flows():
        try:
            days = gr.reference.trades(ticker.value, *EVER, venues="binance-um").select(pl.col("ts").dt.date().unique()).collect().to_series().sort().to_list()
        except gr.Refused:
            days = []
        out = []
        for d in days:
            w = (f"{d}T00:00Z", f"{d}T23:59:59.999999Z")
            parts = []
            for venue in ("binance-um", "bybit-linear", "okx-swap"):
                try:
                    tr = gr.reference.trades(ticker.value, *w, venues=venue, rpi=False).select("venue", "ticker", "ts", "price", "size", "aggressor").collect()
                except gr.Refused:
                    tr = None
                # OKX's day runs 16:00 to 16:00 UTC: it joins only where both of its files reach this UTC day's ends.
                whole = tr is not None and tr.height and (venue != "okx-swap" or (tr["ts"].min().hour, tr["ts"].max().hour) == (0, 23))
                if whole:
                    parts.append(gr.liquidity.flow(tr, "1m"))
            if {p["venue"][0] for p in parts} >= {"binance-um", "bybit-linear"}:
                out.extend(parts)
        for d in ("2026-09-25", "2026-09-26"):
            try:
                tr = gr.market.trades(ticker.value, f"{d}T00:00Z", f"{d}T23:59:59.999Z").select("venue", "ticker", "ts", "price", "size", "aggressor").collect()
            except gr.Refused:
                continue
            if tr.height:
                out.append(gr.liquidity.flow(tr, "1m"))
        return pl.concat(out).with_columns(pl.col("ts").dt.hour().alias("hod"), pl.col("ts").dt.year().alias("year")) if out else None

    with mo.persistent_cache(name=f"flow-3v-{ticker.value}-{HELD}"):
        flows = _flows()
    mo.stop(flows is None, mo.md("No signed trades held on two venues for this ticker."))
    return (flows,)


@app.cell
def _(alt, gr, mo, pl, flows, ticker):
    _venue = gr.liquidity.kyle_lambda(flows, by="venue").join(gr.liquidity.amihud(flows, by="venue").drop("n"), on="venue")
    _days = flows.group_by("venue").agg(pl.col("ts").dt.date().n_unique().alias("days"))
    _auto = (
        flows.sort("venue", "ts")
        .with_columns(pl.col("imbalance").shift(1).over("venue").alias("_prev"), pl.col("return_bps").shift(-1).over("venue").alias("next_return_bps"))
        .group_by("venue")
        .agg(pl.corr("imbalance", "_prev").round(3).alias("imbalance lag-1 autocorr"))
    )
    # This minute's signed flow against the next minute's return: an in-sample association, not a strategy.
    _next = gr.liquidity.kyle_lambda(
        flows.sort("venue", "ts").with_columns(pl.col("return_bps").shift(-1).over("venue").alias("return_bps")), by="venue"
    ).select("venue", pl.col("lambda_bps_per_m").round(4).alias("next-minute slope"), pl.col("t").round(1).alias("next-minute t"))
    lambda_venue = _venue.join(_days, on="venue").join(_auto, on="venue").join(_next, on="venue").sort("venue")
    lambda_year = gr.liquidity.kyle_lambda(flows, by=["venue", "year"]).select("venue", "year", pl.col("lambda_bps_per_m").round(3), pl.col("r2").round(3), "n")
    _hour = gr.liquidity.kyle_lambda(flows, by=["venue", "hod"])
    _chart = (
        alt.Chart(_hour)
        .mark_line(point=True)
        .encode(x=alt.X("hod:O", title="hour, UTC"), y=alt.Y("lambda_bps_per_m:Q", title="λ, bps per $1M", scale=alt.Scale(zero=False)), color="venue:N",
                tooltip=["venue:N", "hod:O", alt.Tooltip("lambda_bps_per_m:Q", format=".3f"), alt.Tooltip("r2:Q", format=".2f"), "n:Q"])  # fmt: skip
        .properties(height=200, width="container")
    )
    mo.vstack([
        mo.md(f"### {ticker.value}: Kyle's λ and Amihud by venue (the tape's two days are few)"),
        lambda_venue, mo.md("By year:"), lambda_year, mo.md("By UTC hour:"), _chart,
    ])  # fmt: skip
    return lambda_venue, lambda_year


@app.cell
def _(mo):
    mo.md(r"""
    ## ⑭ How fast the book refills

    **Resiliency** is how fast the book returns to its normal shape after it
    is hit (Large 2007). Obizhaeva and Wang (2013) show that the best
    execution schedule depends on it, and not on the static spread or depth.

    **Shocks** are the seconds in the top 0.1% of a day's |signed flow| on
    Bybit (`gr.liquidity.shocks`), on the days its book is held. For each,
    the book side the flow took is followed for 60 s: its depth within 2 bps
    against its level just before.

    **Read only against a placebo.** Any second's 2 bps depth dips and
    refills within a minute. Random *quiet* seconds, as many per day as
    shocks and each given a random side, "recover" in about as long as
    shocks do. So the finding is the **excess**: the shock curve over the
    placebo curve, and how fast that excess fades (`gr.liquidity.placebo`).
    """)
    return


@app.cell
def _(EVER, HELD, gr, mo, pl, ticker):
    def _resilience():
        try:
            days = sorted(gr.reference.book(ticker.value, *EVER).select(pl.col("ts").dt.date().unique()).collect().to_series().to_list())
        except gr.Refused:
            return None
        rows, books, events = [], [], {"shock": [], "placebo": []}
        for d in days:
            w = (f"{d}T00:00Z", f"{d}T23:59:59.999999Z")
            try:
                tr = gr.reference.trades(ticker.value, *w, venues="bybit-linear", rpi=False).select("venue", "ticker", "ts", "price", "size", "aggressor").collect()
            except gr.Refused:
                continue
            book = gr.reference.book(ticker.value, *w).collect()
            if tr.is_empty() or book.is_empty():
                continue
            fl = gr.liquidity.flow(tr, "1s")
            hit = gr.liquidity.shocks(fl, share=0.001)
            calm = gr.liquidity.placebo(fl, hit)
            books.append(book)
            for kind, ev in (("shock", hit), ("placebo", calm)):
                events[kind].append(ev)
                rows.append(gr.liquidity.resilience(book, ev, horizon=60).with_columns(pl.lit(kind).alias("kind"), pl.lit(d).alias("day")))
        if not rows:
            return None
        # One pooled median curve per kind, over every event of every day.
        allbooks = pl.concat(books)
        curves = pl.concat([gr.liquidity.resilience_curve(allbooks, pl.concat(events[k]), horizon=60).with_columns(pl.lit(k).alias("kind")) for k in events])
        return pl.concat(rows), curves

    with mo.persistent_cache(name=f"resilience-median-{ticker.value}-{HELD}"):
        _r = _resilience()
    mo.stop(_r is None, mo.md("No Bybit book held for this ticker."))
    refill, refill_curves = _r
    return refill, refill_curves


@app.cell
def _(alt, mo, pl, refill, refill_curves, ticker):
    _curve = (
        refill_curves.pivot(on="kind", index="k", values="ratio").sort("k")
        .with_columns((pl.col("shock") / pl.col("placebo")).alias("excess"))
    )  # fmt: skip
    _gap0 = 1 - _curve.filter(pl.col("k") == 1)["excess"].item()
    _half = _curve.filter((pl.col("k") >= 1) & (1 - pl.col("excess") <= _gap0 / 2))
    half_life = None if _half.is_empty() else _half["k"][0]
    _long = _curve.unpivot(index="k", on=["shock", "placebo", "excess"], variable_name="curve", value_name="ratio")
    _chart = (
        alt.Chart(_long)
        .mark_line()
        .encode(x=alt.X("k:Q", title="seconds after"), y=alt.Y("ratio:Q", title="2 bps depth ÷ its level before"), color="curve:N",
                tooltip=["curve:N", "k:Q", alt.Tooltip("ratio:Q", format=".3f")])  # fmt: skip
        .properties(height=200, width="container")
    )
    _table = (
        refill.with_columns((pl.col("day") < pl.date(2025, 8, 21)).alias("ob500"))
        .group_by("kind", "ob500")
        .agg(pl.len().alias("events"), pl.col("depth_min_ratio").median().round(3).alias("median low point"),
             pl.col("recovery_s").median().alias("median recovery s"), pl.col("recovery_s").is_null().mean().round(3).alias("not back in 60 s"),
             pl.col("spread_peak_ratio").median().round(2).alias("spread peak ÷ before"))  # fmt: skip
        .sort("ob500", "kind")
    )
    _hour = (
        refill.with_columns(pl.col("ts").dt.hour().alias("hod"))
        .group_by("kind", "hod").agg(pl.col("recovery_s").is_null().mean().alias("not back"))
        .pivot(on="kind", index="hod", values="not back").sort("hod")
        .with_columns((pl.col("shock") - pl.col("placebo")).round(3).alias("excess unrecovered"))
    )  # fmt: skip
    mo.vstack([
        mo.md(f"### {ticker.value} on Bybit: the book side a shock took, second by second, against quiet seconds"),
        _chart,
        mo.md(f"The shock's excess depletion at 1 s is **{_gap0:.0%}** of the placebo's depth; it halves by **{half_life if half_life is not None else '> 60'} s**."),
        _table, mo.md("By UTC hour, the share not back within 60 s, shocks less placebo:"), _hour,
    ])  # fmt: skip
    return (half_life,)


@app.cell
def _(mo):
    mo.md(r"""
    ## ⑱ Depth and volatility

    How much does depth give way when volatility rises, and which moves first?
    Hourly, on Binance: ±1% depth, and realized variance from 1-minute
    returns (`gr.timeseries.log_elasticity`, `lagged_correlation`).

    The day's shape is taken out first (hour-of-day mean and weekday effect,
    in logs). Otherwise the shared rhythm (volatility up at 14 UTC while depth
    dips) would be all the correlation said. **Across the day** (the 24
    hour-of-day means, unadjusted) is set beside **within deviations**
    (adjusted). The lead-lag is on levels and on first differences, because
    depth is persistent (⑯). A positive lag means depth moves first.
    """)
    return


@app.cell
def _(EVER, flows, gr, mo, pl, ticker):
    def _series():
        try:
            k = gr.reference.candles(ticker.value, "2023-01-01T00:00Z", EVER[1], venues="binance-um").collect()
            depth = (
                gr.reference.depth(ticker.value, "2023-01-01T00:00Z", EVER[1], venues="binance-um")
                .filter(pl.col("band_pct").abs() == 1.0).group_by("ts").agg(pl.col("notional").sum())
                .group_by(pl.col("ts").dt.truncate("1h")).agg(pl.col("notional").median().alias("depth"), pl.len().alias("n"))
                .filter(pl.col("n") >= 100).drop("n").collect()
            )  # fmt: skip
        except gr.Refused:
            return None
        r = gr.timeseries.returns(k, kind="log")
        rv = (
            r.group_by(pl.col("ts").dt.truncate("1h")).agg((pl.col("return") ** 2).sum().alias("rv"), pl.col("return").is_not_null().sum().alias("n"))
            .filter(pl.col("n") >= 58).drop("n")
        )  # fmt: skip
        return depth.join(rv, on="ts").filter((pl.col("rv") > 0) & (pl.col("depth") > 0)).sort("ts")

    dv = _series()
    mo.stop(dv is None, mo.md("No Binance depth or bars held for this ticker."))
    _within = gr.timeseries.log_elasticity(dv, "depth", "rv")
    _prof = dv.group_by(pl.col("ts").dt.hour().alias("hod")).agg(pl.col("depth").mean(), pl.col("rv").mean())
    _across = _prof.select(pl.cov(pl.col("depth").log(), pl.col("rv").log()) / pl.col("rv").log().var()).item()
    _years = pl.DataFrame([
        {"year": y, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in gr.timeseries.log_elasticity(g, "depth", "rv").items()}}
        for (y,), g in dv.group_by(pl.col("ts").dt.year()) if g.height > 500
    ]).sort("year")  # fmt: skip
    _lags = list(range(-12, 13))
    _lev = gr.timeseries.lagged_correlation(dv, "depth", "rv", _lags).rename({"corr": "levels"})
    _diff_frame = dv.with_columns(pl.col("depth").log().diff().exp().alias("depth"), pl.col("rv").log().diff().exp().alias("rv")).drop_nulls()
    _dif = gr.timeseries.lagged_correlation(_diff_frame, "depth", "rv", _lags, seasonal=False).rename({"corr": "differences"})
    lead_lag_dv = _lev.join(_dif, on="lag").with_columns(pl.col("levels").round(3), pl.col("differences").round(3))

    def _daily_lambda():
        if flows is None:
            return None
        lam = gr.liquidity.kyle_lambda(flows.filter(pl.col("venue") == "binance-um").with_columns(pl.col("ts").dt.date().alias("day")), by="day").filter(pl.col("lambda_bps_per_m") > 0)
        rvd = dv.group_by(pl.col("ts").dt.date().alias("day")).agg(pl.col("rv").sum())
        j = lam.join(rvd, on="day")
        if j.height < 20:
            return None
        return j.select(pl.cov(pl.col("lambda_bps_per_m").log(), pl.col("rv").log()) / pl.col("rv").log().var()).item(), j.height

    _lam = _daily_lambda()
    mo.vstack([
        mo.md(
            f"### {ticker.value}: how depth moves with volatility\n"
            f"- **Across the day** (24 hourly means): elasticity of depth to realized variance **{_across:+.3f}**.\n"
            f"- **Within deviations** (the day's shape off): **{_within['beta']:+.3f}** (se {_within['se']:.3f}, R² {_within['r2']:.2f}, n {_within['n']:,}).\n"
            + (f"- **Daily λ** (Binance, {_lam[1]} trade days): elasticity to daily realized variance **{_lam[0]:+.3f}**." if _lam else "")
        ),
        mo.md("By year:"), _years,
        mo.md("Lagged correlation, depth at t with variance at t + lag (positive lag: depth first):"), lead_lag_dv,
    ])  # fmt: skip
    return (lead_lag_dv,)


if __name__ == "__main__":
    app.run()
