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
def _(mo):
    mo.md(r"""
    # When is the market liquid? (I: the liquidity week)

    At which hour of the week is BTC, ETH, HYPE or gold most liquid, on which
    venue, and has that changed since 2023?

    The record alone cannot say. It holds one venue (Hyperliquid), 1h bars
    from 2026-03 and a week of top of book. So this study also reads **other
    venues' published archives**, fetched by `galata-fetch` into the reference
    store (settled point 8, as restated 2026-09-27):

    | source | what | here |
    |---|---|---|
    | Binance USDⓈ-M `bookDepth` | cumulative depth at ±0.2 … ±5%, every 30 s | every day from 2023-01-01 |
    | Binance 1m klines | volume and trade count | every day |
    | Bybit linear trades | every execution, signed | every third day (`every:3`, which turns through the week) |
    | Hyperliquid, the tape | 1h bars | from 2026-03-01 |

    **Clock.** Everything is on UTC. US daylight saving moves the New York open
    between 13:30 and 14:30 UTC, which smears a UTC profile across two hours.
    The next part of the study re-draws these profiles on the New York clock.

    **What the literature says to expect.** These expectations are written
    down before the cells below test them.

    1. **Volume and trade count** peak at 13:30–17:00 UTC, the US open and the
       "tea time" peak at 16–17 UTC (Brauneis, Mestel and Theissen 2025, online 2024, 38
       exchanges). They trough at 21:00–00:00 UTC and have smaller humps at the
       Asia and Europe opens.
    2. **Depth is not volume.** Depth near the touch is best around
       10:00–13:00 UTC and worst after the US close. Binance ±10 bps depth was
       about 1.4× higher at 11:00 UTC than at 21:00 UTC in Jul–Aug 2025
       (Amberdata). Talos finds execution cheapest at 11–13 UTC across 50+
       venues.
    3. **Weekends are thinner in volume** (13–16% of the week's BTC volume in
       2024; Kaiko), but depth need not be: Amberdata saw Binance weekend depth
       slightly *above* weekday depth.
    4. **Clock-tick spikes** at the top of the hour, from algorithmic and TWAP
       execution (Wątorek et al. 2023).
    5. **The profile has become more US-centred over time.** The US-hours share
       of BTC daily variance rose from 38% (2016–18) to 51% (2022–25).
    """)
    return


@app.cell
def _(gr, mo, pl):
    EVER = ("2019-01-01T00:00Z", "2100-01-01T00:00Z")
    cover = gr.reference.coverage()
    tape = (
        gr.market.candles(None, "1h", *EVER)
        .group_by("venue", "ticker")
        .agg(pl.col("ts").min().dt.date().alias("first"), pl.col("ts").max().dt.date().alias("last"), pl.len().alias("bars"))
        .with_columns(pl.lit("candles 1h (tape)").alias("kind"))
        .collect()
    )
    held = pl.concat(
        [
            cover.select("kind", "venue", "ticker", "first", "last"),
            tape.select("kind", "venue", "ticker", "first", "last"),
        ]
    ).with_columns(pl.concat_str("kind", "venue", "ticker", separator=" · ").alias("series"))
    mo.vstack([mo.md("## ① What is held"), cover, gr.frontier()])
    return EVER, cover, held


@app.cell
def _(alt, held, mo):
    span = (
        alt.Chart(held)
        .mark_bar(height=8)
        .encode(
            x=alt.X("first:T", title=None),
            x2="last:T",
            y=alt.Y("series:N", title=None, sort=None),
            color=alt.Color("venue:N"),
            tooltip=["series:N", "first:T", "last:T"],
        )
        .properties(height=22 * held.height, width="container")
    )
    mo.vstack(
        [
            span,
            mo.md(
                "Sampled series (`every:3`) are drawn from their first to their last day, but hold one day in three. "
                "`coverage()` counts the rest as `days_missing`, never fetched, as distinct from `days_absent`: days the archive did not have."
            ),
        ]
    )
    return


@app.cell
def _(EVER, gr, mo, pl):
    def _bands():
        try:
            lf = gr.reference.depth(None, *EVER)
        except gr.Refused:
            return None
        return (
            lf.group_by("ticker", pl.col("band_pct").abs().alias("band"))
            .agg(pl.col("ts").min().dt.date().alias("first"), pl.col("ts").max().dt.date().alias("last"))
            .sort("ticker", "band")
            .collect()
        )

    depth_bands = _bands()
    mo.vstack(
        [
            mo.md(
                "**Binance's depth bands changed.** Its early files hold ±1 … ±5% only. The ±0.2% band begins later "
                "(`first` below), so a profile over years uses ±1%, and ±0.2% is read only over the days it exists."
            ),
            depth_bands if depth_bands is not None else mo.md("No depth fetched."),
        ]
    )
    return


@app.cell
def _(cover, mo):
    tickers = sorted(set(cover["ticker"]) | {"BTC"})
    ticker = mo.ui.dropdown(tickers, value="BTC", label="ticker")
    norm = mo.ui.dropdown(["day", "week"], value="day", label="normalise by the")
    clock = mo.ui.dropdown({"UTC": "UTC", "New York": "America/New_York", "London": "Europe/London"}, value="UTC", label="hours on the clock of")
    reps = mo.ui.slider(50, 400, value=200, step=50, label="bootstrap draws")
    mo.hstack([ticker, norm, clock, reps])
    return clock, norm, reps, ticker


@app.cell
def _(EVER, gr, pl, ticker):
    def _hourly_depth():
        try:
            lf = gr.reference.depth(ticker.value, *EVER)
        except gr.Refused:
            return None
        # Bid and ask together: the notional within ±x% of the price, per snapshot, then the hour's median snapshot.
        return (
            lf.with_columns(pl.col("band_pct").abs().alias("band"))
            .filter(pl.col("band").is_in([0.2, 1.0]))
            .group_by("venue", "ticker", "ts", "band")
            .agg(pl.col("notional").sum())
            .group_by("venue", "ticker", "band", pl.col("ts").dt.truncate("1h").alias("hour"))
            .agg(pl.col("notional").median().alias("value"), pl.len().alias("n"))
            # A full hour holds 120 snapshots; take an hour only when most of it is there.
            .filter(pl.col("n") >= 100)
            .select(
                "venue", "ticker", "hour",
                pl.format("depth ±{}%", pl.col("band").round(1)).alias("measure"),
                "value",
            )
            .collect()
        )  # fmt: skip

    def _hourly_klines():
        try:
            lf = gr.reference.candles(ticker.value, *EVER)
        except gr.Refused:
            return None
        return _flow(lf.with_columns((pl.col("volume") * pl.col("close")).alias("usd")), "trade_count")

    def _hourly_trades():
        # OKX's file runs 16:00 to 16:00 UTC, so the UTC days either side of a fetched one are partial: only whole days count.
        try:
            lf = gr.reference.trades(ticker.value, *EVER, venues=["bybit-linear", "okx-swap"], rpi=False)
        except gr.Refused:
            return None
        return (
            lf.group_by("venue", "ticker", pl.col("ts").dt.truncate("1h").alias("hour"))
            .agg((pl.col("price") * pl.col("size")).sum().alias("volume $"), pl.len().cast(pl.Float64).alias("trades"))
            .filter(pl.len().over("venue", pl.col("hour").dt.date()) == 24)
            .unpivot(index=["venue", "ticker", "hour"], variable_name="measure", value_name="value")
            .collect()
        )

    def _hourly_tape():
        try:
            lf = gr.market.candles(ticker.value, "1h", *EVER).with_columns((pl.col("volume") * pl.col("close")).alias("usd"))
        except gr.Refused:
            return None
        return _flow(lf.rename({"ts": "hour"}), "trade_count", truncate=False)

    def _flow(lf, count, truncate=True):
        hour = pl.col("ts").dt.truncate("1h").alias("hour") if truncate else pl.col("hour")
        return (
            lf.group_by("venue", "ticker", hour)
            .agg(pl.col("usd").sum().alias("volume $"), pl.col(count).sum().cast(pl.Float64).alias("trades"))
            .unpivot(index=["venue", "ticker", "hour"], variable_name="measure", value_name="value")
            .collect()
        )

    parts = [p for p in (_hourly_depth(), _hourly_klines(), _hourly_trades(), _hourly_tape()) if p is not None and p.height]
    hourly = pl.concat(parts).with_columns(pl.col("hour").dt.date().alias("day")) if parts else None
    return (hourly,)


@app.cell
def _(clock, hourly, mo, norm, pl):
    mo.stop(hourly is None, mo.md("Nothing held for this ticker: fetch it with `galata-fetch`."))

    def normalised(frame, by):
        # Each value over the mean of its own day (or ISO week), so a busy
        # regime does not outweigh a quiet one. Only complete days or weeks count.
        unit = pl.col("day") if by == "day" else pl.col("hour").dt.truncate("1w").alias("unit")
        full = 24 if by == "day" else 168
        return (
            frame.with_columns(unit.alias("unit"))
            .with_columns(pl.len().over("venue", "measure", "unit").alias("held"), pl.col("value").mean().over("venue", "measure", "unit").alias("mean"))
            .filter((pl.col("held") == full) & (pl.col("mean") > 0))
            .with_columns(
                (pl.col("value") / pl.col("mean")).alias("x"),
                # The cell of the week on the chosen clock; the day it is normalised by stays the UTC day.
                pl.col("hour").dt.convert_time_zone(clock.value).dt.weekday().alias("weekday"),
                pl.col("hour").dt.convert_time_zone(clock.value).dt.hour().alias("hod"),
                pl.col("hour").dt.truncate("1w").alias("week"),
                pl.col("hour").dt.year().alias("year"),
            )
            .drop("held", "mean")
        )

    x = normalised(hourly, norm.value)
    mo.stop(x.is_empty(), mo.md(f"No complete {norm.value} for this ticker: a sampled series holds no complete week."))
    return (x,)


@app.cell
def _(gr, pl, random, reps):
    def bands(frame, keys, draws, block=4.0, seed=20260927):
        """The mean of `x` per cell, and a 90% band from a stationary bootstrap over weeks.

        Weeks, not hours, are resampled, in blocks of mean length `block`, so
        the band carries the dependence between days of the same fortnight.
        """
        weekly = frame.group_by("venue", "measure", "week", *keys).agg(pl.col("x").mean())
        weeks = sorted(weekly["week"].unique())
        rng = random.Random(seed)
        out = []
        for r in range(draws):
            picked = gr.timeseries.stationary_bootstrap_indices(len(weeks), block, rng)
            counts = pl.DataFrame({"week": [weeks[i] for i in picked]}).group_by("week").len()
            out.append(
                weekly.join(counts, on="week")
                .group_by("venue", "measure", *keys)
                .agg(((pl.col("x") * pl.col("len")).sum() / pl.col("len").sum()).alias("x"))
                .with_columns(pl.lit(r).alias("draw"))
            )
        spread = pl.concat(out).group_by("venue", "measure", *keys).agg(
            pl.col("x").quantile(0.05).alias("lo"), pl.col("x").quantile(0.95).alias("hi")
        )
        point = weekly.group_by("venue", "measure", *keys).agg(pl.col("x").mean().alias("mean"), pl.col("week").n_unique().alias("weeks"))
        return point.join(spread, on=["venue", "measure", *keys]).sort("venue", "measure", *keys)

    draws = reps.value
    return bands, draws


@app.cell
def _(mo, x):
    measures = sorted(x["measure"].unique())
    measure = mo.ui.dropdown(measures, value="depth ±1.0%" if "depth ±1.0%" in measures else measures[0], label="measure")
    mo.vstack([mo.md("## ② The liquidity week"), measure])
    return (measure,)


@app.cell
def _(alt, bands, draws, measure, mo, norm, pl, ticker, x):
    week = bands(x.filter(pl.col("measure") == measure.value), ["weekday", "hod"], draws)
    DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    shown = week.with_columns(pl.col("weekday").map_elements(lambda d: DAYS[d - 1], return_dtype=pl.String).alias("day"))
    heat = (
        alt.Chart(shown)
        .mark_rect()
        .encode(
            x=alt.X("hod:O", title="hour, UTC"),
            y=alt.Y("day:N", sort=DAYS, title=None),
            color=alt.Color("mean:Q", title=f"÷ {norm.value} mean", scale=alt.Scale(scheme="blueorange", domainMid=1)),
            row=alt.Row("venue:N", title=None),
            tooltip=["venue:N", "day:N", "hod:O", alt.Tooltip("mean:Q", format=".2f"), alt.Tooltip("lo:Q", format=".2f"),
                     alt.Tooltip("hi:Q", format=".2f"), "weeks:Q"],
        )  # fmt: skip
        .properties(height=150, width="container")
    )
    mo.vstack(
        [
            mo.md(
                f"### {ticker.value}: {measure.value}, each hour over its {norm.value}'s mean\n"
                f"Blue is below the {norm.value}'s mean, orange above it. Hover a cell for its 90% band ({draws} draws over weeks)."
            ),
            heat,
        ]
    )
    return (DAYS,)


@app.cell
def _(alt, bands, draws, mo, norm, pl, ticker, x):
    day = bands(x, ["hod"], draws)
    _base = alt.Chart(day).encode(x=alt.X("hod:O", title="hour, UTC"), color=alt.Color("venue:N"))
    profile = (
        (_base.mark_area(opacity=0.2).encode(y="lo:Q", y2="hi:Q") + _base.mark_line(point=True).encode(y=alt.Y("mean:Q", title=f"÷ {norm.value} mean", scale=alt.Scale(zero=False))))
        .properties(height=180, width=280)
        .facet(facet=alt.Facet("measure:N", title=None), columns=2)
        .resolve_scale(y="independent")
    )
    peaks = (
        day.sort("mean", descending=True)
        .group_by("venue", "measure", maintain_order=True)
        .agg(pl.col("hod").first().alias("best hour"), pl.col("hod").last().alias("worst hour"),
             (pl.col("mean").max() / pl.col("mean").min()).round(2).alias("best ÷ worst"))
        .sort("measure", "venue")
    )  # fmt: skip
    mo.vstack([mo.md(f"### {ticker.value}: the hour of the day, every venue and measure, with 90% bands"), profile, peaks])
    return (day,)


@app.cell
def _(DAYS, mo, pl, x):
    weekend = (
        x.with_columns((pl.col("weekday") >= 6).alias("weekend"))
        .group_by("venue", "measure", "weekend")
        .agg(pl.col("x").mean())
        .pivot(on="weekend", index=["venue", "measure"], values="x")
        .rename({"true": "weekend", "false": "weekday"}, strict=False)
    )
    note = (
        "Normalised by the **day**, every day averages 1, so the weekend and the weekday cannot differ in level. "
        "Switch to **week** to compare them."
        if x["unit"].dtype == pl.Date
        else "Normalised by the **week**: a weekend hour above a weekday hour is more liquid in level, not only in shape."
    )
    mo.vstack([mo.md(f"### Weekend and weekday ({', '.join(DAYS[5:])} against the rest)"), weekend.sort("measure", "venue"), mo.md(note)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## ③ Has the day changed shape?

    One hour-of-day profile per calendar year, each normalised by its own days.
    The table gives the Spearman rank correlation of each year's 24 hours with
    the next year's. A value near 1 means the same hours lead. **Binance's
    zero-fee window, 2022-07-08 → 2023-03-22**, inflated its volume with churn
    (Kaiko). It overlaps the first quarter of the depth sample, so 2023's
    volume profile is read with that in mind. The depth profile is not
    affected in the same way.
    """)
    return


@app.cell
def _(alt, measure, mo, pl, ticker, x):
    yearly = (
        x.filter(pl.col("measure") == measure.value)
        .group_by("venue", "year", "hod")
        .agg(pl.col("x").mean().alias("mean"), pl.col("day").n_unique().alias("days"))
        .sort("venue", "year", "hod")
    )
    lines = (
        alt.Chart(yearly)
        .mark_line(point=True)
        .encode(
            x=alt.X("hod:O", title="hour, UTC"),
            y=alt.Y("mean:Q", title="÷ mean", scale=alt.Scale(zero=False)),
            color=alt.Color("year:O", scale=alt.Scale(scheme="viridis")),
            tooltip=["year:O", "hod:O", alt.Tooltip("mean:Q", format=".2f"), "days:Q"],
        )
        .properties(height=200, width=300)
        .facet(facet=alt.Facet("venue:N", title=None), columns=2)
    )
    _wide = yearly.pivot(on="year", index=["venue", "hod"], values="mean").sort("venue", "hod")
    years = sorted(c for c in _wide.columns if c not in ("venue", "hod"))
    rows = []
    for venue in _wide["venue"].unique().sort():
        w = _wide.filter(pl.col("venue") == venue)
        for a, b in zip(years, years[1:], strict=False):
            pair = w.select(a, b).drop_nulls()
            if pair.height == 24:
                rho = pair.select(pl.corr(a, b, method="spearman")).item()
                rows.append({"venue": venue, "from": a, "to": b, "spearman ρ": round(rho, 3)})
    rho = pl.DataFrame(rows) if rows else pl.DataFrame(schema={"venue": pl.String, "from": pl.String, "to": pl.String, "spearman ρ": pl.Float64})
    mo.vstack([mo.md(f"### {ticker.value}: {measure.value} by year"), lines, rho])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## ④ Hyperliquid alongside

    The tape's 1h bars start on 2026-03-01. Here every venue is cut to the days
    Hyperliquid holds, so the three are compared on the same calendar. Does
    Hyperliquid's day follow Binance's and Bybit's, or keep its own hours?
    """)
    return


@app.cell
def _(alt, bands, draws, mo, pl, ticker, x):
    hl_days = x.filter(pl.col("venue") == "hyperliquid")["day"].unique()
    same = x.filter(pl.col("day").is_in(hl_days.implode()) & pl.col("measure").is_in(["volume $", "trades"]))
    mo.stop(hl_days.is_empty(), mo.md("Hyperliquid holds no complete day for this ticker."))
    together = bands(same, ["hod"], draws)
    _base = alt.Chart(together).encode(x=alt.X("hod:O", title="hour, UTC"), color=alt.Color("venue:N"))
    chart = (
        (_base.mark_area(opacity=0.15).encode(y="lo:Q", y2="hi:Q") + _base.mark_line(point=True).encode(y=alt.Y("mean:Q", title="÷ day mean", scale=alt.Scale(zero=False))))
        .properties(height=200, width=300)
        .facet(facet=alt.Facet("measure:N", title=None), columns=2)
    )
    _wide = together.filter(pl.col("measure") == "volume $").pivot(on="venue", index="hod", values="mean").sort("hod")
    others = [c for c in _wide.columns if c not in ("hod", "hyperliquid")]
    agree = [
        {"against": c, "spearman ρ (volume $, 24 hours)": round(_wide.select(pl.corr("hyperliquid", c, method="spearman")).item(), 3)}
        for c in others
        if "hyperliquid" in _wide.columns and _wide.select(c).null_count().item() == 0
    ]
    mo.vstack([mo.md(f"### {ticker.value}, {hl_days.len()} days in common"), chart, pl.DataFrame(agree) if agree else mo.md("No other venue holds those days.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Where the study goes from here

    The study continues in four more notebooks. Each runs on its own, and the
    sections keep their numbers:

    - `liquidity_costs.py`: ⑤ the spread, ⑩ what a size costs, ⑪ what the
      flow moves, ⑭ how fast the book refills, ⑱ depth and volatility.
    - `liquidity_clock.py`: ⑥ whose clock, ⑦ where the price jumps, ⑧ around
      a release, ⑫ the weekend reopen.
    - `liquidity_venues.py`: ⑨ who moves first, ⑬ whose price is the price.
    - `liquidity_stress.py`: ⑮ when it breaks, ⑯ tomorrow's liquidity, ⑰ when
      to work an order.

    **Not yet said:**
    - funding across venues, which waits on datawatch walking Hyperliquid's
      settled funding;
    - Hyperliquid on more than two tape days, which waits on the archive's
      missing days;
    - release surprises, which need a consensus source.
    """)
    return


if __name__ == "__main__":
    app.run()
