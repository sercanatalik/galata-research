import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from datetime import UTC, datetime, timedelta

    import altair as alt
    import marimo as mo
    import polars as pl

    import galata_research as gr

    return UTC, alt, datetime, gr, mo, pl, timedelta


@app.cell
def _(mo):
    mo.md("""
    # Candles

    Bars from the record, one row per **closed** bar: re-fetched bars counted
    once (the latest receipt), closure decided by the record rather than
    `is_final`, and trade-less bars dropped. `ts` is the bar's open;
    `close_ts` is when it is known.
    """)
    return


@app.cell
def _(gr, mo):
    mo.vstack([mo.md("## How far the record goes"), gr.frontier()])
    return


@app.cell
def _(UTC, datetime, gr, mo):
    EVER = datetime(2020, 1, 1, tzinfo=UTC), datetime(2100, 1, 1, tzinfo=UTC)
    held = sorted(gr.market.candles(None, "1d", *EVER, traded_only=False).collect()["ticker"].unique())
    interval = mo.ui.dropdown(["1m", "1h", "4h", "1d"], value="4h", label="interval")
    tickers = mo.ui.multiselect(held, value=[t for t in ["BTC", "ETH"] if t in held], label="tickers")
    mo.hstack([interval, tickers], justify="start")
    return EVER, interval, tickers


@app.cell
def _(EVER, gr, interval, mo, tickers):
    mo.stop(not tickers.value, mo.md("Pick at least one ticker."))
    bars = gr.market.candles(tickers.value, interval.value, *EVER).collect()
    return (bars,)


@app.cell
def _(bars, gr, interval, mo, pl):
    width = pl.duration(microseconds=gr.market.INTERVALS[interval.value])
    coverage = (
        bars.group_by("ticker")
        .agg(
            pl.col("ts").min().alias("first"),
            pl.col("ts").max().alias("last"),
            pl.len().alias("bars"),
        )
        .with_columns(
            (((pl.col("last") - pl.col("first")) / width).cast(pl.Int64) + 1).alias("span"),
        )
        .with_columns((pl.col("span") - pl.col("bars")).alias("missing"))
        .sort("ticker")
    )
    mo.vstack(
        [
            mo.md(f"## Coverage at {interval.value}"),
            mo.md("`missing` counts grid positions with no traded, closed bar: a hole, a session closure, or a trade-less bar."),
            coverage,
        ]
    )
    return


@app.cell
def _(alt, bars, interval, mo):
    chart = (
        alt.Chart(bars.select("ticker", "close_ts", "close"))
        .mark_line(strokeWidth=1)
        .encode(x=alt.X("close_ts:T", title="close_ts"), y=alt.Y("close:Q", scale=alt.Scale(zero=False)))
        .properties(height=160, width="container")
        .facet(row="ticker:N")
        .resolve_scale(y="independent")
    )
    mo.vstack([mo.md(f"## Close, {interval.value}, placed at each bar's close"), chart])
    return


@app.cell
def _(mo):
    mo.md("""
    ## A bar is known at its close

    Move `as_of` through the last hours. A 1h bar appears only once
    `as_of` reaches its `close_ts`, never at its open.
    """)
    return


@app.cell
def _(EVER, gr, mo, tickers):
    ticker = (tickers.value or ["BTC"])[0]
    recent = gr.market.candles(ticker, "1h", *EVER).collect().tail(4)
    first = recent["ts"][0]
    as_of_minutes = mo.ui.slider(0, 4 * 60, step=15, value=90, label=f"as_of, minutes after {first:%Y-%m-%d %H:%M} UTC")
    as_of_minutes
    return as_of_minutes, first, ticker


@app.cell
def _(EVER, as_of_minutes, first, gr, mo, ticker, timedelta):
    as_of = first + timedelta(minutes=as_of_minutes.value)
    known = gr.market.candles(ticker, "1h", *EVER, as_of=as_of).collect().tail(3)
    mo.vstack([mo.md(f"**as_of = {as_of:%H:%M} UTC**: the last {ticker} 1h bars known then"), known.select("ts", "close_ts", "close")])
    return


if __name__ == "__main__":
    app.run()
