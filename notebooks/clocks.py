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
    # Two clocks

    Trades, quotes, candles and **settled** funding carry the venue's time,
    `ts`. The mark, oracle, mid, open interest, premium and **live** funding
    rate come in Hyperliquid's asset context, which carries no time at all,
    so they exist only as received, `recv_ts`. `join_recv` is the one way to
    put them side by side: each trade gets the last mark **received by** its
    `ts`, and the columns it brings say `_recv`.
    """)
    return


@app.cell
def _(UTC, datetime, gr, mo, pl):
    EVER = datetime(2020, 1, 1, tzinfo=UTC), datetime(2100, 1, 1, tzinfo=UTC)
    trades = gr.market.trades("BTC", *EVER).collect()
    busiest = trades.group_by(pl.col("ts").dt.truncate("1m").alias("minute")).len().sort("len", descending=True)["minute"][0]
    mo.md(f"## BTC, the busiest minute: {busiest:%Y-%m-%d %H:%M} UTC")
    return EVER, busiest


@app.cell
def _(alt, busiest, gr, mo, pl, timedelta):
    window = busiest, busiest + timedelta(minutes=1)
    joined = gr.join_recv(gr.market.trades("BTC", *window), gr.market.marks("BTC", window[0] - timedelta(seconds=10), window[1])).collect()
    lines = joined.unpivot(index="ts", on=["price", "mark_recv", "oracle_recv", "mid_recv"], variable_name="series", value_name="px")
    prices = (
        alt.Chart(lines)
        .mark_line(interpolate="step-after", strokeWidth=1)
        .encode(x="ts:T", y=alt.Y("px:Q", scale=alt.Scale(zero=False)), color="series:N")
        .properties(height=280, width="container")
    )
    stale = (joined["ts"] - joined["matched_recv_ts"]).dt.total_milliseconds()
    mo.vstack(
        [
            prices,
            mo.md(f"Every trade matched to a mark received at or before it: staleness median {stale.median():.0f} ms, max {stale.max():.0f} ms, over {joined.height:,} trades."),
        ]
    )
    return


@app.cell
def _(EVER, alt, gr, mo, pl):
    settled = gr.market.funding("BTC", *EVER).collect().select(pl.col("ts").alias("at"), "rate", pl.lit("settled (ts)").alias("series"))
    live = gr.market.funding_live("BTC", *EVER, collapse=True).collect().select(pl.col("recv_ts").alias("at"), "rate", pl.lit("live (recv_ts)").alias("series"))
    premium = (
        gr.market.marks("BTC", *EVER, collapse=True)
        .group_by_dynamic("recv_ts", every="1h")
        .agg(pl.col("premium").mean().alias("rate"))
        .collect()
        .select(pl.col("recv_ts").alias("at"), "rate", pl.lit("premium, hourly mean (recv_ts)").alias("series"))
    )
    rates = pl.concat([settled, live, premium])
    funding_chart = (
        alt.Chart(rates)
        .mark_line(interpolate="step-after", point=True, strokeWidth=1)
        .encode(x=alt.X("at:T", title="time (each series on its own clock)"), y="rate:Q", color="series:N")
        .properties(height=240, width="container")
    )
    mo.vstack(
        [
            mo.md("## BTC funding: settled, live, and the premium it comes from"),
            mo.md("At a neutral premium, Hyperliquid's hourly rate is its interest floor, 0.0000125 (0.01% per 8 h)."),
            funding_chart,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
