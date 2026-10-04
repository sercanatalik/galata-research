import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from datetime import UTC, datetime

    import altair as alt
    import marimo as mo
    import polars as pl

    import galata_research as gr

    return UTC, alt, datetime, gr, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # Gaps

    Every window capture was not receiving, as the record published it: a
    restart names the window it missed, as `downtime` (a clean stop) or
    `crash_unflushed` (a kill). The bounds are **receipt** times. Masking
    marks rows inside a gap and drops none. A bar the venue restated after
    the gap (the walk, after every restart) is whole, and is not marked.
    """)
    return


@app.cell
def _(UTC, alt, datetime, gr, mo):
    EVER = datetime(2020, 1, 1, tzinfo=UTC), datetime(2100, 1, 1, tzinfo=UTC)
    gaps = gr.market.gaps(None, *EVER).collect()
    timeline = (
        alt.Chart(gaps)
        .mark_bar(height=8)
        .encode(
            x=alt.X("from_recv_ts:T", title="receipt time"),
            x2="to_recv_ts:T",
            y=alt.Y("ticker:N"),
            color="cause:N",
            row=alt.Row("series:N"),
            tooltip=["ticker", "series", "cause", "from_recv_ts", "to_recv_ts"],
        )
        .properties(height=90, width="container")
    )
    mo.vstack([mo.md(f"## {gaps.height} gaps"), timeline])
    return (EVER,)


@app.cell
def _(EVER, gr, mo, pl):
    loaders = {
        "trades": gr.market.trades(None, *EVER),
        "quotes": gr.market.quotes(None, *EVER),
        "candles": gr.market.candles(None, "1m", *EVER),
    }
    shares = pl.concat(
        [
            gr.mask_gaps(lf, name)
            .select(pl.lit(name).alias("dataset"), pl.len().alias("rows"), pl.col("in_gap").sum().alias("in_gap"))
            .collect()
            for name, lf in loaders.items()
        ]
    ).with_columns((pl.col("in_gap") / pl.col("rows") * 100).round(3).alias("pct"))
    mo.vstack(
        [
            mo.md("## What masking marks"),
            mo.md("Trades inside a gap are the venue's replay on reconnect: real, received after the outage, with incomplete surroundings. Candles (1m here) are restated by the walk, so none are marked."),
            shares,
        ]
    )
    return


@app.cell
def _(EVER, alt, gr, mo, pl):
    masked = gr.mask_gaps(gr.market.trades("BTC", *EVER), "trades").collect()
    per_minute = masked.group_by_dynamic("ts", every="10m", group_by="in_gap").agg(pl.len().alias("trades"))
    chart = (
        alt.Chart(per_minute)
        .mark_bar()
        .encode(x="ts:T", y="trades:Q", color=alt.Color("in_gap:N", scale=alt.Scale(domain=[False, True], range=["#9aa5b1", "#d64545"])))
        .properties(height=200, width="container")
    )
    mo.vstack([mo.md("## BTC trades per 10 minutes, marked"), chart])
    return


if __name__ == "__main__":
    app.run()
