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
    # Ticks

    Trades counted once per execution (the first receipt of each
    `trade_id`), in the venue's order. Quotes are the book's top as received.
    `ts` is the venue's time, which Hyperliquid stamps to the millisecond.
    """)
    return


@app.cell
def _(UTC, datetime, gr, mo):
    EVER = datetime(2020, 1, 1, tzinfo=UTC), datetime(2100, 1, 1, tzinfo=UTC)
    frontier = gr.frontier()
    held = sorted(gr.market.trades(None, *EVER).select("ticker").unique().collect()["ticker"])
    ticker = mo.ui.dropdown(held, value="BTC" if "BTC" in held else held[0], label="ticker")
    mo.vstack([frontier, ticker])
    return EVER, ticker


@app.cell
def _(EVER, gr, pl, ticker):
    trades = gr.market.trades(ticker.value, *EVER).collect()
    busiest = (
        trades.group_by(pl.col("ts").dt.truncate("1m").alias("minute"))
        .agg(pl.len().alias("trades"))
        .sort("trades", descending=True)
        .head(1)
    )
    return busiest, trades


@app.cell
def _(busiest, mo, ticker):
    minute = busiest["minute"].item()
    mo.md(f"## The busiest minute for {ticker.value}: {minute:%Y-%m-%d %H:%M} UTC, {busiest['trades'].item():,} executions")
    return (minute,)


@app.cell
def _(alt, gr, minute, mo, pl, ticker, timedelta, trades):
    window = minute, minute + timedelta(minutes=1)
    book = gr.market.quotes(ticker.value, *window).collect()
    prints = trades.filter(pl.col("ts").is_between(*window, closed="left"))
    sides = book.unpivot(index="ts", on=["bid_px", "ask_px"], variable_name="side", value_name="px")
    top = (
        alt.Chart(sides)
        .mark_line(interpolate="step-after", strokeWidth=1)
        .encode(x="ts:T", y=alt.Y("px:Q", scale=alt.Scale(zero=False)), color="side:N")
    )
    marks = (
        alt.Chart(prints)
        .mark_circle(opacity=0.5)
        .encode(x="ts:T", y="price:Q", size=alt.Size("size:Q", legend=None), color=alt.Color("aggressor:N"))
    )
    mo.vstack([(top + marks).properties(height=320, width="container"), mo.md(f"{book.height:,} quotes and {prints.height:,} trades in the minute.")])
    return


@app.cell
def _(gr, mo, pl):
    # Replays: raw rows in the tape over executions after the dedupe.
    raw = pl.scan_parquet(gr._root.root() / "tape" / "kind=trades" / "**" / "*.parquet", hive_partitioning=True)
    replay = (
        raw.group_by("date")
        .agg(pl.len().alias("rows"), pl.struct("venue", "ticker", "trade_id").n_unique().alias("executions"))
        .with_columns(((pl.col("rows") / pl.col("executions") - 1) * 100).round(2).alias("replayed_pct"))
        .sort("date")
        .collect()
    )
    mo.vstack([mo.md("## Replay share per day"), mo.md("Rows a naive read would count, over executions. The excess is the venue re-sending recent history on reconnect."), replay])
    return


@app.cell
def _(EVER, alt, gr, mo, pl, ticker):
    spread = (
        gr.market.quotes(ticker.value, *EVER)
        .with_columns(((pl.col("ask_px") - pl.col("bid_px")) / ((pl.col("ask_px") + pl.col("bid_px")) / 2) * 1e4).alias("spread_bps"))
        .group_by_dynamic("ts", every="5m")
        .agg(pl.col("spread_bps").mean().alias("mean_bps"), pl.col("spread_bps").max().alias("max_bps"))
        .collect()
    )
    chart = (
        alt.Chart(spread.unpivot(index="ts", on=["mean_bps", "max_bps"], variable_name="stat", value_name="bps"))
        .mark_line(strokeWidth=1)
        .encode(x="ts:T", y="bps:Q", color="stat:N")
        .properties(height=220, width="container")
    )
    mo.vstack([mo.md(f"## {ticker.value} spread, basis points of mid, per 5 minutes"), chart])
    return


if __name__ == "__main__":
    app.run()
