import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import polars as pl

    import galata_research as gr
    from galata_research import studies

    return alt, gr, mo, pl, studies


@app.cell
def _(mo):
    mo.md("""
    # Moving-average crossover

    Long while the fast moving average of the close is above the slow one:
    `long_flat` is flat otherwise, and `long_short` is short. The position is
    decided at a bar's close and earns the **next** bar. Costs are the
    0.045% taker fee on every change of position. **Modelled, and gross of
    funding**: the record holds only days of settled funding, and longs paid
    about 11.6% a year at the interest floor.

    This is one family of trials. `deflated_sharpe.py` asks how much of its
    best result survives having tried them all.
    """)
    return


@app.cell
def _(mo):
    interval = mo.ui.dropdown({"daily": "1d", "4 hours": "4h"}, value="daily", label="bars")
    side = mo.ui.dropdown(["long_flat", "long_short"], value="long_flat", label="side")
    mo.hstack([interval, side], justify="start")
    return interval, side


@app.cell
def _(gr, interval, pl, studies):
    EVER = ("2020-01-01T00:00Z", "2100-01-01T00:00Z")
    PER_YEAR = {"1d": 365, "4h": 2190}[interval.value]
    FASTS, SLOWS = [5, 10, 20, 50], [20, 50, 100, 200]
    bars = gr.market.candles(["BTC", "ETH"], interval.value, *EVER)
    trials = studies.moving_average(bars, FASTS, SLOWS)
    scores = studies.summary(trials, PER_YEAR).with_columns(
        pl.col("trial").str.extract(r"ma (\d+)/").cast(pl.Int32).alias("fast"),
        pl.col("trial").str.extract(r"/(\d+) ").cast(pl.Int32).alias("slow"),
        pl.col("trial").str.extract(r" (\w+)$").alias("side"),
    )
    return PER_YEAR, scores, trials


@app.cell
def _(alt, mo, pl, scores, side):
    grid = scores.filter(pl.col("side") == side.value)
    heat = (
        alt.Chart(grid)
        .mark_rect()
        .encode(
            x=alt.X("slow:O"),
            y=alt.Y("fast:O", sort="descending"),
            color=alt.Color("sharpe_annual:Q", scale=alt.Scale(scheme="redblue", domainMid=0), title="Sharpe, annual"),
            column=alt.Column("ticker:N"),
            tooltip=["trial", "ticker", "sharpe_annual", "total_net", "total_gross", "periods"],
        )
        .properties(width=220, height=160)
    )
    mo.vstack(
        [
            mo.md(f"## The landscape, {side.value}: annualized Sharpe of net returns"),
            mo.md("Read the landscape, not its peak (galata-legacy's `permute-the-lookback`): a lone bright cell among dark ones is selection, not an edge."),
            heat,
            grid.select("trial", "ticker", "periods", "sharpe_annual", "total_net", "total_gross").sort("sharpe_annual", descending=True, nulls_last=True),
        ]
    )
    return


@app.cell
def _(alt, mo, pl, scores, trials):
    best = scores.sort("sharpe", descending=True, nulls_last=True).row(0, named=True)
    path = (
        trials.filter((pl.col("trial") == best["trial"]) & (pl.col("ticker") == best["ticker"]))
        .drop_nulls("net")
        .with_columns(
            ((pl.col("net") + 1).cum_prod()).alias("net"),
            ((pl.col("gross") + 1).cum_prod()).alias("gross"),
        )
        .unpivot(index="ts", on=["gross", "net"], variable_name="series", value_name="growth of 1")
    )
    curve = (
        alt.Chart(path)
        .mark_line(strokeWidth=1)
        .encode(x="ts:T", y=alt.Y("growth of 1:Q", scale=alt.Scale(type="log")), color="series:N")
        .properties(height=240, width="container")
    )
    mo.vstack(
        [
            mo.md(f"## The best trial across both sides: `{best['trial']}` on {best['ticker']}, Sharpe {best['sharpe_annual']:.2f} a year, before deflation"),
            curve,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
