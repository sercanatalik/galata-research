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
    # Time-series momentum

    Long when the close is above its level `lookback` bars ago, short when
    below, rebalanced at every close and earning the next bar. Costs are the
    0.045% taker fee per unit of position change. **Modelled, gross of
    funding.**

    The one momentum backtest before this (vade-trader, vt/D-067) lost
    −6.74 modelled over 31 hours of testnet, "dominated by churn": it flipped
    constantly and paid the spread each time. Here the bars are daily and 4h,
    and the churn is visible as the gap between gross and net.
    """)
    return


@app.cell
def _(mo):
    interval = mo.ui.dropdown({"daily": "1d", "4 hours": "4h"}, value="daily", label="bars")
    interval
    return (interval,)


@app.cell
def _(gr, interval, pl, studies):
    EVER = ("2020-01-01T00:00Z", "2100-01-01T00:00Z")
    PER_YEAR = {"1d": 365, "4h": 2190}[interval.value]
    LOOKBACKS = [5, 10, 20, 40, 60, 90, 120]
    bars = gr.market.candles(["BTC", "ETH"], interval.value, *EVER)
    trials = studies.momentum(bars, LOOKBACKS)
    scores = (
        studies.summary(trials, PER_YEAR)
        .with_columns(pl.col("trial").str.extract(r"(\d+)").cast(pl.Int32).alias("lookback"))
        # A lookback of L bars over T bars holds about T / L non-overlapping periods.
        .with_columns((pl.col("periods") / pl.col("lookback")).floor().cast(pl.Int32).alias("independent_periods"))
        .with_columns((pl.col("total_gross") - pl.col("total_net")).alias("lost_to_fees"))
        .sort("ticker", "lookback")
    )
    return scores, trials


@app.cell
def _(alt, interval, mo, scores):
    bars_chart = (
        alt.Chart(scores)
        .mark_bar()
        .encode(
            x=alt.X("lookback:O"),
            y=alt.Y("sharpe_annual:Q", title="Sharpe, annual (net)"),
            color="ticker:N",
            xOffset="ticker:N",
            tooltip=["trial", "ticker", "sharpe_annual", "independent_periods", "total_net", "lost_to_fees"],
        )
        .properties(height=240, width="container")
    )
    mo.vstack(
        [
            mo.md(f"## Sharpe by lookback, {interval.value}"),
            bars_chart,
            mo.md("`independent_periods` is how many non-overlapping lookbacks the record holds. Where it is ten or so, a Sharpe is a handful of observations and says so."),
            scores.select("trial", "ticker", "periods", "independent_periods", "sharpe_annual", "total_net", "total_gross", "lost_to_fees"),
        ]
    )
    return


@app.cell
def _(mo, scores):
    lookback = mo.ui.dropdown([str(v) for v in scores["lookback"].unique().sort()], value="20", label="lookback")
    lookback
    return (lookback,)


@app.cell
def _(alt, lookback, mo, pl, trials):
    path = (
        trials.filter(pl.col("trial") == f"mom {lookback.value}")
        .drop_nulls("net")
        .with_columns(
            ((pl.col("net") + 1).cum_prod().over("ticker")).alias("net"),
            ((pl.col("gross") + 1).cum_prod().over("ticker")).alias("gross"),
        )
        .unpivot(index=["ts", "ticker"], on=["gross", "net"], variable_name="series", value_name="growth of 1")
    )
    curves = (
        alt.Chart(path)
        .mark_line(strokeWidth=1)
        .encode(x="ts:T", y=alt.Y("growth of 1:Q", scale=alt.Scale(type="log")), color="series:N", row="ticker:N")
        .properties(height=160, width="container")
    )
    mo.vstack([mo.md(f"## Momentum {lookback.value}: gross and net of fees"), curves])
    return


if __name__ == "__main__":
    app.run()
