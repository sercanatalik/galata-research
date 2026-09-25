import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import time

    import altair as alt
    import marimo as mo
    import polars as pl

    import galata_research as gr
    from galata_research import stats, studies

    return alt, gr, mo, pl, stats, studies, time


@app.cell
def _(mo):
    mo.md(r"""
    # Does choosing on the past choose well?

    The Deflated Sharpe Ratio asks whether the best trial's Sharpe survives
    the number of trials. The **Probability of Backtest Overfitting** (Bailey,
    Borwein, López de Prado and Zhu, 2017) asks something sharper: if you pick
    the best trial on one part of the record, where does it rank on the rest?

    **Combinatorially symmetric cross-validation.** Split the shared calendar
    into $S$ blocks. For every choice of $S/2$ of them as the training set:
    - take the best trial on it, and its rank $\bar r$ among the $N$ trials on
      the other half;
    - compute $\omega = \bar r/(N+1)$ and the logit $\lambda = \ln\frac{\omega}{1-\omega}$.

    **PBO** is the share of $\lambda \le 0$: the in-sample winner at or below
    the median out of sample. Near 0, selection generalises; near ½ it is a
    coin; above ½, choosing the best is worse than choosing at random.
    """)
    return


@app.cell
def _(gr, pl, studies):
    EVER = ("2020-01-01T00:00Z", "2100-01-01T00:00Z")
    PER_YEAR = 365
    bars = gr.market.candles(["BTC", "ETH"], "1d", *EVER)
    trials = pl.concat(
        [
            studies.moving_average(bars, [5, 10, 20, 50], [20, 50, 100, 200]),
            studies.momentum(bars, [5, 10, 20, 40, 60, 90, 120]),
        ]
    )
    grid = studies.matrix(trials)
    return PER_YEAR, grid, trials


@app.cell
def _(mo):
    blocks = mo.ui.dropdown({"8 blocks (70 splits)": 8, "12 blocks (924 splits)": 12, "16 blocks (12,870 splits)": 16}, value="16 blocks (12,870 splits)", label="S")
    blocks
    return (blocks,)


@app.cell
def _(blocks, grid, mo, stats, time):
    started = time.perf_counter()
    result = stats.pbo(grid, blocks=blocks.value)
    elapsed = time.perf_counter() - started
    mo.vstack(
        [
            mo.md(
                f"## {result['trials']} trials on a shared calendar of {grid.height:,} days "
                f"({grid['ts'].min():%Y-%m-%d} → {grid['ts'].max():%Y-%m-%d}), "
                f"{result['blocks']} blocks of {result['rows_per_block']} days, {result['dropped']} earliest days dropped, "
                f"{result['combinations'].height:,} splits in {elapsed:.1f} s"
            ),
            mo.hstack(
                [
                    mo.stat(f"{result['pbo']:.0%}", label="PBO: best IS at or below the OOS median"),
                    mo.stat(f"{result['prob_loss']:.0%}", label="P(loss): best IS loses OOS"),
                    mo.stat(f"{result['slope']:.2f}", label="degradation: OOS Sharpe per unit IS Sharpe"),
                ],
                justify="start",
            ),
            mo.md(
                "**Selection is not generalising here.**" if result["pbo"] > 0.5 else
                "**Selection generalises better than chance here.**" if result["pbo"] < 0.2 else
                "**Selection is close to a coin here.**"
            ),
        ]
    )
    return (result,)


@app.cell
def _(alt, mo, result):
    logits = (
        alt.Chart(result["combinations"])
        .mark_bar()
        .encode(x=alt.X("logit:Q", bin=alt.Bin(maxbins=40), title="λ, the in-sample winner's out-of-sample logit"), y="count():Q")
        .properties(height=200, width="container")
    )
    zero = alt.Chart(result["combinations"].head(1)).mark_rule(strokeDash=[4, 4]).encode(x=alt.datum(0))
    mo.vstack([mo.md("## The logit distribution: left of the dashed line is overfit"), logits + zero])
    return


@app.cell
def _(PER_YEAR, alt, mo, pl, result):
    pairs = result["combinations"].drop_nulls(["is_sharpe", "oos_sharpe"]).with_columns(
        (pl.col("is_sharpe") * PER_YEAR**0.5).alias("in-sample Sharpe, annual"),
        (pl.col("oos_sharpe") * PER_YEAR**0.5).alias("out-of-sample Sharpe, annual"),
    )
    sample = pairs.sample(n=min(3000, pairs.height), seed=1)
    points = alt.Chart(sample).mark_circle(size=12, opacity=0.35).encode(
        x="in-sample Sharpe, annual:Q", y="out-of-sample Sharpe, annual:Q", color=alt.Color("best:N", legend=None), tooltip=["best"]
    )
    fit = points.transform_regression("in-sample Sharpe, annual", "out-of-sample Sharpe, annual").mark_line(color="black")
    winners = result["combinations"].group_by("best").len().sort("len", descending=True).head(8)
    mo.vstack(
        [
            mo.md(f"## Performance degradation: slope {result['slope']:.2f}"),
            mo.md("Each point is one split: the winner's Sharpe in the half it was chosen on against the half it was not. A falling line means the better a trial looked, the worse it did after."),
            (points + fit).properties(height=300, width="container"),
            mo.md("Which trials win in-sample, and how often:"),
            winners,
        ]
    )
    return


@app.cell
def _(PER_YEAR, grid, mo, pl, studies, trials):
    # The naive version of the same question: choose on the first half, report the second.
    cut = grid["ts"][grid.height // 2]
    aligned = trials.filter(pl.col("ts").is_in(grid["ts"].implode()))
    first = studies.summary(aligned.filter(pl.col("ts") < cut), PER_YEAR)
    second = studies.summary(aligned.filter(pl.col("ts") >= cut), PER_YEAR)
    joined = (
        first.select("trial", "ticker", pl.col("sharpe_annual").alias("first_half"))
        .join(second.select("trial", "ticker", pl.col("sharpe_annual").alias("second_half")), on=["trial", "ticker"])
        .with_columns(pl.col("second_half").rank(descending=True).alias("second_half_rank"))
        .sort("first_half", descending=True)
    )
    chosen = joined.row(0, named=True)
    mo.vstack(
        [
            mo.md(f"## One hold-out, split at {cut:%Y-%m-%d}"),
            mo.md(
                f"Chosen on the first half: `{chosen['trial']}` on {chosen['ticker']}, Sharpe {chosen['first_half']:.2f}. "
                f"On the second half it earned {chosen['second_half']:.2f}, ranking {chosen['second_half_rank']:.0f} of {joined.height}. "
                "One split is one draw; PBO is this over every split."
            ),
            joined.head(10),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
