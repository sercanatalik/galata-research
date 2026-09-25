import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import polars as pl

    import galata_research as gr
    from galata_research import backtest, stats, studies

    return alt, backtest, gr, mo, pl, stats, studies


@app.cell
def _(mo):
    mo.md(r"""
    # Does anything in the set beat its benchmark?

    DSR and the random twins judge one trial at a time. **White's Reality
    Check** (2000) and **Hansen's SPA** (2005) ask of the whole family at once:
    after searching every trial, is the best one's excess over its benchmark
    more than the search itself would produce? Both resample days with the
    **stationary bootstrap** (Politis and Romano, 1994; mean block √T), which
    keeps the returns' dependence.

    - Reality Check: $\max_k \sqrt{T}\,\bar d_k$, not studentized.
    - SPA: $\max(\max_k \sqrt{T}\,\bar d_k/\hat\omega_k,\ 0)$, studentized, with
      three recenterings: *lower* (liberal), *consistent* (Hansen's), and
      *upper* (conservative).

    A p-value is the share of bootstrap replicates at or above the observed
    statistic. The Reality Check is cross-checked against `arch` in the tests.
    """)
    return


@app.cell
def _(backtest, gr, pl, studies):
    EVER = ("2020-01-01T00:00Z", "2100-01-01T00:00Z")
    bars = gr.market.candles(["BTC", "ETH"], "1d", *EVER)
    trials = pl.concat(
        [
            studies.moving_average(bars, [5, 10, 20, 50], [20, 50, 100, 200]),
            studies.momentum(bars, [5, 10, 20, 40, 60, 90, 120]),
            studies.donchian_ensemble(bars, sized=True),
            studies.donchian_ensemble(bars, sized=False),
            studies._trial(bars, pl.lit(1.0), "buy and hold", backtest.TAKER_FEE),
        ]
    )
    return (trials,)


@app.cell
def _(mo):
    benchmark = mo.ui.dropdown({"buy-and-hold, same ticker": "buy and hold", "cash (zero)": "cash"}, value="buy-and-hold, same ticker", label="benchmark")
    benchmark
    return (benchmark,)


@app.cell
def _(benchmark, mo, pl, stats, studies, trials):
    excess = studies.excess(trials, benchmark.value)
    root = int(excess.height**0.5)
    runs = []
    for block in (max(1, int(0.5 * root)), root, 2 * root, "auto"):
        r = stats.reality_check(excess, reps=1000, block=block, seed=0)
        runs.append({"choice": "Politis–White (auto)" if block == "auto" else f"{block / root:.1f} × √T", "block": float(r["block"]), "reality_check": r["reality_check"], **{f"spa_{k}": v for k, v in r["spa"].items()}, "best": r["best"]})
    table = pl.DataFrame(runs)
    main = stats.reality_check(excess, reps=1000, seed=0)
    worst = max(main["spa"]["consistent"], main["reality_check"])
    mo.vstack(
        [
            mo.md(f"## {excess.width - 1} trials against {benchmark.value}, over {excess.height:,} shared days (1,000 replicates, seed 0)"),
            mo.hstack(
                [
                    mo.stat(f"{main['reality_check']:.3f}", label="Reality Check p"),
                    mo.stat(f"{main['spa']['consistent']:.3f}", label="SPA p (consistent)"),
                    mo.stat(main["best"], label="the best trial, studentized"),
                ],
                justify="start",
            ),
            mo.md("**Nothing in the set beats its benchmark at 5%.**" if worst > 0.05 else "**At least one trial beats its benchmark at 5%.**"),
            mo.md("The same test at half, one and two times the default block (√T), and at the block the data chooses (Politis and White, 2004, corrected 2009; the median of the columns' blocks). A verdict that flips with the block is not a verdict."),
            table,
        ]
    )
    return (main,)


@app.cell
def _(alt, main, mo):
    columns = main["columns"].sort("t", descending=True)
    chart = (
        alt.Chart(columns.head(25))
        .mark_bar()
        .encode(x=alt.X("t:Q", title="studentized mean excess, √T·d̄/ω̂"), y=alt.Y("column:N", sort="-x", title=None))
        .properties(height=420, width="container")
    )
    mo.vstack(
        [
            mo.md("## The 25 highest studentized excesses: the ones the search would pick"),
            chart,
            mo.md(
                "Against buy-and-hold, each crossover's `long_short` bar equals its `long_flat` bar. That is an identity, not a coincidence: "
                "with the long/flat position L ∈ {0, 1}, long/short is 2L − 1, so its excess over holding, (2L − 1)r − r = 2(Lr − r), "
                "is exactly twice long/flat's (up to fees), and studentizing divides the 2 out."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
