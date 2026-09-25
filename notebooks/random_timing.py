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
    # Against its random twins

    A Sharpe says how well a strategy did; it does not say how much of that
    came from **when** it held, rather than from **how much** it held in a
    market that rose. Each trial here is scored against 1,000 random twins:
    its own runs of position (every holding stretch and every flat one),
    shuffled into a random order and replayed over the same bars with the
    same fees. Trade count, holding times, side mix and exposure match
    exactly; only the timing is random (Masters' Monte Carlo permutation;
    galata-legacy's `score-against-random`).

    **Percentile** = (1 + twins at or above the strategy) / 1,001. Small means
    the timing beat chance. The seed is declared, so every twin can be
    reproduced.
    """)
    return


@app.cell
def _(gr, pl, studies):
    EVER = ("2020-01-01T00:00Z", "2100-01-01T00:00Z")
    PER_YEAR = 365
    SEED = 0
    bars = gr.market.candles(["BTC", "ETH"], "1d", *EVER)
    trials = pl.concat(
        [
            studies.donchian_ensemble(bars, sized=True),
            studies.donchian_ensemble(bars, sized=False),
            studies.moving_average(bars, [5], [100], sides=["long_flat"]),
            studies.momentum(bars, [60]),
        ]
    )
    scored = studies.random_timing(trials, samples=1000, seed=SEED).with_columns(
        (pl.col("observed") * PER_YEAR**0.5).alias("sharpe"),
        (pl.col("random_median") * PER_YEAR**0.5).alias("twins_median"),
        (pl.col("random_p95") * PER_YEAR**0.5).alias("twins_p95"),
        (pl.col("observed") - pl.col("random_median")).mul(PER_YEAR**0.5).alias("from_timing"),
    )
    return PER_YEAR, SEED, scored, trials


@app.cell
def _(SEED, mo, pl, scored):
    table = scored.select("trial", "ticker", "runs", "sharpe", "twins_median", "from_timing", "twins_p95", "percentile").sort("percentile")
    best = scored.sort("percentile").row(0, named=True)
    mo.vstack(
        [
            mo.md(f"## Every trial, annualized Sharpe, against 1,000 twins each (seed {SEED})"),
            table,
            mo.md(
                f"`twins_median` is what the same exposure earns with random timing; `from_timing` is the rest. "
                f"The lowest percentile is `{best['trial']}` on {best['ticker']} at **{best['percentile']:.1%}**: "
                + ("**none beats its twins at 5%.**" if best["percentile"] > 0.05 else "**at least one beats its twins at 5%.**")
            ),
            mo.md(f"Null: *{scored['null'][0]}*. A volatility-sized position changes most days, so its runs are a bar or two long and its twins are close to a bar shuffle; the unsized rows keep real holding times."),
        ]
    )
    return


@app.cell
def _(mo, scored):
    choice = mo.ui.dropdown({f"{r['trial']} | {r['ticker']}": (r["trial"], r["ticker"]) for r in scored.iter_rows(named=True)}, value=f"{scored.sort('percentile')['trial'][0]} | {scored.sort('percentile')['ticker'][0]}", label="trial")
    choice
    return (choice,)


@app.cell
def _(PER_YEAR, SEED, alt, choice, mo, pl, studies, trials):
    import random

    from galata_research import stats

    trial, ticker = choice.value
    kept = trials.filter((pl.col("trial") == trial) & (pl.col("ticker") == ticker)).sort("ts").drop_nulls(["position", "bar_return"])
    positions, returns = kept["position"].to_list(), kept["bar_return"].to_list()
    runs = studies._runs(positions)
    twins = []
    for k in range(1000):
        order = runs[:]
        random.Random(f"{SEED}:{trial}:{ticker}:{k}").shuffle(order)
        twins.append(stats.sharpe(studies._replay([v for v, n in order for _ in range(n)], returns, 0.00045)))
    observed = stats.sharpe(studies._replay(positions, returns, 0.00045)) * PER_YEAR**0.5
    cloud = pl.DataFrame({"sharpe": [t * PER_YEAR**0.5 for t in twins if t is not None]})
    hist = alt.Chart(cloud).mark_bar(opacity=0.7).encode(x=alt.X("sharpe:Q", bin=alt.Bin(maxbins=40), title="annualized Sharpe of a random twin"), y="count():Q")
    rule = alt.Chart(pl.DataFrame({"x": [observed]})).mark_rule(color="#d64545", strokeWidth=2).encode(x="x:Q")
    mo.vstack([mo.md(f"## `{trial}` on {ticker}: the strategy (red) among its twins"), (hist + rule).properties(height=240, width="container")])
    return


if __name__ == "__main__":
    app.run()
