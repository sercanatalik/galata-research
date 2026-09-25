import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import random
    import time

    import altair as alt
    import marimo as mo
    import polars as pl

    import galata_research as gr
    from galata_research import studies

    return alt, gr, mo, pl, random, studies, time


@app.cell
def _(mo):
    mo.md("""
    # Against markets with no structure

    Random timing kept the market and randomized the strategy. This keeps the
    strategy and randomizes the **market**: the bars are permuted (Masters'
    Monte Carlo permutation, as `neurotrader888/mcpt` does it). Each bar's
    shape and the gaps between bars are kept, and so are the drift (the last
    close is unchanged) and the BTC–ETH co-movement (one permutation serves
    both). But their **order** is random, so trend persistence and volatility
    clustering are gone. Then **the whole search is re-run** on every fake
    market, and its best is compared with the real best.

    If the real market has structure the search exploits, the real best
    should stand out from the permuted bests.
    """)
    return


@app.cell
def _(gr, pl, studies, time):
    EVER = ("2020-01-01T00:00Z", "2100-01-01T00:00Z")
    PER_YEAR = 365
    bars = gr.market.candles(["BTC", "ETH"], "1d", *EVER).collect()
    searches = {
        "the 66 example trials": lambda b: pl.concat([studies.moving_average(b, [5, 10, 20, 50], [20, 50, 100, 200]), studies.momentum(b, [5, 10, 20, 40, 60, 90, 120])]),
        "the 4 pre-registered Donchian trials": lambda b: pl.concat([studies.donchian_ensemble(b, sized=True), studies.donchian_ensemble(b, sized=False)]),
    }
    results, elapsed = {}, {}
    for _name, _search in searches.items():
        _started = time.perf_counter()
        results[_name] = studies.permutation_test(bars, _search, samples=200, seed=0)
        elapsed[_name] = time.perf_counter() - _started
    return PER_YEAR, bars, elapsed, results


@app.cell
def _(PER_YEAR, alt, elapsed, mo, pl, results):
    parts = []
    for _name, _r in results.items():
        real = _r["best"]["sharpe"] * PER_YEAR**0.5
        cloud = pl.DataFrame({"sharpe": [b * PER_YEAR**0.5 for b in _r["permuted_bests"]]})
        hist = alt.Chart(cloud).mark_bar(opacity=0.7).encode(x=alt.X("sharpe:Q", bin=alt.Bin(maxbins=30), title="best annualized Sharpe of the search, on a permuted market"), y="count():Q")
        rule = alt.Chart(pl.DataFrame({"x": [real]})).mark_rule(color="#d64545", strokeWidth=2).encode(x="x:Q")
        median = sorted(cloud["sharpe"].to_list())[len(cloud) // 2]
        parts += [
            mo.md(f"## {_name[0].upper() + _name[1:]}"),
            mo.md(
                f"Real best: `{_r['best']['trial']}` on {_r['best']['ticker']}, **{real:.2f}** (red). "
                f"The best on a structureless market has a median of **{median:.2f}**. "
                f"**p = {_r['p_best']:.3f}** over {_r['samples']} permutations ({elapsed[_name]:.0f} s). "
                + ("The real market gave the search nothing a permuted one would not." if _r["p_best"] > 0.05 else "The real market's structure is found.")
            ),
            (hist + rule).properties(height=200, width="container"),
        ]
    mo.vstack([*parts, mo.md(f"Null: *{next(iter(results.values()))['null']}*. `p` counts permuted markets whose best is at or above the real best, so it is selection-aware.")])
    return


@app.cell
def _(PER_YEAR, mo, pl, results):
    per_trial = pl.concat([r["trials"].with_columns(pl.lit(name).alias("search")) for name, r in results.items()]).with_columns((pl.col("sharpe") * PER_YEAR**0.5).alias("annual"))
    mo.vstack(
        [
            mo.md("## Each trial against itself on permuted markets"),
            mo.md("Not selection-aware: a low p here among 70 trials is what 70 tries produce."),
            per_trial.select("search", "trial", "ticker", "annual", "p").sort("p").head(12),
        ]
    )
    return


@app.cell
def _(alt, bars, mo, pl, random, studies):
    fake = studies.permute_bars(bars, random.Random("0:0")).with_columns(pl.lit("permuted (seed 0:0)").alias("market"))
    both = pl.concat([bars.select("ticker", "ts", "close").with_columns(pl.lit("real").alias("market")), fake.select("ticker", "ts", "close", "market")])
    chart = (
        alt.Chart(both)
        .mark_line(strokeWidth=1)
        .encode(x="ts:T", y=alt.Y("close:Q", scale=alt.Scale(type="log", zero=False)), color="market:N", row="ticker:N")
        .properties(height=160, width="container")
        .resolve_scale(y="independent")
    )
    mo.vstack([mo.md("## One permuted market beside the real one: the same start, the same end, no memory in between"), chart])
    return


if __name__ == "__main__":
    app.run()
