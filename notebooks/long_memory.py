import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    import galata_research as gr

    return gr, mo, pl


@app.cell
def _(mo):
    mo.md(r"""
    # Is the volatility's persistence true long memory, or level shifts?

    Six years of Binance 1h bars left GARCH persistence at one, with the
    daily cycle removed (item 29) and between κ₂ breaks (item 31). True long
    memory and small, frequent level shifts both look like that in the time
    domain. They differ near frequency zero, which is where Qu's (2011) test
    looks: a local Whittle fit whose score drifts across the lowest
    frequencies means the long-memory model does not hold there.

    Registered in `planning/preregistered/long-memory-or-level-shifts.md`
    (`d6509ae`) before the test existed in code. It was validated on
    simulations first (`tests/memory.py`).
    - **The test.** On x = log |r|, W > 1.252 rejects true long memory at 5%
      (m = ⌊1 + T^0.7⌋, ε = 0.02).
    - **The reading.** *True long memory* if no cell rejects, *level shifts*
      if three or four do.
    """)
    return


@app.cell
def _(gr, mo, pl):
    from galata_research.models import memory

    SAMPLE = ("2020-01-01T00:00Z", "2026-09-27T00:00Z")

    def _series(ticker, interval):
        bars = gr.timeseries.resample(gr.reference.candles(ticker, *SAMPLE, venues="binance-um").collect(), interval)
        r = gr.timeseries.returns(bars, kind="log")
        if interval == "1h":
            r = gr.timeseries.deseasonalize(r, gr.timeseries.seasonal_factors(r, fit=(bars["ts"].min(), bars["close_ts"].max()))).select(
                "ticker", "ts", "close_ts", pl.col("deseasonalized").alias("return")
            )
        r = r.drop_nulls("return").sort("ts")
        zeros = int((r["return"] == 0).sum())
        return r.filter(pl.col("return") != 0).select(pl.col("return").abs().log())["return"].to_numpy(), zeros

    def _study():
        rows, bands = [], []
        for t in ("BTC", "ETH"):
            for i in ("1h", "1d"):
                x, zeros = _series(t, i)
                q = memory.qu_test(x)
                rows.append({"ticker": t, "bars": i, "T": x.size, "zeros dropped": zeros, "m": q["m"], "d̂ (m = T^0.7)": q["d"], "W": q["W"],
                             "verdict": "spurious (level shifts or trend)" if q["reject_5pct"] else "long memory not rejected"})  # fmt: skip
                for power in (0.5, 0.6, 0.7, 0.8):
                    lw = memory.local_whittle(x, m=int(1 + x.size**power))
                    bands.append({"ticker": t, "bars": i, "m = T^": power, "m": lw["m"], "d̂": lw["d"], "se": lw["se"]})
        return pl.DataFrame(rows), pl.DataFrame(bands)

    with mo.persistent_cache(name="long-memory-or-level-shifts"):
        cells, bandwidths = _study()
    _n = cells.filter(pl.col("verdict") != "long memory not rejected").height
    reading = "true long memory" if _n == 0 else "level shifts" if _n >= 3 else "mixed"
    mo.vstack([
        mo.md("## The four registered cells"), cells,
        mo.md(f"**{_n} of 4 cells reject true long memory: {reading}** (the registered reading)."),
        mo.md("Context, not decided: d̂ as the bandwidth grows. Level shifts make it fall with m (Perron and Qu 2010); long memory keeps it level."),
        bandwidths.pivot(on="m = T^", index=["ticker", "bars"], values="d̂"),
    ])  # fmt: skip
    return bandwidths, cells, reading


if __name__ == "__main__":
    app.run()
