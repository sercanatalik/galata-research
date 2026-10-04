import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import random

    import marimo as mo
    import polars as pl

    import galata_research as gr

    return gr, mo, pl, random


@app.cell
def _(mo):
    mo.md(r"""
    # Is 1h persistence of one on six years neglected variance breaks?

    On six years of Binance 1h bars, deseasonalised GARCH-t persistence
    stays at 1.0000 (item 29), so the daily cycle does not explain it.
    Neglected breaks in the unconditional variance push α+β toward one
    (Lamoureux and Lastrapes 1990; Hillebrand 2005).

    Fitting between breaks also shortens every sample, and κ₂ over-detects
    under persistent GARCH. So the segments at κ₂'s breaks are set against
    20 draws of random cuts, as many as κ₂ found.

    Registered in `planning/preregistered/persistence-1h-breaks.md`
    (`f227e8b`) before any segment was fitted.
    - **Breaks explain it** if the full-sample α+β ≥ 0.999, the median
      segment α+β at the breaks is below 0.99, and it is below at least 19
      of the 20 placebo medians.
    - **Segment length explains it** if the median is below 0.99 but not
      below the placebo.
    """)
    return


@app.cell
def _():
    SAMPLE = ("2020-01-01T00:00Z", "2026-09-27T00:00Z")
    MIN_SEGMENT, MIN_FIT, DRAWS = 720, 2000, 20
    return DRAWS, MIN_FIT, MIN_SEGMENT, SAMPLE


@app.cell
def _(DRAWS, MIN_FIT, MIN_SEGMENT, SAMPLE, gr, mo, pl, random):
    from galata_research.models import vol

    def _returns(ticker):
        bars = gr.timeseries.resample(gr.reference.candles(ticker, *SAMPLE, venues="binance-um").collect(), "1h")
        r = gr.timeseries.returns(bars, kind="log")
        d = gr.timeseries.deseasonalize(r, gr.timeseries.seasonal_factors(r, fit=(bars["ts"].min(), bars["close_ts"].max())))
        return d.select("ticker", "ts", "close_ts", pl.col("deseasonalized").alias("return")).drop_nulls("return").sort("ts")

    def _median(fits):
        seg = fits.filter((pl.col("segment") != "full") & ~pl.col("skipped"))
        return float(seg["persistence"].median()) if seg.height else None, seg.height

    def _random_cuts(r, k, seed):
        rng = random.Random(seed)
        n = r.height
        while True:  # re-drawn only when a segment would be shorter than MIN_SEGMENT bars
            idx = sorted(rng.sample(range(1, n), k))
            edges = [0, *idx, n]
            if min(b - a for a, b in zip(edges, edges[1:], strict=False)) >= MIN_SEGMENT:
                return pl.DataFrame({"ts": r["ts"].gather(idx)})

    def _study():
        rows, fits_all = [], []
        for t in ("BTC", "ETH"):
            r = _returns(t)
            breaks = gr.timeseries.variance_breaks(r, statistic="kappa2", min_segment=MIN_SEGMENT)
            fits = vol.segmented(r, breaks, model="garch", dist="t", min_obs=MIN_FIT)
            full = float(fits.filter(pl.col("segment") == "full")["persistence"][0])
            m_break, fitted = _median(fits)
            placebo = []
            for seed in range(DRAWS):
                pf = vol.segmented(r, _random_cuts(r, breaks.height, seed), model="garch", dist="t", min_obs=MIN_FIT)
                placebo.append(_median(pf)[0])
            below = sum(m_break is not None and p is not None and m_break < p for p in placebo)
            if m_break is None or m_break >= 0.99:
                reading = "neither"
            elif full >= 0.999 and below >= 19:
                reading = "breaks explain the persistence"
            else:
                reading = "segment length explains it"
            rows.append({"ticker": t, "returns": r.height, "breaks": breaks.height, "segments fitted": fitted, "full α+β": full,
                         "median α+β at breaks": m_break, "placebo medians (min, median, max)": f"{min(placebo):.4f}, {sorted(placebo)[DRAWS // 2]:.4f}, {max(placebo):.4f}",
                         "below placebo": f"{below} of {DRAWS}", "reading": reading})  # fmt: skip
            fits_all.append(fits.with_columns(pl.lit(t).alias("ticker")))
        return pl.DataFrame(rows), pl.concat(fits_all, how="diagonal_relaxed")

    with mo.persistent_cache(name="persistence-1h-breaks"):
        readings, segment_fits = _study()
    mo.vstack([
        mo.md("## The registered readings"), readings,
        mo.md("GARCH-t on the full sample and on each segment between κ₂ breaks:"),
        segment_fits.select("ticker", "segment", "from", "to", "skipped", "nobs", "persistence", "half_life"),
    ])  # fmt: skip
    return readings, segment_fits


if __name__ == "__main__":
    app.run()
