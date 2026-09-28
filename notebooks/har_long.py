import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import polars as pl

    import galata_research as gr

    return alt, gr, mo, pl


@app.cell
def _(gr):
    # Cached results are keyed on the days the store holds, so a fetch recomputes them.
    try:
        HELD = int(gr.reference.coverage()["days_ok"].sum())
    except gr.Refused:
        HELD = 0
    return (HELD,)


@app.cell
def _(mo):
    mo.md(r"""
    # HAR against GARCH on six years of BTC and ETH

    Registered in `planning/preregistered/har-vs-garch-long-history.md`,
    committed alone (`3e02cfb`) on 2026-09-28 before any bar below existed.
    This notebook runs it as registered:
    - **Data.** Binance USDⓈ-M 1m klines from `gr.reference`, 2020-01-01 to
      2026-09-27, built into whole 1d, 4h, 1h and 5m bars
      (`gr.timeseries.resample`, a partial bucket dropped).
    - **Walk.** Out of sample from 2024-09-01, walked and scored with
      `garch.py`'s calls: `ewma, garch, gjr, egarch, har, harq`, Student-t,
      refit every 5 (1d) or 6 (4h) bars, horizons 1/7/30 and 1/6/42.
    - **HAR's RV, two ways.** (A) from 4h bars at 1d and 1h bars at 4h, as the
      record does it; (B) from 5-minute returns.
    - **Score.** QLIKE against squared returns.

    | # | claim | rule |
    |---|---|---|
    | H1 | HAR beats GARCH, RV (A) | *consistent* if min(HAR, HARQ) QLIKE < GARCH at every horizon; *contradicts* if at none |
    | H2 | HAR beats GARCH, RV (B) | the same |
    | H3 | HARQ beats GARCH at every horizon, RV (A) | *yes* if uSPA p < 0.05 (499 reps) |
    """)
    return


@app.cell
def _():
    SAMPLE = ("2020-01-01T00:00Z", "2026-09-27T00:00Z")
    SPLIT = "2024-09-01T00:00:00+00:00"
    TICKERS = ("BTC", "ETH")
    BARS = ("1d", "4h")
    HORIZONS = {"1d": (1, 7, 30), "4h": (1, 6, 42)}
    EVERY = {"1d": 5, "4h": 6}
    FINE_A = {"1d": "4h", "4h": "1h"}  # garch.py's walk(): the record's fine bars
    MODELS = ("ewma", "garch", "gjr", "egarch", "har", "harq")
    return BARS, EVERY, FINE_A, HORIZONS, MODELS, SAMPLE, SPLIT, TICKERS


@app.cell
def _(BARS, FINE_A, SAMPLE, TICKERS, gr, mo, pl):
    def _bars():
        out = {}
        for t in TICKERS:
            m = gr.reference.candles(t, *SAMPLE, venues="binance-um").collect()
            for width in sorted(set(BARS) | set(FINE_A.values()) | {"5m"}):
                out[(t, width)] = gr.timeseries.resample(m, width)
        return out

    bars = _bars()
    _rows = [{"ticker": t, "bars": w, "count": b.height, "first": b["ts"].min(), "last": b["ts"].max()} for (t, w), b in sorted(bars.items())]
    mo.vstack([mo.md("## The bars built from the 1m klines (whole buckets only)"), pl.DataFrame(_rows)])
    return (bars,)


@app.cell
def _(EVERY, FINE_A, HELD, HORIZONS, MODELS, SPLIT, bars, gr, mo, pl):
    from galata_research.models import vol

    def _walk(ticker, interval, model, rv):
        # garch.py's walk(), line for line, on these bars
        hs = HORIZONS[interval]
        if model in ("har", "harq"):
            fine = FINE_A[interval] if rv == "A" else "5m"
            measures = gr.timeseries.realized_from(bars[(ticker, fine)], interval)
            return vol.har(measures, model=model, split=SPLIT, every=EVERY[interval], horizons=hs)
        r = gr.timeseries.returns(bars[(ticker, interval)], kind="log")
        hs = hs[:1] if model == "egarch" else hs  # EGARCH-t has no variance beyond one step (item 22)
        return vol.walk_forward(r, model=model, dist="t", split=SPLIT, every=EVERY[interval], horizons=hs, simulations=500, min_obs=250)

    def _all():
        parts, skipped = [], []
        for (ticker, interval) in [(t, i) for t in ("BTC", "ETH") for i in ("1d", "4h")]:
            for model in MODELS:
                for rv in (("A", "B") if model in ("har", "harq") else ("-",)):
                    try:
                        f = _walk(ticker, interval, model, rv)
                        parts.append(f.with_columns(pl.lit(interval).alias("interval"), pl.lit(model).alias("model"), pl.lit(rv).alias("rv")))
                    except gr.Refused as why:  # a model the data cannot support is reported, not hidden
                        skipped.append({"ticker": ticker, "interval": interval, "model": model, "rv": rv, "why": str(why)})
        return pl.concat(parts, how="diagonal_relaxed"), pl.DataFrame(skipped, schema={"ticker": pl.String, "interval": pl.String, "model": pl.String, "rv": pl.String, "why": pl.String})

    with mo.persistent_cache(name=f"har-long-walk-{HELD}"):
        walked, skipped = _all()
    mo.vstack([
        mo.md(f"**{walked.height:,}** forecasts; `fitted_through ≤ close_ts` on every row: **{bool((walked['fitted_through'] <= walked['close_ts']).all())}**."),
        mo.md("Not walked:") if skipped.height else mo.md(""), skipped if skipped.height else mo.md(""),
    ])  # fmt: skip
    return skipped, walked


@app.cell
def _(bars, gr, pl, walked):
    ev = gr.models.evaluate

    def aligned_for(ticker, interval, rv, frame=None):
        # One frame per cell and RV version: the GARCH family, plus HAR and HARQ built with that RV.
        f = (walked if frame is None else frame).filter((pl.col("ticker") == ticker) & (pl.col("interval") == interval) & pl.col("rv").is_in(["-", rv]))
        proxy = ev.proxies(bars[(ticker, interval)], "r2")  # garch.py's default proxy
        return ev.align(f.drop("interval", "rv"), proxy)

    return aligned_for, ev


@app.cell
def _(BARS, TICKERS, aligned_for, ev, pl):
    _rows, _cards = [], []
    for _t in TICKERS:
        for _i in BARS:
            for _rv, _hyp in (("A", "H1"), ("B", "H2")):
                _al = aligned_for(_t, _i, _rv)
                _card = ev.scorecard(_al, benchmark="garch").with_columns(pl.lit(_t).alias("ticker"), pl.lit(_i).alias("bars"), pl.lit(_rv).alias("rv"))
                _cards.append(_card)
                # garch.py line 1134, the registered rule
                _q = _card.select("model", "h", "qlike")
                _har = _q.filter(pl.col("model").is_in(["har", "harq"])).group_by("h").agg(pl.col("qlike").min().alias("har"))
                _hg = _har.join(_q.filter(pl.col("model") == "garch").select("h", pl.col("qlike").alias("garch")), on="h").sort("h")
                if _hg.height:
                    _wins = int((_hg["har"] < _hg["garch"]).sum())
                    _v = "consistent" if _wins == _hg.height else "contradicts" if _wins == 0 else "mixed"
                    # Patton's QLIKE is negative at these scales, so the ratio shown is the scorecard's textbook one (below 1: better).
                    _best = _card.filter(pl.col("model").is_in(["har", "harq"])).sort("qlike").group_by("h", maintain_order=True).first().sort("h")
                    _m = f"{_wins} of {_hg.height} horizons; best HAR ÷ GARCH " + ", ".join(f"h{h} {r:.3f} ({m})" for h, r, m in _best.select("h", "qlike_ratio", "model").iter_rows())
                else:
                    _v, _m = "can't tell", "HAR or GARCH not walked"
                _rows.append({"#": _hyp, "ticker": _t, "bars": _i, "verdict": _v, "measured": _m})
                if _rv == "A":
                    try:
                        _p = ev.uspa(_al, model="harq", benchmark="garch", reps=499)["p_value"]
                        _rows.append({"#": "H3", "ticker": _t, "bars": _i, "verdict": "yes" if _p < 0.05 else "no", "measured": f"uSPA p {_p:.3f}"})
                    except Exception as _why:  # a test the data cannot form is reported, not hidden
                        _rows.append({"#": "H3", "ticker": _t, "bars": _i, "verdict": "can't tell", "measured": str(_why)[:80]})
    verdicts = pl.DataFrame(_rows).sort("#", "ticker", "bars")
    cards = pl.concat(_cards)
    return cards, verdicts


@app.cell
def _(mo, pl, verdicts):
    _h1 = verdicts.filter(pl.col("#") == "H1")["verdict"].to_list()
    _headline = (
        "**The claim survives on long history**: H1 is consistent in all four cells." if _h1.count("consistent") == 4
        else "**Venue- or sample-specific**: H1 contradicts in two or more cells." if _h1.count("contradicts") >= 2
        else "**Mixed**: neither all consistent nor two contradicting."
    )  # fmt: skip
    mo.vstack([mo.md("## The twelve registered verdicts"), mo.ui.table(verdicts, selection=None, page_size=12), mo.md(_headline + " (the registration's *What would count*)")])
    return


@app.cell
def _(alt, cards, mo, pl):
    _c = cards.filter(pl.col("model") != "garch").with_columns(pl.concat_str("ticker", "bars", pl.lit("RV"), "rv", separator=" ").alias("cell"))
    _chart = (
        alt.Chart(_c).mark_rect()
        .encode(x=alt.X("h:O", title="horizon (bars)"), y=alt.Y("model:N"), color=alt.Color("qlike_ratio:Q", scale=alt.Scale(scheme="redblue", domainMid=1, reverse=True), title="QLIKE / GARCH"),
                tooltip=["cell:N", "model:N", "h:O", alt.Tooltip("qlike_ratio:Q", format=".3f"), alt.Tooltip("dm_p:Q", format=".3f")])
        .properties(width=110, height=120).facet(facet=alt.Facet("cell:N", title=None), columns=4)
    )  # fmt: skip
    mo.vstack([mo.md("## Every model against GARCH (blue: better than GARCH)"), _chart, mo.md("Not registered: context for the verdicts, which use only the rules above.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The level, removed (item 27)

    Registered in `planning/preregistered/har-level-matched.md` (`68ae3e3`),
    after the verdicts above. HAR forecasts its own measure's level, and
    5-minute RV sits above the squared return it is scored against. Here each
    HAR and HARQ forecast is multiplied by c = Σr² / ΣRV over the bars closing
    by the split, per ticker, bars and RV version. Nothing else changes.

    | # | claim | rule |
    |---|---|---|
    | H4 | rescaled HAR beats GARCH, RV from 5 minutes | as H1 |
    | H5 | rescaled HAR beats GARCH, RV as the record builds it | as H1 |
    | H6 | rescaled HARQ beats GARCH at every horizon, RV from 5 minutes | uSPA p < 0.05 |
    """)
    return


@app.cell
def _(BARS, FINE_A, SPLIT, TICKERS, bars, gr, pl):
    _rows = []
    _cut = pl.lit(SPLIT).str.to_datetime(time_zone="UTC")
    for _t in TICKERS:
        for _i in BARS:
            _r2 = gr.timeseries.returns(bars[(_t, _i)], kind="log").select("ts", (pl.col("return") ** 2).alias("r2"))
            for _rv, _fine in (("A", FINE_A[_i]), ("B", "5m")):
                _m = gr.timeseries.realized_from(bars[(_t, _fine)], _i).select("ts", "close_ts", "rv")
                _j = _m.join(_r2, on="ts").drop_nulls().filter(pl.col("close_ts") <= _cut)
                _rows.append({"ticker": _t, "interval": _i, "rv": _rv, "c": _j["r2"].sum() / _j["rv"].sum(), "bars": _j.height})
    scales = pl.DataFrame(_rows)
    return (scales,)


@app.cell
def _(BARS, TICKERS, aligned_for, ev, mo, pl, scales, walked):
    _scaled = walked.join(scales.select("ticker", "interval", "rv", "c"), on=["ticker", "interval", "rv"], how="left").with_columns(
        pl.when(pl.col("model").is_in(["har", "harq"])).then(pl.col(c) * pl.col("c")).otherwise(pl.col(c)).alias(c) for c in ("variance", "cum_variance")
    ).drop("c")
    _rows = []
    for _t in TICKERS:
        for _i in BARS:
            for _rv, _hyp in (("B", "H4"), ("A", "H5")):
                _al = aligned_for(_t, _i, _rv, frame=_scaled)
                _card = ev.scorecard(_al, benchmark="garch")
                _q = _card.select("model", "h", "qlike")
                _har = _q.filter(pl.col("model").is_in(["har", "harq"])).group_by("h").agg(pl.col("qlike").min().alias("har"))
                _hg = _har.join(_q.filter(pl.col("model") == "garch").select("h", pl.col("qlike").alias("garch")), on="h").sort("h")
                _wins = int((_hg["har"] < _hg["garch"]).sum())
                _best = _card.filter(pl.col("model").is_in(["har", "harq"])).sort("qlike").group_by("h", maintain_order=True).first().sort("h")
                _rows.append({"#": _hyp, "ticker": _t, "bars": _i, "verdict": "consistent" if _wins == _hg.height else "contradicts" if _wins == 0 else "mixed",
                              "measured": f"{_wins} of {_hg.height} horizons; best HAR ÷ GARCH " + ", ".join(f"h{h} {r:.3f} ({m})" for h, r, m in _best.select("h", "qlike_ratio", "model").iter_rows())})  # fmt: skip
                if _rv == "B":
                    _p = ev.uspa(_al, model="harq", benchmark="garch", reps=499)["p_value"]
                    _rows.append({"#": "H6", "ticker": _t, "bars": _i, "verdict": "yes" if _p < 0.05 else "no", "measured": f"uSPA p {_p:.3f}"})
    level_verdicts = pl.DataFrame(_rows).sort("#", "ticker", "bars")
    _h4 = level_verdicts.filter(pl.col("#") == "H4")
    _n4 = _h4["verdict"].to_list().count("consistent")
    _both_1d = _h4.filter((pl.col("bars") == "1d") & (pl.col("verdict") == "consistent")).height == 2
    _reading = ("**The level explains H2**: H4 consistent in three or more cells, both at 1d." if _n4 >= 3 and _both_1d
                else "**The level does not explain H2**: H4 has no more consistent cells than H2 (1)." if _n4 <= 1 else "**Neither**, by the registered reading.")  # fmt: skip
    mo.vstack([mo.md("### The scale c, fitted before the split"), scales, mo.md("### The twelve verdicts of item 27"), mo.ui.table(level_verdicts, selection=None, page_size=12), mo.md(_reading)])
    return (level_verdicts,)

if __name__ == "__main__":
    app.run()
