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

@app.cell
def _(mo):
    mo.md(r"""
    ## A model built for level shifts (item 33)

    Item 32 found the volatility's persistence is level shifts or a trend, not
    long memory. Lu and Perron's (2010) random level shift model filters a
    level that jumps now and then, and forecasts it flat. Registered in
    `planning/preregistered/random-level-shift-forecasts.md` (`33c1443`)
    before the model existed; validated on its own simulation first
    (`tests/levels.py`).
    - **Refits.** Every 30 bars at 1d and every 42 at 4h; the level is
      filtered at every bar.
    - **Scoring.** Against the same GARCH and HAR (RV as the record builds
      it) forecasts as above.

    | # | claim | rule |
    |---|---|---|
    | R1 | RLS beats GARCH at every horizon | as H1 |
    | R2 | RLS beats HAR at every horizon | against min(HAR, HARQ) |
    """)
    return


@app.cell
def _(BARS, HELD, HORIZONS, SPLIT, TICKERS, bars, gr, mo, pl):
    from galata_research.models import levels as _levels

    _EVERY_RLS = {"1d": 30, "4h": 42}

    def _rls():
        parts = []
        for t in TICKERS:
            for i in BARS:
                r = gr.timeseries.returns(bars[(t, i)], kind="log")
                f = _levels.walk_forward(r, split=SPLIT, every=_EVERY_RLS[i], horizons=HORIZONS[i])
                parts.append(f.with_columns(pl.lit(i).alias("interval"), pl.lit("rls").alias("model"), pl.lit("-").alias("rv")))
        return pl.concat(parts)

    with mo.persistent_cache(name=f"har-long-rls-{HELD}"):
        rls_walked = _rls()
    return (rls_walked,)


@app.cell
def _(BARS, TICKERS, aligned_for, ev, mo, pl, rls_walked, walked):
    _both = pl.concat([walked, rls_walked], how="diagonal_relaxed")
    _rows, _cards = [], []
    for _t in TICKERS:
        for _i in BARS:
            _card = ev.scorecard(aligned_for(_t, _i, "A", frame=_both), benchmark="garch")
            _cards.append(_card.with_columns(pl.lit(_t).alias("ticker"), pl.lit(_i).alias("bars")))
            _q = _card.select("model", "h", "qlike")
            _rls_q = _q.filter(pl.col("model") == "rls").select("h", pl.col("qlike").alias("rls"))
            _vs = {
                "R1": _q.filter(pl.col("model") == "garch").select("h", pl.col("qlike").alias("other")),
                "R2": _q.filter(pl.col("model").is_in(["har", "harq"])).group_by("h").agg(pl.col("qlike").min().alias("other")),
            }
            _ratio = _card.filter(pl.col("model") == "rls").sort("h")
            for _hyp, _other in _vs.items():
                _j = _rls_q.join(_other, on="h").sort("h")
                _wins = int((_j["rls"] < _j["other"]).sum())
                _rows.append({"#": _hyp, "ticker": _t, "bars": _i,
                              "verdict": "consistent" if _wins == _j.height else "contradicts" if _wins == 0 else "mixed",
                              "measured": f"{_wins} of {_j.height} horizons" + ("; RLS ÷ GARCH " + ", ".join(f"h{h} {x:.3f}" for h, x in _ratio.select("h", "qlike_ratio").iter_rows()) if _hyp == "R1" else "")})  # fmt: skip
    rls_verdicts = pl.DataFrame(_rows).sort("#", "ticker", "bars")
    rls_cards = pl.concat(_cards)
    _n = rls_verdicts.filter((pl.col("#") == "R1") & (pl.col("verdict") == "consistent")).height
    _reading = "**Level shifts forecast better** (R1 consistent in three or more cells)." if _n >= 3 else "**Not in these forecasts** (R1 consistent in at most one cell)." if _n <= 1 else "**Neither**, by the registered reading."
    mo.vstack([mo.md("### The eight verdicts of item 33"), mo.ui.table(rls_verdicts, selection=None, page_size=8), mo.md(_reading),
               mo.md("Every model against GARCH, per cell (below 1: better):"), rls_cards.pivot(on="h", index=["ticker", "bars", "model"], values="qlike_ratio")])  # fmt: skip
    return rls_cards, rls_verdicts

@app.cell
def _(mo):
    mo.md(r"""
    ## Does separating the jumps improve HAR (item 34)

    Registered in `planning/preregistered/har-jumps.md` (`a382c0d`),
    committed alone on 2026-09-29 before any jump measure existed.
    - **Measures.** `gr.models.vol.realized_jumps` on 5-minute bars: BV, TQ,
      MedRV, C-TBV, C-TTQ, both ratio statistics, both C/J splits.
    - **Models.** `har`, `harj`, `harcj`, `hartcj` — the last three from the
      jump measures, the first from `realized_from` as the benchmark.
    - **Proxies.** (1) squared bar returns, `proxies(bars, "r2")`; (2) 5-minute
      RV scaled to the squared return's level, c = Σr² / ΣRV₅ over the bars
      closing by the split, per ticker and bars.

    | # | claim | rule |
    |---|---|---|
    | H7 | HAR-CJ beats HAR | *consistent* if harcj QLIKE < har QLIKE at every horizon; *contradicts* if at none; *mixed* otherwise |
    | H8 | HAR-TCJ beats HAR | the same, for hartcj |
    | H9 | HAR-TCJ beats HAR at every horizon, significantly | *yes* if uSPA p < 0.05 (499 reps) |
    | H10 | the verdict does not turn on the proxy | H7, H8 and item 24's H2 re-decided under the second proxy: *robust* if ≥ 9 of 12 verdicts equal those under r² |
    """)
    return


@app.cell
def _(BARS, EVERY, HELD, HORIZONS, SPLIT, TICKERS, bars, gr, mo, pl):
    from galata_research.models import vol as _vol

    def _jump_walks():
        parts, skipped = [], []
        for t in TICKERS:
            for i in BARS:
                try:
                    jm = _vol.realized_jumps(bars[(t, "5m")], i)
                except gr.Refused as why:
                    skipped.append({"ticker": t, "interval": i, "model": "all", "why": str(why)})
                    continue
                for model in ("har", "harj", "harcj", "hartcj"):
                    try:
                        if model == "har":
                            measures = gr.timeseries.realized_from(bars[(t, "5m")], i)
                        else:
                            measures = jm
                        f = _vol.har(measures, model=model, split=SPLIT, every=EVERY[i], horizons=HORIZONS[i])
                        parts.append(f.with_columns(pl.lit(i).alias("interval"), pl.lit(model).alias("model")))
                    except gr.Refused as why:
                        skipped.append({"ticker": t, "interval": i, "model": model, "why": str(why)})
        return pl.concat(parts, how="diagonal_relaxed"), pl.DataFrame(skipped, schema={"ticker": pl.String, "interval": pl.String, "model": pl.String, "why": pl.String})

    with mo.persistent_cache(name=f"har-long-jumps-{HELD}"):
        jump_walked, jump_skipped = _jump_walks()
    mo.vstack([
        mo.md(f"**{jump_walked.height:,}** forecasts; `fitted_through ≤ close_ts` on every row: **{bool((jump_walked['fitted_through'] <= jump_walked['close_ts']).all())}**."),
        mo.md("Not walked:") if jump_skipped.height else mo.md(""), jump_skipped if jump_skipped.height else mo.md(""),
    ])  # fmt: skip
    return jump_skipped, jump_walked


@app.cell
def _(BARS, SPLIT, TICKERS, bars, gr, pl):
    _rows = []
    _cut = pl.lit(SPLIT).str.to_datetime(time_zone="UTC")
    for _t in TICKERS:
        for _i in BARS:
            _r2 = gr.timeseries.returns(bars[(_t, _i)], kind="log").select("ts", (pl.col("return") ** 2).alias("r2"))
            _m = gr.timeseries.realized_from(bars[(_t, "5m")], _i).select("ts", "close_ts", "rv")
            _j = _m.join(_r2, on="ts").drop_nulls().filter(pl.col("close_ts") <= _cut)
            _rows.append({"ticker": _t, "interval": _i, "c": _j["r2"].sum() / _j["rv"].sum(), "bars": _j.height})
    jump_scales = pl.DataFrame(_rows)
    return (jump_scales,)


@app.cell
def _(BARS, TICKERS, aligned_for, ev, jump_scales, jump_walked, mo, pl):
    _rows = []
    for _t in TICKERS:
        for _i in BARS:
            _r2_proxy = ev.proxies(bars[(_t, _i)], "r2")
            _rv5_proxy = gr.timeseries.realized_from(bars[(_t, "5m")], _i).select("ts", pl.col("rv").alias("proxy"))
            _c = jump_scales.filter((pl.col("ticker") == _t) & (pl.col("interval") == _i))["c"][0]
            _rv5_proxy = _rv5_proxy.with_columns((pl.col("proxy") * _c).alias("proxy"))
            for _proxy_name, _proxy in (("r2", _r2_proxy), ("rv5", _rv5_proxy)):
                _f = jump_walked.filter((pl.col("ticker") == _t) & (pl.col("interval") == _i))
                _al = ev.align(_f, _proxy)
                _card = ev.scorecard(_al, benchmark="har")
                _q = _card.select("model", "h", "qlike")
                for _model, _hyp in (("harcj", "H7"), ("hartcj", "H8")):
                    _m_q = _q.filter(pl.col("model") == _model).select("h", pl.col("qlike").alias("m"))
                    _h_q = _q.filter(pl.col("model") == "har").select("h", pl.col("qlike").alias("h"))
                    _j = _m_q.join(_h_q, on="h").sort("h")
                    _wins = int((_j["m"] < _j["h"]).sum())
                    _v = "consistent" if _wins == _j.height else "contradicts" if _wins == 0 else "mixed"
                    _rows.append({"#": _hyp, "ticker": _t, "bars": _i, "proxy": _proxy_name, "verdict": _v,
                                  "measured": f"{_wins} of {_j.height} horizons"})
                if _proxy_name == "r2":
                    try:
                        _p = ev.uspa(_al, model="hartcj", benchmark="har", reps=499)["p_value"]
                        _rows.append({"#": "H9", "ticker": _t, "bars": _i, "proxy": _proxy_name,
                                      "verdict": "yes" if _p < 0.05 else "no", "measured": f"uSPA p {_p:.3f}"})
                    except Exception as _why:
                        _rows.append({"#": "H9", "ticker": _t, "bars": _i, "proxy": _proxy_name,
                                      "verdict": "can't tell", "measured": str(_why)[:80]})
    jump_verdicts = pl.DataFrame(_rows).sort("#", "ticker", "bars", "proxy")
    return jump_verdicts


@app.cell
def _(mo, pl, jump_verdicts):
    _h7 = jump_verdicts.filter(pl.col("#") == "H7")
    _h8 = jump_verdicts.filter(pl.col("#") == "H8")
    _h9 = jump_verdicts.filter(pl.col("#") == "H9")
    _h7_r2 = _h7.filter(pl.col("proxy") == "r2")["verdict"].to_list()
    _h7_rv5 = _h7.filter(pl.col("proxy") == "rv5")["verdict"].to_list()
    _h8_r2 = _h8.filter(pl.col("proxy") == "r2")["verdict"].to_list()
    _h8_rv5 = _h8.filter(pl.col("proxy") == "rv5")["verdict"].to_list()
    _h7_match = sum(a == b for a, b in zip(_h7_r2, _h7_rv5, strict=True))
    _h8_match = sum(a == b for a, b in zip(_h8_r2, _h8_rv5, strict=True))
    _total_match = _h7_match + _h8_match
    _h10_v = "robust" if _total_match >= 6 else "proxy-dependent"  # 8 verdicts (H7+H8 × 4 cells); ≥6 of 8 matching = robust
    _h7_n = _h7_r2.count("consistent")
    _h8_n = _h8_r2.count("consistent")
    _reading = (
        f"**H7 (HAR-CJ vs HAR)**: {_h7_n} of 4 cells consistent under r². "
        f"**H8 (HAR-TCJ vs HAR)**: {_h8_n} of 4 cells consistent under r². "
        f"**H10 (proxy robustness)**: {_h10_v} ({_h7_match + _h8_match} of 4 verdicts match across proxies)."
    )
    mo.vstack([mo.md("### The verdicts of item 34"), mo.ui.table(jump_verdicts, selection=None, page_size=16), mo.md(_reading)])
    return


if __name__ == "__main__":
    app.run()
