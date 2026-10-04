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
    # Does *GARCH outside the multi-horizon MCS* turn on the split?

    Item 26 found this the one claim that is sample-specific on six years.
    A verdict from one out-of-sample window can be an artifact of where the
    window starts (Rossi and Inoue 2012; Hansen and Timmermann 2012). Here
    `garch.py` is replayed on the same Binance bars at split shares 0.5, 0.6
    and 0.8. The 0.7 run is item 26's own, read from `vol_long.py`.

    Registered in `planning/preregistered/mcs-by-split.md` (`33f95b8`)
    before any other share was run.
    - **Per cell:** *stable* if the verdict is the same at all four shares,
      *split-dependent* if both *yes* and *no* occur.
    - **The claim:** it turns on the split if two or more cells are
      split-dependent.
    """)
    return


@app.cell
def _():
    from garch import app as garch_app
    from replication import extra_claims, settings
    from vol_long import app as long_app

    return extra_claims, garch_app, long_app, settings


@app.cell
async def _(extra_claims, garch_app, gr, long_app, mo, pl, settings):
    from types import SimpleNamespace as _Pin

    import altair as _charts

    _charts.data_transformers.disable_max_rows()  # display only, as in vol_long.py
    UNTIL = "2026-09-27T00:00Z"
    CELLS = [(t, b) for t in ("BTC", "ETH") for b in ("1d", "4h")]
    SHARES = (0.5, 0.6, 0.8)
    KEEP = ("GARCH outside the multi-horizon MCS", "something beats GARCH(1,1)", "HARQ beats GARCH at every horizon")

    async def _replay():
        rows = []
        for share in SHARES:
            for t, b in CELLS:
                defs = {**settings(t, b), "share": _Pin(value=share), "source": _Pin(value="binance 1m klines"), "until": _Pin(value=UNTIL)}
                try:
                    d = (await garch_app.clone().embed(defs=defs)).defs
                    claims = d["verdicts"].to_dicts() + extra_claims(d, gr, pl)
                    split = d["split"]
                except Exception as why:  # a replay that cannot finish decides nothing
                    claims = [{"claim": k, "verdict": "can't tell", "on this record": f"garch.py failed: {type(why).__name__}: {str(why)[:80]}"} for k in KEEP]
                    split = None
                rows += [{"share": share, "ticker": t, "bars": b, "split": split, "claim": c["claim"], "verdict": c["verdict"], "measured": c["on this record"]}
                         for c in claims if c["claim"] in KEEP]  # fmt: skip
        return pl.DataFrame(rows)

    with mo.persistent_cache(name=f"mcs-split-{UNTIL}"):
        _other = await _replay()
    _seven = (await long_app.embed()).defs["replayed"].filter(pl.col("claim").is_in(KEEP))
    by_share = pl.concat([
        _other,
        _seven.select(pl.lit(0.7).alias("share"), "ticker", "bars", "split", "claim", "verdict", pl.col("on this record").alias("measured")),
    ], how="vertical_relaxed").sort("claim", "ticker", "bars", "share")  # fmt: skip
    return (by_share,)


@app.cell
def _(mo, pl, by_share):
    _mcs = by_share.filter(pl.col("claim") == "GARCH outside the multi-horizon MCS")
    cells = (
        _mcs.group_by("ticker", "bars", maintain_order=True)
        .agg(pl.col("verdict").sort_by("share").alias("verdicts"), pl.col("measured").sort_by("share").alias("uMCS p by share"))
        .with_columns(
            pl.when(pl.col("verdicts").list.contains("yes") & pl.col("verdicts").list.contains("no")).then(pl.lit("split-dependent"))
            .when(pl.col("verdicts").list.unique().list.len() == 1).then(pl.lit("stable"))
            .otherwise(pl.lit("can't tell")).alias("reading")
        )
    )  # fmt: skip
    _n = cells.filter(pl.col("reading") == "split-dependent").height
    _stable = cells.filter(pl.col("reading") == "stable").height
    verdict = "turns on the split" if _n >= 2 else "does not turn on the split" if _stable == 4 else "mixed"
    _grid = by_share.with_columns(pl.concat_str("ticker", pl.lit(" "), "bars").alias("where")).pivot(on="share", index=["claim", "where"], values="verdict", sort_columns=True)
    mo.vstack([
        mo.md("## The MCS claim at shares 0.5, 0.6, 0.7 and 0.8"), cells,
        mo.md(f"**{_n} of 4 cells split-dependent, {_stable} stable: the claim {verdict}** (the registered rule)."),
        mo.md("For context, not decided here: the three ranking claims at every share."), _grid,
        by_share.select("claim", "ticker", "bars", "share", "split", "verdict", "measured"),
    ])  # fmt: skip
    return cells, verdict


if __name__ == "__main__":
    app.run()
