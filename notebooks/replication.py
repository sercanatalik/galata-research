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
    # Does the BTC volatility study replicate on ETH and HYPE?

    `garch.py` reached its verdicts on BTC, and BTC was the only asset looked
    at. Each verdict was a finding on one series. Here the same notebook is
    replayed, unchanged, for ETH and HYPE at 1d, 4h and 1h. For each run it
    is embedded with the ticker and interval set and every model walked. The
    ticker only changes the data, never a rule.

    Before ETH or HYPE was run, `planning/registered/replication.md` fixed
    three things: the claims, BTC's verdict on each (the prediction), and what
    counts as a replication. Replication is a harder test than it sounds. Hou,
    Xue and Zhang (2020) re-ran 452 published anomalies, and 65% failed a
    single t > 1.96 hurdle.

    The claims are ⑬'s seven rows and four more, computed from the embedded
    run's own frames:
    - **GARCH outside the uniform multi-horizon MCS** (Quaedvlieg 2021, at 90%);
    - **HARQ beats GARCH at every horizon at once**: uSPA p < 0.05;
    - **feedback tracks the target better**: its |ln(realized/τ)| is below
      open loop's for every model (Devanathan, Rueter, Boyd et al. 2026);
    - **feedback's Sharpe gain is not significant**: no model's
      Ledoit–Wolf p is below 0.05 after Holm (the refutation in ⑭).

    ETH and BTC move together (Katsiampa 2019). ETH is the easier test.
    HYPE is younger (daily from 2024-12), thinner and a different kind of
    token, so it is the harder one.
    """)
    return


@app.cell
def _():
    from garch import app as garch_app

    return (garch_app,)


@app.function
def settings(ticker: str, interval: str) -> dict:
    """The embedded run's UI values, as garch.py's defaults compute them for this ticker and interval."""
    from types import SimpleNamespace as _Value

    horizons = {"1h": {"1 bar": 1, "1 day": 24, "1 week": 168}, "4h": {"1 bar": 1, "1 day": 6, "1 week": 42}, "1d": {"1 day": 1, "1 week": 7, "1 month": 30}}[interval]
    models = ["ewma", "garch", "gjr", "egarch"] + (["har", "harq"] if interval != "1h" else [])
    return {
        "ticker": _Value(value=ticker), "interval": _Value(value=interval), "share": _Value(value=0.7), "deseason": _Value(value=False),
        "HORIZONS": horizons, "EVERY": {"1h": 24, "4h": 6, "1d": 5}[interval], "horizon": _Value(value=1),
        "walk_models": _Value(value=models), "go": _Value(value=True),
    }  # fmt: skip


@app.function
def extra_claims(defs, gr, pl) -> list[dict]:
    """The four claims beyond ⑬, each by its registered rule, from an embedded run's definitions."""
    import numpy as np

    ev = gr.models.evaluate
    aligned, targeted, tau, per_year = defs["aligned"], defs["targeted"], defs["tau"], defs["per_year"]
    walked = set(aligned["model"].unique().to_list())
    rows = []

    def row(claim, verdict, number, rule):
        rows.append({"claim": claim, "verdict": verdict, "on this record": number, "rule": rule})

    try:
        u = ev.mcs_horizons(aligned, uniform=True)
        g = u.filter(pl.col("model") == "garch")
        row("GARCH outside the multi-horizon MCS", "yes" if not g["included"][0] else "no", f"uMCS p {g['pvalue'][0]:.3f}", "yes if GARCH is outside the 90% uniform multi-horizon MCS")
    except Exception as why:  # a set the data cannot form is reported, not hidden
        row("GARCH outside the multi-horizon MCS", "can't tell", str(why)[:60], "yes if GARCH is outside the 90% uniform multi-horizon MCS")
    if {"harq", "garch"} <= walked:
        p = ev.uspa(aligned, model="harq", benchmark="garch", reps=499)["p_value"]
        row("HARQ beats GARCH at every horizon", "yes" if p < 0.05 else "no", f"uSPA p {p:.3f}", "yes if uSPA p < 0.05")
    else:
        row("HARQ beats GARCH at every horizon", "can't tell", "—", "needs HARQ walked (no 1h RV at 1h)")

    scored = targeted.drop_nulls("net")
    models = sorted({t.split(" ")[0] for t in scored["trial"].unique().to_list() if t.endswith("feedback band 0")})
    better, pvalues = 0, []
    for m in models:
        fb = scored.filter(pl.col("trial") == f"{m} feedback band 0").select("ts", pl.col("net").alias("a"))
        iv = scored.filter(pl.col("trial") == f"{m} inverse_vol band 0").select("ts", pl.col("net").alias("b"))
        pair = fb.join(iv, on="ts").sort("ts")
        err = [abs(float(np.log(float(pair[c].std()) * per_year**0.5 / tau))) for c in ("a", "b")]
        better += err[0] < err[1]
        pvalues.append(ev.sharpe_difference(pair["a"], pair["b"])["p_boot"])
    row("feedback tracks the target better", "yes" if models and better == len(models) else "no" if models else "can't tell", f"{better} of {len(models)} models", "yes if feedback's vol error is lower for every model")
    k = len(pvalues)
    holm = sum(1 for i, p in enumerate(sorted(pvalues)) if all(q <= 0.05 / (k - j) for j, q in enumerate(sorted(pvalues)[: i + 1])))
    row("feedback's Sharpe gain is not significant", "yes" if k and holm == 0 else "no" if k else "can't tell", f"min p {min(pvalues):.3f}, {holm} Holm rejections" if k else "—", "yes if no model's Ledoit–Wolf p rejects after Holm at 5%")
    return rows


@app.cell
def _(mo):
    tickers = mo.ui.multiselect(["BTC", "ETH", "HYPE"], value=["BTC", "ETH", "HYPE"], label="tickers")
    intervals = mo.ui.multiselect(["1d", "4h", "1h"], value=["1d", "4h", "1h"], label="bars")
    go = mo.ui.run_button(label="replay the study")
    mo.hstack([tickers, intervals, go])
    return go, intervals, tickers


@app.cell
async def _(garch_app, go, gr, intervals, mo, pl, tickers):
    mo.stop(not go.value, mo.md("Press **replay the study**: each (ticker, bars) replays garch.py from start to end, minutes each."))
    _rows = []
    for _t in tickers.value:
        for _i in intervals.value:
            _run = await garch_app.clone().embed(defs=settings(_t, _i))
            _d = _run.defs
            _claims = _d["verdicts"].to_dicts() + extra_claims(_d, gr, pl)
            _rows += [{"ticker": _t, "bars": _i, "frontier": _d["bars"]["close_ts"].max(), **c} for c in _claims]
    replayed = pl.DataFrame(_rows)
    return (replayed,)


@app.cell
def _(mo, pl, replayed):
    _btc = replayed.filter(pl.col("ticker") == "BTC").select("bars", "claim", pl.col("verdict").alias("btc"))
    compared = (
        replayed.filter(pl.col("ticker") != "BTC")
        .join(_btc, on=["bars", "claim"], how="left")
        .with_columns(
            pl.when(pl.col("btc").is_null() | (pl.col("btc") == "can't tell") | (pl.col("verdict") == "can't tell"))
            .then(pl.lit("can't tell"))
            .when(pl.col("verdict") == pl.col("btc"))
            .then(pl.lit("replicates"))
            .otherwise(pl.lit("differs"))
            .alias("replication")
        )
    )
    _grid = compared.select("claim", "bars", "ticker", "replication").pivot(on=["ticker", "bars"], index="claim", values="replication")
    mo.vstack(
        [
            mo.md("## Each claim, by ticker and bar: does ETH's or HYPE's verdict equal BTC's?"),
            _grid,
            mo.md("Every verdict, with its number and rule:"),
            replayed.select("ticker", "bars", "claim", "verdict", "on this record", "rule"),
        ]
    )
    return (compared,)


@app.cell
def _(mo):
    mo.md(r"""
    ## References

    Hou, Xue and Zhang 2020, "Replicating Anomalies", *RFS* 33(5):2019–2133
    · Katsiampa 2019, "Volatility co-movement between Bitcoin and Ether",
    *FRL* 30:221–227 · Quaedvlieg 2021, *JBES* 39(1):40–53 · Ledoit and Wolf
    2008, *J. Empirical Finance* 15(5):850–859 · Holm 1979, *Scand. J.
    Statistics* 6(2):65–70 · Devanathan, Rueter, Boyd, Candès, Hastie,
    Kochenderfer et al. 2026, arXiv 2603.01298. The seven claims of ⑬ carry
    their sources in `garch.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
