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
    # Does the volatility study hold on six years of BTC and ETH?

    Every verdict of `garch.py` rests on the record, whose 1d bars start in
    2023-02 and 4h bars in 2024-06. Here the same notebook is replayed,
    unchanged, on Binance's archived 1m klines from 2020-01, resampled to
    whole bars. The replay uses `replication.py`'s settings and its four
    extra claims; the only new values are the source and the end.

    Registered in `planning/preregistered/vol-study-long-history.md`,
    committed alone (`2d8d580`) before the run. Each cell's prediction is
    that ticker's verdict on the record. *HAR beats GARCH* and *HARQ beats
    GARCH* were already run on nearly this data by item 24, so they are
    shown but not counted.
    """)
    return


@app.cell
def _():
    from garch import app as garch_app
    from replication import extra_claims, settings

    return extra_claims, garch_app, settings


@app.cell
def _():
    UNTIL = "2026-09-27T00:00Z"
    CELLS = [(t, b) for t in ("BTC", "ETH") for b in ("1d", "4h")]
    NOT_BLIND = ("HAR beats GARCH", "HARQ beats GARCH at every horizon")
    # The registration's prediction table: each ticker's verdict on the record (— undecided there).
    RECORD = {
        "t beats normal": ("yes", "yes", "yes", "yes"),
        "no leverage effect": ("yes", "no", "yes", "yes"),
        "α+β≈1 intraday is the daily cycle": ("—", "no", "—", "no"),
        "HAR beats GARCH": ("yes", "—", "yes", "—"),
        "something beats GARCH(1,1)": ("no", "yes", "yes", "yes"),
        "better σ ≠ better P&L": ("yes", "yes", "yes", "yes"),
        "targeting does not cut drawdown per vol": ("no", "no", "no", "no"),
        "GARCH outside the multi-horizon MCS": ("yes", "yes", "yes", "no"),
        "HARQ beats GARCH at every horizon": ("yes", "—", "yes", "—"),
        "feedback tracks the target better": ("yes", "yes", "no", "yes"),
        "feedback's Sharpe gain is not significant": ("yes", "yes", "yes", "yes"),
    }
    return CELLS, NOT_BLIND, RECORD, UNTIL


@app.cell
async def _(CELLS, UNTIL, extra_claims, garch_app, gr, mo, pl, settings):
    from types import SimpleNamespace as _Value

    import altair as _alt

    # An embedded run renders every chart; altair refuses one over 5,000 rows, and six years of 4h bars is ~14,600.
    # Display only: no rule or number of garch.py depends on it. Under marimo's kernel the limit is already off;
    # a plain-Python embed (the first run of this, 2026-09-28) failed at 4h on it.
    _alt.data_transformers.disable_max_rows()

    async def _replay():
        rows = []
        for t, b in CELLS:
            defs = {**settings(t, b), "source": _Value(value="binance 1m klines"), "until": _Value(value=UNTIL)}
            try:
                d = (await garch_app.clone().embed(defs=defs)).defs
                claims = d["verdicts"].to_dicts() + extra_claims(d, gr, pl)
                span = (d["bars"]["ts"].min(), d["bars"]["close_ts"].max(), d["split"])
            except Exception as why:  # a replay that cannot finish decides nothing: every claim is can't tell
                claims = [{"claim": "the replay", "verdict": "can't tell", "on this record": f"garch.py failed: {type(why).__name__}: {str(why)[:80]}", "rule": "—"}]
                span = (None, None, None)
            rows += [{"ticker": t, "bars": b, "first": span[0], "last": span[1], "split": span[2], **c} for c in claims]
        return pl.DataFrame(rows)

    with mo.persistent_cache(name=f"vol-long-v2-{UNTIL}"):
        replayed = await _replay()
    return (replayed,)


@app.cell
def _(CELLS, NOT_BLIND, RECORD, mo, pl, replayed):
    _holds = {"consistent": "yes", "yes": "yes", "contradicts": "no", "no": "no"}
    _named = replayed.with_columns(pl.col("claim").str.replace(r"at \d+h ", "intraday "))
    _long = {(r["claim"], r["ticker"], r["bars"]): _holds.get(r["verdict"], "—") for r in _named.iter_rows(named=True)}
    _rows = []
    for _claim, _rec in RECORD.items():
        for (_t, _b), _r in zip(CELLS, _rec, strict=True):
            _l = _long.get((_claim, _t, _b), "—")
            _cmp = "can't tell" if "—" in (_l, _r) else "repeats" if _l == _r else "differs"
            _rows.append({"claim": _claim, "ticker": _t, "bars": _b, "record": _r, "long history": _l, "result": _cmp, "counted": _claim not in NOT_BLIND})
    compared = pl.DataFrame(_rows)
    _per = (
        compared.filter(pl.col("result") != "can't tell").group_by("claim", maintain_order=True)
        .agg((pl.col("result") == "repeats").sum().alias("repeats"), pl.len().alias("decided"), pl.col("counted").first())
        .with_columns(
            pl.when(pl.col("repeats") == pl.col("decided")).then(pl.lit("holds on long history"))
            .when((pl.col("decided") - pl.col("repeats")) * 2 >= pl.col("decided")).then(pl.lit("sample-specific"))
            .otherwise(pl.lit("mixed")).alias("reading")
        )
    )  # fmt: skip
    _head = compared.filter(pl.col("counted") & (pl.col("result") != "can't tell"))
    headline = f"**{int((_head['result'] == 'repeats').sum())} of {_head.height}** of the record's decided cells repeat on six years (the two claims item 24 already ran are left out)."
    _grid = compared.with_columns(pl.concat_str("record", pl.lit(" → "), "long history").alias("cell"), pl.concat_str("ticker", pl.lit(" "), "bars").alias("where")).pivot(on="where", index="claim", values="cell")
    mo.vstack([
        mo.md("## Each claim: the record's verdict → six years' verdict"), _grid,
        mo.md(headline), mo.md("By the registered rule, per claim:"), _per,
        mo.md("The samples, and every verdict with its number and rule:"),
        replayed.select("ticker", "bars", "first", "last", "split").unique().sort("ticker", "bars"),
        replayed.select("ticker", "bars", "claim", "verdict", "on this record", "rule"),
    ])  # fmt: skip
    return compared, headline


if __name__ == "__main__":
    app.run()
