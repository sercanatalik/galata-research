import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import polars as pl

    import galata_research as gr
    from galata_research import backtest, overlays, stats, studies, trades

    return alt, backtest, gr, mo, overlays, pl, stats, studies, trades


@app.cell
def _(mo):
    mo.md("""
    # A pre-registered test: do stops help?

    Kaminski and Lo (2014) show that a stop-loss with re-entry adds to
    expected return only when returns have momentum, and costs return under a
    random walk. Trading frameworks ship stops, take-profits and cooldowns as
    defaults anyway. The bases, the overlay grid, N and what would count as
    support were frozen in `planning/preregistered/overlay-the-positions.md`
    and committed alone (`f96b81a`) before this notebook or `gr.overlays`
    existed. It runs exactly that, once.

    - **Three bases:** buy-and-hold, `ma 20/100 long_flat`, `donchian unsized`,
      on BTC and ETH daily.
    - **12 overlays each:** stop at 5, 10 or 20%, fixed or trailing, with and
      without a 25% take-profit, then 10 bars flat before the base resumes.
    - **Filled inside the bar:** at the level, or at the open when the bar
      opens beyond it. Levels come from the entry close and earlier bars
      only. A bar that touches both levels is taken as stopped.
    - 0.045% taker fee on every change, exits included. **Funding is not charged.**
    - **H1, per base:** the Reality Check over its 12 overlays against the
      base itself has p < 0.05 on both tickers. **H2:** the best of all 78 has
      DSR ≥ 0.95 and PBO < 0.5.
    - The expectation, registered: **not supported**. The permuted-bars test
      found no structure on this record, which is Kaminski and Lo's case
      where a stop costs.
    """)
    return


@app.cell
def _(backtest, gr, overlays, pl, studies):
    EVER = ("2020-01-01T00:00Z", "2100-01-01T00:00Z")
    PER_YEAR = gr.timeseries.periods_per_year("1d")
    BASES = ["buy and hold", "ma 20/100 long_flat", "donchian unsized"]
    bars = studies.shared_days(gr.market.candles(["BTC", "ETH"], "1d", *EVER))
    bases = pl.concat(
        [
            studies.trial(bars, pl.lit(1.0), "buy and hold", fee=backtest.TAKER_FEE),
            studies.moving_average(bars, [20], [100], sides=["long_flat"]),
            studies.donchian_ensemble(bars, sized=False),
        ]
    )
    overlaid = overlays.grid(bars, bases, cooldown=10)
    every = pl.concat([bases, overlaid.select(bases.columns)])
    return BASES, PER_YEAR, bars, bases, every, overlaid


@app.cell
def _(BASES, every, mo, pl, stats, studies):
    _rows = []
    for _base in BASES:
        _family = every.filter((pl.col("trial") == _base) | pl.col("trial").str.starts_with(f"{_base} + "))
        for _ticker in ("BTC", "ETH"):
            _excess = studies.excess(_family.filter(pl.col("ticker") == _ticker), _base)
            # An overlay that never fired is the base: its excess is zero on every day, with nothing to test.
            _fired = [c for c in _excess.columns if c != "ts" and (_excess[c].abs().max() or 0) > 0]
            _rc = stats.reality_check(_excess.select("ts", *_fired), reps=1000, seed=0) if _fired else None
            _rows.append(
                {
                    "base": _base,
                    "ticker": _ticker,
                    "overlays": _excess.width - 1,
                    "never_fired": _excess.width - 1 - len(_fired),
                    "days": _excess.height,
                    "p": _rc["reality_check"] if _rc else 1.0,
                    "best": _rc["best"].split(" | ")[0] if _rc else None,
                }
            )
    h1 = pl.DataFrame(_rows)
    h1_by_base = h1.group_by("base", maintain_order=True).agg((pl.col("p") < 0.05).all().alias("supported"))
    h1_supported = bool(h1_by_base["supported"].any())
    mo.vstack(
        [
            mo.md("## H1: does any overlay beat its own base, after the search over 12?"),
            h1,
            h1_by_base,
            mo.md(
                "White's Reality Check over each base's 12 overlays, net returns in excess of the base on the "
                "same days (1,000 replicates, Politis–White block, seed 0). An overlay whose stop and take-profit "
                "never fired is the base itself: its excess is zero every day, so it is left out of that test "
                "and counted in `never_fired`. The registration did not foresee this, and it is stated here. "
                "With none fired, p is 1. **H1 is "
                + ("supported**" if h1_supported else "not supported**")
                + " on this record."
            ),
        ]
    )
    return h1, h1_supported


@app.cell
def _(PER_YEAR, every, mo, stats, studies):
    scores = studies.summary(every, PER_YEAR)
    deflated = stats.deflate(scores)
    overfit = stats.pbo(studies.matrix(every), blocks=16)
    h2 = deflated["dsr"] >= 0.95 and overfit["pbo"] < 0.5
    mo.vstack(
        [
            mo.md(f"## H2: the whole set of {deflated['trials']}"),
            mo.hstack(
                [
                    mo.stat(f"{deflated['dsr']:.3f}", label=f"DSR, best of {deflated['trials']}"),
                    mo.stat(f"{overfit['pbo']:.2f}", label=f"PBO, {overfit['trials']} columns"),
                ],
                justify="start",
            ),
            mo.md(
                f"The best is `{deflated['trial']}` on {deflated['ticker']} (annualised Sharpe "
                f"{deflated['sharpe'] * PER_YEAR**0.5:.2f}). H2 is **{'met' if h2 else 'not met'}**. "
                "It cannot support the claim alone: a base can meet it without any overlay."
            ),
        ]
    )
    return (scores,)


@app.cell
def _(BASES, PER_YEAR, alt, mo, pl, scores):
    _base = pl.col("trial").str.split(" + ").list.first().alias("base")
    _plot = scores.with_columns(
        _base,
        pl.col("trial").str.contains(" + ", literal=True).alias("overlaid"),
        (pl.col("sharpe") * PER_YEAR**0.5).alias("annual"),
    ).drop_nulls("annual")
    _dots = (
        alt.Chart(_plot)
        .mark_point(filled=True, size=60)
        .encode(
            x=alt.X("annual:Q", title="annualised Sharpe, net"),
            y=alt.Y("base:N", sort=BASES, title=None),
            color=alt.Color("overlaid:N", title="overlay", scale=alt.Scale(domain=[False, True], range=["#222", "#e45756"])),
            shape="ticker:N",
            tooltip=["trial:N", "ticker:N", alt.Tooltip("annual:Q", format=".2f")],
        )
        .properties(height=180, width=560)
    )
    mo.vstack([mo.md("## Each overlay against its base"), _dots, mo.md("Black: the base. Red: its 12 overlays.")])
    return


@app.cell
def _(PER_YEAR, bars, every, mo, overlaid, pl, stats, trades):
    _described = stats.describe(every, PER_YEAR).with_columns(pl.col("trial").str.split(" + ").list.first().alias("base"))
    _base_rows = _described.filter(pl.col("trial") == pl.col("base")).select("base", "ticker", pl.col("max_drawdown").alias("_mdd"), pl.col("ulcer").alias("_ulcer"))
    _exits = overlaid.group_by("trial", "ticker").agg(
        (pl.col("exit") == "stop").sum().alias("stops"), (pl.col("exit") == "take").sum().alias("takes")
    )
    _tr = trades.summary(overlaid.select(every.columns), bars, periods_per_year=PER_YEAR).select("trial", "ticker", "closed", "win_rate", "profit_factor")
    described = (
        _described.join(_base_rows, on=["base", "ticker"])
        .filter(pl.col("trial") != pl.col("base"))
        .join(_exits, on=["trial", "ticker"])
        .join(_tr, on=["trial", "ticker"])
        .select(
            "trial", "ticker", "sharpe",
            (pl.col("max_drawdown") / pl.col("_mdd")).alias("drawdown_vs_base"),
            (pl.col("ulcer") / pl.col("_ulcer")).alias("ulcer_vs_base"),
            "closed",
            (pl.col("stops") / pl.col("closed")).alias("stopped_share"),
            (pl.col("takes") / pl.col("closed")).alias("taken_share"),
            "win_rate", "profit_factor",
        )  # fmt: skip
        .sort("trial", "ticker")
    )
    mo.vstack(
        [
            mo.md("## Described, not tested: what each overlay did to the drawdown"),
            described,
            mo.md(
                "`drawdown_vs_base` and `ulcer_vs_base` below 1 mean a shallower drawdown than the base. "
                "`stopped_share` is the share of closed trades the stop ended. A stop that cuts the "
                "drawdown and the Sharpe together has bought insurance, not edge."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
