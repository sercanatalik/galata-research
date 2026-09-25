import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import polars as pl

    import galata_research as gr
    from galata_research import backtest, stats, studies

    return alt, backtest, gr, mo, pl, stats, studies


@app.cell
def _(mo):
    mo.md("""
    # A pre-registered test: the Donchian ensemble

    Zarattini, Pagani and Barbon (SSRN 5209907, 2025) report a Donchian
    ensemble with volatility sizing at a Sharpe of 1.58 on a 20-coin
    rotation. The rules, the trials (N = 4) and what would count as support
    were frozen in `planning/preregistered/donchian-ensemble.md`, and
    committed before any of this ran. This notebook runs exactly that, once.

    - Nine lookbacks (5 … 360 days); a breakout above the previous L closes'
      high opens each one; a trailing stop at the channel midpoint closes it.
    - Equal-weight ensemble; 25% volatility target on 90-day vol, capped at 1×.
    - Long-only, daily, 0.045% taker fees. **Funding is not charged.**
    """)
    return


@app.cell
def _(backtest, gr, pl, studies):
    EVER = ("2020-01-01T00:00Z", "2100-01-01T00:00Z")
    PER_YEAR = 365
    bars = gr.market.candles(["BTC", "ETH"], "1d", *EVER)
    trials = pl.concat(
        [
            studies.donchian_ensemble(bars, sized=True),
            studies.donchian_ensemble(bars, sized=False),
            # Buy-and-hold, the null PBO ranks against. It is not one of the registered N.
            studies._trial(bars, pl.lit(1.0), "buy and hold", backtest.TAKER_FEE),
        ]
    )
    # Every trial is scored over the same days: from when the sized ensemble first has a position.
    grid = studies.matrix(trials)
    aligned = trials.filter(pl.col("ts").is_in(grid["ts"].implode()))
    scores = studies.summary(aligned, PER_YEAR).sort("ticker", "trial")
    return PER_YEAR, aligned, bars, grid, scores


@app.cell
def _(grid, mo, scores):
    mo.vstack(
        [
            mo.md(f"## The trials, over the same {grid.height:,} days ({grid['ts'].min():%Y-%m-%d} → {grid['ts'].max():%Y-%m-%d})"),
            scores.select("trial", "ticker", "periods", "sharpe_annual", "total_net", "total_gross", "skew", "kurt"),
        ]
    )
    return


@app.cell
def _(alt, aligned, mo, pl):
    curves = (
        aligned.drop_nulls("net")
        .with_columns(((pl.col("net") + 1).cum_prod().over("trial", "ticker")).alias("growth of 1"))
        .select("ts", "ticker", "trial", "growth of 1")
    )
    chart = (
        alt.Chart(curves)
        .mark_line(strokeWidth=1)
        .encode(x="ts:T", y=alt.Y("growth of 1:Q", scale=alt.Scale(type="log")), color="trial:N", row="ticker:N")
        .properties(height=180, width="container")
    )
    mo.vstack([mo.md("## Growth of 1, net of fees, gross of funding"), chart])
    return


@app.cell
def _(PER_YEAR, grid, mo, pl, scores, stats):
    registered = scores.filter(pl.col("trial").str.starts_with("donchian"))
    btc = scores.filter((pl.col("trial") == "donchian sized") & (pl.col("ticker") == "BTC")).row(0, named=True)
    hold = scores.filter((pl.col("trial") == "buy and hold") & (pl.col("ticker") == "BTC")).row(0, named=True)
    variance = float(registered["sharpe"].var())
    dsr4 = stats.dsr(btc["sharpe"], btc["periods"], btc["skew"], btc["kurt"], 4, variance)
    dsr70 = stats.dsr(btc["sharpe"], btc["periods"], btc["skew"], btc["kurt"], 70, variance)
    overfit = stats.pbo(grid, blocks=16)
    criteria = [
        ("1. DSR ≥ 0.95 at N = 4", f"{dsr4:.3f}", dsr4 >= 0.95),
        ("2. Sharpe above buy-and-hold BTC", f"{btc['sharpe_annual']:.2f} vs {hold['sharpe_annual']:.2f}", btc["sharpe"] > hold["sharpe"]),
        ("3. PBO < 0.5 (weak with five columns)", f"{overfit['pbo']:.3f}", overfit["pbo"] < 0.5),
    ]
    supported = all(ok for _, _, ok in criteria)
    table = pl.DataFrame({"criterion": [c for c, _, _ in criteria], "measured": [m for _, m, _ in criteria], "met": [ok for _, _, ok in criteria]})
    mo.vstack(
        [
            mo.md("## The registered criteria, for the sized ensemble on BTC"),
            table,
            mo.md(f"For reference, DSR at N = 70 (every trial run in this repository): **{dsr70:.3f}**. V[SR] is taken across the four registered trials: {variance:.2e} per day²."),
            mo.md("### Verdict: **supported on this record**" if supported else "### Verdict: **not supported on this record**"),
        ]
    )
    return btc, dsr4, dsr70, hold, overfit, supported


@app.cell
def _(aligned, mo, pl):
    # Grobys et al.: volatility management "does not change the tail risk". Compare tails
    # relative to each trial's own volatility, so sizing's smaller scale is not mistaken for safety.
    tails = (
        aligned.drop_nulls("net")
        .filter(pl.col("trial").str.starts_with("donchian"))
        .group_by("trial", "ticker")
        .agg(
            pl.col("net").std().alias("sd"),
            pl.col("net").min().alias("worst_day"),
            pl.col("net").filter(pl.col("net") <= pl.col("net").quantile(0.01)).mean().alias("es_1pct"),
        )
        .with_columns(
            (pl.col("worst_day") / pl.col("sd")).alias("worst_in_sd"),
            (pl.col("es_1pct") / pl.col("sd")).alias("es_in_sd"),
        )
        .sort("ticker", "trial")
    )
    mo.vstack(
        [
            mo.md("## Tail risk, in units of each trial's own volatility"),
            mo.md("If sizing tamed the tail, the sized rows would sit closer to zero than the unsized ones. Grobys et al. say they will not."),
            tails.select("trial", "ticker", "worst_day", "worst_in_sd", "es_1pct", "es_in_sd"),
        ]
    )
    return


@app.cell
def _(bars, mo, pl, studies):
    closes = bars.filter(pl.col("ticker") == "BTC").sort("ts").collect()
    active = closes.select("ts", pl.Series("open lookbacks", [round(x * 9) for x in studies._donchian_signal(closes["close"].to_list(), list(studies.DONCHIAN_LOOKBACKS))]))
    mo.vstack([mo.md("## BTC: how many of the nine lookbacks are open"), mo.ui.altair_chart(__import__("altair").Chart(active).mark_area(opacity=0.6).encode(x="ts:T", y="open lookbacks:Q").properties(height=160, width="container"))])
    return


if __name__ == "__main__":
    app.run()
