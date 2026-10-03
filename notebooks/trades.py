import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import polars as pl

    import galata_research as gr
    from galata_research import backtest, stats, studies, trades

    return alt, backtest, gr, mo, pl, stats, studies, trades


@app.cell
def _(mo):
    mo.md("""
    # Score the trades

    A Sharpe ratio summarises a return series. It does not say how often a
    trial wins, what a winning trade is worth against a losing one, how far a
    trade goes against the position before it closes, or how long a
    drawdown lasts. This notebook gives those figures for the trials the
    other studies run. The metrics come from investing-algorithm-framework's
    list (`planning/investing-algorithm-framework.md`), and each one is pinned
    to a hand-worked case in `tests/stats.py` and `tests/trades.py`.

    - **A trade** is a stretch of bars held on one side. A resize stays in
      the trade, a flip starts a new one. Each trade is charged its own
      turnover, the exit included, and funding when it is charged.
    - **MAE and MFE** are the price's worst and best move from the entry
      close over the held bars' lows and highs. They are per unit of price,
      not scaled by the position.
    - **These are descriptions, not verdicts.** Whether an edge survives the
      search that found it is a question for the Deflated Sharpe Ratio and
      PBO (`deflated_sharpe.py`, `overfitting.py`), asked of every trial.
    - Daily bars, 0.045% taker fee on turnover. **Funding is not charged.**
    """)
    return


@app.cell
def _(backtest, gr, pl, studies):
    EVER = ("2020-01-01T00:00Z", "2100-01-01T00:00Z")
    PER_YEAR = gr.timeseries.periods_per_year("1d")
    bars = gr.market.candles(["BTC", "ETH"], "1d", *EVER)
    trials = pl.concat(
        [
            studies.donchian_ensemble(bars, sized=True),
            studies.donchian_ensemble(bars, sized=False),
            studies.moving_average(bars, [20], [100], sides=["long_flat", "long_short"]),
            studies.momentum(bars, [60]),
            studies.trial(bars, pl.lit(1.0), "buy and hold", fee=backtest.TAKER_FEE),
        ]
    )
    return PER_YEAR, bars, trials


@app.cell
def _(PER_YEAR, mo, stats, trials):
    described = stats.describe(trials, PER_YEAR).sort("ticker", "sharpe", descending=[False, True])
    mo.vstack(
        [
            mo.md("## The return series, per trial and ticker"),
            described,
            mo.md(
                "Sharpe and Sortino are annualised by √365. Calmar is CAGR over the worst drawdown, and Omega "
                "is gains over losses about zero. The Ulcer index is the root mean square of the drawdown, so "
                "it counts depth and length together. `drawdown_periods` is the longest time spent below a "
                "previous peak, in days."
            ),
        ]
    )
    return


@app.cell
def _(PER_YEAR, bars, mo, trades, trials):
    scored = trades.summary(trials, bars, periods_per_year=PER_YEAR).sort("ticker", "trial")
    mo.vstack(
        [
            mo.md("## The trades, per trial and ticker"),
            scored,
            mo.md(
                "Win rates and ratios count closed trades only: a trade still held at the end is unrealized. "
                "`profit_factor` is the sum of winning trades over the sum of losing ones, and `win_loss` is "
                "the mean win over the mean loss. A ratio with nothing to divide by is null, not infinite. "
                "`exposure` is the share of days held. A volatility-sized ensemble holds a changing size, not "
                "a changing side, so its trades are its holding stretches."
            ),
        ]
    )
    return


@app.cell
def _(mo, trials):
    names = sorted(trials["trial"].unique().to_list())
    pick = mo.ui.dropdown(names, value="donchian unsized", label="trial")
    ticker = mo.ui.dropdown(sorted(trials["ticker"].unique().to_list()), value="BTC", label="ticker")
    mo.hstack([pick, ticker], justify="start")
    return pick, ticker


@app.cell
def _(alt, bars, mo, pick, pl, ticker, trades, trials):
    _one = trials.filter((pl.col("trial") == pick.value) & (pl.col("ticker") == ticker.value))
    _table = trades.table(_one, bars).filter(~pl.col("open"))
    _scatter = (
        alt.Chart(_table.with_columns(pl.when(pl.col("net") > 0).then(pl.lit("win")).otherwise(pl.lit("loss")).alias("outcome")))
        .mark_circle(size=40, opacity=0.7)
        .encode(
            x=alt.X("mae:Q", title="MAE (worst move against, from entry)", axis=alt.Axis(format="%")),
            y=alt.Y("net:Q", title="trade net return", axis=alt.Axis(format="%")),
            color=alt.Color("outcome:N", scale=alt.Scale(domain=["win", "loss"])),
            tooltip=["entry_ts:T", "exit_ts:T", "bars:Q", alt.Tooltip("net:Q", format=".2%"), alt.Tooltip("mae:Q", format=".2%"), alt.Tooltip("mfe:Q", format=".2%")],
        )
        .properties(height=280, width=520)
    )
    mo.vstack(
        [
            mo.md(f"## `{pick.value}` on {ticker.value}: each closed trade against its worst excursion"),
            _scatter,
            mo.md(
                f"{_table.height} closed trades. If the losers cluster beyond an MAE the winners rarely reach, "
                "a stop there would have cut losses without cutting wins. Any such stop is a new trial, though, "
                "and has to be counted as one."
            ),
        ]
    )
    return


@app.cell
def _(alt, mo, pick, pl, stats, ticker, trials):
    _one = trials.filter((pl.col("trial") == pick.value) & (pl.col("ticker") == ticker.value))
    _months = stats.period_returns(_one, "1mo").with_columns(
        pl.col("period").dt.year().cast(pl.String).alias("year"),
        pl.col("period").dt.strftime("%b").alias("month"),
    )
    _order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    _heat = (
        alt.Chart(_months)
        .mark_rect()
        .encode(
            x=alt.X("month:O", sort=_order, title=None),
            y=alt.Y("year:O", title=None),
            color=alt.Color("return:Q", scale=alt.Scale(scheme="redblue", domainMid=0), legend=alt.Legend(format="%")),
            opacity=alt.condition("datum.full", alt.value(1.0), alt.value(0.4)),
            tooltip=["year:O", "month:O", alt.Tooltip("return:Q", format=".2%"), "n:Q", "bars:Q", "full:N"],
        )
        .properties(height=200, width=520)
    )
    _full = _months.filter(pl.col("full"))
    _share = _full.filter(pl.col("return") > 0).height / _full.height if _full.height else None
    mo.vstack(
        [
            mo.md("## Month by month"),
            _heat,
            mo.md(
                "Faded cells are partial months, cut by the start or end of the sample or by a hole. "
                + (f"**{_share:.0%}** of the {_full.height} full months were positive." if _share is not None else "No month is full.")
            ),
        ]
    )
    return


@app.cell
def _(PER_YEAR, alt, mo, pick, pl, stats, ticker, trials):
    WINDOW = 180
    _one = trials.filter((pl.col("trial") == pick.value) & (pl.col("ticker") == ticker.value))
    _rolling = stats.rolling_sharpe(_one, WINDOW).with_columns((pl.col("sharpe") * PER_YEAR**0.5).alias("sharpe")).drop_nulls("sharpe")
    _line = (
        alt.Chart(_rolling)
        .mark_line(strokeWidth=1)
        .encode(x=alt.X("ts:T", title=None), y=alt.Y("sharpe:Q", title=f"annualised Sharpe, last {WINDOW} days"))
        .properties(height=220, width=520)
    )
    _zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    mo.vstack(
        [
            mo.md(f"## Is the edge steady, or one stretch? Sharpe over a rolling {WINDOW} days"),
            _line + _zero,
            mo.md(
                "A window that holds a hole is left blank rather than shortened. A {WINDOW}-day Sharpe has a "
                "standard error of about 1.4 annualised, so the swings here are mostly noise."
                .replace("{WINDOW}", str(WINDOW))
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
