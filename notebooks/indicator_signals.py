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
    # A pre-registered test: five indicator families

    EMA crossovers, RSI and Bollinger mean reversion, MACD and Supertrend are
    the standard vocabulary of the trading side: they are what
    investing-algorithm-framework and its `pyindicators` ship as strategies.
    The families, their grids, N and what would count as support were frozen
    in `planning/preregistered/indicator-signals.md` and committed alone
    (`020ef19`), before this notebook or the code it runs existed. It runs
    exactly that, once.

    - **50 rules × {BTC, ETH} = N = 100**, daily, long/flat and long/short.
    - Indicators from `gr.indicators`, evaluated per contiguous stretch, so
      none spans a hole and none trades before it exists.
    - Next-bar execution, 0.045% taker fee on turnover. **Funding is not charged.**
    - **Support needs all four:** DSR ≥ 0.95 at N = 100, PBO < 0.5, Reality
      Check p < 0.05 against buy-and-hold, and a permuted-bars p_best < 0.05.
    - The expectation, registered: **not supported.** The 66 moving-average
      and momentum trials here gave a DSR of 0.67.
    """)
    return


@app.cell
def _(backtest, gr, pl, studies):
    EVER = ("2020-01-01T00:00Z", "2100-01-01T00:00Z")
    TICKERS = ["BTC", "ETH"]
    PER_YEAR = gr.timeseries.periods_per_year("1d")
    N_BEFORE = 70  # trials run in this repository before this test (whole_set.py)
    bars = studies.shared_days(gr.market.candles(TICKERS, "1d", *EVER))
    trials = studies.indicator_signals(bars)
    hold_bh = studies.trial(bars, pl.lit(1.0), "buy and hold", fee=backtest.TAKER_FEE)
    return N_BEFORE, PER_YEAR, TICKERS, bars, hold_bh, trials


@app.cell
def _(N_BEFORE, PER_YEAR, bars, hold_bh, mo, pl, stats, studies, trials):
    scores = studies.summary(trials, PER_YEAR)
    deflated = stats.deflate(scores)
    dsr_wider = stats.dsr(deflated["sharpe"], deflated["periods"], deflated["skew"], deflated["kurt"], deflated["trials"] + N_BEFORE, deflated["variance"])
    overfit = stats.pbo(studies.matrix(trials), blocks=16)
    rc = stats.reality_check(studies.excess(pl.concat([trials, hold_bh]), "buy and hold"), reps=1000, seed=0)
    bh = studies.summary(hold_bh, PER_YEAR).select("ticker", pl.col("sharpe_annual").alias("buy_and_hold"))
    mo.vstack(
        [
            mo.md(f"## {deflated['trials']} trials over {bars['ts'].min():%Y-%m-%d} → {bars['close_ts'].max():%Y-%m-%d}"),
            mo.hstack(
                [
                    mo.stat(f"{deflated['dsr']:.3f}", label=f"DSR, best of {deflated['trials']}"),
                    mo.stat(f"{dsr_wider:.3f}", label=f"DSR at N = {deflated['trials'] + N_BEFORE}"),
                    mo.stat(f"{overfit['pbo']:.2f}", label=f"PBO, {overfit['trials']} columns"),
                    mo.stat(f"{rc['reality_check']:.3f}", label="Reality Check p vs buy-and-hold"),
                ],
                justify="start",
            ),
            mo.md(
                f"The best is `{deflated['trial']}` on {deflated['ticker']}, annualised Sharpe "
                f"**{deflated['sharpe'] * PER_YEAR**0.5:.2f}**, against a luck benchmark of "
                f"{deflated['benchmark'] * PER_YEAR**0.5:.2f} for the best of {deflated['trials']}."
            ),
            scores.sort("sharpe", descending=True, nulls_last=True).join(bh, on="ticker").head(15),
        ]
    )
    return deflated, dsr_wider, overfit, rc, scores


@app.cell
def _(PER_YEAR, alt, mo, pl, scores):
    _family = pl.col("trial").str.split(" ").list.first().alias("family")
    _side = pl.col("trial").str.split(" ").list.last().alias("side")
    _chart = (
        alt.Chart(scores.with_columns(_family, _side, (pl.col("sharpe") * PER_YEAR**0.5).alias("annual")).drop_nulls("annual"))
        .mark_tick(thickness=2, size=18)
        .encode(
            x=alt.X("annual:Q", title="annualised Sharpe, net of fees"),
            y=alt.Y("family:N", title=None),
            color=alt.Color("side:N"),
            tooltip=["trial:N", "ticker:N", alt.Tooltip("annual:Q", format=".2f")],
        )
        .properties(height=200, width=560)
    )
    mo.vstack([mo.md("## Every trial, by family"), _chart])
    return


@app.cell
def _(mo):
    permute_go = mo.ui.run_button(label="run the permuted-bars test (200 markets, ~2 min)")
    permute_go
    return (permute_go,)


@app.cell
def _(bars, mo, permute_go, studies):
    mo.stop(not permute_go.value, mo.md("*The fourth criterion runs behind the button: the whole search, 200 times.*"))
    permuted = studies.permutation_test(bars, studies.indicator_signals, samples=200, seed=0)
    mo.md(f"**Permuted-bars p_best = {permuted['p_best']:.3f}** over {permuted['samples']} markets (seed {permuted['seed']}): *{permuted['null']}*.")
    return (permuted,)


@app.cell
def _(deflated, mo, overfit, permuted, pl, rc):
    verdicts = pl.DataFrame(
        {
            "criterion": ["DSR ≥ 0.95 at N = 100", "PBO < 0.5", "Reality Check p < 0.05", "permuted-bars p_best < 0.05"],
            "value": [deflated["dsr"], overfit["pbo"], rc["reality_check"], permuted["p_best"]],
            "met": [deflated["dsr"] >= 0.95, overfit["pbo"] < 0.5, rc["reality_check"] < 0.05, permuted["p_best"] < 0.05],
        }
    )
    supported = bool(verdicts["met"].all())
    mo.vstack(
        [
            mo.md("## The verdict, against the registered criteria"),
            verdicts,
            mo.md("**Supported on this record.**" if supported else "**Not supported on this record.**"),
        ]
    )
    return


@app.cell
def _(PER_YEAR, bars, mo, pl, scores, stats, trades, trials):
    _family = pl.col("trial").str.split(" ").list.first()
    _best = scores.drop_nulls("sharpe").sort("sharpe", descending=True).group_by(_family.alias("family"), maintain_order=True).first()
    _keys = _best.select("trial", "ticker")
    _picked = trials.join(_keys, on=["trial", "ticker"])
    _described = stats.describe(_picked, PER_YEAR).join(trades.summary(_picked, bars, periods_per_year=PER_YEAR), on=["trial", "ticker"])
    mo.vstack(
        [
            mo.md("## Described, not tested: the best trial of each family"),
            _described.select(
                "trial", "ticker", "sharpe", "sortino", "max_drawdown", "drawdown_periods",
                "trades", "win_rate", "profit_factor", "win_loss", "mean_bars", "exposure", "mean_mae", "mean_mfe",
            ),  # fmt: skip
            mo.md(
                "Picked as the best of each family, so these are the selected figures: each is the most "
                "flattering row of its own search."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    binance_go = mo.ui.run_button(label="run the secondary: six years of Binance (reads 1m klines, several minutes)")
    binance_go
    return (binance_go,)


@app.cell
def _(PER_YEAR, TICKERS, binance_go, gr, mo, pl, stats, studies):
    mo.stop(not binance_go.value, mo.md("*The secondary sample runs behind the button. It cannot rescue a failure on the record.*"))
    # To the end of the last whole day held for every ticker.
    _end = (
        gr.reference.coverage()
        .filter((pl.col("kind") == "candles") & (pl.col("venue") == "binance-um") & pl.col("ticker").is_in(TICKERS))
        .select((pl.col("last").min() + pl.duration(days=1)).dt.strftime("%Y-%m-%dT00:00Z"))
        .item()
    )
    _fine = gr.reference.candles(TICKERS, "2020-01-01T00:00Z", _end, venues="binance-um").collect()
    binance = studies.shared_days(pl.concat([gr.timeseries.resample(_fine.filter(pl.col("ticker") == t), "1d") for t in TICKERS]))
    b_trials = studies.indicator_signals(binance)
    b_hold = studies.trial(binance, pl.lit(1.0), "buy and hold")
    b_scores = studies.summary(b_trials, PER_YEAR)
    b_deflated = stats.deflate(b_scores)
    b_pbo = stats.pbo(studies.matrix(b_trials), blocks=16)
    b_rc = stats.reality_check(studies.excess(pl.concat([b_trials, b_hold]), "buy and hold"), reps=1000, seed=0)
    b_perm = studies.permutation_test(binance, studies.indicator_signals, samples=200, seed=0)
    mo.vstack(
        [
            mo.md(f"## Secondary: Binance, {binance['ts'].min():%Y-%m-%d} → {binance['close_ts'].max():%Y-%m-%d}"),
            pl.DataFrame(
                {
                    "criterion": ["DSR ≥ 0.95 at N = 100", "PBO < 0.5", "Reality Check p < 0.05", "permuted-bars p_best < 0.05"],
                    "value": [b_deflated["dsr"], b_pbo["pbo"], b_rc["reality_check"], b_perm["p_best"]],
                }
            ),
            mo.md(
                f"Best: `{b_deflated['trial']}` on {b_deflated['ticker']}, annualised Sharpe "
                f"{b_deflated['sharpe'] * PER_YEAR**0.5:.2f}."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
