import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import polars as pl

    import galata_research as gr
    from galata_research import stats, studies

    return alt, gr, mo, pl, stats, studies


@app.cell
def _(mo):
    mo.md(r"""
    # The Deflated Sharpe Ratio

    Run enough trials and the best one looks good by chance. Bailey and
    López de Prado (2014) give the Sharpe ratio the best of **N** trials
    reaches when every true Sharpe is zero:

    $$SR_0 = \sqrt{V[\widehat{SR}_n]}\left((1-\gamma)\,\Phi^{-1}\!\left(1-\tfrac1N\right) + \gamma\,\Phi^{-1}\!\left(1-\tfrac1{Ne}\right)\right)$$

    and the **DSR** is the probability that the chosen trial's true Sharpe
    beats that, given its length $T$, skewness $\hat\gamma_3$ and (raw)
    kurtosis $\hat\gamma_4$:

    $$DSR = \Phi\!\left(\frac{(\widehat{SR}-SR_0)\sqrt{T-1}}{\sqrt{1-\hat\gamma_3\widehat{SR}+\tfrac{\hat\gamma_4-1}{4}\widehat{SR}^2}}\right)$$

    `galata_research.stats` reproduces the paper's example exactly (SR₀
    0.1132 a day, DSR 0.9004 at N=100, 0.9505 at N=46). Below, **every**
    trial from both example families counts toward N.
    """)
    return


@app.cell
def _(mo):
    interval = mo.ui.dropdown({"daily": "1d", "4 hours": "4h"}, value="daily", label="bars")
    interval
    return (interval,)


@app.cell
def _(gr, interval, pl, stats, studies):
    EVER = ("2020-01-01T00:00Z", "2100-01-01T00:00Z")
    PER_YEAR = {"1d": 365, "4h": 2190}[interval.value]
    bars = gr.market.candles(["BTC", "ETH"], interval.value, *EVER)
    trials = pl.concat(
        [
            studies.moving_average(bars, [5, 10, 20, 50], [20, 50, 100, 200]),
            studies.momentum(bars, [5, 10, 20, 40, 60, 90, 120]),
        ]
    )
    scores = studies.summary(trials, PER_YEAR)
    verdict = stats.deflate(scores)
    return PER_YEAR, scores, verdict


@app.cell
def _(PER_YEAR, mo, stats, verdict):
    annual = lambda sr: stats.annualize(sr, PER_YEAR)  # noqa: E731
    mo.vstack(
        [
            mo.md(f"## The best of {verdict['trials']} trials: `{verdict['trial']}` on {verdict['ticker']}"),
            mo.hstack(
                [
                    mo.stat(f"{annual(verdict['sharpe']):.2f}", label="its Sharpe, annual"),
                    mo.stat(f"{annual(verdict['benchmark']):.2f}", label=f"SR₀: the best of {verdict['trials']} by luck"),
                    mo.stat(f"{verdict['dsr']:.1%}", label="DSR: P(true Sharpe > SR₀)"),
                ],
                justify="start",
            ),
            mo.md(
                f"Over {verdict['periods']:,} returns with skewness {verdict['skew']:.2f} and kurtosis {verdict['kurt']:.1f}. "
                + ("**Above 95%: worth a closer look.**" if verdict["dsr"] > 0.95 else "**Below 95%: this is what the best of many trials looks like when there may be nothing there.**")
            ),
        ]
    )
    return


@app.cell
def _(alt, mo, pl, scores, stats, verdict):
    # Trials in these families are correlated, so the effective N is smaller than the count.
    # How the verdict moves with N, holding the best trial and V[SR] fixed:
    rows = [
        {
            "N": n,
            "DSR": stats.dsr(verdict["sharpe"], verdict["periods"], verdict["skew"], verdict["kurt"], n, verdict["variance"]),
        }
        for n in [2, 3, 5, 8, 12, 20, 30, 45, 66, 100, 200, 500]
    ]
    sensitivity = pl.DataFrame(rows)
    line = alt.Chart(sensitivity).mark_line(point=True).encode(x=alt.X("N:Q", scale=alt.Scale(type="log")), y=alt.Y("DSR:Q", scale=alt.Scale(domain=[0, 1])))
    rule = alt.Chart(pl.DataFrame({"y": [0.95]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    mo.vstack(
        [
            mo.md("## How the verdict depends on N"),
            mo.md(f"The count of trials run is {verdict['trials']}. Correlated trials are fewer independent ones, so the true N lies somewhere below it. The dashed line is 95%."),
            (line + rule).properties(height=240, width="container"),
        ]
    )
    return


@app.cell
def _(alt, mo, scores):
    spread = (
        alt.Chart(scores.drop_nulls("sharpe_annual"))
        .mark_bar()
        .encode(x=alt.X("sharpe_annual:Q", bin=alt.Bin(maxbins=30), title="Sharpe, annual"), y="count():Q", color="ticker:N")
        .properties(height=200, width="container")
    )
    mo.vstack([mo.md("## Every trial's Sharpe: the variance SR₀ is built from"), spread])
    return


if __name__ == "__main__":
    app.run()
