import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    import galata_research as gr
    from galata_research import factors, stats, studies

    return factors, gr, mo, pl, stats, studies


@app.cell
def _(mo):
    mo.md("""
    # A pre-registered test: weighting a book of perpetuals

    Allocation, not timing. The book is the ten most liquid Binance
    perpetuals at the end of 2021, chosen on what was known that day. Each
    week it is weighted three ways (inverse variance, Hierarchical Risk
    Parity, long-only minimum variance) on three estimates of the 7-day
    covariance (a 90-day sample, RiskMetrics' EWMA, DCC walked forward). That
    makes 9 trials, against equal weight. Frozen in
    `planning/preregistered/build-the-portfolio.md` and committed alone
    (`b63b2ad`) before `gr.models.portfolio` or this notebook existed.

    - **H1** (DeMiguel, Garlappi and Uppal 2009): does any optimised
      portfolio beat 1/N after the search? Expected: no.
    - **H2** (López de Prado 2016): is HRP's realised variance below
      minimum variance's, under every estimator? Expected: no.
    - **H3** (Engle 2002): does DCC's Σ give minimum variance a lower
      realised variance than the 90-day sample? Expected: uncertain.

    0.05% taker on turnover, funding charged daily, weights fixed between
    rebalances. A coin delisted in the span is held to its last bar, and
    the book continues on the survivors.
    """)
    return


@app.cell
def _(factors, gr):
    SPAN = ("2019-09-01T00:00Z", "2026-10-01T00:00Z")
    START, END = "2022-01-01T00:00Z", "2026-10-01T00:00Z"
    PER_YEAR = 365
    panel = factors.panel(gr.reference.daily(None, *SPAN), gr.reference.funding(None, *SPAN))
    members = factors.members(panel, exclude=factors.INDEX_AND_STABLE)
    BOOK = gr.models.portfolio.book(panel, members, "2021-12-31T00:00Z")
    on = gr.models.portfolio.schedule(START, END)
    trials, weights = gr.models.portfolio.study(panel, BOOK, on, END)
    return BOOK, PER_YEAR, panel, trials, weights


@app.cell
def _(BOOK, mo, panel, pl):
    _last = panel.filter(pl.col("ticker").is_in(BOOK)).group_by("ticker").agg(pl.col("ts").max().alias("last_bar")).sort("last_bar")
    mo.vstack([mo.md(f"## The book: {', '.join(BOOK)}"), _last, mo.md("A coin whose last bar is before the span's end was delisted inside it.")])
    return


@app.cell
def _(gr, mo, pl, stats, studies, trials):
    h1 = stats.reality_check(studies.excess(trials, gr.models.portfolio.EQUAL), reps=1000, seed=0)
    h1_supported = h1["reality_check"] < 0.05

    def one_sided(loss, benchmark):
        """DM on squared daily net returns, h = 7: p one-sided for `loss` below `benchmark`."""
        a = trials.filter(pl.col("trial") == loss).sort("ts").select("ts", (pl.col("net") ** 2).alias("a"))
        b = trials.filter(pl.col("trial") == benchmark).sort("ts").select("ts", (pl.col("net") ** 2).alias("b"))
        both = a.join(b, on="ts").drop_nulls()
        r = gr.models.evaluate.dm(both["a"].to_numpy(), both["b"].to_numpy(), h=7)
        p = r["p_value"] / 2 if r["statistic"] < 0 else 1 - r["p_value"] / 2
        return {"loss": loss, "against": benchmark, "n": r["n"], "dm": r["statistic"], "p_one_sided": p}

    h2 = pl.DataFrame([one_sided(f"hrp {e}", f"minvar {e}") for e in gr.models.portfolio.ESTIMATORS])
    h2_supported = bool(((h2["dm"] < 0) & (h2["p_one_sided"] < 0.05 / 3)).all())
    h3 = pl.DataFrame([one_sided("minvar dcc", "minvar sample")])
    h3_supported = bool(((h3["dm"] < 0) & (h3["p_one_sided"] < 0.05)).all())
    mo.vstack(
        [
            mo.md("## The registered tests"),
            mo.md(
                f"**H1**: the Reality Check over the 9 against 1/N gives p = **{h1['reality_check']:.3f}** "
                f"(best `{h1['best']}`). **{'Supported' if h1_supported else 'Not supported'}.**"
            ),
            mo.md("**H2**: HRP's squared daily net returns against minimum variance's, per estimator (a negative DM favours HRP):"),
            h2,
            mo.md(f"**{'Supported' if h2_supported else 'Not supported'}** (needs p < 0.05/3 under all three)."),
            mo.md("**H3**: minimum variance on DCC against the 90-day sample:"),
            h3,
            mo.md(f"**{'Supported' if h3_supported else 'Not supported'}.**"),
        ]
    )
    return


@app.cell
def _(PER_YEAR, mo, pl, stats, trials, weights):
    _eff = weights.group_by("trial", "rebalance").agg((1 / (pl.col("w") ** 2).sum()).alias("eff_n")).group_by("trial").agg(pl.col("eff_n").mean())
    _ops = trials.group_by("trial").agg(
        pl.col("turnover").filter(pl.col("turnover") > 0).mean().alias("turnover_per_rebalance"),
        pl.col("funding").sum().alias("funding_paid"),
    )
    described = stats.describe(trials, PER_YEAR).join(_eff, on="trial").join(_ops, on="trial")
    _vol = trials.group_by("trial").agg((pl.col("net").std() * PER_YEAR**0.5).alias("volatility"))
    mo.vstack(
        [
            mo.md("## Described, not tested"),
            described.join(_vol, on="trial").select(
                "trial", "volatility", "sharpe", "sortino", "cagr", "max_drawdown", "ulcer", "eff_n", "turnover_per_rebalance", "funding_paid"
            ).sort("volatility"),
            mo.md("`eff_n` is the mean effective number of coins, 1/Σw². Volatility is annualised from daily net returns."),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
