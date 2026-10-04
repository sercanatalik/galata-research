import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import polars as pl

    import galata_research as gr
    from galata_research import factors, stats, studies

    return alt, factors, gr, mo, pl, stats, studies


@app.cell
def _(mo):
    mo.md("""
    # A pre-registered test: ranking every Binance perpetual

    This is the repository's first study across many tickers (D8 lifted,
    2026-10-04). The universe, the rules, N and what would count as support
    were frozen in `planning/preregistered/rank-the-universe.md` and
    committed alone (`d333e55`) before `gr.factors` or this notebook existed.

    - **Every USDT perpetual Binance's archive lists, delisted ones
      included.** Each day, the 50 most liquid with 90 days of history are
      traded. The universe on a day is the one a reader had that day.
    - **12 trials:** momentum over 7, 30 and 90 days, low volatility,
      residual momentum against BTC, and small (low dollar volume), each
      long/short (top fifth against bottom fifth, gross 1) and long-only (top
      fifth, against the equal-weight traded set).
    - Rebalanced weekly, 0.05% taker on turnover, funding charged daily.
    - **Primary span 2020–2024**, before Binance listed stocks and metals.
      Support needs DSR ≥ 0.95, a Reality Check p < 0.05 within the best
      rule's side, and PBO < 0.5. **Secondary span 2025-01 to 2026-09**,
      reported the same way; it cannot rescue a failure. Expected: not
      supported.
    """)
    return


@app.cell
def _(factors, gr):
    SPAN = ("2019-09-01T00:00Z", "2026-10-01T00:00Z")
    PRIMARY = ("2020-01-01T00:00Z", "2025-01-01T00:00Z")
    SECONDARY = ("2025-01-01T00:00Z", "2026-10-01T00:00Z")
    PER_YEAR = 365
    panel = factors.panel(gr.reference.daily(None, *SPAN), gr.reference.funding(None, *SPAN))
    return PER_YEAR, PRIMARY, SECONDARY, panel


@app.cell
def _(PRIMARY, SECONDARY, factors, panel):
    members_primary = factors.members(panel, exclude=factors.INDEX_AND_STABLE)
    members_secondary = factors.members(panel, exclude=factors.INDEX_AND_STABLE | factors.NON_CRYPTO)
    primary = factors.rules(panel, members_primary, *PRIMARY)
    secondary = factors.rules(panel, members_secondary, *SECONDARY)
    return members_primary, members_secondary, primary, secondary


@app.cell
def _(PER_YEAR, factors, pl, stats, studies):
    def judge(trials):
        """The registered figures for one span: the best of the 12, its DSR, its side's Reality Check, and PBO."""
        twelve = trials.filter(pl.col("trial") != factors.BENCHMARK)
        scores = studies.summary(twelve, PER_YEAR)
        deflated = stats.deflate(scores)
        side = "long_short" if deflated["trial"].endswith("long_short") else "long_only"
        same_side = twelve.filter(pl.col("trial").str.ends_with(side))
        if side == "long_short":
            excess = studies.excess(same_side, "cash")
        else:
            excess = studies.excess(pl.concat([same_side, trials.filter(pl.col("trial") == factors.BENCHMARK)]), factors.BENCHMARK)
        rc = stats.reality_check(excess, reps=1000, seed=0)
        overfit = stats.pbo(studies.matrix(twelve), blocks=16)
        met = {
            "DSR ≥ 0.95 at N = 12": deflated["dsr"] >= 0.95,
            f"Reality Check p < 0.05 ({side})": rc["reality_check"] < 0.05,
            "PBO < 0.5": overfit["pbo"] < 0.5,
        }
        return {
            "scores": studies.summary(trials, PER_YEAR).sort("sharpe", descending=True, nulls_last=True),
            "best": deflated["trial"],
            "sharpe": deflated["sharpe"] * PER_YEAR**0.5,
            "dsr": deflated["dsr"],
            "rc": rc["reality_check"],
            "pbo": overfit["pbo"],
            "met": met,
            "supported": all(met.values()),
            "days": twelve["ts"].n_unique(),
        }

    return (judge,)


@app.cell
def _(judge, mo, primary, secondary):
    verdict_primary = judge(primary)
    verdict_secondary = judge(secondary)

    def show(title, v):
        return mo.vstack(
            [
                mo.md(f"## {title}: {v['days']:,} days"),
                mo.hstack(
                    [
                        mo.stat(f"{v['sharpe']:.2f}", label=f"best: {v['best']}"),
                        mo.stat(f"{v['dsr']:.3f}", label="DSR, N = 12"),
                        mo.stat(f"{v['rc']:.3f}", label="Reality Check p, its side"),
                        mo.stat(f"{v['pbo']:.2f}", label="PBO"),
                    ],
                    justify="start",
                ),
                mo.md(", ".join(f"{k}: {'yes' if ok else 'no'}" for k, ok in v["met"].items()) + f". **{'Supported' if v['supported'] else 'Not supported'}.**"),
                v["scores"],
            ]
        )

    mo.vstack([show("Primary, 2020–2024 (the test)", verdict_primary), show("Secondary, 2025-01 to 2026-09 (reported, cannot rescue)", verdict_secondary)])
    return


@app.cell
def _(PER_YEAR, alt, mo, pl, primary, secondary):
    _both = pl.concat([primary, secondary])
    by_year = (
        _both.group_by("trial", pl.col("ts").dt.year().alias("year"))
        .agg(
            (pl.col("net").mean() / pl.col("net").std() * PER_YEAR**0.5).alias("sharpe"),
            ((1 + pl.col("net")).product() - 1).alias("net_return"),
            (-pl.col("funding").sum()).alias("funding_received"),
            pl.col("turnover").filter(pl.col("turnover") > 0).mean().alias("turnover_per_rebalance"),
            (pl.col("funding_missing").sum() / pl.len()).alias("funding_missing_per_day"),
        )
        .sort("trial", "year")
    )
    _chart = (
        alt.Chart(by_year.filter(pl.col("trial").str.ends_with("long_short") | (pl.col("trial") == "ew universe")))
        .mark_line(point=True)
        .encode(x=alt.X("year:O", title=None), y=alt.Y("sharpe:Q", title="annualised Sharpe, net, within the year"), color="trial:N", tooltip=["trial:N", "year:O", alt.Tooltip("sharpe:Q", format=".2f"), alt.Tooltip("net_return:Q", format=".1%")])
        .properties(height=260, width=600)
    )
    mo.vstack(
        [
            mo.md("## Described, not tested: each rule by year"),
            _chart,
            by_year,
            mo.md(
                "`turnover_per_rebalance` is the sum of |Δw| at a rebalance, on a gross book of 1. "
                "`funding_received` is negative where the rule paid. `funding_missing_per_day` is the mean "
                "count of held coins a day with no funding archive, charged nothing."
            ),
        ]
    )
    return


@app.cell
def _(factors, members_primary, members_secondary, mo, panel, pl):
    _open = factors.members(panel, exclude=factors.INDEX_AND_STABLE).filter(pl.col("ts") >= pl.datetime(2025, 1, 1, time_zone="UTC"))
    excluded_entering = (
        _open.filter(pl.col("traded") & pl.col("ticker").is_in(list(factors.NON_CRYPTO)))
        .group_by("ticker")
        .agg(pl.len().alias("days_in_top_50"))
        .sort("days_in_top_50", descending=True)
    )
    kept_2025 = (
        members_secondary.filter(pl.col("traded") & (pl.col("ts") >= pl.datetime(2025, 1, 1, time_zone="UTC")))
        .group_by("ticker")
        .agg(pl.len().alias("days_traded"))
        .sort("days_traded", descending=True)
    )
    primary_names = members_primary.filter(pl.col("traded") & (pl.col("ts") < pl.datetime(2025, 1, 1, time_zone="UTC")))["ticker"].n_unique()
    mo.vstack(
        [
            mo.md("## Described: who was in the universe"),
            mo.md(f"**{primary_names}** different perpetuals were traded at some point in 2020–2024."),
            mo.md("Excluded non-crypto names that would have entered the top 50 from 2025, had they not been excluded:"),
            excluded_entering,
            mo.md("Every name traded in the secondary span, for a reader to check that none is a stock, a metal or FX:"),
            kept_2025,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
