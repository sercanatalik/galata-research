import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from datetime import timedelta

    import altair as alt
    import marimo as mo
    import polars as pl

    import galata_research as gr
    from galata_research import carry, stats, studies

    return alt, carry, gr, mo, pl, stats, studies, timedelta


@app.cell
def _(mo):
    mo.md("""
    # A pre-registered test: trading the funding

    A perpetual's funding pays one side to hold it. This notebook tests three
    claims, frozen in `planning/preregistered/trade-the-funding.md` and
    committed alone (`d82ee2b`) before `gr.carry` or this notebook existed.
    They are tested on Binance BTC and ETH, 2020 to the last whole month held.

    - **H1, carry.** A position hedged against the index earns the funding
      and bears the premium's moves. Do 13 carry rules per ticker (N = 26)
      survive DSR, the Reality Check against cash, and PBO? Expected:
      supported over the whole span.
    - **H2, persistence.** Is daily funding's lag-one autocorrelation
      positive at p < 0.01 on both tickers? This is carry's mechanism.
    - **H3, the fade.** Does an unhedged perp position against extreme
      funding beat buy-and-hold after its search (N = 16)? Expected: not
      supported.

    Binance's base taker fees: 0.05% the perp, plus 0.10% spot on a hedged
    change. The index stands in for a spot fill, and a short spot leg's
    borrow is not charged. Returns are per unit of notional.
    """)
    return


@app.cell
def _(gr, pl, timedelta):
    TICKERS = ["BTC", "ETH"]
    START = "2020-01-01T00:00Z"
    PER_YEAR = gr.timeseries.periods_per_year("1d")
    _cov = gr.reference.coverage().filter((pl.col("venue") == "binance-um") & pl.col("ticker").is_in(TICKERS))
    _last = dict(_cov.filter(pl.col("kind").is_in(["funding", "premium"])).group_by("kind").agg(pl.col("last").min()).iter_rows())
    # Funding is held a month at a time under the month's first day: the span ends where that month does,
    # or at the last whole premium day if that is earlier.
    _month_after = (_last["funding"].replace(day=28) + timedelta(days=4)).replace(day=1)
    END = f"{min(_month_after, _last['premium'] + timedelta(days=1)).isoformat()}T00:00Z"
    settled = gr.reference.funding(TICKERS, START, END).collect()
    day = gr.carry.daily(gr.reference.premium(TICKERS, START, END), settled)
    return END, PER_YEAR, START, TICKERS, day, settled


@app.cell
def _(END, START, day, mo, pl):
    _held = day.group_by("ticker").agg(pl.len().alias("days"), pl.col("ts").min().alias("first"), pl.col("ts").max().alias("last")).sort("ticker")
    mo.vstack([mo.md(f"## The days held, {START[:10]} → {END[:10]}"), _held, mo.md("A day missing a minute of premium or a settlement is a hole, not a day.")])
    return


@app.cell
def _(carry, day, mo, pl):
    h2 = carry.persistence(day)
    h2_supported = bool(h2.select(((pl.col("rho") > 0) & (pl.col("p") < 0.01)).all()).item())
    mo.vstack(
        [
            mo.md("## H2: does funding persist?"),
            h2,
            mo.md(f"Lag-one autocorrelation of daily funding, Fisher's z, one-sided. **H2 is {'supported' if h2_supported else 'not supported'}.**"),
        ]
    )
    return


@app.cell
def _(PER_YEAR, carry, day, mo, stats, studies):
    hedged = carry.rules(day)
    h1_scores = studies.summary(hedged, PER_YEAR)
    h1_deflated = stats.deflate(h1_scores)
    h1_rc = stats.reality_check(studies.excess(hedged, "cash"), reps=1000, seed=0)
    h1_pbo = stats.pbo(studies.matrix(hedged), blocks=16)
    h1_met = {
        f"DSR ≥ 0.95 at N = {h1_deflated['trials']}": h1_deflated["dsr"] >= 0.95,
        "Reality Check p < 0.05 vs cash": h1_rc["reality_check"] < 0.05,
        "PBO < 0.5": h1_pbo["pbo"] < 0.5,
    }
    mo.vstack(
        [
            mo.md(f"## H1: hedged carry, {h1_deflated['trials']} trials"),
            mo.hstack(
                [
                    mo.stat(f"{h1_deflated['dsr']:.3f}", label="DSR, the best"),
                    mo.stat(f"{h1_rc['reality_check']:.3f}", label="Reality Check p vs cash"),
                    mo.stat(f"{h1_pbo['pbo']:.2f}", label="PBO"),
                ],
                justify="start",
            ),
            mo.md(
                f"The best is `{h1_deflated['trial']}` on {h1_deflated['ticker']}, annualised Sharpe "
                f"**{h1_deflated['sharpe'] * PER_YEAR**0.5:.2f}**. "
                + ", ".join(f"{k}: {'yes' if v else 'no'}" for k, v in h1_met.items())
                + f". **H1 is {'supported' if all(h1_met.values()) else 'not supported'}.**"
            ),
            h1_scores.sort("sharpe", descending=True, nulls_last=True),
        ]
    )
    return (hedged,)


@app.cell
def _(PER_YEAR, alt, hedged, mo, pl):
    _year = pl.col("ts").dt.year().alias("year")
    by_year = (
        hedged.drop_nulls("net")
        .group_by("trial", "ticker", _year)
        .agg(
            (pl.col("net").mean() / pl.col("net").std() * PER_YEAR**0.5).alias("sharpe"),
            (-pl.col("funding").sum()).alias("funding_earned"),
            pl.col("gross").sum().alias("basis"),
            pl.col("cost").sum().alias("costs"),
        )
        .sort("trial", "ticker", "year")
    )
    _chart = (
        alt.Chart(by_year.filter(pl.col("trial").is_in(["always", "carry 7 0 short_only", "carry 30 0 both"])))
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title=None),
            y=alt.Y("sharpe:Q", title="annualised Sharpe, net, within the year"),
            color="trial:N",
            strokeDash="ticker:N",
            tooltip=["trial:N", "ticker:N", "year:O", alt.Tooltip("sharpe:Q", format=".2f"), alt.Tooltip("funding_earned:Q", format=".2%")],
        )
        .properties(height=240, width=560)
    )
    mo.vstack(
        [
            mo.md("## Described, not tested: was it one regime? Each rule's Sharpe and income by year"),
            _chart,
            by_year,
            mo.md(
                "`funding_earned` is the sum of the funding received, `basis` the sum of the premium's moves "
                "against the position, and `costs` both legs' fees. Simple sums, per unit of notional. "
                "A carry that is mostly funding in a few years is a regime, however high its whole-span Sharpe."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    fade_go = mo.ui.run_button(label="run H3: the fade (builds daily bars from six years of 1m klines, several minutes)")
    fade_go
    return (fade_go,)


@app.cell
def _(END, PER_YEAR, START, TICKERS, carry, day, fade_go, gr, mo, pl, settled, stats, studies):
    mo.stop(not fade_go.value, mo.md("*H3 runs behind the button.*"))
    _fine = gr.reference.candles(TICKERS, START, END, venues="binance-um").collect()
    perp = studies.shared_days(pl.concat([gr.timeseries.resample(_fine.filter(pl.col("ticker") == t), "1d") for t in TICKERS]))
    _hours = gr.reference.funding_hours(settled)
    faded = carry.fades(perp, day, settled)
    _hold = studies.trial(perp, pl.lit(1.0), "buy and hold", fee=carry.PERP_FEE, funding=_hours)
    h3_scores = studies.summary(faded, PER_YEAR)
    h3_deflated = stats.deflate(h3_scores)
    h3_rc = stats.reality_check(studies.excess(pl.concat([faded, _hold]), "buy and hold"), reps=1000, seed=0)
    h3_pbo = stats.pbo(studies.matrix(faded), blocks=16)
    _met = {
        f"DSR ≥ 0.95 at N = {h3_deflated['trials']}": h3_deflated["dsr"] >= 0.95,
        "Reality Check p < 0.05 vs buy-and-hold": h3_rc["reality_check"] < 0.05,
        "PBO < 0.5": h3_pbo["pbo"] < 0.5,
    }
    mo.vstack(
        [
            mo.md(f"## H3: the fade, {h3_deflated['trials']} trials, funding charged"),
            mo.md(
                f"The best is `{h3_deflated['trial']}` on {h3_deflated['ticker']}, annualised Sharpe "
                f"{h3_deflated['sharpe'] * PER_YEAR**0.5:.2f}; DSR {h3_deflated['dsr']:.3f}, Reality Check p "
                f"{h3_rc['reality_check']:.3f}, PBO {h3_pbo['pbo']:.2f}. "
                + ", ".join(f"{k}: {'yes' if v else 'no'}" for k, v in _met.items())
                + f". **H3 is {'supported' if all(_met.values()) else 'not supported'}.**"
            ),
            h3_scores.sort("sharpe", descending=True, nulls_last=True),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
