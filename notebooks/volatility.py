import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import polars as pl

    import galata_research as gr

    return alt, gr, mo, pl


@app.cell
def _(mo):
    mo.md(r"""
    # Realized volatility, five ways

    What was the volatility? The answer depends on what a bar is allowed to
    say. Close-to-close reads one number per bar. The range estimators also
    read the high and the low, which carry most of what happened inside the
    bar.

    | estimator | per bar, log prices | efficiency vs close-to-close* |
    |---|---|---|
    | close-to-close | $r_t=\ln(C_t/C_{t-1})$, sample std | 1 |
    | Parkinson (1980) | $\ln(H/L)^2/(4\ln 2)$ | ≈ 5.2 |
    | Garman–Klass (1980) | $\tfrac12\ln(H/L)^2-(2\ln2-1)\ln(C/O)^2$ | ≈ 7.4 |
    | Rogers–Satchell (1991) | $\ln\tfrac HC\ln\tfrac HO+\ln\tfrac LC\ln\tfrac LO$ | ≈ 6, unbiased under drift |
    | Yang–Zhang (2000) | $\sigma_o^2+k\sigma_c^2+(1-k)\overline{RS}$, $k=\frac{0.34}{1.34+\frac{n+1}{n-1}}$ | up to ≈ 14 |

    \*Commonly cited, from secondary sources. The efficiencies assume a price
    observed continuously. A bar's high and low are sampled, so every range
    estimator reads **low** (Molnár 2012). legacy's testnet week found the
    opposite on three of four markets: range above close-to-close
    (`design/tower/econometrics.md`). The record says which, below.

    Every estimator counts the same bars: one with a contiguous return, not in
    a gap. A window has a figure only when it is full.
    """)
    return


@app.cell
def _(gr, pl):
    EVER = ("2020-01-01T00:00Z", "2100-01-01T00:00Z")
    TICKERS = gr.market.candles(None, "1d", *EVER).select("ticker").unique().sort("ticker").collect()["ticker"].to_list()
    return EVER, TICKERS


@app.cell
def _(TICKERS, mo):
    ticker = mo.ui.dropdown(TICKERS, value="BTC", label="ticker")
    interval = mo.ui.dropdown(["1h", "4h", "1d"], value="1d", label="bars")
    window = mo.ui.slider(10, 180, value=30, step=5, label="window (bars)")
    mo.hstack([ticker, interval, window])
    return interval, ticker, window


@app.cell
def _(EVER, gr, interval, pl, ticker, window):
    bars = gr.mask_gaps(gr.market.candles([ticker.value], interval.value, *EVER), "candles").collect()
    per_year = gr.timeseries.periods_per_year(interval.value)
    scale = per_year**0.5
    frames = [
        gr.timeseries.realized(bars, est, window.value).select("ts", (pl.col("sigma") * scale).alias("sigma"), pl.lit(est).alias("estimator"))
        for est in gr.timeseries.ESTIMATORS
    ]
    frames.append(gr.timeseries.ewma_vol(bars).select("ts", (pl.col("sigma") * scale).alias("sigma"), pl.lit("ewma 0.94").alias("estimator")))
    frames.append(gr.timeseries.ewma_max(bars).select("ts", (pl.col("sigma") * scale).alias("sigma"), pl.lit("ewma max(0.94, 0.97)").alias("estimator")))
    vols = pl.concat(frames).drop_nulls("sigma")
    return bars, per_year, vols


@app.cell
def _(alt, interval, mo, ticker, vols, window):
    chart = (
        alt.Chart(vols)
        .mark_line(strokeWidth=1)
        .encode(
            x=alt.X("ts:T", title=None),
            y=alt.Y("sigma:Q", title="σ, annualised", scale=alt.Scale(type="log")),
            color=alt.Color("estimator:N", sort=None),
            tooltip=["ts:T", "estimator:N", alt.Tooltip("sigma:Q", format=".1%")],
        )
        .properties(height=320, width="container")
    )
    mo.vstack([mo.md(f"## {ticker.value}, {interval.value} bars, {window.value}-bar window"), chart])
    return


@app.cell
def _(mo, pl, vols):
    wide = vols.pivot(on="estimator", index="ts", values="sigma").drop_nulls()
    ratios = pl.DataFrame(
        {
            "estimator": [c for c in wide.columns if c not in ("ts", "close_to_close")],
            "median ratio to close-to-close": [
                (wide[c] / wide["close_to_close"]).median() for c in wide.columns if c not in ("ts", "close_to_close")
            ],
            "share of windows above": [
                (wide[c] > wide["close_to_close"]).mean() for c in wide.columns if c not in ("ts", "close_to_close")
            ],
        }
    )
    ranges = ratios.filter(pl.col("estimator").is_in(["parkinson", "garman_klass", "rogers_satchell"]))
    above = ranges.filter(pl.col("median ratio to close-to-close") > 1).height
    verdict = (
        f"On this record, **{above} of 3** range estimators read above close-to-close at the median, over {wide.height} windows. "
        + ("That is what legacy's testnet week found, against the textbook's downward bias." if above >= 2 else "That is the textbook's downward bias, not what legacy's testnet week found.")
    )
    mo.vstack([mo.md("## Against close-to-close"), ratios, mo.md(verdict)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Realized variance from finer bars

    Summing squared returns inside a bar measures the bar's variance with the
    bar's own path: $RV=\sum_i r_i^2$. The realized range sums each fine bar's
    Parkinson term instead, $RR=\sum_i \ln(H_i/L_i)^2/(4\ln 2)$, and is several
    times more efficient at the same sampling (Christensen and Podolskij 2007;
    Martens and van Dijk 2007). A bucket missing one fine return has no figure.
    These are what the GARCH study scores its forecasts against.
    """)
    return


@app.cell
def _(EVER, alt, gr, mo, pl, ticker):
    hourly = gr.market.candles([ticker.value], "1h", *EVER).collect()
    daily_rv = gr.timeseries.realized_from(hourly, "1d").drop_nulls("rv")
    daily = gr.timeseries.returns(gr.market.candles([ticker.value], "1d", *EVER), kind="log")
    joined = daily_rv.join(daily, on=["ticker", "ts"]).with_columns((pl.col("return") ** 2).alias("r²"))
    long = joined.select("ts", pl.col("r²"), pl.col("rv").alias("RV (24 × 1h)"), pl.col("rr").alias("RR (24 × 1h)")).unpivot(
        index="ts", variable_name="measure", value_name="variance"
    )
    proxy = (
        alt.Chart(long.with_columns((pl.col("variance") * 365).sqrt().alias("σ, annualised")))
        .mark_line(strokeWidth=0.8)
        .encode(x=alt.X("ts:T", title=None), y=alt.Y("σ, annualised:Q", scale=alt.Scale(type="log")), color="measure:N")
        .properties(height=240, width="container")
    )
    noise = joined.select(
        pl.corr(pl.col("rv"), pl.col("rr")).alias("corr(RV, RR)"),
        (pl.col("r²").std() / pl.col("r²").mean()).alias("cv of r²"),
        (pl.col("rv").std() / pl.col("rv").mean()).alias("cv of RV"),
        (pl.col("rr").std() / pl.col("rr").mean()).alias("cv of RR"),
        pl.len().alias("days"),
    )
    mo.vstack([mo.md(f"### {ticker.value}: one day's variance, three proxies"), proxy, noise, mo.md("A lower coefficient of variation is a less noisy proxy for the same day's variance: the reason the study scores against RV or RR and not r².")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The signature plot

    Mean daily RV against the sampling interval (Andersen, Bollerslev, Diebold
    and Labys 2000). With no microstructure noise it is flat. Where bid–ask
    bounce dominates, finer sampling inflates it. The finest interval where the
    curve has flattened is the finest usable for a realized-variance proxy,
    conventionally 5 minutes (Liu, Patton and Sheppard 2015). The record holds
    1m bars only since 2026-09-20, so this is a first look over a handful of
    days, not a finding.
    """)
    return


@app.cell
def _(EVER, alt, gr, mo, pl, ticker):
    minute = gr.market.candles([ticker.value], "1m", *EVER).collect()
    sig = gr.timeseries.signature(minute, [1, 2, 3, 5, 10, 15, 30, 60]).with_columns(
        (pl.col("mean_rv") * 365).sqrt().alias("σ, annualised")
    )
    plot = (
        alt.Chart(sig)
        .mark_line(point=True)
        .encode(x=alt.X("minutes:Q", scale=alt.Scale(type="log"), title="sampling, minutes"), y=alt.Y("σ, annualised:Q"), tooltip=["minutes", "days", alt.Tooltip("σ, annualised:Q", format=".1%")])
        .properties(height=220, width="container")
    )
    mo.vstack([mo.md(f"### {ticker.value}: {sig['days'].max() if sig.height else 0} whole days of 1m bars"), plot, sig])
    return


if __name__ == "__main__":
    app.run()
