import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import time

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl

    import galata_research as gr

    return alt, gr, mo, np, pl, time


@app.cell
def _(mo):
    mo.md(r"""
    # Correlation from the volatility fit

    How do the instruments co-move **under the model that sized their
    volatility**? A covariance Σ = D·R·D whose σ comes from a GARCH and whose
    ρ comes from an equal-weight window has two halves on two clocks. The
    GARCH reacts to yesterday's shock; the window weighs last month like
    today. This notebook fits the correlation from the volatility model itself
    (roadmap D13): Engle's (2002) two-step DCC.

    **Step 1.** Each ticker's GJR-t, as `gr.models.vol` fits it. Its σ
    standardises the returns: $z_{i,t} = (r_{i,t}-\mu_i)/\sigma_{i,t}$.

    **Step 2.** On z, by Gaussian quasi-likelihood,

    $$Q_t = (1-a-b)\bar Q + a\,z_{t-1}z_{t-1}^\top + b\,Q_{t-1},\qquad R_t = \operatorname{diag}(Q_t)^{-1/2}\,Q_t\,\operatorname{diag}(Q_t)^{-1/2}.$$

    a is the reaction to the latest co-movement, b the persistence, and
    $\bar Q$ the long-run level R reverts to. Only two parameters for the
    whole matrix. Aielli's (2013) cDCC runs the recursion on
    $z^*_t = \operatorname{diag}(Q_t)^{1/2}z_t$, which makes the target
    consistent; it is shown beside DCC. Multi-step R reverts as
    $R_{t+h} = \bar R + (a+b)^{h-1}(R_{t+1}-\bar R)$ (Engle and Sheppard 2001).

    The implementation reproduces rmgarch's `dccfit` path and likelihood at its
    own parameters (`tests/corr.py`, fixture in `tests/data`).

    **The joint sample.** The recursion needs every $z_t$ in full, so a bar
    missing for any ticker is dropped for all of them. The sample starts at
    the youngest ticker's first return, which is what it costs the older ones.
    """)
    return


@app.cell
def _(gr, pl):
    EVER = ("2020-01-01T00:00Z", "2100-01-01T00:00Z")
    INTERVAL = "4h"
    bars = gr.mask_gaps(gr.market.candles(None, INTERVAL, *EVER), "candles").collect()
    returns = gr.timeseries.returns(bars, kind="log")
    MAIN = ["BTC", "ETH", "HYPE"]
    firsts = returns.drop_nulls("return").group_by("ticker").agg(pl.col("ts").min().alias("first return"), pl.len().alias("returns")).sort("first return")
    return INTERVAL, MAIN, firsts, returns


@app.cell
def _(INTERVAL, MAIN, firsts, gr, mo, pl, returns):
    tickers_all, joint_all, youngest_all = gr.models.corr.joint(returns)
    _, joint_main, youngest_main = gr.models.corr.joint(returns.filter(pl.col("ticker").is_in(MAIN)))
    mo.vstack(
        [
            mo.md(f"### The joint sample, {INTERVAL} bars"),
            mo.ui.table(firsts, selection=None),
            mo.md(
                f"All six: **{joint_all.height:,}** joint returns from {joint_all['ts'][0]:%Y-%m-%d}, started by **{youngest_all}**. "
                f"The main dex alone ({', '.join(MAIN)}): **{joint_main.height:,}** from {joint_main['ts'][0]:%Y-%m-%d}, started by {youngest_main}."
            ),
        ]
    )
    return (tickers_all,)


@app.cell
def _(MAIN, gr, np, pl, returns, time):
    def _fit(sub, corr):
        t0 = time.perf_counter()
        f = gr.models.corr.fit(sub, model="gjr", dist="t", corr=corr)
        return f, time.perf_counter() - t0

    samples = {"all six": returns, "main dex": returns.filter(pl.col("ticker").is_in(MAIN))}
    fits = {(name, corr): _fit(sub, corr) for name, sub in samples.items() for corr in ("dcc", "cdcc")}
    summary = pl.DataFrame(
        [
            {
                "sample": name,
                "corr": corr,
                "joint returns": f.nobs,
                "a": f.a,
                "b": f.b,
                "a + b": f.persistence,
                "half-life (bars)": f.half_life,
                "converged": f.converged,
                "seconds": secs,
            }
            for (name, corr), (f, secs) in fits.items()
        ]
    )

    def rbar(f):
        d = np.sqrt(np.diag(f.qbar))
        return f.qbar / np.outer(d, d)

    return fits, rbar, summary


@app.cell
def _(fits, mo, pl, rbar, summary):
    _f = fits[("all six", "dcc")][0]
    _R = rbar(_f)
    long_run = pl.DataFrame({"": _f.tickers, **{t: _R[:, k].round(2) for k, t in enumerate(_f.tickers)}})
    mo.vstack(
        [
            mo.md("### The fits (GJR-t margins)"),
            mo.ui.table(summary, selection=None),
            mo.md("### R̄, the level R reverts to (all six, DCC)"),
            mo.ui.table(long_run, selection=None),
        ]
    )
    return


@app.cell
def _(mo, tickers_all):
    pair = mo.ui.dropdown([f"{a}–{b}" for i, a in enumerate(tickers_all) for b in tickers_all[i + 1 :]], value="BTC–ETH", label="pair")
    window = mo.ui.slider(30, 360, value=180, step=30, label="rolling window (bars)")
    mo.hstack([pair, window])
    return pair, window


@app.cell
def _(fits, gr, pair, pl, returns, window):
    _a, _b = pair.value.split("–")
    _f = fits[("all six", "dcc")][0]
    dcc_path = _f.series.filter((pl.col("ticker_i") == _a) & (pl.col("ticker_j") == _b)).select("ts", pl.col("correlation").alias("rho"), pl.lit("DCC (GJR-t)").alias("estimator"))
    _, _joint, _ = gr.models.corr.joint(returns)
    rolling = _joint.select("ts", pl.rolling_corr(pl.col(_a), pl.col(_b), window_size=window.value).alias("rho"), pl.lit(f"sample, {window.value} bars").alias("estimator"))
    _e = gr.models.corr.ewma(returns, lam=0.94).series
    ewma_path = _e.filter((pl.col("ticker_i") == _a) & (pl.col("ticker_j") == _b)).select("ts", "correlation", pl.lit("EWMA 0.94").alias("estimator")).rename({"correlation": "rho"})
    paths = pl.concat([dcc_path, rolling, ewma_path]).drop_nulls("rho")
    return (paths,)


@app.cell
def _(alt, mo, pair, paths):
    _chart = (
        alt.Chart(paths)
        .mark_line(strokeWidth=1)
        .encode(x=alt.X("ts:T", title=None), y=alt.Y("rho:Q", title="ρ"), color=alt.Color("estimator:N", title=None))
        .properties(width="container", height=280, title=f"{pair.value}: three correlations on one joint sample")
    )
    mo.vstack(
        [
            mo.ui.altair_chart(_chart),
            mo.md(
                "DCC moves at the pace its fitted b allows and reverts to R̄. The rolling sample ρ jumps when a shock "
                "enters and again when it leaves the window. EWMA at λ = 0.94 has an n_eff of 32 bars "
                "(`gr.models.corr.ewma_n_eff(0.94)`), about five days of 4h bars, and is the noisiest."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Walked forward: what galata-datawatch's Tier 16 would publish

    From a split at 70% of the joint sample, refitting every 6 bars (daily at
    4h) and filtering in between, forecasting 1 and 6 bars ahead. Each row is a
    pair, a horizon and an origin; `fitted_through ≤ close_ts` on every one.
    Below: the last origin's σ (annualised) and ρ at h = 6.
    """)
    return


@app.cell
def _(INTERVAL, gr, np, pl, returns, time):
    _, _joint, _ = gr.models.corr.joint(returns)
    split = _joint["close_ts"][int(_joint.height * 0.7)]
    _t0 = time.perf_counter()
    walked = gr.models.corr.walk_forward(returns, model="gjr", dist="t", split=split, every=6, horizons=[1, 6])
    walk_seconds = time.perf_counter() - _t0
    last = walked.filter((pl.col("close_ts") == pl.col("close_ts").max()) & (pl.col("h") == 6))
    per_year = gr.timeseries.periods_per_year(INTERVAL)
    sigmas = last.filter(pl.col("ticker_i") == pl.col("ticker_j")).select(pl.col("ticker_i").alias("ticker"), (pl.col("covariance") * per_year).sqrt().alias("σ annualised"))
    rho_last = last.filter(pl.col("ticker_i") != pl.col("ticker_j")).select("ticker_i", "ticker_j", "correlation", "n_eff", "a", "b", "fitted_through")
    refits = walked.filter(pl.col("refit"))["close_ts"].n_unique()
    return last, refits, rho_last, sigmas, split, walk_seconds, walked


@app.cell
def _(mo, refits, rho_last, sigmas, split, walk_seconds, walked):
    mo.vstack(
        [
            mo.md(f"Split at {split:%Y-%m-%d %H:%M}; **{refits}** refits and {walked['close_ts'].n_unique():,} origins in **{walk_seconds:.0f} s**."),
            mo.hstack([mo.ui.table(sigmas, selection=None), mo.ui.table(rho_last, selection=None)]),
        ]
    )
    return


@app.cell
def _(fits, mo, walk_seconds):
    _six, _secs = fits[("all six", "dcc")]
    _main = fits[("main dex", "dcc")][0]
    _c6 = fits[("all six", "cdcc")][0]
    _cm = fits[("main dex", "cdcc")][0]
    mo.md(f"""
    ### What this shows, on this record

    - **Correlation moves slowly here.** All six at 4h: a = {_six.a:.4f}, b = {_six.b:.4f}, a half-life of
      {_six.half_life:.0f} bars ({_six.half_life / 6:.1f} days). The main dex alone, on a sample
      {_main.nobs / _six.nobs:.1f}× longer: a = {_main.a:.4f}, b = {_main.b:.4f}. Engle and Sheppard (2001)
      report a ≈ 0.01–0.03 and b ≈ 0.94–0.975 on daily equity sectors.
    - **DCC and cDCC agree on all six and part on the main dex.** All six: cDCC's a = {_c6.a:.4f},
      b = {_c6.b:.4f}. The main dex: DCC's a + b = {_main.persistence:.4f}, all but a unit root, against
      cDCC's a = {_cm.a:.4f}, b = {_cm.b:.4f} (a + b = {_cm.persistence:.4f}). That is the direction Aielli's
      inconsistency argument predicts on a long sample, but one sample decides nothing; the registered
      comparison is where it would be decided.
    - **A fit costs about a second** ({_secs:.1f} s for all six); the walk-forward above took
      {walk_seconds:.0f} s. A 30-minute cadence has room for it.
    - **Not shown here, and not yet claimed:** whether DCC forecasts covariance better than EWMA or the
      sample ρ out of sample. That needs a registered comparison (roadmap item 25), on minimum-variance
      portfolio variance and a multivariate QLIKE.
    """)
    return


if __name__ == "__main__":
    app.run()
