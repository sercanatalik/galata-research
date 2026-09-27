import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import polars as pl

    import galata_research as gr

    vol = gr.models.vol
    return alt, gr, mo, pl, vol


@app.cell
def _(mo):
    mo.md(r"""
    # GARCH, GARCH-t and variations

    A GARCH model is a filter: it updates the next bar's variance from this
    bar's shock. The variants differ in three ways: how the variance reacts to
    a shock, whether falls and rises are treated alike, and what distribution
    the standardised shock $z$ is given.

    $$r_t=\mu+\varepsilon_t,\qquad \varepsilon_t=\sigma_t z_t,\qquad z_t\sim D(0,1)$$

    This notebook is the **in-sample** half of the study. The models are fitted
    on an estimation period and read: their parameters, persistence, tails,
    asymmetry and residuals. Forecasting out of sample comes next
    (`planning/roadmap.md`, items 5–10). Parameters are shown on returns × 100,
    the scale `arch` fits them on. Every σ is in return units, annualised in
    calendar time.
    """)
    return


@app.cell
def _(gr):
    EVER = ("2020-01-01T00:00Z", "2100-01-01T00:00Z")
    TICKERS = gr.market.candles(None, "1d", *EVER).select("ticker").unique().sort("ticker").collect()["ticker"].to_list()
    return EVER, TICKERS


@app.cell
def _(TICKERS, mo):
    ticker = mo.ui.dropdown(TICKERS, value="BTC", label="ticker")
    interval = mo.ui.dropdown(["1h", "4h", "1d"], value="1d", label="bars")
    share = mo.ui.slider(0.4, 0.9, value=0.7, step=0.05, label="estimation period (share of history)")
    deseason = mo.ui.checkbox(value=False, label="deseasonalise (hour × weekday, fitted on the estimation period)")
    mo.hstack([ticker, interval, share, deseason])
    return deseason, interval, share, ticker


@app.cell
def _(EVER, deseason, gr, interval, pl, share, ticker):
    bars = gr.market.candles([ticker.value], interval.value, *EVER).collect()
    returns = gr.timeseries.returns(bars, kind="log")
    first, last = bars["ts"].min(), bars["close_ts"].max()
    split = bars["close_ts"][int(bars.height * share.value) - 1]
    estimation = (first, split)
    column = "return"
    if deseason.value and interval.value != "1d":
        factors = gr.timeseries.seasonal_factors(returns, fit=estimation)
        returns = gr.timeseries.deseasonalize(returns, factors)
        column = "deseasonalized"
    in_sample = returns.filter(pl.col("close_ts") <= split)
    per_year = gr.timeseries.periods_per_year(interval.value)
    return bars, column, estimation, first, in_sample, per_year, returns, split


@app.cell
def _(alt, bars, first, interval, mo, pl, split, ticker):
    price = (
        alt.Chart(bars.select("ts", "close"))
        .mark_line(strokeWidth=1)
        .encode(x=alt.X("ts:T", title=None), y=alt.Y("close:Q", scale=alt.Scale(type="log"), title="close"))
    )
    rule = alt.Chart(pl.DataFrame({"ts": [split]})).mark_rule(strokeDash=[4, 4]).encode(x="ts:T")
    mo.vstack(
        [
            mo.md(f"## ① {ticker.value}, {interval.value} bars: estimation period {first:%Y-%m-%d} to {split:%Y-%m-%d} (left of the rule)"),
            (price + rule).properties(height=180, width="container"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## ② The returns: fat tails and clustering

    Two stylised facts make GARCH necessary. Returns have almost no
    autocorrelation, but their squares have a great deal: large moves follow
    large moves (Mandelbrot 1963; Engle 1982). And the distribution has far
    fatter tails than a normal. On BTC the kurtosis is 6 at 1d and 14 at 1h,
    against 3 for a normal. A GARCH with normal shocks explains part of the
    kurtosis by the changing σ. Whatever is left needs a fat-tailed $z$
    (Bollerslev 1987).
    """)
    return


@app.cell
def _(alt, column, in_sample, mo, pl):
    _r = in_sample.drop_nulls(column)[column]
    acf_rows = []
    for _name, _s in (("r", _r), ("r²", _r**2)):
        _d = _s - _s.mean()
        _denom = float((_d * _d).sum())
        for _k in range(1, 31):
            acf_rows.append({"series": _name, "lag": _k, "acf": float((_d.slice(_k) * _d.slice(0, _d.len() - _k)).sum()) / _denom})
    band = 1.96 / _r.len() ** 0.5
    acf = (
        alt.Chart(pl.DataFrame(acf_rows))
        .mark_bar()
        .encode(x="lag:O", y=alt.Y("acf:Q", title="autocorrelation"), color="series:N", xOffset="series:N")
        .properties(height=180, width="container")
    )
    kurt = float(_r.kurtosis(fisher=False))
    mo.vstack(
        [
            acf,
            mo.md(
                f"±{band:.3f} is the 95% band for white noise. Kurtosis **{kurt:.1f}** over {_r.len()} returns "
                f"(normal: 3). r² has {sum(1 for x in acf_rows if x['series'] == 'r²' and x['acf'] > band)} of 30 lags above the band; "
                f"r has {sum(1 for x in acf_rows if x['series'] == 'r' and abs(x['acf']) > band)}."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The models

    | model | variance recursion (arch's form) | what it adds |
    |---|---|---|
    | EWMA | $\sigma_t^2=\lambda\sigma_{t-1}^2+(1-\lambda)\varepsilon_{t-1}^2$, λ = 0.94 | RiskMetrics' baseline, fixed; integrated |
    | GARCH | $\sigma_t^2=\omega+\alpha\varepsilon_{t-1}^2+\beta\sigma_{t-1}^2$ | Bollerslev 1986; persistence α+β |
    | GJR | $+\gamma\varepsilon_{t-1}^2\,\mathbb 1[\varepsilon_{t-1}<0]$ | Glosten, Jagannathan, Runkle 1993: falls raise σ more |
    | EGARCH | $\ln\sigma_t^2=\omega+\alpha(\lvert z_{t-1}\rvert-\sqrt{2/\pi})+\gamma z_{t-1}+\beta\ln\sigma_{t-1}^2$ | Nelson 1991: asymmetry in logs; persistence β |
    | APARCH | $\sigma_t^\delta=\omega+\alpha(\lvert\varepsilon_{t-1}\rvert-\gamma\varepsilon_{t-1})^\delta+\beta\sigma_{t-1}^\delta$ | Ding, Granger, Engle 1993: the power δ is estimated |
    | FIGARCH | $(1-\beta L)\sigma_t^2=\omega+[1-\beta L-\phi L(1-L)^d]\varepsilon_t^2$ | Baillie, Bollerslev, Mikkelsen 1996: hyperbolic decay, *d* |
    | RiskMetrics 2006 | a sum of EWMAs at many horizons | Zumbach 2006: long memory with no fit |

    Distributions of $z$: normal; Student-*t* with ν degrees of freedom,
    scaled to unit variance (ν ≤ 4 means the fourth moment does not exist);
    Hansen's (1994) skewed *t* (η, λ); and the GED.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Why persistence differs by model": mo.md(r"""
            Persistence is the coefficient on yesterday's variance in the
            expected recursion, $E_{t-1}[\sigma_{t+1}^2]$.
            - GARCH: α+β.
            - GJR: the γ term is on only when $z<0$, so it enters with weight
              $\kappa=E[z^2\mathbb 1(z<0)]$. That is ½ for a symmetric $z$, and
              **not** $P(z<0)$ when $z$ is skewed.
            - EGARCH recurses on $\ln\sigma^2$, so persistence is β alone.
              arch centres $\lvert z\rvert$ at $\sqrt{2/\pi}$, its normal mean,
              whatever the distribution, so under *t* the intercept absorbs the
              difference.
            - APARCH: $\alpha E[(\lvert z\rvert-\gamma z)^\delta]+\beta$.
            - FIGARCH has none: its weights decay hyperbolically, and *d* is the
              summary.

            The half-life is $\ln 0.5/\ln(\text{persistence})$ bars. The
            expectations are integrals over the fitted distribution's quantile
            function.
            """),
            "Why a near-unit persistence misstates the level": mo.md(r"""
            The unconditional variance is $\omega/(1-\text{persistence})$. With
            persistence 0.99 the denominator is 0.01, so a small error in
            α or β moves the level a lot. On BTC daily, GARCH-*t* implies σ̄ =
            0.038 against a sample deviation of 0.024. The same model with
            normal shocks implies 0.0244. The fitted σ path is fine; its
            long-run anchor is not well determined.
            """),
        }
    )
    return


@app.cell
def _(column, estimation, in_sample, vol):
    specs = [(m, "t") for m in vol.MODELS] + [("garch", d) for d in ("normal", "skewt", "ged")] + [("gjr", "skewt")]
    fits = {}
    for model, dist in specs:
        try:
            fits[(model, dist)] = vol.fit(in_sample, model=model, dist=dist, column=column, fit=estimation, min_obs=250)
        except Exception as error:  # a fit that fails is reported, not hidden
            fits[(model, dist)] = error
    good = [f for f in fits.values() if isinstance(f, vol.Fit)]
    failed = {k: str(v) for k, v in fits.items() if not isinstance(v, vol.Fit)}
    table = vol.table(good).sort("bic")
    return failed, good, table


@app.cell
def _(failed, mo, pl, table):
    t_normal = table.filter((pl.col("model") == "garch") & pl.col("dist").is_in(["normal", "t"]))
    d_bic = None
    if t_normal.height == 2:
        by = dict(zip(t_normal["dist"], t_normal["bic"]))
        d_bic = by["t"] - by["normal"]
    gjr = table.filter((pl.col("model") == "gjr") & (pl.col("dist") == "t"))
    gamma = gjr["gamma[1]"][0] if gjr.height and "gamma[1]" in gjr.columns else None
    best = table.row(0, named=True)
    lines = [
        f"Best by BIC: **{best['model']}-{best['dist']}**.",
        f"*t* against normal in GARCH: ΔBIC **{d_bic:+.0f}**, "
        + ("consistent with the literature's verdict that fat tails are necessary (Troster et al. 2019)." if d_bic is not None and d_bic < -10 else "not the decisive gap the literature leads one to expect.")
        if d_bic is not None
        else "",
        f"GJR's γ is **{gamma:+.3f}**: "
        + ("the leverage effect, falls raising σ more." if gamma > 0.02 else "no leverage effect to speak of, consistent with crypto's absent or reversed asymmetry (Cheikh, Ben Zaied, Chevallier 2020).")
        if gamma is not None
        else "",
    ]
    cols = ["model", "dist", "nobs", "bic", "persistence", "half_life", "sigma_bar", "converged", "omega", "alpha[1]", "gamma[1]", "beta[1]", "delta", "d", "nu", "eta", "lambda"]
    mo.vstack(
        [
            mo.md("## ⑤ The fit table (estimation period)"),
            table.select([c for c in cols if c in table.columns]),
            mo.md("**On this record:** " + " ".join(x for x in lines if x)),
            mo.md(f"Failed fits: {failed}") if failed else mo.md(""),
        ]
    )
    return


@app.cell
def _(good, mo):
    choice = mo.ui.dropdown({f"{f.model}-{f.dist}": i for i, f in enumerate(good)}, value="garch-t" if any(f.model == "garch" and f.dist == "t" for f in good) else None, label="model to inspect")
    choice
    return (choice,)


@app.cell
def _(alt, bars, choice, good, mo, per_year, pl, split):
    chosen = good[choice.value] if choice.value is not None else good[0]
    proxy = bars.filter(pl.col("close_ts") <= split).select(
        "ts", ((pl.col("high") / pl.col("low")).log().pow(2) / (4 * 0.6931471805599453)).sqrt().alias("parkinson")
    )
    _s = chosen.series.join(proxy, on="ts", how="left").with_columns(
        (pl.col("sigma") * per_year**0.5).alias("σ fitted"), (pl.col("parkinson") * per_year**0.5).alias("|range| per bar")
    )
    lines_ = (
        alt.Chart(_s)
        .mark_line(strokeWidth=1, color="#1f77b4")
        .encode(x=alt.X("ts:T", title=None), y=alt.Y("σ fitted:Q", scale=alt.Scale(type="log"), title="σ, annualised"))
    )
    dots = alt.Chart(_s.drop_nulls("|range| per bar").filter(pl.col("|range| per bar") > 0)).mark_point(size=4, opacity=0.25, color="gray").encode(x="ts:T", y="|range| per bar:Q")
    zstrip = alt.Chart(_s).mark_bar(width=1).encode(x=alt.X("ts:T", title=None), y=alt.Y("z:Q", title="z")).properties(height=90, width="container")
    mo.vstack(
        [
            mo.md(f"## ④ {chosen.model}-{chosen.dist}: fitted σ against each bar'_s Parkinson range (grey)"),
            (dots + lines_).properties(height=240, width="container"),
            zstrip,
            mo.md("A good filter leaves $z$ with no visible clusters."),
        ]
    )
    return (chosen,)


@app.cell
def _(alt, chosen, mo, pl, vol):
    from arch.univariate import GeneralizedError, Normal, SkewStudent, StudentsT

    z = chosen.series["z"].sort()
    n = z.len()
    dist_ = {"normal": Normal, "t": StudentsT, "skewt": SkewStudent, "ged": GeneralizedError}[chosen.dist]()
    shape = {"normal": [], "t": ["nu"], "skewt": ["eta", "lambda"], "ged": ["nu"]}[chosen.dist]
    import numpy as np

    u = (np.arange(1, n + 1) - 0.5) / n
    theo = dist_.ppf(u, np.array([chosen.params[k] for k in shape], dtype=float))
    qq = (
        alt.Chart(pl.DataFrame({"theoretical": theo, "empirical": z.to_numpy()}))
        .mark_point(size=6)
        .encode(x="theoretical:Q", y="empirical:Q")
        .properties(height=220, width=260, title=f"QQ of z against the fitted {chosen.dist}")
    )
    curves = []
    grid = [x / 10 for x in range(-40, 41)]
    return curves, grid, qq


@app.cell
def _(alt, curves, good, grid, mo, pl, qq, vol):
    for _f in good:
        if _f.dist == "t" and _f.model in ("garch", "gjr", "egarch", "aparch", "ewma"):
            curves.append(vol.news_impact(_f, grid).with_columns(pl.lit(_f.model).alias("model")))
    nic = (
        alt.Chart(pl.concat(curves))
        .mark_line()
        .encode(x=alt.X("z:Q", title="shock, in σ̄"), y=alt.Y("sigma2:Q", title="next σ²"), color="model:N")
        .properties(height=220, width=320, title="News impact (Engle and Ng 1993), σₜ₋₁ at σ̄")
    )
    mo.vstack(
        [
            mo.md(r"""
            ## ⑤ Tails and asymmetry

            **QQ.** If the fitted distribution is right, the points lie on
            the diagonal. A normal's QQ bends away in both tails on crypto;
            a *t* with ν ≈ 3 should not.

            **News impact.** The next bar's variance after a shock of *z*
            σ̄, holding today's σ at σ̄. GARCH is a symmetric parabola, GJR a
            kinked one, EGARCH exponential. A curve lower on the left than
            on the right is the reversed asymmetry the crypto literature
            reports (Cheikh et al. 2020).
            """),
            mo.hstack([qq, nic]),
        ]
    )
    return


@app.cell
def _(chosen, mo, vol):
    diag = vol.diagnose(chosen)
    passed = (diag["p_value"] > 0.05).sum()
    mo.vstack(
        [
            mo.md(f"### Residual diagnostics, {chosen.model}-{chosen.dist}"),
            diag,
            mo.md(
                f"{passed} of {diag.height} tests do not reject at 5%. Ljung–Box on z² uses df reduced by the volatility parameters "
                "(the common McLeod–Li-style correction, not Li and Mak's 1994 statistic). A rejection on z² means clustering the model left behind."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## ⑤b Is the persistence real?

    A GARCH fitted through a deterministic pattern or a break in the level
    reads it as persistence. Unmodelled shifts inflate α+β (Lamoureux and
    Lastrapes 1990), and long memory and IGARCH effects can both be artifacts
    of non-stationarity (Mikosch and Stărică 2004). At 1h the daily cycle
    alone makes BTC look integrated (Andersen and Bollerslev 1997). Two checks
    follow:
    - α+β and γ re-estimated on rolling windows, next to the full-sample
      value. A stable α+β well below 1 in each window, against 1 over the
      whole, points to breaks.
    - α+β before and after deseasonalising (1h and 4h).
    """)
    return


@app.cell
def _(alt, column, gr, in_sample, interval, mo, pl, returns, vol):
    kept = in_sample.drop_nulls(column)
    width = {"1h": 1500, "4h": 1000, "1d": 500}[interval.value]
    step = max(width // 5, 50)
    rolling_rows = []
    for start in range(0, max(kept.height - width, 0) + 1, step):
        window_ = kept.slice(start, width)
        g = vol.fit(window_, model="gjr", dist="t", column=column, min_obs=min(width, 250))
        rolling_rows.append({"ts": window_["close_ts"][-1], "α+β": g.params["alpha[1]"] + g.params["beta[1]"], "γ": g.params["gamma[1]"]})
    full = vol.fit(kept, model="gjr", dist="t", column=column, min_obs=250)
    roll = pl.DataFrame(rolling_rows).unpivot(index="ts", variable_name="parameter", value_name="value")
    rolled = (
        alt.Chart(roll)
        .mark_line(point=True)
        .encode(x=alt.X("ts:T", title="window end"), y="value:Q", color="parameter:N")
        .properties(height=200, width="container")
    )
    compare = None
    if interval.value != "1d" and column == "return":
        est = (kept["ts"].min(), kept["close_ts"].max())
        des = gr.timeseries.deseasonalize(returns, gr.timeseries.seasonal_factors(returns, fit=est)).filter(pl.col("close_ts") <= est[1])
        raw_g = vol.fit(kept, model="garch", dist="t", min_obs=250)
        des_g = vol.fit(des, model="garch", dist="t", column="deseasonalized", min_obs=250)
        compare = pl.DataFrame(
            {
                "returns": ["raw", "deseasonalised"],
                "α+β": [raw_g.persistence, des_g.persistence],
                "α": [raw_g.params["alpha[1]"], des_g.params["alpha[1]"]],
                "half-life (bars)": [raw_g.half_life, des_g.half_life],
            }
        )
    mo.vstack(
        [
            mo.md(f"GJR-t on rolling windows of {width} bars, every {step}. Full estimation period: α+β = **{full.params['alpha[1]'] + full.params['beta[1]']:.4f}**, γ = **{full.params['gamma[1]']:+.3f}**."),
            rolled,
            mo.md("GARCH-t, estimation period, raw against deseasonalised (hour × weekday factor fitted on the same period):") if compare is not None else mo.md("Deseasonalising applies to 1h and 4h bars, with the checkbox off."),
            compare if compare is not None else mo.md(""),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
