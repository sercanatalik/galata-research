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

    The study in four parts. **In sample** (①–⑤b), the models are fitted on
    an estimation period and read: parameters, persistence, tails, asymmetry,
    residuals. **Out of sample** (⑥–⑦), they are walked forward from the
    split, every forecast stating the fit it came from. **Scored** (⑧, ⑨, ⑪),
    against a realized proxy, as a set and over time. **Traded** (⑩, ⑫),
    sizing a position net of fees. ⑬ says what survives. Built in ten changes
    (`planning/roadmap.md`). Parameters are shown on returns × 100, the scale
    `arch` fits them on. Every σ is in return units, annualised in calendar
    time.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The claims under test

    What the literature says, and where the record answers it. ⑬ at the end
    computes each verdict from this notebook's results by the rule beside it.
    These are the literature's claims checked after the fact, not
    pre-registered ones.

    | claim | source | section |
    |---|---|---|
    | Fat tails: Student-*t* beats normal | Troster, Tiwari, Shahbaz, Macedo 2019 (heavy tails for 1% VaR on BTC). **Against:** Chu, Chan, Nadarajah, Osterrieder 2017 found IGARCH with *normal* innovations best for Bitcoin | ⑤ |
    | No leverage effect in crypto, or a reversed one | Cheikh, Ben Zaied, Chevallier 2020 | ⑤, ⑤b |
    | α+β ≈ 1 at 1h is the daily cycle, not memory | Andersen and Bollerslev 1997. **In tension:** Rambaccussing and Mazibas 2020 find long memory in crypto *volatility* genuine | ⑤b |
    | HAR beats GARCH when intraday data exist | Bergsli, Lind, Molnár, Polasik 2022 | ⑪ |
    | Does anything beat GARCH(1,1)? | Hansen and Lunde 2005 | ⑪ (MCS) |
    | A better σ forecast is not a better P&L | Becker, Clements, Doolan, Hurn 2015 | ⑫ |
    | Vol targeting cuts drawdown, but not drawdown per unit of vol | Harvey et al. 2018; Bloomberg (Ghia and Hou) 2021: BTC 0.90 → 1.28 | ⑩ |
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

    Hand-written here, because arch does not have them:

    | model | recursion | why |
    |---|---|---|
    | component GARCH | $q_t=\omega+\rho(q_{t-1}-\omega)+\phi(\varepsilon_{t-1}^2-\sigma_{t-1}^2)$, $\sigma_t^2=q_t+\alpha(\varepsilon_{t-1}^2-q_{t-1})+\beta(\sigma_{t-1}^2-q_{t-1})$ | Engle and Lee 1999: a long-run level that moves; best for BTC in Katsiampa 2017 |
    | Beta-*t*-EGARCH | $\lambda_{t+1}=\omega+\phi\lambda_t+\kappa u_t+\kappa^*\operatorname{sgn}(-\varepsilon_t)(u_t+1)$, $u_t\in[-1,\nu]$ the *t* score | Harvey and Chakravarty 2008: one outlier moves σ a bounded amount |
    | CARR | $\lambda_t=\omega+\alpha R_{t-1}+\beta\lambda_{t-1}$ on $R=\ln(H/L)$ | Chou 2005: models the range; σ = λ/√(8/π) |
    | Realized GARCH | $\log h_t=\omega+\beta\log h_{t-1}+\gamma\log x_{t-1}$, $\log x_t=\xi+\phi\log h_t+\tau(z_t)+u_t$ | Hansen, Huang and Shek 2012: a realized measure drives the variance; π = β + φγ (1d, RV from 4h) |
    | MS-GARCH | two GARCH(1,1) variances updated in parallel, $\sigma^2_{k,t}=\omega_k+\alpha_k\varepsilon^2_{t-1}+\beta_k\sigma^2_{k,t-1}$, a Markov chain choosing the regime | Haas, Mittnik and Paolella 2004 (R's MSGARCH): regimes without path dependence (1d) |

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
def _(EVER, column, estimation, gr, in_sample, interval, ticker, vol):
    # MODELS includes the hand-written cgarch and betat; rgarch needs a daily realized measure, so it is fitted at 1d only.
    # rgarch needs a daily realized measure and msgarch takes ~40 s a fit, so both are fitted at 1d only.
    specs = [(m, "t") for m in vol.MODELS if m not in ("rgarch", "msgarch")] + [("garch", d) for d in ("normal", "skewt", "ged")] + [("gjr", "skewt")]
    measures = None
    if interval.value == "1d":
        measures = gr.timeseries.realized_from(gr.market.candles([ticker.value], "4h", *EVER).collect(), "1d")
        specs += [("rgarch", "normal"), ("msgarch", "t")]
    fits = {}
    for model, dist in specs:
        try:
            fits[(model, dist)] = vol.fit(
                in_sample, model=model, dist=dist, column=column, fit=estimation, min_obs=250, measures=measures if model == "rgarch" else None
            )
        except Exception as error:  # a fit that fails is reported, not hidden
            fits[(model, dist)] = error
    good = [f for f in fits.values() if isinstance(f, vol.Fit)]
    failed = {k: str(v) for k, v in fits.items() if not isinstance(v, vol.Fit)}
    table = vol.table(good).sort("bic")
    return failed, good, table


@app.function
def katsiampa_table(table):
    import marimo as mo
    import polars as pl

    parts = []
    rg = table.filter(pl.col("model") == "rgarch")
    if rg.height:
        r = rg.row(0, named=True)
        parts += [
            mo.md("**Realized GARCH against Hansen, Huang and Shek's SPY estimates** (Table II; daily RV here is from six 4h returns, so σᵤ is expected larger). `nobs` differs from the other rows: only days with a complete RV."),
            pl.DataFrame(
                {
                    "": ["HHS 2012, SPY", "this record"],
                    "β": [0.55, r["beta"]],
                    "γ": [0.41, r["gamma"]],
                    "φ": [1.04, r["phi"]],
                    "τ₁": [-0.07, r["tau1"]],
                    "τ₂": [0.07, r["tau2"]],
                    "σᵤ": [0.38, r["sigma_u"]],
                    "π": [0.975, r["persistence"]],
                }
            ),
        ]
    ours = table.filter(pl.col("model") == "cgarch")
    if ours.height == 0:
        return mo.vstack(parts) if parts else mo.md("")
    row = ours.row(0, named=True)
    compare = pl.DataFrame(
        {
            "": ["Katsiampa 2017, BTC daily 2010–2016", "this record, estimation period"],
            "α": [0.1825, row["alpha"]],
            "β": [0.7855, row["beta"]],
            "ρ": [0.9999, row["rho"]],
            "φ": [0.0549, row["phi"]],
        }
    )
    return mo.vstack(
        parts
        + [
            mo.md("**Component GARCH against the paper that found it best for BTC.** Katsiampa's AR(1)-CGARCH(1,1) estimates (Economics Letters 158, 2017) beside ours, with a constant mean and *t* innovations here:"),
            compare,
        ]
    )


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
        + ("a leverage effect, falls raising σ more (γ ≥ 0.1, the rule ⑬ uses)." if gamma >= 0.1 else "below 0.1, no leverage effect to speak of, consistent with crypto's absent or reversed asymmetry (Cheikh, Ben Zaied, Chevallier 2020).")
        if gamma is not None
        else "",
    ]
    cols = ["model", "dist", "nobs", "bic", "persistence", "half_life", "sigma_bar", "converged", "omega", "alpha[1]", "gamma[1]", "beta[1]", "delta", "d", "alpha", "beta", "rho", "phi", "kappa", "kappa_star", "gamma", "xi", "tau1", "tau2", "sigma_u", "omega1", "alpha1", "beta1", "omega2", "alpha2", "beta2", "p11", "p22", "nu", "eta", "lambda"]
    mo.vstack(
        [
            mo.md("## ⑤ The fit table (estimation period)"),
            table.select([c for c in cols if c in table.columns]),
            mo.md("**On this record:** " + " ".join(x for x in lines if x)),
            mo.md(f"Failed fits: {failed}") if failed else mo.md(""),
            katsiampa_table(table),
        ]
    )
    return


@app.cell
def _(alt, bars, good, mo, pl):
    _ms = [f for f in good if f.model == "msgarch"]
    if not _ms:
        _out = mo.md("")
    else:
        _f = _ms[0]
        _p = _f.params
        _u1 = _p["omega1"] / (1 - _p["alpha1"] - _p["beta1"])
        _u2 = _p["omega2"] / (1 - _p["alpha2"] - _p["beta2"])
        _d1, _d2 = 1 / (1 - _p["p11"]), 1 / (1 - _p["p22"])
        _joined = _f.series.select("ts", "p_high").join(bars.select("ts", "close"), on="ts")
        _prob = alt.Chart(_joined).mark_area(opacity=0.35, color="#d62728").encode(x=alt.X("ts:T", title=None), y=alt.Y("p_high:Q", title="P(volatile regime)", scale=alt.Scale(domain=[0, 1])))
        _price = alt.Chart(_joined).mark_line(strokeWidth=1, color="black").encode(x="ts:T", y=alt.Y("close:Q", scale=alt.Scale(type="log"), title="close"))
        _out = mo.vstack(
            [
                mo.md(rf"""
                ### Two regimes (Haas, Mittnik and Paolella 2004)

                Two GARCH variances run side by side on the same shocks, and a
                Markov chain picks which one the market is in. Regime 1 is the
                calm one: unconditional σ² {_u1:.2f} against {_u2:.2f} (on
                returns × 100). Expected stay **{_d1:.1f}** and **{_d2:.1f}**
                days. Stays of a few days mean the two regimes act as a
                *mixture*, a second source of fat tails, rather than as calm
                and turbulent epochs. Ardia, Bluteau and Rüede (2019) found
                regime changes in Bitcoin's GARCH dynamics.
                """),
                alt.layer(_prob, _price).resolve_scale(y="independent").properties(height=220, width="container"),
            ]
        )
    _out
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
    return (compare,)


@app.cell
def _(mo):
    mo.md(r"""
    ## ⑥ Out of sample: walking forward

    From the split onward, every bar's close is an **origin**. Each model
    forecasts the next bars from there using only returns through the origin.
    Its parameters are re-estimated every *k* origins on the data so far and
    held fixed in between (arch's fixed-parameter filtered forecasts). Every
    row carries `fitted_through`, the close of the fit it used, which is never
    past its own origin.

    The forecast for horizon *h* is $E_t[\sigma^2_{t+h}]$. For GARCH it decays
    to the long-run level geometrically,
    $\sigma^2_{t+h|t}=\bar\sigma^2+(\alpha+\beta)^{h-1}(\sigma^2_{t+1|t}-\bar\sigma^2)$
    (Andersen, Bollerslev, Christoffersen and Diebold 2006). EGARCH and APARCH
    have no such formula beyond one step, so they are simulated: 500 seeded
    paths. The grey points are each target bar's Parkinson range, a noisy
    proxy, which is why they are dots and not a line.

    **HAR, SHAR, HARQ** forecast the *realized* variance built from finer
    bars (daily RV from six 4h returns at 1d, 4h RV from four 1h returns at
    4h), by regressing it on its own last value and its weekly and monthly
    means (Corsi 2009):
    $RV_{t+h}=\beta_0+\beta_d RV_t+\beta_w RV^{(w)}_t+\beta_m RV^{(m)}_t$,
    one direct regression per horizon. SHAR splits $RV_t$ by the sign of the
    returns (Patton and Sheppard 2015). HARQ shrinks $\beta_d$ when the
    realized quarticity says $RV_t$ was measured noisily (Bollerslev, Patton
    and Quaedvlieg 2016). The literature expects these to beat GARCH at short
    horizons on Bitcoin (Bergsli et al. 2022). Item 8 scores whether they do
    here. A forecast outside its training range is replaced by the training
    mean (the insanity filter) and marked `filtered`.
    """)
    return


@app.cell
def _(interval, mo):
    HORIZONS = {"1h": {"1 bar": 1, "1 day": 24, "1 week": 168}, "4h": {"1 bar": 1, "1 day": 6, "1 week": 42}, "1d": {"1 day": 1, "1 week": 7, "1 month": 30}}[interval.value]
    EVERY = {"1h": 24, "4h": 6, "1d": 5}[interval.value]
    horizon = mo.ui.dropdown(HORIZONS, value=list(HORIZONS)[0], label="horizon")
    walk_models = mo.ui.multiselect(
        ["ewma", "garch", "gjr", "egarch", "aparch", "figarch", "rm2006", "cgarch", "betat", "carr"] + (["har", "shar", "harq"] if interval.value != "1h" else []) + (["rgarch", "msgarch"] if interval.value == "1d" else []),
        value=["ewma", "garch", "gjr", "egarch"] + (["har", "harq"] if interval.value != "1h" else []),
        label="models",
    )
    go = mo.ui.run_button(label="walk forward")
    mo.hstack([horizon, walk_models, go, mo.md(f"refit every **{EVERY}** bars (FIGARCH every {EVERY * 7})")])
    return EVERY, HORIZONS, go, horizon, walk_models


@app.cell
def _(EVER, gr, mo, pl, vol):
    @mo.cache
    def walk(ticker_, interval_, split_iso, model_, every_, horizons_, deseason_):
        if model_ in ("har", "shar", "harq"):
            _fine = {"1d": "4h", "4h": "1h"}[interval_]
            _measures = gr.timeseries.realized_from(gr.market.candles([ticker_], _fine, *EVER).collect(), interval_)
            return vol.har(_measures, model=model_, split=split_iso, every=every_, horizons=horizons_).with_columns(pl.lit(model_).alias("model"))
        _bars = gr.market.candles([ticker_], interval_, *EVER).collect()
        if model_ == "rgarch":
            _m4 = gr.timeseries.realized_from(gr.market.candles([ticker_], "4h", *EVER).collect(), "1d")
            return vol.walk_forward(
                gr.timeseries.returns(_bars, kind="log"), model="rgarch", dist="normal", measures=_m4, split=split_iso, every=every_, horizons=horizons_, simulations=500, min_obs=250
            ).with_columns(pl.lit(model_).alias("model"))
        if model_ == "carr":
            return vol.carr(_bars, split=split_iso, every=every_, horizons=horizons_, min_obs=250).with_columns(pl.lit(model_).alias("model"))
        _r = gr.timeseries.returns(_bars, kind="log")
        _factors = None
        if deseason_ and interval_ != "1d":
            _factors = gr.timeseries.seasonal_factors(_r, fit=(_bars["ts"].min(), split_iso))
        return vol.walk_forward(
            _r, model=model_, dist="t", split=split_iso, every=every_, horizons=horizons_, factors=_factors, simulations=500, min_obs=250
        ).with_columns(pl.lit(model_).alias("model"))

    return (walk,)


@app.cell
def _(EVERY, HORIZONS, deseason, go, gr, interval, mo, pl, split, ticker, walk, walk_models):
    mo.stop(not go.value, mo.md("Press **walk forward** to fit and forecast out of sample (cached once run)."))
    hs = tuple(sorted(set(HORIZONS.values())))
    _parts, skipped = [], {}
    for _m in walk_models.value:
        try:
            _parts.append(walk(ticker.value, interval.value, split.isoformat(), _m, EVERY * (18 if _m == "msgarch" else 7 if _m == "figarch" else 4 if _m in ("cgarch", "betat") else 1), hs, deseason.value))
        except gr.Refused as _why:  # a model the record cannot support here is reported, not hidden
            skipped[_m] = str(_why)
    walked = pl.concat(_parts)
    return skipped, walked


@app.cell
def _(alt, bars, horizon, mo, per_year, pl, skipped, walked):
    _proxy = bars.select(
        pl.col("ts").alias("target_ts"), ((pl.col("high") / pl.col("low")).log().pow(2) / (4 * 0.6931471805599453)).sqrt().alias("range")
    )
    _at = walked.filter(pl.col("h") == horizon.value).with_columns((pl.col("variance").sqrt() * per_year**0.5).alias("σ̂"))
    _lines = alt.Chart(_at).mark_line(strokeWidth=1).encode(
        x=alt.X("target_ts:T", title="target bar"), y=alt.Y("σ̂:Q", scale=alt.Scale(type="log"), title="σ, annualised"), color="model:N"
    )
    _dots = (
        alt.Chart(_proxy.join(_at.select("target_ts").unique(), on="target_ts").filter(pl.col("range") > 0).with_columns((pl.col("range") * per_year**0.5).alias("range")))
        .mark_point(size=5, opacity=0.3, color="gray")
        .encode(x="target_ts:T", y="range:Q")
    )
    _refits = alt.Chart(_at.filter(pl.col("refit") & (pl.col("model") == _at["model"][0]))).mark_tick(color="black", opacity=0.4, thickness=1).encode(x="close_ts:T")
    mo.vstack(
        [
            mo.md(f"### σ̂ at horizon {horizon.value} bar(s), plotted at the bar it forecasts"),
            (_dots + _lines).properties(height=260, width="container"),
            _refits.properties(height=20, width="container"),
            mo.md(f"{walked.filter(pl.col('h') == horizon.value).height} forecasts; `fitted_through ≤ close_ts` on every row: **{bool((walked['fitted_through'] <= walked['close_ts']).all())}**."),
            mo.md("Not walked: " + "; ".join(f"**{k}**: {v}" for k, v in skipped.items())) if skipped else mo.md(""),
        ]
    )
    return


@app.cell
def _(mo, walked):
    _origins = walked["close_ts"].unique().sort()
    origin = mo.ui.slider(0, _origins.len() - 1, value=_origins.len() // 2, label="origin (out-of-sample bar)")
    fan_length = mo.ui.slider(5, 60, value=30, label="fan length (bars)")
    mo.vstack([mo.md(r"""
    ## ⑦ The forecast fan from one origin

    Every model's $\sigma_{t+h|t}$ for *h* = 1…H from the chosen origin. A
    mean-reverting model bends toward its long-run level (dashed where it
    exists), and EWMA stays flat, since it has no level to revert to. The dots
    are what happened. One-step forecasts hide exactly this difference: the
    models disagree most at long horizons.
    """), mo.hstack([origin, fan_length])])
    return fan_length, origin


@app.cell
def _(EVER, alt, bars, column, deseason, estimation, fan_length, gr, in_sample, interval, mo, origin, per_year, pl, returns, skipped, ticker, vol, walk_models, walked):
    _origins = walked["close_ts"].unique().sort()
    _o = _origins[origin.value]
    _fans, _levels = [], []
    for _m in [m for m in walk_models.value if m not in skipped]:
        if _m == "carr":
            _w = vol.carr(bars, split=_o, every=10**9, horizons=range(1, fan_length.value + 1), min_obs=250)
        elif _m == "rgarch":
            _m4 = gr.timeseries.realized_from(gr.market.candles([ticker.value], "4h", *EVER).collect(), "1d")
            _w = vol.walk_forward(returns.select("ticker", "ts", "close_ts", "return"), model="rgarch", dist="normal", measures=_m4, split=_o, every=10**9, horizons=range(1, fan_length.value + 1), simulations=500, min_obs=250)
        elif _m in ("har", "shar", "harq"):
            _fine = {"1d": "4h", "4h": "1h"}[interval.value]
            _meas = gr.timeseries.realized_from(gr.market.candles([ticker.value], _fine, *EVER).collect(), interval.value)
            _w = vol.har(_meas, model=_m, split=_o, every=10**9, horizons=range(1, fan_length.value + 1))
        else:
            _w = vol.walk_forward(returns.select("ticker", "ts", "close_ts", "return"), model=_m, split=_o, every=10**9, horizons=range(1, fan_length.value + 1), simulations=500, min_obs=250)
        _fans.append(_w.filter(pl.col("close_ts") == _o).with_columns(pl.lit(_m).alias("model")))
        if _m in ("garch", "gjr", "aparch"):
            _fit = vol.fit(returns.filter(pl.col("close_ts") <= _o), model=_m, dist="t", min_obs=250)
            if _fit.sigma_bar is not None:
                _levels.append({"model": _m, "level": _fit.sigma_bar * per_year**0.5})
    _fan = pl.concat(_fans).with_columns((pl.col("variance").sqrt() * per_year**0.5).alias("σ̂"))
    _real = (
        bars.select(pl.col("ts").alias("target_ts"), ((pl.col("high") / pl.col("low")).log().pow(2) / (4 * 0.6931471805599453)).sqrt().alias("range"))
        .join(_fan.select("target_ts", "h").unique(), on="target_ts")
        .with_columns((pl.col("range") * per_year**0.5).alias("range"))
    )
    _chart = alt.Chart(_fan).mark_line(point=True).encode(x=alt.X("h:Q", title="bars ahead"), y=alt.Y("σ̂:Q", title="σ, annualised"), color="model:N")
    _dots = alt.Chart(_real).mark_point(color="gray", filled=True, size=20).encode(x="h:Q", y="range:Q")
    _layers = _chart + _dots
    if _levels:
        _layers = _layers + alt.Chart(pl.DataFrame(_levels)).mark_rule(strokeDash=[4, 4]).encode(y="level:Q", color="model:N")
    mo.vstack([mo.md(f"### From the close of {_o:%Y-%m-%d %H:%M}"), _layers.properties(height=260, width="container")])
    return


@app.cell
def _(interval, mo):
    mo.md(r"""
    ## ⑪ Scoring: which forecasts were better?

    Every forecast is scored against a **proxy** for the variance it
    forecast. A forecast for *h* bars is scored against the proxy summed over
    exactly those *h* bars, and left blank if one is missing. The loss is
    **QLIKE**, $\tilde\sigma^2/h+\ln h$. With a noisy but unbiased proxy it
    ranks models as the true variance would, and it has the most power
    (Patton 2011; Patton and Sheppard 2009). MSE on squared returns is known
    to pick the wrong model (Hansen and Lunde 2006). With about sixteen
    models, a single pairwise test is not enough. The **Model Confidence
    Set** (Hansen, Lunde and Nason 2011) is the set that contains the best
    model with 90% confidence, and a large set is an honest answer.
    """)
    proxy_kind = mo.ui.dropdown(["r2", "parkinson"] + (["rv"] if interval.value != "1h" else []), value="r2", label="proxy")
    benchmark = mo.ui.dropdown(["ewma", "garch"], value="ewma", label="benchmark")
    mo.hstack([proxy_kind, benchmark])
    return benchmark, proxy_kind


@app.cell
def _(EVER, bars, gr, interval, proxy_kind, ticker, walked):
    ev = gr.models.evaluate
    if proxy_kind.value == "rv":
        _fine = {"1d": "4h", "4h": "1h"}[interval.value]
        _proxy = ev.proxies(gr.timeseries.realized_from(gr.market.candles([ticker.value], _fine, *EVER).collect(), interval.value), "rv")
    else:
        _proxy = ev.proxies(bars, proxy_kind.value)
    aligned = ev.align(walked, _proxy)
    return aligned, ev


@app.cell
def _(aligned, alt, benchmark, ev, mo, pl, walked):
    _bench = benchmark.value if benchmark.value in walked["model"].unique().to_list() else walked["model"][0]
    card = ev.scorecard(aligned, benchmark=_bench)
    sets = []
    for _h in sorted(aligned["h"].unique().to_list()):
        try:
            sets.append(ev.mcs(aligned, _h, reps=500, seed=0).with_columns(pl.lit(_h).alias("h")))
        except Exception:
            sets.append(pl.DataFrame({"model": [], "pvalue": [], "included": [], "rows": [], "h": []}))
    mcs_table = pl.concat([x for x in sets if x.height]) if any(x.height for x in sets) else None
    heat = card.join(mcs_table.select("model", "h", "included"), on=["model", "h"], how="left") if mcs_table is not None else card.with_columns(pl.lit(None).alias("included"))
    heat = heat.with_columns(
        pl.format(
            "{}{}{}",
            pl.col("qlike_ratio").round(3).cast(pl.String),
            pl.when(pl.col("dm_p") < 0.05).then(pl.lit("*")).otherwise(pl.lit("")),
            pl.when(pl.col("included")).then(pl.lit(" ●")).otherwise(pl.lit("")),
        ).alias("label")
    )
    _base = alt.Chart(heat).encode(x=alt.X("h:O", title="horizon (bars)"), y=alt.Y("model:N", sort="ascending"))
    _chart = _base.mark_rect().encode(color=alt.Color("qlike_ratio:Q", scale=alt.Scale(scheme="redblue", domainMid=1, reverse=True), title=f"QLIKE / {_bench}")) + _base.mark_text(fontSize=11).encode(text="label:N")
    mo.vstack(
        [
            _chart.properties(height=26 * heat["model"].n_unique(), width="container"),
            mo.md(f"Cell: QLIKE relative to **{_bench}** on the same origins (below 1 is better). `*` Diebold–Mariano p < 0.05 (Harvey–Leybourne–Newbold corrected); `●` in the 90% Model Confidence Set at that horizon."),
            card,
        ]
    )
    return card, mcs_table


@app.cell
def _(aligned, benchmark, ev, mo, pl, walked):
    _bench = benchmark.value if benchmark.value in walked["model"].unique().to_list() else walked["model"][0]
    _rows = []
    for _m in walked["model"].unique().sort().to_list():
        if _m == _bench:
            continue
        try:
            _u = ev.uspa(aligned, model=_m, benchmark=_bench, reps=499)
            _a = ev.aspa(aligned, model=_m, benchmark=_bench, reps=499)
            _rows.append({"model": _m, "rows": _u["rows"], "uSPA t": _u["statistic"], "uSPA p": _u["p_value"], "aSPA t": _a["statistic"], "aSPA p": _a["p_value"]})
        except Exception as _why:
            _rows.append({"model": _m, "rows": 0, "uSPA t": None, "uSPA p": None, "aSPA t": None, "aSPA p": None})
    mo.vstack(
        [
            mo.md(rf"""
            ### One verdict across the horizon path

            The heatmap tests each horizon on its own. Quaedvlieg (2021) asks
            once, across all of them. The **uniform** test (uSPA) asks
            whether the model is better than {_bench} at *every* horizon:
            $t=\min_h \sqrt T\,\bar d_h/\hat\omega_h$. The **average** test
            (aSPA) asks whether it is better on the equal-weighted average.
            Both use a moving-block bootstrap (block 3), with QLIKE losses on
            the origins where every horizon is scored. A small p-value is
            evidence *for* the model.
            """),
            pl.DataFrame(_rows),
        ]
    )
    return


@app.cell
def _(aligned, benchmark, ev, mo, pl, walked):
    _bench = benchmark.value if benchmark.value in walked["model"].unique().to_list() else walked["model"][0]
    _h = int(aligned["h"].min())
    _rows = []
    for _m in walked["model"].unique().sort().to_list():
        if _m == _bench:
            continue
        try:
            _g = ev.gw(aligned, _h, model=_m, benchmark=_bench)
            _rows.append({"model": _m, "n": _g["n"], "GW p": _g["p_value"], "coef on log forecast": _g["coefficients"]["log_forecast"], "rule picks model": _g["decision_share"]})
        except Exception:
            continue
    mo.vstack(
        [
            mo.md(rf"""
            ### When does a model win?

            Being better *on average* can hide being better only in some
            states. Giacomini and White (2006) ask whether the loss difference
            against {_bench} can be predicted from what is known at the
            origin: its own last known value, and the volatility state (the
            benchmark's forecast log-variance). A small p-value says *when* a
            model wins is predictable. A positive coefficient on the log
            forecast says it gains as volatility rises. The last column is how
            often their decision rule would have picked the model.
            *{ev.GW_THEORY}* The walks here are expanding unless changed, so
            read these as indicative.
            """),
            pl.DataFrame(_rows) if _rows else mo.md("No pair had enough origins."),
        ]
    )
    return


@app.cell
def _(aligned, alt, benchmark, ev, mo, pl, walked):
    _models = walked["model"].unique().sort().to_list()
    _bench = benchmark.value if benchmark.value in _models else _models[0]
    _h = int(aligned["h"].min())
    _lines, _flucs = [], []
    for _m in _models:
        if _m == _bench:
            continue
        _w = ev.losses(aligned, _h).select("close_ts", _m, _bench)
        _lines.append(_w.select("close_ts", (pl.col(_bench) - pl.col(_m)).cum_sum().alias("cumulative"), pl.lit(_m).alias("model")))
        try:
            _flucs.append(ev.fluctuation(aligned, _h, model=_m, benchmark=_bench, mu=0.3).with_columns(pl.lit(_m).alias("model")))
        except Exception:
            pass
    _cum = alt.Chart(pl.concat(_lines)).mark_line(strokeWidth=1).encode(x=alt.X("close_ts:T", title=None), y=alt.Y("cumulative:Q", title=f"Σ QLIKE({_bench}) − QLIKE(model)"), color="model:N")
    _fl = pl.concat(_flucs) if _flucs else None
    _parts = [
        mo.md(rf"""
        ## ⑨ When a model wins

        The running sum of the benchmark's QLIKE minus each model's, at the
        shortest horizon. Rising means the model is winning; a jump is one
        episode doing the work. The fluctuation test (Giacomini and Rossi
        2010) turns that into a test: a rolling Diebold–Mariano statistic over
        30% of the sample. It crosses ±the dashed band when relative
        performance was not stable. Critical values are from Rossi's own code.
        """),
        _cum.properties(height=240, width="container"),
    ]
    if _fl is not None:
        _band = _fl.select(pl.col("critical_5").first()).item()
        _stat = alt.Chart(_fl).mark_line(strokeWidth=1).encode(x=alt.X("close_ts:T", title=None), y=alt.Y("statistic:Q", title="fluctuation statistic"), color="model:N")
        _rules = alt.Chart(pl.DataFrame({"y": [_band, -_band]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
        _parts.append((_stat + _rules).properties(height=200, width="container"))
    mo.vstack(_parts)
    return


@app.cell
def _(aligned, alt, bars, ev, good, mo, pl, returns, walked):
    _fitted = {f"{f.model}": f for f in good if f.dist == "t"}
    _candidates = [m for m in walked["model"].unique().sort().to_list() if m in _fitted]
    if not _candidates:
        _out = mo.md("⑧ needs a walked model with an in-sample fit (the GARCH family, cgarch, betat).")
    else:
        _m = "garch" if "garch" in _candidates else _candidates[0]
        _rows = walked.filter((pl.col("model") == _m) & (pl.col("h") == 1) & ~pl.col("after_gap"))
        _var = ev.value_at_risk(_rows, _fitted[_m].series["z"], 0.01)
        _r = returns.select(pl.col("ts").alias("target_ts"), pl.col("return").alias("r"))
        _joined = _var.join(_r, on="target_ts", how="inner").drop_nulls("r")
        _test = ev.var_backtest(_joined["r"], _joined["var"], _joined["es"], 0.01)
        _plot = _joined.with_columns((pl.col("r") < pl.col("var")).alias("breach"))
        _bars = alt.Chart(_plot).mark_bar(width=1, color="gray").encode(x=alt.X("target_ts:T", title=None), y=alt.Y("r:Q", title="return"))
        _v = alt.Chart(_plot).mark_line(color="#d62728", strokeWidth=1).encode(x="target_ts:T", y="var:Q")
        _e = alt.Chart(_plot).mark_line(color="#9467bd", strokeWidth=1, strokeDash=[3, 3]).encode(x="target_ts:T", y="es:Q")
        _b = alt.Chart(_plot.filter(pl.col("breach"))).mark_point(shape="triangle-down", color="#d62728", size=40, filled=True).encode(x="target_ts:T", y="r:Q")
        _out = mo.vstack(
            [
                mo.md(rf"""
                ## ⑧ Value at risk and expected shortfall, 1%, one bar ahead

                {_m}'s σ̂ times the 1% quantile of its own in-sample standardised
                residuals (filtered historical simulation), so the fitted
                distribution is not assumed right. That is what is being
                tested. ES is the mean of the residuals beyond that quantile.
                Kupiec asks whether the breach rate is 1%. Christoffersen asks
                whether breaches cluster. DQ (Engle and Manganelli 2004) asks
                both, with more power. FZ0 scores VaR and ES together (Patton,
                Ziegel and Chen 2019).
                """),
                (_bars + _v + _e + _b).properties(height=240, width="container"),
                mo.md(f"Breaches **{_test['hits']}** of {_test['n']} (expected {_test['expected']:.1f})."),
                pl.DataFrame({k: [v] for k, v in _test.items()}),
            ]
        )
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## ⑩ Does a better σ earn anything?

    Each model's one-step σ̂ sizes a long position,
    $w_t=\min(\tau/\hat\sigma_{t+1\mid t},\,2)$. It is decided at the close and
    held through the next bar, and pays 0.045% on every change. τ is the
    realized volatility of the estimation period, known at the split. A
    full-sample scale is the lookahead Liu, Tang and Zhou (2019) found in
    Moreira and Muir (2017). *Conditional* targeting scales only when σ̂ is
    in the top or bottom quintile of its own history, and holds 1× otherwise
    (Bongaerts, Kang and van Dijk 2020). The band is a no-trade region.
    Beside Sharpe:
    - drawdown per unit of volatility: a smaller drawdown from a smaller
      position is not protection (Harvey et al. 2018; Bloomberg 2021 on BTC);
    - Fleming, Kirby and Ostdiek's fee, the basis points a year a
      quadratic-utility investor would pay to switch from holding.

    Every trial counts toward the Deflated Sharpe Ratio.

    *Expected-shortfall* sizing divides by the 1% ES of the refitted
    Student-t rather than by σ̂: $w=\min(\tau\,\mathrm{ES}(\nu_0)/(\hat\sigma\,\mathrm{ES}(\hat\nu)),2)$.
    It equals inverse vol while ν̂ stays at its value at the split, and holds
    less when a refit finds a fatter tail. The chart below shows how far ν̂
    moved.

    *Feedback* closes the loop (Devanathan, Rueter, Boyd et al. 2026): the
    leverage is multiplied by $e^{\kappa}$, and κ moves against the gap
    between the position's own realized volatility (EWMA, half-life 126) and
    the target: $\kappa_k=(1-\theta)\,\mathrm{clip}(-g\,e_k)+\theta\kappa_{k-1}$,
    with g = 55 and θ = 0.6, the paper's values. On the S&P it cut the
    tracking error from 2.3% to 0.4%. `vol_error` in the table,
    |ln(realized/target)|, is how close each trial came.
    """)
    return


@app.cell
def _(bars, per_year, split, vol, walked):
    targeted = vol.trials(bars, walked, split=split, rules=("inverse_vol", "conditional", "expected_shortfall", "feedback"), bands=(0.0, 0.25))
    tau = vol.estimation_target(bars, split, per_year)
    econ, deflated = vol.economics(targeted, periods_per_year=per_year, target=tau)
    econ = econ.sort("sharpe_annual", descending=True, nulls_last=True)
    return deflated, econ, targeted, tau


@app.cell
def _(alt, mo, pl, walked):
    _nu = walked.filter((pl.col("h") == pl.col("h").min()) & pl.col("nu").is_not_null()).select("close_ts", "model", "nu")
    mo.vstack(
        [mo.md("ν̂ of each refit over the walk (t, skew-t η, and the hand-written t models): what expected-shortfall sizing responds to."),
         alt.Chart(_nu).mark_line(strokeWidth=1, interpolate="step-after").encode(x=alt.X("close_ts:T", title=None), y=alt.Y("nu:Q", title="ν̂"), color="model:N").properties(height=160, width="container")]
    ) if _nu.height else mo.md("")
    return


@app.cell
def _(alt, deflated, econ, mo, pl, targeted):
    _show = ["hold"] + [t for t in econ["trial"].to_list() if t != "hold"][:3]
    _eq = (
        targeted.filter(pl.col("trial").is_in(_show))
        .drop_nulls("net")
        .with_columns(((pl.col("net") + 1).cum_prod().over("trial")).alias("value"), (pl.col("position")).alias("position"))
        .with_columns((pl.col("value") / pl.col("value").cum_max().over("trial") - 1).alias("drawdown"))
    )
    _value = alt.Chart(_eq).mark_line(strokeWidth=1).encode(x=alt.X("ts:T", title=None), y=alt.Y("value:Q", scale=alt.Scale(type="log"), title="growth of 1, net"), color="trial:N").properties(height=220, width="container")
    _pos = alt.Chart(_eq).mark_line(strokeWidth=0.8).encode(x=alt.X("ts:T", title=None), y=alt.Y("position:Q"), color="trial:N").properties(height=90, width="container")
    _dd = alt.Chart(_eq).mark_area(opacity=0.3).encode(x=alt.X("ts:T", title=None), y=alt.Y("drawdown:Q"), color="trial:N").properties(height=90, width="container")
    mo.vstack(
        [
            mo.md(f"Hold and the three best trials by net Sharpe, of **{deflated['trials']}** run. Deflated Sharpe Ratio of the best ({deflated['trial']}): **{deflated['dsr']:.3f}**."),
            _value,
            _pos,
            _dd,
            econ,
        ]
    )
    return


@app.cell
def _(card, econ, mo, pl):
    _one = card.filter(pl.col("h") == pl.col("h").min()).select("model", "qlike")
    _money = econ.filter(pl.col("trial").str.ends_with("inverse_vol band 0")).with_columns(pl.col("trial").str.split(" ").list.first().alias("model")).select("model", "sharpe_annual", "fee_bp_g10")
    _both = _one.join(_money, on="model").with_columns(
        pl.col("qlike").rank().alias("rank_qlike"),
        pl.col("sharpe_annual").rank(descending=True).alias("rank_sharpe"),
        pl.col("fee_bp_g10").rank(descending=True).alias("rank_fee"),
    )
    _rho_s = _both.select(pl.corr("rank_qlike", "rank_sharpe")).item() if _both.height > 2 else None
    _rho_f = _both.select(pl.corr("rank_qlike", "rank_fee")).item() if _both.height > 2 else None
    _verdict = (
        "The statistical and the economic rankings **agree**."
        if _rho_s is not None and _rho_s > 0.5
        else "The statistical and the economic rankings **do not agree**: a better σ forecast did not reliably earn more, as Becker, Clements, Doolan and Hurn (2015) warn."
    )
    mo.vstack(
        [
            mo.md(r"""
            ## ⑫ Do the two rankings agree?

            Each model's one-step QLIKE rank (1 = best forecast) against the
            rank of its inverse-vol trial's net Sharpe and FKO fee (γ = 10).
            Spearman's ρ near 1 means better forecasts earned more. Near 0 or
            negative means they did not.
            """),
            _both.sort("rank_qlike"),
            mo.md(f"ρ(QLIKE, Sharpe) = **{_rho_s if _rho_s is None else round(_rho_s, 2)}**, ρ(QLIKE, fee) = **{_rho_f if _rho_f is None else round(_rho_f, 2)}**, over {_both.height} models. {_verdict}"),
        ]
    )
    return


@app.cell
def _(card, compare, econ, good, interval, mcs_table, mo, pl, table):
    def _row(claim, verdict, number, rule):
        return {"claim": claim, "verdict": verdict, "on this record": number, "rule": rule}

    _rows = []
    # Fat tails
    _g = {r["dist"]: r["bic"] for r in table.filter(pl.col("model") == "garch").iter_rows(named=True)}
    if {"t", "normal"} <= set(_g):
        _d = _g["t"] - _g["normal"]
        _rows.append(_row("t beats normal", "consistent" if _d < -10 else "contradicts" if _d > 0 else "can't tell", f"ΔBIC {_d:+.0f}", "consistent if t's BIC is lower by > 10"))
    else:
        _rows.append(_row("t beats normal", "can't tell", "—", "needs GARCH-t and GARCH-normal fits"))
    # Leverage
    _gjr = [f for f in good if f.model == "gjr" and f.dist == "t"]
    if _gjr:
        _gm, _se = _gjr[0].params["gamma[1]"], _gjr[0].std_err.get("gamma[1]")
        _sig = _se is not None and abs(_gm) > 2 * _se
        _rows.append(_row("no leverage effect", "contradicts" if _gm >= 0.1 and _sig else "consistent", f"γ {_gm:+.3f} (se {_se:.3f})" if _se else f"γ {_gm:+.3f}", "contradicts if γ ≥ 0.1 and 2 se from 0"))
    # Persistence at 1h
    if compare is not None:
        _raw, _des = compare["α+β"][0], compare["α+β"][1]
        _rows.append(_row(f"α+β≈1 at {interval.value} is the daily cycle", "consistent" if _raw - _des > 0.002 else "contradicts", f"{_raw:.4f} → {_des:.4f}", "consistent if deseasonalising lowers α+β by > 0.002"))
    else:
        _rows.append(_row("α+β≈1 intraday is the daily cycle", "can't tell", "—", f"run at 1h or 4h with the checkbox off (now {interval.value})"))
    # HAR vs GARCH
    _q = card.select("model", "h", "qlike")
    _har = _q.filter(pl.col("model").is_in(["har", "harq"])).group_by("h").agg(pl.col("qlike").min().alias("har"))
    _gar = _q.filter(pl.col("model") == "garch").select("h", pl.col("qlike").alias("garch"))
    _hg = _har.join(_gar, on="h")
    if _hg.height:
        _wins = int((_hg["har"] < _hg["garch"]).sum())
        _rows.append(_row("HAR beats GARCH", "consistent" if _wins == _hg.height else "contradicts" if _wins == 0 else "mixed", f"{_wins} of {_hg.height} horizons", "consistent if HAR or HARQ has lower QLIKE at every horizon"))
    else:
        _rows.append(_row("HAR beats GARCH", "can't tell", "—", "walk garch and har/harq forward (1d or 4h)"))
    # Anything beats GARCH(1,1)
    if mcs_table is not None and "garch" in mcs_table["model"].to_list():
        _out = mcs_table.filter((pl.col("model") == "garch") & ~pl.col("included"))["h"].to_list()
        _rows.append(_row("something beats GARCH(1,1)", "yes" if _out else "no", f"GARCH outside the MCS at h = {_out}" if _out else "GARCH in every MCS", "yes if GARCH is excluded from the 90% MCS at some horizon"))
    else:
        _rows.append(_row("something beats GARCH(1,1)", "can't tell", "—", "walk garch forward with other models"))
    # Better σ ≠ better P&L
    _one = card.filter(pl.col("h") == pl.col("h").min()).select("model", "qlike")
    _money = econ.filter(pl.col("trial").str.ends_with("inverse_vol band 0")).with_columns(pl.col("trial").str.split(" ").list.first().alias("model")).select("model", "sharpe_annual")
    _both = _one.join(_money, on="model").with_columns(pl.col("qlike").rank().alias("a"), pl.col("sharpe_annual").rank(descending=True).alias("b"))
    if _both.height > 2:
        _rho = _both.select(pl.corr("a", "b")).item()
        _rows.append(_row("better σ ≠ better P&L", "consistent" if _rho < 0.5 else "contradicts", f"ρ = {_rho:.2f} over {_both.height}", "consistent if ρ(QLIKE rank, Sharpe rank) < 0.5"))
    else:
        _rows.append(_row("better σ ≠ better P&L", "can't tell", "—", "needs three or more walked models"))
    # Drawdown per vol
    _hold = econ.filter(pl.col("trial") == "hold")["drawdown_per_vol"]
    _best = econ.filter(pl.col("trial") != "hold").sort("sharpe_annual", descending=True, nulls_last=True)["drawdown_per_vol"]
    if _hold.len() and _best.len():
        _rows.append(_row("targeting does not cut drawdown per vol", "consistent" if _best[0] >= _hold[0] else "contradicts", f"{_best[0]:.2f} vs hold {_hold[0]:.2f}", "consistent if the best trial's drawdown per vol ≥ hold's"))
    verdicts = pl.DataFrame(_rows)
    mo.vstack(
        [
            mo.md(r"""
            ## ⑬ What survives

            Each claim from the top, decided by the rule in its row from the
            results above. The rules were written down before the notebook
            computed them. A borderline number can flip a verdict, so the
            number is shown beside it. "Can't tell" means a section was not
            run, or had too little to decide.
            """),
            verdicts,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## References

    Verified 2026-09-27 against publisher or repository pages. Where a paper
    was out of reach, the source used is named.

    **Models.** Engle 1982, *Econometrica* 50(4):987–1007 · Bollerslev 1986,
    *J. Econometrics* 31:307–327 · Bollerslev 1987, *REStat* 69(3):542–547 ·
    Nelson 1991, *Econometrica* 59(2):347–370 · Glosten, Jagannathan, Runkle
    1993, *J. Finance* 48(5):1779–1801 · Ding, Granger, Engle 1993, *J.
    Empirical Finance* 1:83–106 · Hansen 1994, *Int. Economic Review*
    35(3):705–730 · Baillie, Bollerslev, Mikkelsen 1996, *J. Econometrics*
    74:3–30 · Engle and Lee 1999, in *Cointegration, Causality and
    Forecasting* · Harvey and Chakravarty 2008, Cambridge WP 0840 · Chou 2005,
    *J. Money, Credit and Banking* 37(3):561–582 · Corsi 2009, *J. Financial
    Econometrics* 7(2):174–196 · Patton and Sheppard 2015, *REStat*
    97(3):683–697 · Bollerslev, Patton, Quaedvlieg 2016, *J. Econometrics*
    192(1):1–18 · Andersen, Bollerslev, Diebold 2007, *REStat* 89(4):701–720.

    **Crypto evidence.** Katsiampa 2017, *Economics Letters* 158:3–6 · Chu,
    Chan, Nadarajah, Osterrieder 2017, *JRFM* 10(4):17 (IGARCH-normal best for
    BTC) · Troster, Tiwari, Shahbaz, Macedo 2019, *FRL* 30:187–193 · Cheikh,
    Ben Zaied, Chevallier 2020, *FRL* 35:101293 · Rambaccussing and Mazibas
    2020, *JRFM* 13(9):186 · Bergsli, Lind, Molnár, Polasik 2022, *RIBAF*
    59:101540 · Hansen, Kim, Kimbrough 2021, "Periodicity in Cryptocurrency
    Volatility and Liquidity", arXiv 2109.12142.

    **Seasonality and persistence.** Andersen and Bollerslev 1997, *J.
    Empirical Finance* 4:115–158 · Lamoureux and Lastrapes 1990, *JBES*
    8(2):225–234 · Mikosch and Stărică 2004, *REStat* 86(1):378–390.

    **Realized measures.** Parkinson 1980, *J. Business* 53(1):61–65 ·
    Garman and Klass 1980, *J. Business* 53(1):67–78 · Rogers and Satchell
    1991, *Ann. Applied Probability* 1(4):504–512 · Yang and Zhang 2000, *J.
    Business* 73(3):477–491 · Christensen and Podolskij 2007, *J.
    Econometrics* 141(2):323–349 · Martens and van Dijk 2007, *J.
    Econometrics* 138(1):181–207 · Liu, Patton, Sheppard 2015, *J.
    Econometrics* 187(1):293–311.

    **Evaluation.** Diebold and Mariano 1995, *JBES* 13(3):253–263 · Harvey,
    Leybourne, Newbold 1997, *IJF* 13(2):281–291 · Hansen 2005, *JBES*
    23(4):365–380 · Hansen and Lunde 2005, *JAE* 20(7):873–889 · Hansen and
    Lunde 2006, *J. Econometrics* 131:97–121 · Patton and Sheppard 2009, in
    *Handbook of Financial Time Series* · Patton 2011, *J. Econometrics*
    160(1):246–256 · Hansen, Lunde, Nason 2011, *Econometrica* 79(2):453–497
    · Giacomini and Rossi 2010, *JAE* 25(4):595–620 (critical values from
    Rossi's `giacross.ado`) · Goyal and Welch 2008, *RFS* 21(4):1455–1508 ·
    Kupiec 1995, *J. Derivatives* 3(2):73–84 · Christoffersen 1998, *Int.
    Economic Review* 39(4):841–862 · Engle and Manganelli 2004, *JBES*
    22(4):367–381 · Patton, Ziegel, Chen 2019, *J. Econometrics*
    211(2):388–413.

    **Targeting and economic value.** Fleming, Kirby, Ostdiek 2001, *J.
    Finance* 56(1):329–352 · Moreira and Muir 2017, *J. Finance*
    72(4):1611–1644 · Liu, Tang, Zhou 2019, *JPM* 46(1):38–51 · Harvey,
    Hoyle, Korgaonkar, Rattray, Sargaison, Van Hemert 2018, *JPM*
    45(1):14–33 · Bongaerts, Kang, van Dijk 2020, *FAJ* 76(4):54–71 ·
    Becker, Clements, Doolan, Hurn 2015, *IJF* 31(3):849–861 · Ghia and Hou
    2021, "Crypto Insights: The Impact of Volatility Targeting", Bloomberg ·
    Grobys, Kolari, Sandretto, Shahzad, Äijö 2025, "Cryptocurrency momentum
    has (not) its moments", *FMPM* 39(4) · Devanathan, Rueter, Boyd, Candès,
    Hastie, Kochenderfer et al. 2026, "Single-Asset Adaptive Leveraged
    Volatility Control", arXiv 2603.01298.
    """)
    return


if __name__ == "__main__":
    app.run()
