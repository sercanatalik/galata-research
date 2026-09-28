import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import time
    from datetime import UTC, datetime

    import marimo as mo
    import polars as pl

    import galata_research as gr

    return UTC, datetime, gr, mo, pl, time


@app.cell
def _(mo):
    mo.md(r"""
    # Does DCC forecast covariance better than the simpler estimators?

    **The registered procedure, run as written** —
    `planning/preregistered/correlation-models.md`, committed before any
    forecast was scored. BTC, ETH and HYPE at 4h, one bar ahead, from the split
    2025-11-01T04:00Z, refit daily.

    | family | DCC against | why |
    |---|---|---|
    | H1, Σ | RiskMetrics EWMA (λ 0.94), the 180-bar sample covariance | what a desk would use instead |
    | H2, R alone | constant correlation, EWMA on z (λ 0.94) | the same GJR-t σ, so QLIKE differs only by the correlation likelihood |
    | H3, economic | CCC, EWMA | the minimum-variance portfolio's realised risk (Engle and Colacito 2006) |

    Losses with r rᵀ as the proxy: multivariate QLIKE, ln|H| + rᵀH⁻¹r, and
    Frobenius, both robust to a noisy proxy (Patton and Sheppard 2009;
    Laurent, Rombouts and Violante 2013). DM with a Newey–West variance and
    Holm across each family; the MCS at 90%.
    """)
    return


@app.cell
def _(UTC, datetime, gr, pl):
    CUT = datetime(2026, 9, 28, tzinfo=UTC)
    MAIN = ["BTC", "ETH", "HYPE"]
    bars = gr.mask_gaps(gr.market.candles(MAIN, "4h", "2020-01-01T00:00Z", "2100-01-01T00:00Z"), "candles").collect().filter(pl.col("close_ts") <= CUT)
    returns = gr.timeseries.returns(bars, kind="log")
    _, joint, _ = gr.models.corr.joint(returns)
    split = joint["close_ts"][joint.height // 2 - 1]
    assert joint.height == 3969 and str(split) == "2025-11-01 04:00:00+00:00", (joint.height, split)
    return CUT, returns, split


@app.cell
def _(gr, returns, split, time):
    def _walk(**kw):
        t0 = time.perf_counter()
        w = gr.models.corr.walk_forward(returns, split=split, horizons=[1], every=6, seed=0, min_obs=500, **kw)
        return w, time.perf_counter() - t0

    gjr = dict(model="gjr", dist="t")
    arms = {
        "dcc": dict(corr="dcc", **gjr),
        "cdcc": dict(corr="cdcc", **gjr),
        "ccc": dict(corr="ccc", **gjr),
        "iewma": dict(corr="iewma", lam=0.94, **gjr),
        "ewma": dict(corr="ewma", lam=0.94),
        "sample": dict(corr="sample", sample_window=180),
    }
    walked = {name: _walk(**kw) for name, kw in arms.items()}
    forecasts = {name: w for name, (w, _) in walked.items()}
    seconds = {name: s for name, (_, s) in walked.items()}
    scores = gr.models.corr.score(forecasts, returns)
    return scores, seconds


@app.cell
def _(gr, pl, scores):
    def holm(ps):
        order = sorted(range(len(ps)), key=lambda i: ps[i])
        adj, running = [0.0] * len(ps), 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
            adj[i] = running
        return adj

    def family(loss, against):
        rows = []
        for alt in against:
            t = gr.models.corr.compare(scores.filter(pl.col("model").is_in(["dcc", alt])), loss=loss, benchmark=alt, reps=1000)
            dcc = t.filter(pl.col("model") == "dcc").row(0, named=True)
            other = t.filter(pl.col("model") == alt).row(0, named=True)
            rows.append({"against": alt, "loss": loss, "dcc_mean": dcc["mean_loss"], "alt_mean": other["mean_loss"], "dm": dcc["dm"], "p": dcc["dm_p"], "bars": dcc["bars"]})
        adj = holm([r["p"] for r in rows])
        for r, a in zip(rows, adj):
            r["p_holm"] = a
            better = r["dcc_mean"] < r["alt_mean"]
            r["verdict"] = ("supported" if better else "refuted") if a < 0.05 else "undecided"
        return pl.DataFrame(rows)

    primary = family("stein", ["ewma", "sample", "ccc", "iewma"])
    secondary = family("gmv", ["ccc", "ewma"])
    frobenius = family("frobenius", ["ewma", "sample", "ccc", "iewma"]).drop("verdict", "p_holm")
    mcs = gr.models.corr.compare(scores, loss="stein", benchmark="dcc", reps=10_000, seed=0)
    return frobenius, mcs, primary, secondary


@app.cell
def _(frobenius, mcs, mo, primary, secondary, seconds):
    mo.vstack(
        [
            mo.md("### Primary: multivariate QLIKE, Holm across four"),
            mo.ui.table(primary, selection=None),
            mo.md("### Secondary: the minimum-variance portfolio, Holm across two"),
            mo.ui.table(secondary, selection=None),
            mo.md("### Frobenius, reported, not tested"),
            mo.ui.table(frobenius, selection=None),
            mo.md("### The Model Confidence Set at 90% on QLIKE (10,000 replicates)"),
            mo.ui.table(mcs, selection=None),
            mo.md(f"Seconds per walk: {', '.join(f'{k} {v:.0f}' for k, v in seconds.items())}."),
        ]
    )
    return


@app.cell
def _(mo, primary, secondary):
    _p = {r["against"]: r for r in primary.iter_rows(named=True)}
    _s = {r["against"]: r for r in secondary.iter_rows(named=True)}
    mo.md(f"""
    ### What the registered rule decides

    - **H1a, DCC against EWMA (Σ):** {_p["ewma"]["verdict"]} (Holm p {_p["ewma"]["p_holm"]:.3f})
    - **H1b, DCC against the 180-bar sample:** {_p["sample"]["verdict"]} (Holm p {_p["sample"]["p_holm"]:.3f})
    - **H2a, DCC against constant correlation (R alone):** {_p["ccc"]["verdict"]} (Holm p {_p["ccc"]["p_holm"]:.3f})
    - **H2b, DCC against EWMA on z (R alone):** {_p["iewma"]["verdict"]} (Holm p {_p["iewma"]["p_holm"]:.3f})
    - **H3, minimum-variance portfolio, against CCC:** {_s["ccc"]["verdict"]}; **against EWMA:** {_s["ewma"]["verdict"]}
    """)
    return


@app.cell
def _(CUT, gr, mo, pl):
    # The registered secondary: all six instruments, split at joint row 799, the same models; described, not tested.
    _bars = gr.mask_gaps(gr.market.candles(None, "4h", "2020-01-01T00:00Z", "2100-01-01T00:00Z"), "candles").collect().filter(pl.col("close_ts") <= CUT)
    six_returns = gr.timeseries.returns(_bars, kind="log")
    _, _joint, _ = gr.models.corr.joint(six_returns)
    _split = _joint["close_ts"][799]
    _gjr = dict(model="gjr", dist="t")
    _arms = {
        "dcc": dict(corr="dcc", **_gjr), "cdcc": dict(corr="cdcc", **_gjr), "ccc": dict(corr="ccc", **_gjr),
        "iewma": dict(corr="iewma", lam=0.94, **_gjr), "ewma": dict(corr="ewma", lam=0.94), "sample": dict(corr="sample", sample_window=180),
    }  # fmt: skip
    _forecasts = {n: gr.models.corr.walk_forward(six_returns, split=_split, horizons=[1], every=6, seed=0, min_obs=500, **kw) for n, kw in _arms.items()}
    six = gr.models.corr.score(_forecasts, six_returns).group_by("model").agg(pl.col("stein").mean(), pl.col("gmv").mean(), pl.len().alias("bars")).sort("stein")
    mo.vstack([mo.md(f"### Secondary, descriptive: all six instruments, split {_split:%Y-%m-%d %H:%M}"), mo.ui.table(six, selection=None)])
    return (six,)


if __name__ == "__main__":
    app.run()
