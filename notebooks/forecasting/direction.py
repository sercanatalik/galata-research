import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import marimo as mo
    import polars as pl

    import galata_research as gr
    from galata_research import momentum
    from galata_research.models import direction

    return direction, gr, mo, momentum, os, pl


@app.cell
def _(gr):
    # Cached results are keyed on the days the store holds, so a fetch recomputes them.
    try:
        HELD = int(gr.reference.coverage()["days_ok"].sum())
    except gr.Refused:
        HELD = 0
    return (HELD,)


@app.cell
def _(mo):
    mo.md(r"""
    # Does momentum tell the next move's direction?

    Registered in `planning/preregistered/forecast-the-direction.md`, committed
    alone (`d14857e`) on 2026-10-26 before any feature, target or forecast
    existed. This notebook runs it as registered.

    **The question.** Momentum on this record has failed as a trading rule
    (`indicator_signals.py`, `donchian_ensemble.py`, `universe.py`). This asks
    something else: does a ladder of short (15m, 1h, 4h) and long (1d, 3d, 1w,
    4w) momentum z's, measured on 1-minute data, carry information about the
    sign of the next 5m, 30m, 5h or 1d return **beyond volatility and the
    clock**? Model A is a logistic regression on log σ and the time of day and
    week; B adds the momentum. The claim is B over A.

    **Theory, briefly.**
    - Time-series momentum (Moskowitz, Ooi and Pedersen 2012) is the sign of an
      asset's own past return. Baz et al. (2015) scale it by volatility and
      squash it, z·exp(−z²/4)/0.89, so that the strongest trends are not the
      largest bets.
    - Kim, Tse and Wald (2016) find most of its published alpha is the
      volatility scaling. That is why A already has σ, and why the economic
      test is judged against the inverse-vol book, not only against holding.
    - At intraday horizons the evidence is mixed. In Bitcoin the first
      half-hour (on a volume clock) predicts the last (Shen, Urquhart and Wang
      2022), while 1–4 h returns mean-revert (De Nicola 2021). The order flow
      is predictable, but the price mostly is not (Lillo and Farmer 2004;
      Bouchaud et al. 2004): the S parts of `flow.py` held and every E part
      lost.

    **The arithmetic that sets the expectation.** A taker flipping every period
    needs a hit rate of 89% at 5m, 66% at 30m, 55% at 5h and 52.3% at 1d. The
    out-of-sample span can detect 50.2% at 5m but only 52.7% at 1d.
    Expected: **S supported at 5m and 30m, mixed at 5h, not at 1d; E not
    supported.**

    | part | horizons | supported when |
    |---|---|---|
    | S | all | B < A on log loss (DM, h − 1 lags, Holm p < 0.05 over 24 cells), Pesaran–Timmermann p < 0.05, the same sign on both tickers and both halves |
    | E | 5h, 1d | the sign rule net of 4.75 bps per unit and funding: DSR ≥ 0.95 at N = 24, PBO < 0.5, Sharpe above hold and inverse vol |
    | (b) | all | secondary: S criteria 1 and 3 on the three-class (±9.5 bps) multinomial log loss |
    """)
    return


@app.cell
def _(os):
    # GR_DIRECTION_DRY=1 checks the code on the estimation period only, which the registration allows.
    DRY = os.environ.get("GR_DIRECTION_DRY") == "1"
    if DRY:
        SAMPLE = ("2020-01-01T00:00Z", "2023-01-01T00:00Z")
        SPLIT, HALVES = "2022-01-01T00:00Z", "2022-07-01T00:00Z"
    else:
        SAMPLE = ("2020-01-01T00:00Z", "2026-10-01T00:00Z")
        SPLIT, HALVES = "2023-01-01T00:00Z", "2025-01-01T00:00Z"
    FEATURES_FROM = "2020-03-01T00:00Z"
    TICKERS = ("BTC", "ETH")
    HORIZONS = ("5m", "30m", "5h", "1d")
    ECONOMIC = ("5h", "1d")
    ENCODINGS = ("binary", "ternary", "score")
    return DRY, ECONOMIC, ENCODINGS, FEATURES_FROM, HALVES, HORIZONS, SAMPLE, SPLIT, TICKERS


@app.cell
def _(DRY, FEATURES_FROM, HELD, SAMPLE, TICKERS, gr, mo, momentum, pl):
    def _load():
        feats, untraded, funding = {}, {}, {}
        for t in TICKERS:
            bars = gr.reference.candles(t, *SAMPLE, venues="binance-um").collect()
            untraded[t] = momentum.untraded(bars)
            f = momentum.features(bars)
            feats[t] = f.filter(pl.col("ts") >= pl.lit(FEATURES_FROM).str.to_datetime(time_zone="UTC"))
            settled = gr.reference.funding(t, *SAMPLE, venues="binance-um").collect()
            funding[t] = gr.reference.funding_hours(settled)
        return feats, untraded, funding

    with mo.persistent_cache(name=f"direction-features-{HELD}-{DRY}"):
        feats, untraded, funding = _load()

    _zs = [f"z_{n}" for n in momentum.LADDER]
    _lost = pl.concat(
        [
            f.group_by(pl.col("ts").dt.year().alias("year"))
            .agg(pl.len().alias("rows"), pl.any_horizontal(pl.col(_zs).is_null()).sum().alias("rows_with_a_null_z"))
            .with_columns(pl.lit(t).alias("ticker"), (pl.col("rows_with_a_null_z") / pl.col("rows")).alias("share_lost"))
            for t, f in feats.items()
        ]
    ).sort("ticker", "year")
    mo.vstack(
        [
            mo.md(f"## The features\n\nUntraded 1m bars dropped: {', '.join(f'{t} {n:,}' for t, n in untraded.items())}."),
            mo.md("Rows lost to holes (a lookback window that holds one), per year:"),
            _lost,
        ]
    )
    return feats, funding


@app.cell
def _(HORIZONS, SPLIT, direction, feats, mo, momentum, pl):
    # Described, not tested: the 2x2 momentum state on the estimation period, before any model.
    _short = pl.mean_horizontal([f"z_{n}" for n in momentum.SHORT]) > 0
    _long = pl.mean_horizontal([f"z_{n}" for n in momentum.LONG]) > 0
    _split = pl.lit(SPLIT).str.to_datetime(time_zone="UTC")
    _rows = []
    for _t, _f in feats.items():
        for _h in HORIZONS:
            _rows.append(
                direction.targets(_f.filter(pl.col("close_ts") <= _split), _h)
                .drop_nulls(["y", "z_15m", "z_4w"])
                .group_by(_short.alias("short_up"), _long.alias("long_up"))
                .agg(pl.len().alias("n"), pl.col("y").cast(pl.Float64).mean().alias("share_up"))
                .with_columns(pl.lit(_t).alias("ticker"), pl.lit(_h).alias("horizon"))
            )
    mo.vstack(
        [
            mo.md("## The momentum state on the estimation period\n\nShare of up moves by the sign of the mean short and mean long z. Described only."),
            pl.concat(_rows).sort("ticker", "horizon", "short_up", "long_up"),
        ]
    )
    return


@app.cell
def _(DRY, ECONOMIC, ENCODINGS, HALVES, HELD, HORIZONS, SPLIT, direction, feats, mo, momentum, pl):
    def _non_overlapping(f):
        first = f["close_ts"].min()
        step = f["h"][0] * 5
        keep = ((pl.col("close_ts") - first).dt.total_minutes() % step) == 0
        return f.filter(keep).with_columns(pl.lit(1).alias("h"))

    def _three_stats(f):
        p = direction.paired(f, "B", "A").with_columns((pl.col("loss_model") - pl.col("loss_benchmark")).alias("d"))
        at = pl.lit(HALVES).str.to_datetime(time_zone="UTC")
        dm = direction.compare(f, model="B", benchmark="A")
        return {
            "mean_difference": dm["mean_difference"],
            "dm_p": dm["p_value"],
            "first_half": p.filter(pl.col("close_ts") < at)["d"].mean(),
            "second_half": p.filter(pl.col("close_ts") >= at)["d"].mean(),
        }

    def _run():
        s_rows, s_flat, b_rows, rel, coefs, keep = [], [], [], [], [], {}
        for t, f in feats.items():
            for h in HORIZONS:
                for enc in ENCODINGS:
                    frame = direction.targets(momentum.encoded(f, enc), h)
                    cache = {}
                    sign = direction.walk_forward(frame, horizon=h, split=SPLIT, encoding=enc, target="sign", cache=cache)
                    key = {"horizon": h, "encoding": enc, "ticker": t}
                    s_rows.append(key | direction.cell(sign, halves_at=HALVES))
                    s_flat.append(key | direction.cell(_non_overlapping(sign), halves_at=HALVES))
                    s_rows[-1] |= {
                        "brier_skill_B": direction.brier_skill(sign, model="B", benchmark="climatology"),
                        "brier_skill_A": direction.brier_skill(sign, model="A", benchmark="climatology"),
                    }
                    rel.append(direction.reliability(sign.filter(pl.col("set") == "B")).with_columns(**{k: pl.lit(v) for k, v in key.items()}))
                    last = max(r for (tk, hz, which, en, tg, r) in cache if which == "B")
                    beta = cache[(t, h, "B", enc, "sign", last)]["beta"]
                    names = ["intercept", *direction.columns(frame.pipe(direction.clock), "B")]
                    coefs.append(pl.DataFrame({"column": names, "beta": beta}).with_columns(**{k: pl.lit(v) for k, v in key.items()}))
                    if h in ECONOMIC:
                        keep[(t, h, enc)] = _non_overlapping(sign.filter(pl.col("set") == "B")).with_columns(pl.lit(sign["h"][0]).alias("h"))
                    three = direction.walk_forward(frame, horizon=h, split=SPLIT, encoding=enc, target="three")
                    b_rows.append(key | _three_stats(three))
        return pl.DataFrame(s_rows), pl.DataFrame(s_flat), pl.DataFrame(b_rows), pl.concat(rel), pl.concat(coefs), keep

    with mo.persistent_cache(name=f"direction-walk-{HELD}-{DRY}"):
        s_cells, s_flat, b_cells, reliability, coefficients, kept = _run()
    mo.md(f"Walked **{s_cells.height}** sign cells and **{b_cells.height}** three-class cells.")
    return b_cells, coefficients, kept, reliability, s_cells, s_flat


@app.cell
def _(direction, mo, pl, s_cells, s_flat):
    s_judged, s_horizons = direction.s_verdict(s_cells)
    _flat_judged, _flat_horizons = direction.s_verdict(s_flat)
    _cols = ["horizon", "encoding", "ticker", "mean_difference", "dm_statistic", "lags", "holm_p", "pt_p", "hit_rate", "first_half", "second_half"]
    mo.vstack(
        [
            mo.md("## S: does B beat A out of sample?\n\nPer cell (horizon × encoding × ticker): the registered criteria."),
            s_judged.select(*_cols, "brier_skill_A", "brier_skill_B", "c1", "c2", "c3", "met").sort("horizon", "encoding", "ticker"),
            mo.md("**Per horizon, as registered:**"),
            s_horizons,
            mo.md("**Described, not tested:** the same on non-overlapping origins only."),
            _flat_horizons.rename({"supported": "supported_non_overlapping"}),
        ]
    )
    return s_horizons, s_judged


@app.cell
def _(coefficients, mo, pl, reliability):
    mo.vstack(
        [
            mo.md("## Calibration and coefficients (described)\n\nB's reliability per cell: mean forecast against the share up, in bins of P(up) with any rows."),
            reliability.filter(pl.col("n") > 0).sort("horizon", "encoding", "ticker", "bin"),
            mo.md("B's coefficients at each cell's last refit, on the standardised columns. Collinear across the ladder: read none alone."),
            coefficients.pivot(on="column", index=["horizon", "encoding", "ticker"], values="beta").sort("horizon", "encoding", "ticker"),
        ]
    )
    return


@app.cell
def _(ECONOMIC, ENCODINGS, SPLIT, TICKERS, direction, feats, funding, kept, mo):
    def _cells():
        return {
            f"{h} {enc} {t}": direction.economic(feats[t], kept[(t, h, enc)], split=SPLIT, funding=funding[t])
            for h in ECONOMIC
            for enc in ENCODINGS
            for t in TICKERS
        }

    e_cells = _cells()
    e_result = direction.e_verdict(e_cells)
    mo.vstack(
        [
            mo.md("## E: does the sign rule pay at 5h and 1d?\n\nDaily net returns; DSR deflates the best cell by N = 24."),
            e_result["cells"].sort("sharpe_annual", descending=True),
            mo.md(
                f"Best cell **{e_result['best']}**: DSR **{e_result['dsr']:.3f}**, PBO **{e_result['pbo']:.3f}**, "
                f"Sharpe above hold and inverse vol: **{e_result['beats_hold_and_inverse_vol']}**. "
                f"**E supported: {e_result['supported']}.**"
            ),
        ]
    )
    return (e_result,)


@app.cell
def _(b_cells, direction, mo, pl):
    b_judged = b_cells.with_columns(pl.Series("holm_p", direction.holm(b_cells["dm_p"].to_list()))).with_columns(
        ((pl.col("mean_difference") < 0) & (pl.col("holm_p") < 0.05)).alias("c1"),
        ((pl.col("mean_difference") < 0).all().over("horizon", "encoding") & (pl.col("first_half") < 0) & (pl.col("second_half") < 0)).alias("c3"),
    )
    b_horizons = (
        b_judged.with_columns((pl.col("c1") & pl.col("c3")).alias("met"))
        .group_by("horizon", "encoding")
        .agg((pl.col("met").sum() == pl.col("ticker").n_unique()).alias("both"))
        .group_by("horizon")
        .agg(pl.col("both").any().alias("supported"))
        .sort("horizon")
    )
    mo.vstack([mo.md("## (b), secondary: the three-class target at ±9.5 bps"), b_judged.sort("horizon", "encoding", "ticker"), b_horizons])
    return (b_horizons,)


@app.cell
def _(DRY, b_horizons, e_result, mo, pl, s_horizons):
    claims = s_horizons.rename({"supported": "S"}).join(b_horizons.rename({"supported": "(b)"}), on="horizon", how="left")
    claims = claims.with_columns(pl.when(pl.col("horizon").is_in(["5h", "1d"])).then(pl.lit(e_result["supported"])).alias("E"))
    mo.vstack(
        [
            mo.md("## What survives" + (" (DRY RUN on the estimation period: not the registered result)" if DRY else "")),
            claims,
            mo.md("Expected: S at 5m and 30m, mixed at 5h, not at 1d; E not supported. (b) is secondary and changes no verdict."),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
