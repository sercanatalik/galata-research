import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import random
    from datetime import UTC, datetime

    import altair as alt
    import marimo as mo
    import polars as pl

    import galata_research as gr

    return UTC, alt, datetime, gr, mo, pl, random


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
    # The liquidity study: Across venues: who moves first, and whose price is the price (⑨ ⑬)

    Part of the intraday-liquidity study; `liquidity.py` is where it begins (① to ④). The sections keep their numbers from the study, which the README and the archived designs cite. Data: the reference store (`galata-fetch`) beside the record.
    """)
    return

@app.cell
def _(gr, mo):
    EVER = ("2019-01-01T00:00Z", "2100-01-01T00:00Z")
    _held = sorted(set(gr.reference.coverage()["ticker"]) | {"BTC"})
    ticker = mo.ui.dropdown(_held, value="BTC", label="ticker")
    ticker
    return EVER, ticker

@app.cell
def _(mo):
    mo.md(r"""
    ## ⑨ Who moves first?

    Two venues trade the same perp. When the price moves, which one moves
    first? This uses Hoffmann, Rosenbaum and Yoshida's (2013) lead-lag
    estimator:
    - the Hayashi–Yoshida correlation of the two venues' trade prices, each on
      its own clock, with one clock shifted by θ (`gr.leadlag`);
    - the lead is the θ where the correlation peaks;
    - Huth and Abergel's (2014) lead-lag ratio LLR = Σ_{θ>0} ρ² / Σ_{θ<0} ρ²
      summarises the whole curve. Above 1, the first venue leads.

    Resampling to a grid would not do: it pulls the correlation toward zero
    (the Epps effect) and loses the milliseconds a lead lives in.

    **Clocks.** Each venue stamps its own trades, and Hyperliquid's stamp is
    its block time. Legacy measured one venue 35 ms off our clock. So a lead
    of **50 ms or less is within clock skew**: shaded below, and never called a
    lead on its own.

    **Data.**
    - Binance aggTrades against Bybit trades, on the 195 days both are held
      since 2023 (every ninth day, and the book days).
    - OKX trades against both, on the same days, where both of OKX's files
      (16:00 to 16:00 UTC) cover the UTC day.
    - Hyperliquid's tape against both, on its two complete days, 2026-09-25 and 26.
    """)
    return


@app.cell
def _(HELD, gr, mo, pl, ticker):
    def _binance_bybit():
        from galata_research.reference import coverage as _coverage

        cover = _coverage().filter((pl.col("kind") == "trades") & (pl.col("ticker") == ticker.value))
        if cover.filter(pl.col("venue") == "binance-um").is_empty() or cover.filter(pl.col("venue") == "bybit-linear").is_empty():
            return None
        rows = []
        first = cover["first"].min()
        last = cover["last"].max()
        days = pl.date_range(first, last, eager=True).to_list()
        for d in days:
            w = (f"{d}T00:00Z", f"{d}T23:59:59.999999Z")
            bn = gr.reference.trades(ticker.value, *w, venues="binance-um").select("ts", "price").collect()
            if bn.is_empty():
                continue
            bb = gr.reference.trades(ticker.value, *w, venues="bybit-linear", rpi=False).select("ts", "price").collect()
            if bb.is_empty():
                continue
            rows.append(gr.leadlag.lead_lag(bn, bb, every="1h"))
        return pl.concat(rows) if rows else None

    with mo.persistent_cache(name=f"leadlag-binance-bybit-{ticker.value}-{HELD}"):
        pairs = _binance_bybit()
    mo.stop(pairs is None, mo.md("Binance and Bybit trades are not both held for this ticker."))
    return (pairs,)


@app.cell
def _(alt, mo, pairs, pl, ticker):
    _h = pairs.with_columns(pl.col("ts").dt.hour().alias("hod"), pl.col("ts").dt.year().alias("year"))
    _lead = pl.col("lead_ms")
    _by = lambda k: _h.group_by(k).agg(  # noqa: E731
        pl.len().alias("hours"),
        _lead.median().alias("median lead ms"),
        (_lead > 0).mean().round(3).alias("Binance ahead, share"),
        (_lead.abs() <= 50).mean().round(3).alias("within ±50 ms, share"),
        pl.col("llr").median().round(3).alias("median LLR"),
        pl.col("rho_0").median().round(3).alias("ρ at 0"),
        pl.col("rho_lead").median().round(3).alias("ρ at lead"),
    ).sort(k)
    overall = _h.select(
        pl.len().alias("hours"), pl.col("ts").dt.date().n_unique().alias("days"),
        _lead.median().alias("median lead ms"), (_lead > 0).mean().round(3).alias("Binance ahead, share"),
        (_lead.abs() <= 50).mean().round(3).alias("within ±50 ms, share"),
        pl.col("llr").median().round(3).alias("median LLR"),
    )  # fmt: skip
    by_hour, by_year = _by("hod"), _by("year")
    _chart = (
        alt.Chart(by_hour)
        .mark_line(point=True)
        .encode(x=alt.X("hod:O", title="hour, UTC"), y=alt.Y("median LLR:Q", scale=alt.Scale(zero=False)), tooltip=list(by_hour.columns))
        .properties(height=180, width="container")
    )
    mo.vstack([
        mo.md(f"### {ticker.value}: Binance against Bybit, hour by hour (LLR above 1: Binance moves first)"),
        overall, _chart, mo.md("By year:"), by_year,
    ])  # fmt: skip
    return by_hour, by_year, overall


@app.cell
def _(HELD, alt, gr, mo, pl, ticker):
    def _okx():
        # OKX's day runs 16:00 to 16:00 UTC: a UTC day counts only where both of its files reach its ends.
        try:
            okx_days = gr.reference.trades(ticker.value, "2021-11-01T00:00Z", "2100-01-01T00:00Z", venues="okx-swap")
            okx_days = okx_days.group_by(pl.col("ts").dt.date().alias("d")).agg(pl.col("ts").min().dt.hour().alias("a"), pl.col("ts").max().dt.hour().alias("b"))
            okx_days = okx_days.filter((pl.col("a") == 0) & (pl.col("b") == 23)).collect()["d"].sort().to_list()
        except gr.Refused:
            return None
        rows = []
        for d in okx_days:
            w = (f"{d}T00:00Z", f"{d}T23:59:59.999999Z")
            ok = gr.reference.trades(ticker.value, *w, venues="okx-swap", rpi=False).select("ts", "price").collect()
            for name, venue in (("binance", "binance-um"), ("bybit", "bybit-linear")):
                other = gr.reference.trades(ticker.value, *w, venues=venue, rpi=False).select("ts", "price").collect()
                if other.height > 1 and ok.height > 1:
                    rows.append(gr.leadlag.lead_lag(other, ok, every="1h").with_columns(pl.lit(f"{name} → okx").alias("pair")))
        return pl.concat(rows) if rows else None

    with mo.persistent_cache(name=f"leadlag-okx-{ticker.value}-{HELD}"):
        okx_pairs = _okx()
    mo.stop(okx_pairs is None, mo.md("OKX trades are not held for this ticker."))
    _h = okx_pairs.with_columns(pl.col("ts").dt.hour().alias("hod"))
    okx_overall = _h.group_by("pair").agg(
        pl.len().alias("hours"), pl.col("ts").dt.date().n_unique().alias("days"),
        pl.col("lead_ms").median().alias("median lead ms"), (pl.col("lead_ms") > 0).mean().round(3).alias("first ahead, share"),
        (pl.col("lead_ms").abs() <= 50).mean().round(3).alias("within ±50 ms, share"), pl.col("llr").median().round(3).alias("median LLR"),
    ).sort("pair")  # fmt: skip
    okx_by_hour = _h.group_by("pair", "hod").agg(pl.col("llr").median().alias("median LLR"), (pl.col("lead_ms") > 0).mean().alias("first ahead, share")).sort("pair", "hod")
    _chart = (
        alt.Chart(okx_by_hour).mark_line(point=True)
        .encode(x=alt.X("hod:O", title="hour, UTC"), y=alt.Y("median LLR:Q", scale=alt.Scale(zero=False)), color="pair:N", tooltip=list(okx_by_hour.columns))
        .properties(height=180, width="container")
    )  # fmt: skip
    mo.vstack([mo.md(f"### {ticker.value}: Binance and Bybit against OKX, hour by hour (LLR above 1: the first venue moves first)"), okx_overall, _chart])
    return okx_by_hour, okx_overall


@app.cell
def _(alt, gr, mo, pl, ticker):
    def _tape():
        w = ("2026-09-25T00:00Z", "2026-09-27T00:00Z")
        try:
            series = {
                "hyperliquid": gr.market.trades(ticker.value, *w).select("ts", "price").collect(),
                "binance": gr.reference.trades(ticker.value, *w, venues="binance-um").select("ts", "price").collect(),
                "bybit": gr.reference.trades(ticker.value, *w, venues="bybit-linear", rpi=False).select("ts", "price").collect(),
            }
        except gr.Refused:
            return None, None
        series = {k: v for k, v in series.items() if v.height > 1}
        curves, rows = [], []
        for a, b in (("binance", "bybit"), ("binance", "hyperliquid"), ("bybit", "hyperliquid")):
            if a in series and b in series:
                c = gr.leadlag.hayashi_yoshida(series[a], series[b]).with_columns(pl.lit(f"{a} → {b}").alias("pair"))
                curves.append(c)
                s = gr.leadlag.lead_lag(series[a], series[b]).row(0, named=True)
                rows.append({"pair": f"{a} → {b}", "lead ms": s["lead_ms"], "ρ at lead": round(s["rho_lead"], 3),
                             "ρ at 0": round(s["rho_0"], 3), "LLR": round(s["llr"], 3),
                             "reading": "within clock skew" if abs(s["lead_ms"]) <= 50 else ("first leads" if s["lead_ms"] > 0 else "second leads")})  # fmt: skip
        return (pl.concat(curves) if curves else None), (pl.DataFrame(rows) if rows else None)

    tape_curves, tape_pairs = _tape()
    mo.stop(tape_pairs is None, mo.md("The tape's days are not held on the other venues for this ticker."))
    _skew = alt.Chart(pl.DataFrame({"a": [-50], "b": [50]})).mark_rect(opacity=0.12, color="#888").encode(x="a:Q", x2="b:Q")
    _lines = alt.Chart(tape_curves.filter(pl.col("lag_ms").abs() <= 1000)).mark_line(point=True).encode(
        x=alt.X("lag_ms:Q", title="lag, ms (positive: the first venue leads)"), y=alt.Y("rho:Q", title="ρ(θ)"), color="pair:N",
        tooltip=["pair:N", "lag_ms:Q", alt.Tooltip("rho:Q", format=".3f")],
    )
    mo.vstack([
        mo.md(f"### {ticker.value}: Hyperliquid against Binance and Bybit, 2026-09-25 and 26 (grey: within clock skew)"),
        (_skew + _lines).properties(height=220, width="container"),
        tape_pairs,
    ])  # fmt: skip
    return (tape_pairs,)


@app.cell
def _(mo):
    mo.md(r"""
    ## ⑬ Whose price is the price?

    ⑨ says who moves first. The price-discovery shares say how much of the
    common, efficient price each venue carries, from a VECM on both venues'
    prices (`gr.models.discovery`, the `[models]` extra):

    - **Component share** (Gonzalo and Granger 1995): which venue the other
      corrects toward.
    - **Information share** (Hasbrouck 1995): each venue's part of the
      efficient price's innovation variance, bounded because the ordering
      matters. The bounds narrow as sampling gets finer, so 100 ms is the
      primary reading and 1 s is shown beside it.
    - **Information leadership share** (Putniņš 2013): IS and CS both mix
      leadership with noise, and their ratio does not. Above 0.5, the first
      venue leads.

    The model is one per day, the log last trade price per step carried
    forward, 20 lags, and β = (1, −1).
    """)
    return


@app.cell
def _(EVER, HELD, gr, mo, pl, ticker):
    def _shares():
        try:
            from galata_research.models import discovery as _d
        except gr.Refused as e:
            return None, str(e)
        try:
            days = gr.reference.trades(ticker.value, *EVER, venues="binance-um").select(pl.col("ts").dt.date().unique()).collect().to_series().sort().to_list()
        except gr.Refused:
            return None, "no Binance trades held for this ticker"
        rows = []
        for day in days:
            w = (f"{day}T00:00Z", f"{day}T23:59:59.999999Z")
            series = {"binance": gr.reference.trades(ticker.value, *w, venues="binance-um").select("ts", "price").collect()}
            try:
                series["bybit"] = gr.reference.trades(ticker.value, *w, venues="bybit-linear", rpi=False).select("ts", "price").collect()
            except gr.Refused:
                pass
            try:
                series["hyperliquid"] = gr.market.trades(ticker.value, w[0], w[1]).select("ts", "price").collect()
            except gr.Refused:
                pass
            series = {k: v for k, v in series.items() if v.height > 1000}
            for a, b in (("binance", "bybit"), ("binance", "hyperliquid"), ("bybit", "hyperliquid")):
                if a not in series or b not in series:
                    continue
                for every in ("1s", "100ms"):
                    try:
                        s = _d.shares(_d.vecm(_d.grid(series[a], series[b], every), 20)).row(0, named=True)
                    except gr.Refused:
                        continue
                    rows.append({"day": day, "pair": f"{a} vs {b}", "every": every, **{k: s[k] for k in ("cs", "is_low", "is_mid", "is_high", "ils")}})
        return (pl.DataFrame(rows) if rows else None), None

    with mo.persistent_cache(name=f"discovery-{ticker.value}-{HELD}"):
        discovery_days, _why = _shares()
    mo.stop(discovery_days is None, mo.md(f"No price-discovery shares: {_why or 'no pair of venues on the same day'}."))
    return (discovery_days,)


@app.cell
def _(discovery_days, mo, pl, ticker):
    discovery_table = (
        discovery_days.group_by("pair", "every")
        .agg(
            pl.len().alias("days"),
            pl.col("cs").median().round(3).alias("CS, first venue"),
            pl.col("is_low").median().round(3).alias("IS low"),
            pl.col("is_high").median().round(3).alias("IS high"),
            pl.col("ils").median().round(3).alias("ILS, first venue"),
            (pl.col("ils") > 0.5).mean().round(3).alias("days first leads (ILS)"),
        )
        .sort("pair", "every")
    )
    _years = (
        discovery_days.filter((pl.col("pair") == "binance vs bybit") & (pl.col("every") == "100ms"))
        .group_by(pl.col("day").dt.year().alias("year"))
        .agg(pl.len().alias("days"), pl.col("cs").median().round(3).alias("CS"), pl.col("ils").median().round(3).alias("ILS"))
        .sort("year")
    )
    mo.vstack([
        mo.md(f"### {ticker.value}: medians over days (the first venue named leads when a share is above 0.5)"),
        discovery_table, mo.md("Binance against Bybit at 100 ms, by year:"), _years,
    ])  # fmt: skip
    return (discovery_table,)


if __name__ == "__main__":
    app.run()
