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
def _(mo):
    mo.md(r"""
    # The liquidity study: Whose clock the market keeps: New York or UTC, where it jumps, what happens at a release, and the weekend reopen (⑥ ⑦ ⑧ ⑫)

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
    ## ⑥ Whose clock?

    Crypto never closes, so its day could keep UTC, or it could keep New
    York's. US daylight saving moves the New York open from 14:30 to 13:30 UTC
    in March and back in November. A market that follows New York moves with
    it; one that follows UTC does not.

    **Published, and tested here** (CryptoSlate, Kraken XBT/USD hourly data
    2016–2025):

    1. BTC's most volatile hour was **14:00 UTC in US daylight time and 15:00
       UTC in standard time** in 2022–25, and did not move in 2016–18.
    2. On weekday US market holidays, the US-hours share of daily variance
       fell from **55.7% to 41.9%**, toward what a flat day would give.

    The data here are Binance 1m bars since 2019-12-31: volume in dollars,
    and realized variance (Σ r² of 1m log returns), per complete hour, each
    day normalised by its own mean. NYSE's sessions and closures come from
    `gr.calendar` (the `calendars` extra: `uv sync --extra calendars`).
    """)
    return


@app.cell
def _(EVER, gr, mo, pl, ticker):
    def _hours():
        try:
            k = gr.reference.candles(ticker.value, *EVER).collect()
        except gr.Refused:
            return None
        r = gr.timeseries.returns(k, kind="log")
        h = (
            k.join(r.select("ticker", "ts", "return"), on=["ticker", "ts"], how="left")
            .group_by(pl.col("ts").dt.truncate("1h"))
            .agg(
                (pl.col("volume") * pl.col("close")).sum().alias("volume"),
                (pl.col("return") ** 2).sum().alias("variance"),
                pl.col("return").is_not_null().sum().alias("n"),
            )
            # An hour counts when at least 58 of its 60 returns are there.
            .filter(pl.col("n") >= 58)
            .sort("ts")
        )
        h = gr.timeseries.local_clock(h, "America/New_York", "ny")
        h = gr.timeseries.local_clock(h, "Europe/London", "ldn")
        h = h.with_columns(pl.col("ts").dt.date().alias("day"), pl.col("ts").dt.hour().alias("utc_hour"), pl.col("ts").dt.year().alias("year"))
        h = h.with_columns(pl.len().over("day").alias("_k")).filter(pl.col("_k") == 24).drop("_k")
        return h.with_columns(*[(pl.col(m) / pl.col(m).mean().over("day")).alias(f"{m} ÷ day") for m in ("volume", "variance")])

    clock_hours = _hours()
    mo.stop(clock_hours is None, mo.md("No Binance 1m bars held for this ticker."))
    return (clock_hours,)


@app.cell
def _(alt, clock_hours, mo, pl, ticker):
    _rows = []
    for _y in sorted(clock_hours["year"].unique()):
        _hy = clock_hours.filter(pl.col("year") == _y)
        for _m in ("volume ÷ day", "variance ÷ day"):
            _res = {"year": _y, "measure": _m}
            for _clk, _label in (("utc_hour", "ρ on UTC"), ("ny_hour", "ρ on New York")):
                _p = _hy.group_by(_clk, "ny_dst").agg(pl.col(_m).mean()).pivot(on="ny_dst", index=_clk, values=_m).drop_nulls()
                _res[_label] = round(_p.select(pl.corr("true", "false", method="spearman")).item(), 3) if _p.width == 3 and _p.height == 24 else None
            _pk = _hy.group_by("ny_dst", "utc_hour").agg(pl.col(_m).mean()).sort(_m, descending=True).group_by("ny_dst", maintain_order=True).first()
            for _row in _pk.iter_rows(named=True):
                _res["peak UTC, US " + ("daylight" if _row["ny_dst"] else "standard")] = _row["utc_hour"]
            # The variance-weighted centre of the afternoon, 12–20 UTC, away from midnight's wrap.
            for _dst in (True, False):
                _c = _hy.filter(pl.col("ny_dst") == _dst, pl.col("utc_hour").is_between(12, 20)).group_by("utc_hour").agg(pl.col(_m).mean())
                _res["centre, " + ("daylight" if _dst else "standard")] = (
                    round((_c["utc_hour"] * _c[_m]).sum() / _c[_m].sum(), 2) if _c.height else None
                )
            _rows.append(_res)
    lineup = pl.DataFrame(_rows)
    _prof = (
        clock_hours.filter(pl.col("year") >= 2022)
        .unpivot(index=["ny_dst", "utc_hour", "ny_hour"], on=["variance ÷ day"], value_name="x")
        .unpivot(index=["ny_dst", "x"], on=["utc_hour", "ny_hour"], variable_name="clock", value_name="hour")
        .group_by("clock", "ny_dst", "hour")
        .agg(pl.col("x").mean())
        .with_columns(pl.when(pl.col("ny_dst")).then(pl.lit("US daylight")).otherwise(pl.lit("US standard")).alias("regime"))
    )
    _chart = (
        alt.Chart(_prof)
        .mark_line(point=True)
        .encode(
            x=alt.X("hour:O"),
            y=alt.Y("x:Q", title="variance ÷ day mean"),
            color="regime:N",
            tooltip=["clock:N", "regime:N", "hour:O", alt.Tooltip("x:Q", format=".2f")],
        )
        .properties(height=200, width=320)
        .facet(facet=alt.Facet("clock:N", title=None, header=alt.Header(labelExpr="datum.value == 'utc_hour' ? 'UTC clock' : 'New York clock'")), columns=2)
    )
    mo.vstack([
        mo.md(f"### {ticker.value}: US daylight and standard time, on the UTC clock and the New York clock (2022 on)"),
        _chart,
        mo.md(
            "Per year: the Spearman ρ between the daylight and standard-time 24-hour profiles on each clock "
            "(the higher one is the clock the day keeps), the peak hour in each regime, and the variance-weighted "
            "centre of 12–20 UTC."
        ),
        lineup,
    ])
    return (lineup,)


@app.cell
def _(clock_hours, gr, mo, pl, random, ticker):
    def _shares():
        try:
            closed = set(gr.calendar.closures("XNYS", "2019-12-31T00:00Z", "2026-12-31T00:00Z")["date"].to_list())
            early = set(gr.calendar.sessions("XNYS", "2019-12-31T00:00Z", "2026-12-31T00:00Z").filter(pl.col("early_close"))["date"].to_list())
        except gr.Refused as e:
            return None, str(e)
        # US hours: New York 10:00–15:59, the six whole hours inside NYSE's 09:30–16:00; a flat day gives them 25%.
        d = clock_hours.with_columns(pl.col("ny_hour").is_between(10, 15).alias("us"))
        share = (
            d.group_by("ny_date")
            .agg(pl.col("ny_weekday").first(), pl.len().alias("hours"),
                 *[(pl.col(m).filter(pl.col("us")).sum() / pl.col(m).sum()).alias(m) for m in ("volume", "variance")])  # fmt: skip
            .filter((pl.col("ny_weekday") <= 5) & (pl.col("hours") >= 20) & ~pl.col("ny_date").is_in(list(early)))
            .with_columns(pl.col("ny_date").is_in(list(closed)).alias("NYSE closed"), pl.col("ny_date").dt.year().alias("year"))
        )
        return share, None

    share, why = _shares()
    mo.stop(share is None, mo.md(f"Holidays need the calendar: {why}"))

    def _gap(frame, m, draws=2000, seed=20260927):
        # Mean share on NYSE-closed weekdays minus on open weekdays, with a 90% band from resampling days.
        on, off = frame.filter(pl.col("NYSE closed"))[m].to_list(), frame.filter(~pl.col("NYSE closed"))[m].to_list()
        rng = random.Random(seed)
        diffs = sorted(
            sum(rng.choice(on) for _ in on) / len(on) - sum(rng.choice(off) for _ in off) / len(off) for _ in range(draws)
        )
        return sum(on) / len(on), sum(off) / len(off), diffs[int(0.05 * draws)], diffs[int(0.95 * draws)], len(on)

    _rows = []
    for _span, _f in (("all years", share), ("2022 on", share.filter(pl.col("year") >= 2022))):
        for _m in ("volume", "variance"):
            _c, _o, _lo, _hi, _n = _gap(_f, _m)
            _rows.append({"sample": _span, "measure": _m, "closed days": _n, "share, NYSE closed": round(_c, 3), "share, NYSE open": round(_o, 3),
                          "difference": round(_c - _o, 3), "90% band": f"{_lo:+.3f} … {_hi:+.3f}"})  # fmt: skip
    holidays = pl.DataFrame(_rows)
    mo.vstack([
        mo.md(
            f"### {ticker.value}: the US-hours share of the day, on weekdays NYSE was shut\n"
            "US hours here are New York 10:00–15:59, six whole hours, 25% of a flat day. The published study "
            "used 13:00–21:59 UTC (37.5%), so the levels differ; the drop is what is compared. Early-close days are left out."
        ),
        holidays,
    ])
    return (holidays,)


@app.cell
def _(EVER, gr, mo, pl, ticker):
    def _depth_clock():
        try:
            lf = gr.reference.depth(ticker.value, *EVER)
        except gr.Refused:
            return None
        hourly = (
            lf.filter(pl.col("band_pct").abs() == 1.0)
            .group_by("ts").agg(pl.col("notional").sum())
            .group_by(pl.col("ts").dt.truncate("1h")).agg(pl.col("notional").median(), pl.len().alias("n"))
            .filter(pl.col("n") >= 100)
            .collect()
        )  # fmt: skip
        hourly = gr.timeseries.local_clock(hourly, "America/New_York", "ny")
        hourly = gr.timeseries.local_clock(hourly, "Europe/London", "ldn")
        hourly = hourly.with_columns(pl.col("ts").dt.date().alias("day"), pl.col("ts").dt.hour().alias("utc_hour"))
        hourly = hourly.with_columns((pl.col("notional") / pl.col("notional").mean().over("day")).alias("x"))
        # London's summer and winter, as New York's DST split above: the clock on which the two line up is the one depth keeps.
        _rows = []
        for clk, label in (("utc_hour", "UTC"), ("ldn_hour", "London"), ("ny_hour", "New York")):
            prof = hourly.group_by(clk, "ldn_dst").agg(pl.col("x").mean())
            wide = prof.pivot(on="ldn_dst", index=clk, values="x").drop_nulls().sort(clk)
            rho = wide.select(pl.corr("true", "false", method="spearman")).item() if wide.height == 24 else None
            best = {d: prof.filter(pl.col("ldn_dst") == d).sort("x", descending=True)[clk][0] for d in (True, False)}
            _rows.append({"clock": label, "ρ, London summer vs winter": None if rho is None else round(rho, 3),
                          "best hour, summer": best[True], "best hour, winter": best[False]})  # fmt: skip
        return pl.DataFrame(_rows)

    depth_clock = _depth_clock()
    mo.vstack([
        mo.md(
            f"### {ticker.value}: whose clock does Binance ±1% depth keep?\n"
            "London's summer and winter profiles compared on each clock, as New York's were above. If depth keeps "
            "London's morning, the two line up best on the London clock and its best hour holds still there. "
            "Depth's top is a plateau, 09–13 UTC within about half a percent on BTC, so a best hour moving by one is noise; the ρ is the test."
        ),
        depth_clock if depth_clock is not None else mo.md("No depth held for this ticker."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## ⑦ When does the price jump?

    A jump is a return too large for the volatility around it. Lee and
    Mykland (2008) divide each return by a bipower scale made of the returns
    **before** it. A Gumbel threshold then sets the chance of any false jump
    in a day at α = 1%: about 25 false jumps over the whole sample, whatever
    it holds.

    The day's own shape must come out first. Otherwise every busy hour looks
    jumpy, and the map below would redraw ②. So each return is divided by
    Boudt, Croux and Laurent's (2011) robust factor for its 5-minute slot of
    the week. That factor is a weighted standard deviation that the jumps
    themselves cannot inflate (`gr.jumps`).

    **Returns.** 5m log returns from Binance 1m bars since 2019-12-31, complete
    bars only, null across a hole. A 1m book is too thin to test at 1m:
    catch-up moves after quiet minutes look like jumps
    (`legacy/galata-legacy/design/measured.md:1592-1616`). The periodicity is
    fitted on the whole sample. It describes the sample; it is not a forecast.

    **Expected** (Wątorek et al. 2023; Ben Omrane et al. 2023; Saef et al. 2024; Yang and Wang 2026):
    - jumps at US macro releases (08:30 New York) and FOMC (14:00 New York);
    - mostly negative;
    - more on ETH than on BTC.
    """)
    return


@app.cell
def _(EVER, gr, mo, pl, ticker):
    def _jumps():
        try:
            k = gr.reference.candles(ticker.value, *EVER).collect()
        except gr.Refused:
            return None
        bars = (
            k.group_by("ticker", pl.col("ts").dt.truncate("5m").alias("t5"))
            .agg(pl.col("open").first(), pl.col("close").last(), pl.len().alias("n"))
            .filter(pl.col("n") == 5)
            .rename({"t5": "ts"})
            .with_columns(pl.col("ts").dt.offset_by("5m").alias("close_ts"))
            .sort("ts")
        )
        r = gr.timeseries.returns(bars, kind="log")
        f = gr.jumps.periodicity(r, slot="5m", by="time_of_week")
        raw = gr.jumps.lee_mykland(r).select("ts", "return", "L", pl.col("jump").alias("raw jump"))
        adj = gr.jumps.lee_mykland(r, periodicity=f).select("ts", pl.col("L").alias("L adjusted"), pl.col("jump").alias("jump"))
        out = raw.join(adj, on="ts")
        ny = pl.col("ts").dt.convert_time_zone("America/New_York")
        return out.with_columns(
            (ny.dt.hour().cast(pl.Int32) * 60 + ny.dt.minute().cast(pl.Int32)).alias("ny_minute"),
            (pl.col("ts").dt.hour().cast(pl.Int32) * 60 + pl.col("ts").dt.minute().cast(pl.Int32)).alias("utc_minute"),
            pl.col("ts").dt.weekday().alias("weekday"),
            pl.col("ts").dt.hour().alias("hod"),
        )

    tested = _jumps()
    mo.stop(tested is None, mo.md("No Binance 1m bars held for this ticker."))
    return (tested,)


@app.cell
def _(alt, gr, mo, pl, tested, ticker):
    _days = tested["ts"].dt.date().n_unique()
    _summary = pl.DataFrame({
        "returns": [tested.height], "days": [_days],
        "jumps, raw": [int(tested["raw jump"].sum())], "jumps, adjusted": [int(tested["jump"].sum())],
        "false jumps expected (α = 1% a day)": [round(0.01 * _days, 1)],
        "negative share": [round(tested.filter(pl.col("jump"))["return"].lt(0).mean(), 3)],
        "jump share of variance": [round((tested.filter(pl.col("jump"))["return"] ** 2).sum() / (tested["return"] ** 2).sum(), 3)],
    })  # fmt: skip
    _week = (
        tested.group_by("weekday", "hod")
        .agg(pl.col("raw jump").sum().alias("raw"), pl.col("jump").sum().alias("adjusted"))
        .unpivot(index=["weekday", "hod"], variable_name="test", value_name="jumps")
        .with_columns((pl.col("jumps") / pl.col("jumps").mean().over("test")).alias("÷ uniform"))
    )
    _DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    _heat = (
        alt.Chart(_week.with_columns(pl.col("weekday").map_elements(lambda d: _DAYS[d - 1], return_dtype=pl.String).alias("day")))
        .mark_rect()
        .encode(
            x=alt.X("hod:O", title="hour, UTC"),
            y=alt.Y("day:N", sort=_DAYS, title=None),
            color=alt.Color("÷ uniform:Q", scale=alt.Scale(scheme="blueorange", domainMid=1)),
            row=alt.Row("test:N", title=None),
            tooltip=["test:N", "day:N", "hod:O", "jumps:Q", alt.Tooltip("÷ uniform:Q", format=".2f")],
        )
        .properties(height=140, width="container")
    )
    mo.vstack([
        mo.md(f"### {ticker.value}: jumps by hour of the week, before and after the day's shape is taken out"),
        _summary,
        _heat,
        mo.md("Orange cells hold more jumps than a uniform week would. **Raw**, the busy hours light up because they are busy. **Adjusted**, what is left is where returns were large *for their slot*."),
    ])  # fmt: skip
    return


@app.cell
def _(alt, mo, pl, tested, ticker):
    def _by(col, marks, label):
        counts = (
            tested.filter(pl.col("jump"))
            .group_by(col)
            .len()
            .rename({col: "minute", "len": "jumps"})
            .with_columns((pl.col("minute") / 60).alias("hour"))
        )
        bars = alt.Chart(counts).mark_bar(width=2).encode(x=alt.X("hour:Q", title=label, scale=alt.Scale(domain=[0, 24])), y="jumps:Q", tooltip=["minute:Q", "jumps:Q"])
        rules = alt.Chart(pl.DataFrame({"hour": [m / 60 for m, _ in marks], "what": [w for _, w in marks]})).mark_rule(color="#c0392b", strokeDash=[3, 3]).encode(x="hour:Q", tooltip=["what:N"])
        top = counts.sort("jumps", descending=True).head(8).with_columns(pl.format("{}:{}", (pl.col("minute") // 60).cast(pl.String).str.zfill(2), (pl.col("minute") % 60).cast(pl.String).str.zfill(2)).alias("slot")).select("slot", "jumps")
        return (bars + rules).properties(height=180, width="container"), top

    _ny, _ny_top = _by("ny_minute", [(8 * 60 + 30, "US data 08:30"), (9 * 60 + 30, "open 09:30"), (14 * 60, "FOMC 14:00"), (16 * 60, "close 16:00")], "New York time")
    _utc, _utc_top = _by("utc_minute", [(0, "funding 00:00"), (8 * 60, "funding 08:00"), (16 * 60, "funding 16:00")], "UTC")
    mo.vstack([
        mo.md(f"### {ticker.value}: adjusted jumps by 5-minute slot, on New York's clock and on UTC"),
        _ny, mo.md("Busiest New York slots:"), _ny_top,
        _utc, mo.md("Busiest UTC slots (Binance funding settles at 00, 08 and 16 UTC):"), _utc_top,
    ])  # fmt: skip
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## ⑧ Around the release

    ⑦ found the busiest jump slots at 08:30 and 14:00 New York. Is that the
    release days, or the same clock time every day? Here each scheduled event
    is compared with **the same clock time on matched days without it**.

    **Fetched events** (`galata-fetch events` → `gr.reference.events`):
    - CPI and the jobs report at 08:30 New York (BLS);
    - scheduled FOMC statements at 14:00 New York (federalreserve.gov).

    **Rule events,** stated rather than fetched:
    - the NYSE open (from `gr.calendar`), against 09:30 New York on NYSE-closed weekdays;
    - Binance funding at 00/08/16 UTC, against the same minute 1–3 hours either side, since it happens every day;
    - Deribit's Friday 08:00 UTC expiry, against 08:00 UTC on other weekdays.

    **Baseline** for a fetched event on day D: the same New York time on
    weekdays within ±10 days that hold no event of that kind and are not NYSE
    closures.

    **Measures** per 5-minute bin from −60 to +120 minutes:
    - mean |r| of 5m log returns;
    - dollar volume;
    - Binance ±1% depth (2023 on);
    - the share of bins holding a ⑦ jump.

    Each is shown as the mean at the event over the mean at the baseline, with
    a 90% band from resampling events.
    """)
    return


@app.cell
def _(EVER, gr, mo, pl, tested, ticker):
    import datetime as _dtm
    from zoneinfo import ZoneInfo as _Zone

    _td = _dtm.timedelta

    def _bars():
        k = gr.reference.candles(ticker.value, *EVER).collect()
        vol = k.group_by(pl.col("ts").dt.truncate("5m")).agg((pl.col("volume") * pl.col("close")).sum().alias("volume"), pl.len().alias("n")).filter(pl.col("n") == 5)
        out = tested.select("ts", pl.col("return").abs().alias("abs_r"), pl.col("jump").cast(pl.Float64).alias("jump")).join(vol.drop("n"), on="ts")
        try:
            depth = (
                gr.reference.depth(ticker.value, *EVER).filter(pl.col("band_pct").abs() == 1.0)
                .group_by("ts").agg(pl.col("notional").sum())
                .group_by(pl.col("ts").dt.truncate("5m")).agg(pl.col("notional").median().alias("depth"))
                .collect()
            )  # fmt: skip
            out = out.join(depth, on="ts", how="left")
        except gr.Refused:
            out = out.with_columns(pl.lit(None, pl.Float64).alias("depth"))
        return out

    def _anchors(first, last):
        """(kind, event id, role, anchor) rows: each event instant and its baseline instants."""
        ny = "America/New_York"
        rows = []
        try:
            fetched = gr.reference.events(first, last).filter(pl.col("scheduled"))
        except gr.Refused as e:
            return None, str(e)
        try:
            closed = set(gr.calendar.closures("XNYS", first, last)["date"].to_list())
            opens = gr.calendar.sessions("XNYS", first, last).filter(~pl.col("early_close"))
        except gr.Refused as e:
            return None, str(e)
        for kind in ("cpi", "jobs", "fomc"):
            ev = fetched.filter(pl.col("event") == kind)
            days = set(ev["date"].to_list())
            for n, r in enumerate(ev.iter_rows(named=True)):
                local = r["ts"].astimezone(_Zone(ny))
                rows.append((kind, n, "event", r["ts"]))
                for off in range(-10, 11):
                    d = r["date"] + _td(days=off)
                    if off == 0 or d.weekday() >= 5 or d in days or d in closed:
                        continue
                    base = local.replace(year=d.year, month=d.month, day=d.day)
                    rows.append((kind, n, "base", base.astimezone(_dtm.UTC)))
        for n, r in enumerate(opens.iter_rows(named=True)):
            rows.append(("nyse open", n, "event", r["open_ts"]))
        for n, d in enumerate(sorted(closed)):
            t = _dtm.datetime(d.year, d.month, d.day, 9, 30, tzinfo=_Zone(ny))
            rows.append(("nyse open", 10_000 + n, "base", t.astimezone(_dtm.UTC)))
        return rows, None

    fivemin = _bars()
    _first, _last = fivemin["ts"].min(), fivemin["ts"].max()
    anchors, _why = _anchors(_first.isoformat(), _last.isoformat())
    mo.stop(anchors is None, mo.md(f"The event study needs the calendar: {_why}"))
    return anchors, fivemin


@app.cell
def _(anchors, fivemin, pl):
    import datetime as _dt

    def _rule_anchors(first, last):
        rows = []
        day = first.date()
        n = 0
        while day <= last.date():
            midnight = _dt.datetime(day.year, day.month, day.day, tzinfo=_dt.UTC)
            for h in (0, 8, 16):
                t = midnight + _dt.timedelta(hours=h)
                rows.append(("binance funding", n, "event", t))
                for off in (-3, -2, 2, 3):  # the same minute, away from the neighbouring settlements
                    rows.append(("binance funding", n, "base", t + _dt.timedelta(hours=off)))
                n += 1
            t = midnight + _dt.timedelta(hours=8)
            # Friday 08:00 UTC is Deribit's expiry; 08:00 on the other weekdays is its baseline.
            if day.weekday() < 5:
                rows.append(("deribit expiry", n, "event" if day.weekday() == 4 else "base", t))
            n += 1
            day += _dt.timedelta(days=1)
        return rows

    _all = anchors + _rule_anchors(fivemin["ts"].min(), fivemin["ts"].max())
    table = pl.DataFrame(_all, schema={"kind": pl.String, "id": pl.Int64, "role": pl.String, "anchor": pl.Datetime("us", "UTC")}, orient="row")
    bins = pl.DataFrame({"bin": list(range(-12, 24))})
    windows = (
        table.join(bins, how="cross")
        .with_columns((pl.col("anchor") + pl.duration(minutes=5) * pl.col("bin")).alias("ts"))
        .join(fivemin, on="ts", how="inner")
    )
    return (windows,)


@app.cell
def _(alt, mo, pl, random, ticker, windows):
    _ms = ("abs_r", "volume", "depth", "jump")

    def _ratio(frame, kind, draws=1000, seed=20260927):
        # Per event: its values by bin; the baseline: the mean over all baseline instants by bin.
        ev = frame.filter((pl.col("kind") == kind) & (pl.col("role") == "event"))
        base = frame.filter((pl.col("kind") == kind) & (pl.col("role") == "base")).group_by("bin").agg(*[pl.col(m).mean() for m in _ms])
        per = ev.group_by("id", "bin").agg(*[pl.col(m).mean() for m in _ms])
        ids = per["id"].unique().to_list()
        point = per.group_by("bin").agg(*[pl.col(m).mean() for m in _ms]).join(base, on="bin", suffix="_base")
        point = point.with_columns(*[(pl.col(m) / pl.col(f"{m}_base")).alias(f"{m} ratio") for m in _ms])
        # Bands for |r|, resampling events.
        wide = per.pivot(on="bin", index="id", values="abs_r")
        cols = [c for c in wide.columns if c != "id"]
        rows = wide.select(cols).rows()
        rng = random.Random(seed)
        draws_by_bin = {c: [] for c in cols}
        for _ in range(draws):
            pick = [rows[rng.randrange(len(rows))] for _ in rows]
            for j, c in enumerate(cols):
                vals = [p[j] for p in pick if p[j] is not None]
                draws_by_bin[c].append(sum(vals) / len(vals) if vals else None)
        band = pl.DataFrame({
            "bin": [int(c) for c in cols],
            "lo": [sorted(v for v in draws_by_bin[c] if v is not None)[int(0.05 * draws)] for c in cols],
            "hi": [sorted(v for v in draws_by_bin[c] if v is not None)[int(0.95 * draws) - 1] for c in cols],
        }).join(base.select("bin", "abs_r"), on="bin").with_columns((pl.col("lo") / pl.col("abs_r")).alias("lo"), (pl.col("hi") / pl.col("abs_r")).alias("hi")).drop("abs_r")
        return point.join(band, on="bin").with_columns(pl.lit(kind).alias("kind"), (pl.col("bin") * 5).alias("minutes")), len(ids)

    _kinds = ["cpi", "jobs", "fomc", "nyse open", "binance funding", "deribit expiry"]
    _parts, _counts = [], {}
    for _k in _kinds:
        _r, _counts[_k] = _ratio(windows, _k)
        _parts.append(_r)
    around = pl.concat(_parts, how="diagonal_relaxed")
    _long = around.unpivot(index=["kind", "minutes", "lo", "hi"], on=[f"{m} ratio" for m in _ms], variable_name="measure", value_name="ratio")
    _base = alt.Chart(_long)
    # The 90% band is for |r| alone; one data source so the layers can be faceted together.
    _band = _base.transform_filter(alt.datum.measure == "abs_r ratio").mark_area(opacity=0.2, color="#4c78a8").encode(x="minutes:Q", y="lo:Q", y2="hi:Q")
    _lines = _base.mark_line().encode(
        x=alt.X("minutes:Q", title="minutes from the event"),
        y=alt.Y("ratio:Q", title="event ÷ baseline", scale=alt.Scale(type="log")),
        color=alt.Color("measure:N", sort=[f"{m} ratio" for m in _ms]),
        tooltip=["kind:N", "minutes:Q", "measure:N", alt.Tooltip("ratio:Q", format=".2f")],
    )
    _chart = alt.layer(_band, _lines).properties(height=170, width=300).facet(facet=alt.Facet("kind:N", title=None, sort=_kinds), columns=2)

    def _summary(k):
        a = around.filter(pl.col("kind") == k).sort("minutes")
        peak = a.sort("abs_r ratio", descending=True).row(0, named=True)
        after = a.filter((pl.col("minutes") > peak["minutes"]) & (pl.col("abs_r ratio") < 1.5))
        pre = a.filter(pl.col("minutes").is_between(-30, -5))
        return {
            "event": k, "events": _counts[k], "peak |r| ÷ base": round(peak["abs_r ratio"], 2), "at minute": peak["minutes"],
            "volume ÷ base at peak": round(peak["volume ratio"], 2),
            "jump rate at peak": round(peak["jump"], 3), "jump rate, base": round(peak["jump_base"], 4),
            "depth ÷ base, −30 … −5": None if pre["depth ratio"].drop_nulls().is_empty() else round(pre["depth ratio"].mean(), 3),
            "back under 1.5× by minute": None if after.is_empty() else after["minutes"][0],
        }  # fmt: skip

    event_table = pl.DataFrame([_summary(k) for k in _kinds])
    mo.vstack([mo.md(f"### {ticker.value}: around each scheduled event, against the same clock time without it"), event_table, _chart])
    return (event_table,)


@app.cell
def _(mo):
    mo.md(r"""
    ## ⑫ When the underlying sleeps

    Three of the tape's perps (GOLD, CL and XYZ100 on the `xyz` dex) trade
    every hour, on underlyings that close for the weekend and for holidays.
    trade[XYZ] takes external prices Sunday 18:00 → Friday 17:00 New York.
    Outside them, the perp moves inside **discovery bounds** of ±(1/max
    leverage) around a reference that re-anchors a set number of times
    (docs.trade.xyz, "Discovery Bounds").

    Binance's XAUUSDT is a second gold perp that never closes. BTC and ETH,
    which have no underlying session, are run through the same lens as a null.

    **Sessions** come from `gr.calendar`: COMEX for gold, NYMEX for oil, CME
    equity for XYZ100 (and for the controls). The upstream calendar has no
    daily break and closes Fridays an hour after CME's real 17:00 New York
    close; that is stated here, not patched.

    **The question for the reopen:** is the weekend's price a forecast, or
    noise? Take W, the move from the last hour before the closure to the hour
    before the reopen, and R, the move in the first 1 and 4 hours after it.
    - a slope of R on W below 0: the weekend overshot, and the reopen undoes part of it;
    - near 0: the weekend price was as good as the market's;
    - above 0: the weekend under-reacted.
    """)
    return


@app.cell
def _(EVER, gr, mo, pl):
    import datetime as _clock
    import math as _m

    _UNDERLYING = {"GOLD": "COMEX", "CL": "NYMEX", "XYZ100": "CMES", "BTC": "CMES", "ETH": "CMES"}

    def _hours(source, t):
        if source == "tape":
            return gr.market.candles(t, "1h", *EVER).select("ts", "close", (pl.col("volume") * pl.col("close")).alias("volume"), "trade_count").collect()
        k = gr.reference.candles(t, *EVER, venues="binance-um").collect()
        return (
            k.group_by(pl.col("ts").dt.truncate("1h"))
            .agg(pl.col("close").last(), (pl.col("volume") * pl.col("close")).sum().alias("volume"), pl.col("trade_count").sum(), pl.len().alias("n"))
            .filter(pl.col("n") == 60).drop("n").sort("ts")
        )  # fmt: skip

    def _states(h, exchange):
        first, last = h["ts"].min().isoformat(), h["ts"].max().isoformat()
        ses = gr.calendar.sessions(exchange, first, last).sort("open_ts")
        reo = gr.calendar.reopenings(exchange, first, last).sort("open_ts")
        h = h.sort("ts").join_asof(ses.select("open_ts", "close_ts"), left_on="ts", right_on="open_ts", strategy="backward")
        h = h.join_asof(reo.select(pl.col("open_ts").alias("next_open"), "kind"), left_on="ts", right_on="next_open", strategy="forward")
        is_open = (pl.col("ts") >= pl.col("open_ts")) & (pl.col("ts") < pl.col("close_ts"))
        return h.with_columns(
            pl.when(pl.col("ts") == pl.col("next_open")).then(pl.lit("reopen"))
            .when(is_open).then(pl.lit("open"))
            .when(pl.col("kind").str.contains("weekend")).then(pl.lit("weekend"))
            .otherwise(pl.lit("holiday")).alias("state"),
            (1e4 * (pl.col("close") / pl.col("close").shift(1)).log()).alias("r_bps"),
        ).drop("open_ts", "close_ts")  # fmt: skip

    def _all():
        out, gaps = [], []
        for source, t in (("tape", "GOLD"), ("tape", "CL"), ("tape", "XYZ100"), ("binance", "GOLD"), ("tape", "BTC"), ("tape", "ETH")):
            try:
                h = _hours(source, t)
            except gr.Refused:
                continue
            if h.height < 100:
                continue
            ex = _UNDERLYING[t]
            h = _states(h, ex)
            name = f"{t} ({source})"
            out.append(h.with_columns(pl.lit(name).alias("series")))
            reo = gr.calendar.reopenings(ex, h["ts"].min().isoformat(), h["ts"].max().isoformat())
            close = dict(zip(h["ts"].to_list(), h["close"].to_list(), strict=True))
            for r in reo.iter_rows(named=True):
                hour = _clock.timedelta(hours=1)
                before, pre = close.get(r["closed_from"] - hour), close.get(r["open_ts"] - hour)
                after1, after4 = close.get(r["open_ts"]), close.get(r["open_ts"] + 3 * hour)
                if None in (before, pre, after1, after4):
                    continue
                gaps.append({"series": name, "kind": r["kind"], "open_ts": r["open_ts"], "W_bps": 1e4 * _m.log(pre / before),
                             "R1_bps": 1e4 * _m.log(after1 / pre), "R4_bps": 1e4 * _m.log(after4 / pre)})  # fmt: skip
        return (pl.concat(out) if out else None), (pl.DataFrame(gaps) if gaps else None)

    sleep_hours, reopen_moves = _all()
    mo.stop(sleep_hours is None, mo.md("No hourly bars held for the xyz tickers."))
    return reopen_moves, sleep_hours


@app.cell
def _(mo, pl, random, reopen_moves, sleep_hours):
    base = sleep_hours.filter(pl.col("state") == "open").group_by("series").agg(
        pl.col("volume").mean().alias("_v"), pl.col("trade_count").mean().alias("_n"), pl.col("r_bps").abs().mean().alias("_r")
    )
    by_state = (
        sleep_hours.group_by("series", "state")
        .agg(pl.len().alias("hours"), pl.col("volume").mean().alias("v"), pl.col("trade_count").mean().alias("n"), pl.col("r_bps").abs().mean().alias("r"))
        .join(base, on="series")
        .select("series", "state", "hours",
                (pl.col("v") / pl.col("_v")).round(2).alias("volume ÷ open"),
                (pl.col("n") / pl.col("_n")).round(2).alias("trades ÷ open"),
                (pl.col("r") / pl.col("_r")).round(2).alias("|r| ÷ open"))
        .sort("series", "state")
    )  # fmt: skip

    def _slope(frame, _y, draws=2000, seed=20260927):
        pts = list(zip(frame["W_bps"].to_list(), frame[_y].to_list(), strict=True))
        if len(pts) < 5:
            return None, None, None, None
        def fit(p):
            mx, my = sum(a for a, _ in p) / len(p), sum(_b for _, _b in p) / len(p)
            sxx = sum((a - mx) ** 2 for a, _ in p)
            return sum((a - mx) * (_b - my) for a, _b in p) / sxx if sxx else None
        rng = random.Random(seed)
        boots = sorted(v for v in (fit([rng.choice(pts) for _ in pts]) for _ in range(draws)) if v is not None)
        corr = frame.select(pl.corr("W_bps", _y)).item()
        return fit(pts), boots[int(0.05 * len(boots))], boots[int(0.95 * len(boots)) - 1], corr

    _rows = []
    if reopen_moves is not None:
        for (_series,), _g in reopen_moves.filter(pl.col("kind") == "weekend").group_by("series"):
            for _y in ("R1_bps", "R4_bps"):
                _b, _lo, _hi, _c = _slope(_g, _y)
                _rows.append({"series": _series, "weekends": _g.height, "reopen move": _y.replace("_bps", ""),
                             "mean |W| bps": round(_g["W_bps"].abs().mean(), 1), "mean |R| bps": round(_g[_y].abs().mean(), 1),
                             "slope R on W": None if _b is None else round(_b, 3),
                             "90% band": None if _b is None else f"{_lo:+.2f} … {_hi:+.2f}",
                             "corr": None if _c is None else round(_c, 3)})  # fmt: skip
    reopen_table = pl.DataFrame(_rows).sort("series", "reopen move") if _rows else None
    mo.vstack([
        mo.md("### Liquidity and volatility by the underlying's state (per hour, against the hours it is open)"),
        by_state,
        mo.md("### The weekend's move against the reopen's (weekends only; BTC and ETH are the null)"),
        reopen_table if reopen_table is not None else mo.md("Too few weekends held."),
    ])  # fmt: skip
    return by_state, reopen_table


if __name__ == "__main__":
    app.run()
