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
    # The liquidity study: Under stress and ahead of time: extreme hours, tomorrow's liquidity, and when to work an order (⑮ ⑯ ⑰)

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
    ## ⑮ When it breaks

    Every profile above is a typical day; this is the atypical hour. The
    episodes are the ten BTC hours with the largest log range since 2023,
    three days apart, **picked by the data** (`gr.timeseries.extremes`). They
    are crashes and spikes alike.

    Two views, each against a baseline:

    - **Binance, every hour from −24 h to +48 h:** ±1% depth, dollar volume
      and |r|. The baseline is the same hour of day over the 7 days before.
    - **Bybit and λ, the episode day against the day before, hour for hour:**
      time-weighted spread, 2 bps depth, and Kyle's λ from both venues.

    **Summaries:**
    - the depth low point;
    - the first hour from which depth stays at or above 90% of its baseline
      for three hours;
    - the mean depth over the six hours before, to see whether the book
      thinned before the move.
    """)
    return


@app.cell
def _(EVER, gr, mo, pl):
    def _episodes():
        try:
            k = gr.reference.candles("BTC", "2023-01-01T00:00Z", EVER[1], venues="binance-um").collect()
            depth = (
                gr.reference.depth("BTC", "2023-01-01T00:00Z", EVER[1], venues="binance-um")
                .filter(pl.col("band_pct").abs() == 1.0)
                .group_by("ts").agg(pl.col("notional").sum())
                .group_by(pl.col("ts").dt.truncate("1h")).agg(pl.col("notional").median().alias("depth"))
                .collect()
            )  # fmt: skip
        except gr.Refused:
            return None, None
        hours = (
            k.group_by("ticker", pl.col("ts").dt.truncate("1h"))
            .agg(pl.col("open").first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").last(),
                 (pl.col("volume") * pl.col("close")).sum().alias("volume"), pl.len().alias("n"))
            .filter(pl.col("n") == 60).drop("n").sort("ts")
            .with_columns(pl.col("ts").dt.offset_by("1h").alias("close_ts"))
        )  # fmt: skip
        hours = hours.with_columns((1e4 * (pl.col("close") / pl.col("close").shift(1)).log()).abs().alias("abs_r")).join(depth, on="ts", how="left")
        picks = gr.timeseries.extremes(hours, 10, by="range", spacing="3d")
        return hours, picks

    crash_hours, crash_picks = _episodes()
    mo.stop(crash_hours is None, mo.md("No Binance bars or depth held."))
    mo.vstack([mo.md("### The ten episodes (BTC, Binance 1h, largest log range, three days apart)"),
               crash_picks.select("ts", (pl.col("score") * 1e4).round(0).alias("range bps"), (1e4 * (pl.col("close") / pl.col("open")).log()).round(0).alias("open→close bps"))])  # fmt: skip
    return crash_hours, crash_picks


@app.cell
def _(alt, crash_hours, crash_picks, mo, pl):
    _hod = pl.col("ts").dt.hour()
    _rows, _summary = [], []
    for _e in crash_picks["ts"].to_list():
        _day = _e.replace(hour=0)
        _base = (
            crash_hours.filter((pl.col("ts") >= _day - pl.duration(days=7)) & (pl.col("ts") < _day))
            .group_by(_hod.alias("hod")).agg(pl.col("depth").mean().alias("_d"), pl.col("volume").mean().alias("_v"), pl.col("abs_r").mean().alias("_r"))
        )  # fmt: skip
        _w = (
            crash_hours.filter(pl.col("ts").is_between(_e - pl.duration(hours=24), _e + pl.duration(hours=48)))
            .with_columns(_hod.alias("hod"), ((pl.col("ts") - _e).dt.total_hours()).alias("h"))
            .join(_base, on="hod", how="left")
            .select("h", (pl.col("depth") / pl.col("_d")).alias("depth"), (pl.col("volume") / pl.col("_v")).alias("volume"), (pl.col("abs_r") / pl.col("_r")).alias("|r|"))
            .with_columns(pl.lit(_e.strftime("%Y-%m-%d %H:00")).alias("episode"))
        )  # fmt: skip
        _rows.append(_w)
        _d = _w.sort("h").drop_nulls("depth")
        _after = _d.filter(pl.col("h") >= 0)
        _ok = (_after["depth"] >= 0.9).to_list()
        _back = next((int(_after["h"][i]) for i in range(len(_ok) - 2) if all(_ok[i : i + 3])), None)
        _low = _d.sort("depth").row(0, named=True) if _d.height else None
        _summary.append({
            "episode": _e.strftime("%Y-%m-%d %H:00"),
            "depth low ÷ base": None if _low is None else round(_low["depth"], 2), "at hour": None if _low is None else int(_low["h"]),
            "depth back ≥0.9 from hour": _back,
            "depth −6…−1 h ÷ base": round(_d.filter(pl.col("h").is_between(-6, -1))["depth"].mean(), 2) if _d.height else None,
            "volume peak ÷ base": round(_w["volume"].max(), 1), "|r| peak ÷ base": round(_w["|r|"].max(), 1),
        })  # fmt: skip
    crash_windows = pl.concat(_rows)
    crash_table = pl.DataFrame(_summary)
    _chart = (
        alt.Chart(crash_windows.filter(pl.col("depth").is_not_null()))
        .mark_line(opacity=0.6)
        .encode(x=alt.X("h:Q", title="hours from the episode"), y=alt.Y("depth:Q", title="±1% depth ÷ the week before (same hour)", scale=alt.Scale(type="log")),
                color="episode:N", tooltip=["episode:N", "h:Q", alt.Tooltip("depth:Q", format=".2f")])  # fmt: skip
        .properties(height=240, width="container")
    )
    _med = crash_table.select(
        pl.col("depth low ÷ base").median().alias("median low"), pl.col("depth back ≥0.9 from hour").median().alias("median hours to recover"),
        pl.col("depth −6…−1 h ÷ base").median().alias("median depth before"),
    )  # fmt: skip
    mo.vstack([mo.md("### BTC on Binance through each episode"), _chart, crash_table, mo.md("Medians across the ten (ten episodes, read as such):"), _med])
    return crash_table, crash_windows


@app.cell
def _(crash_picks, gr, mo, pl):
    import datetime as _cdt

    def _near():
        rows = []
        for e in crash_picks["ts"].to_list():
            day = e.date()
            for label, d in (("before", day - _cdt.timedelta(days=1)), ("episode", day)):
                w = (f"{d}T00:00Z", f"{d}T23:59:59.999999Z")
                row = {"episode": e.strftime("%Y-%m-%d %H:00"), "day": label}
                try:
                    book = gr.reference.book("BTC", *w).collect()
                except gr.Refused:
                    book = None
                if book is not None and book.height:
                    q = gr.liquidity.quoted(book)
                    row["spread bps (mean)"] = q["spread_bps"].mean()
                    row["2 bps depth (median, BTC)"] = ((q["ask_depth_2bps"] + q["bid_depth_2bps"]) / 2).median()
                lam = []
                for venue, kw in (("binance-um", {}), ("bybit-linear", {"rpi": False})):
                    try:
                        tr = gr.reference.trades("BTC", *w, venues=venue, **kw).select("venue", "ticker", "ts", "price", "size", "aggressor").collect()
                    except gr.Refused:
                        continue
                    if tr.height:
                        lam.append(gr.liquidity.kyle_lambda(gr.liquidity.flow(tr, "1m")).row(0, named=True)["lambda_bps_per_m"])
                        row[f"λ {venue}"] = lam[-1]
                rows.append(row)
        return pl.DataFrame(rows) if rows else None

    crash_near = _near()
    mo.stop(crash_near is None, mo.md("No near-touch data fetched for the episodes."))
    _wide = crash_near.pivot(on="day", index="episode", values=[c for c in crash_near.columns if c not in ("episode", "day")])
    _ratios = _wide.select("episode", *[
        (pl.col(f"{c}_episode") / pl.col(f"{c}_before")).round(2).alias(f"{c}: episode ÷ day before")
        for c in [c for c in crash_near.columns if c not in ("episode", "day")] if f"{c}_episode" in _wide.columns
    ])  # fmt: skip
    mo.vstack([mo.md("### The episode day against the day before: Bybit's book and λ on both venues"), _ratios])
    return (crash_near,)


@app.cell
def _(mo):
    mo.md(r"""
    ## ⑯ Tomorrow's liquidity

    ② is the average week, but a schedule needs the next hour today. That
    means a decomposition, as the intraday-volume literature models it
    (Bialkowski, Darolles and Le Fol 2008; Brownlees, Cipollini and Gallo 2011):
    - log value = an hour-of-day profile plus a weekday effect;
    - the deviation from it follows a HAR on its last hour, day and week
      (`gr.models.intraday`).

    It is forecast one hour ahead and walked forward from 2025-01-01, refitted
    every 30 days on everything before. Two benchmarks:
    - the **seasonal profile** alone, the average week;
    - **persistence**, next hour like this one.

    Scores are squared error in logs, with Diebold–Mariano against each
    benchmark.
    """)
    return


@app.cell
def _(EVER, HELD, gr, mo, pl, ticker):
    def _forecasts():
        try:
            from galata_research.models import intraday as _i
        except gr.Refused as e:
            return None, str(e)
        try:
            depth = (
                gr.reference.depth(ticker.value, "2023-01-01T00:00Z", EVER[1], venues="binance-um")
                .filter(pl.col("band_pct").abs() == 1.0)
                .group_by("ts").agg(pl.col("notional").sum())
                .group_by(pl.col("ts").dt.truncate("1h")).agg(pl.col("notional").median().alias("value"), pl.len().alias("n"))
                .filter(pl.col("n") >= 100).drop("n").collect()
            )  # fmt: skip
            volume = (
                gr.reference.candles(ticker.value, "2023-01-01T00:00Z", EVER[1], venues="binance-um")
                .group_by(pl.col("ts").dt.truncate("1h")).agg((pl.col("volume") * pl.col("close")).sum().alias("value"), pl.len().alias("n"))
                .filter((pl.col("n") == 60) & (pl.col("value") > 0)).drop("n").collect()
            )  # fmt: skip
        except gr.Refused as e:
            return None, str(e)
        out = []
        for name, series in (("±1% depth", depth), ("dollar volume", volume)):
            if series.height < 24 * 400:
                continue
            for m in _i.MODELS:
                out.append(_i.forecast(series, split="2025-01-01T00:00Z", model=m).with_columns(pl.lit(name).alias("series")))
        return (pl.concat(out) if out else None), None

    with mo.persistent_cache(name=f"intraday-forecast-v2-{ticker.value}-{HELD}"):
        liq_forecasts, _why = _forecasts()
    mo.stop(liq_forecasts is None, mo.md(f"No forecasts: {_why or 'too little history'}."))
    return (liq_forecasts,)


@app.cell
def _(liq_forecasts, mo, pl, ticker):
    from galata_research.models import intraday as _intraday

    _tables = []
    for (_name,), _f in liq_forecasts.group_by("series", maintain_order=True):
        for _bench in ("seasonal", "persistence"):
            _tables.append(_intraday.score(_f.drop("series"), benchmark=_bench).with_columns(pl.lit(_name).alias("series"), pl.lit(_bench).alias("against")))
    forecast_scores = pl.concat(_tables).select("series", "against", "model", "n", pl.col("mse").round(4), pl.col("r2_oos").round(3), pl.col("dm_stat").round(1), pl.col("dm_p"))
    _hod = (
        liq_forecasts.with_columns(pl.col("ts").dt.hour().alias("hod"), ((pl.col("forecast") - pl.col("actual")) ** 2).alias("se"))
        .group_by("series", "model", "hod").agg(pl.col("se").mean())
        .pivot(on="model", index=["series", "hod"], values="se")
        .with_columns((1 - pl.col("decomposition") / pl.col("persistence")).round(3).alias("R²oos vs persistence"))
        .sort("series", "hod")
    )  # fmt: skip
    mo.vstack([
        mo.md(f"### {ticker.value} on Binance: one hour ahead, out of sample from 2025-01-01"),
        forecast_scores.sort("series", "against", "mse"),
        mo.md("By UTC hour, the decomposition's gain over persistence:"),
        _hod.select("series", "hod", "R²oos vs persistence").pivot(on="series", index="hod", values="R²oos vs persistence"),
    ])  # fmt: skip
    return (forecast_scores,)


@app.cell
def _(mo):
    mo.md(r"""
    ## ⑰ When to work an order

    A day's buy of $10M or $50M, worked over the 24 UTC hours. Temporary
    impact varies with liquidity, so the risk-neutral optimum trades in
    proportion to it. TWAP is optimal only if liquidity is constant (Almgren
    and Chriss 2000; for volume, VWAP, arXiv 1408.6118).

    Each hour's order is cut into 60 one-minute children, costed on ⑩'s
    linear book against that hour's **realized** ±1% depth per side
    (`gr.liquidity.schedule_cost`). The model overstates BTC's near-touch cost
    (⑩), so the **savings** between plans are the finding, not the levels.

    **Four plans, every day from 2025-01-01:**
    - **uniform** (TWAP);
    - **profile**: ∝ the average depth by UTC hour over every day before;
    - **adaptive**: each hour, the rest of the order is re-planned. The next
      hour gets ⑯'s one-hour-ahead forecast; the later ones get the profile
      at the past week's level;
    - **hindsight**: ∝ the depth that happened, the unattainable floor.
    """)
    return


@app.cell
def _(HELD, crash_picks, gr, liq_forecasts, mo, pl, random, ticker):
    def _run():
        if ticker.value != "BTC":
            return None
        try:
            depth = (
                gr.reference.depth("BTC", "2023-01-01T00:00Z", "2100-01-01T00:00Z", venues="binance-um")
                .filter(pl.col("band_pct").abs() == 1.0).group_by("ts").agg(pl.col("notional").sum())
                .group_by(pl.col("ts").dt.truncate("1h")).agg((pl.col("notional").median() / 2).alias("depth"), pl.len().alias("n"))
                .filter(pl.col("n") >= 100).drop("n").sort("ts").collect()
            )  # fmt: skip
        except gr.Refused:
            return None
        fc = liq_forecasts.filter((pl.col("series") == "±1% depth") & (pl.col("model") == "decomposition")).select("ts", (pl.col("forecast").exp() / 2).alias("f"))
        d = depth.join(fc, on="ts", how="left").with_columns(pl.col("ts").dt.date().alias("day"), pl.col("ts").dt.hour().alias("hod"))
        byday = {k[0]: g.sort("ts") for k, g in d.group_by("day")}
        days = sorted(k for k, g in byday.items() if g.height == 24 and str(k) >= "2025-01-01")
        rows = []
        for day in days:
            g = byday[day]
            past = d.filter(pl.col("day") < day)
            prof = past.group_by("hod").agg(pl.col("depth").mean()).sort("hod")["depth"].to_list()
            week = past.tail(168)["depth"].mean()
            level = week / (sum(prof) / 24)
            actual, f = g["depth"].to_list(), g["f"].to_list()
            if any(v is None for v in f):
                continue
            for q in (10e6, 50e6):
                plans = {"uniform": [1.0] * 24, "profile": prof, "hindsight": actual}
                rem, adaptive = q, []
                for h in range(24):
                    later = sum(prof[k] * level for k in range(h + 1, 24))
                    take = rem * f[h] / (f[h] + later) if h < 23 else rem
                    adaptive.append(take)
                    rem -= take
                costs = {name: gr.liquidity.schedule_cost(pl.DataFrame({"ts": g["ts"], "notional": gr.liquidity.allocate(q, w)}), g.select("ts", "depth")) for name, w in plans.items()}
                costs["adaptive"] = gr.liquidity.schedule_cost(pl.DataFrame({"ts": g["ts"], "notional": adaptive}), g.select("ts", "depth"))
                for name, c in costs.items():
                    rows.append({"day": day, "order": f"${int(q / 1e6)}M", "plan": name, "cost_bps": c["cost_bps"], "over_depth": c["over_depth"]})
        return pl.DataFrame(rows) if rows else None

    with mo.persistent_cache(name=f"schedules-BTC-{HELD}"):
        plan_costs = _run()
    mo.stop(plan_costs is None, mo.md("The schedule study runs on BTC (choose it above)."))
    _wide = plan_costs.pivot(on="plan", index=["day", "order"], values="cost_bps")
    _rng = random.Random(20260928)

    def _band(xs, draws=2000):
        m = [sum(_rng.choice(xs) for _ in xs) / len(xs) for _ in range(draws)]
        m.sort()
        return m[int(0.05 * draws)], m[int(0.95 * draws)]

    _rows = []
    for (_order,), _g in _wide.group_by("order", maintain_order=True):
        for _p in ("profile", "adaptive", "hindsight"):
            _s = ((_g["uniform"] - _g[_p]) / _g["uniform"]).to_list()
            _lo, _hi = _band(_s)
            _rows.append({"order": _order, "plan": _p, "days": len(_s), "mean saving vs TWAP": round(sum(_s) / len(_s), 4), "90% band": f"{_lo:.4f} … {_hi:.4f}",
                          "mean cost bps": round(_g[_p].mean(), 4), "TWAP cost bps": round(_g["uniform"].mean(), 4)})  # fmt: skip
    schedule_table = pl.DataFrame(_rows)
    _hind = schedule_table.filter(pl.col("plan") == "hindsight").select("order", pl.col("mean saving vs TWAP").alias("_h"))
    schedule_table = schedule_table.join(_hind, on="order").with_columns((pl.col("mean saving vs TWAP") / pl.col("_h")).round(2).alias("share of hindsight saving")).drop("_h")
    _episodes = [t.date() for t in crash_picks["ts"].to_list()]
    _ep = _wide.filter(pl.col("day").is_in(_episodes))
    mo.vstack([
        mo.md("### BTC: what depth-aware scheduling saves against TWAP, out of sample from 2025-01-01"),
        schedule_table.sort("order", "plan"),
        mo.md(f"On the ⑮ episode days in this window ({_ep['day'].n_unique()}):"), _ep,
        mo.md(f"Hours where a one-minute child exceeded the hour's ±1% depth: {int(plan_costs['over_depth'].sum())}."),
    ])  # fmt: skip
    return plan_costs, schedule_table


if __name__ == "__main__":
    app.run()
