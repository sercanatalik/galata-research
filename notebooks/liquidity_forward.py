import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import datetime as dt

    import marimo as mo
    import polars as pl

    import galata_research as gr

    return dt, gr, mo, pl


@app.cell
def _(mo):
    mo.md(r"""
    # The liquidity study: seven forward claims

    Registered in `planning/preregistered/liquidity-forward.md`, committed
    alone (`98b1e53`) on 2026-09-28, before any of the data they are tested
    on existed. This notebook scores them exactly as registered:
    - only on data from **2026-09-29 00:00 UTC**;
    - with every profile, periodicity and forecast fitted only on data before;
    - *not yet decidable* until each claim's registered sample is complete.

    **Dry run** replays the same scorer from 2026-06-01. It shows the code
    works and what the verdicts would have been. It is **not a result**, and
    is never written into the registration.

    To keep the claims current, fetch new days as they arrive:
    - `galata-fetch depth binance-um BTC` and `galata-fetch candles binance-um BTC`, daily;
    - `galata-fetch events`, before each FOMC statement;
    - for F6, `galata-fetch trades binance-um BTC` and `galata-fetch trades bybit-linear BTC` on the same `--days`.
    """)
    return


@app.cell
def _(mo):
    mode = mo.ui.dropdown(["registered: from 2026-09-29", "dry run: from 2026-06-01 (not a result)"], value="registered: from 2026-09-29", label="score")
    mode
    return (mode,)


@app.cell
def _(dt, mode):
    DRY = mode.value.startswith("dry")
    SINCE = dt.datetime(2026, 6, 1, tzinfo=dt.UTC) if DRY else dt.datetime(2026, 9, 29, tzinfo=dt.UTC)
    LABEL = "dry run: not a result" if DRY else "registered"
    return DRY, LABEL, SINCE


@app.cell
def _(gr, pl):
    END = "2100-01-01T00:00Z"

    def depth_hours():
        return (
            gr.reference.depth("BTC", "2023-01-01T00:00Z", END, venues="binance-um")
            .filter(pl.col("band_pct").abs() == 1.0).group_by("ts").agg(pl.col("notional").sum())
            .group_by(pl.col("ts").dt.truncate("1h")).agg(pl.col("notional").median().alias("value"), pl.len().alias("n"))
            .filter(pl.col("n") >= 100).drop("n").sort("ts").collect()
        )  # fmt: skip

    def minute_bars():
        return gr.reference.candles("BTC", "2019-12-31T00:00Z", END, venues="binance-um").collect()

    def volume_hours(k):
        return (
            k.group_by(pl.col("ts").dt.truncate("1h")).agg((pl.col("volume") * pl.col("close")).sum().alias("value"), pl.len().alias("n"))
            .filter((pl.col("n") == 60) & (pl.col("value") > 0)).drop("n").sort("ts")
        )  # fmt: skip

    def rv_hours(k):
        return (
            gr.timeseries.returns(k, kind="log").group_by(pl.col("ts").dt.truncate("1h"))
            .agg((pl.col("return") ** 2).sum().alias("value"), pl.col("return").is_not_null().sum().alias("n"))
            .filter((pl.col("n") >= 58) & (pl.col("value") > 0)).drop("n").sort("ts")
        )  # fmt: skip

    def complete_days(h, since, first=None):
        """The first `first` UTC days from `since` holding all 24 hours, and how many there are."""
        d = h.filter(pl.col("ts") >= since).with_columns(pl.col("ts").dt.date().alias("day"))
        full = d.group_by("day").len().filter(pl.col("len") == 24)["day"].sort().to_list()
        if first is not None:
            full = full[:first]
        return d.filter(pl.col("day").is_in(full)), len(full)

    def best_hour(h):
        x = h.with_columns((pl.col("value") / pl.col("value").mean().over("day")).alias("x"))
        return x.group_by(pl.col("ts").dt.hour().alias("hod")).agg(pl.col("x").mean()).sort("x", descending=True)["hod"][0]

    return best_hour, complete_days, depth_hours, minute_bars, rv_hours, volume_hours


@app.cell
def _(LABEL, SINCE, best_hour, complete_days, depth_hours, dt, gr, minute_bars, mo, pl, rv_hours, volume_hours):
    from zoneinfo import ZoneInfo as _Zone

    rows = []

    def verdict(n, claim, held, needed, statistic, ok):
        decided = held >= needed
        rows.append({"#": n, "claim": claim, "held": held, "needed": needed, "statistic": statistic if decided else "—",
                     "verdict": ("supported" if ok else "not supported") if decided else "not yet decidable", "mode": LABEL})  # fmt: skip

    _depth = depth_hours()
    _k = minute_bars()
    _vol, _rv = volume_hours(_k), rv_hours(_k)

    # F1, F2: the day's shape on the first 60 complete days
    _d, _n = complete_days(_depth, SINCE, 60)
    _b = best_hour(_d) if _n >= 60 else None
    verdict("F1", "±1% depth deepest at 9–12 UTC", _n, 60, f"best hour {_b} UTC", _b in (9, 10, 11, 12))
    _v, _n = complete_days(_vol, SINCE, 60)
    _b = best_hour(_v) if _n >= 60 else None
    verdict("F2", "volume peaks at 13–16 UTC", _n, 60, f"best hour {_b} UTC", _b in (13, 14, 15, 16))

    # F3: the most volatile hour in US standard time
    _std = max(SINCE, dt.datetime(2026, 11, 1, tzinfo=dt.UTC))
    _r, _n = complete_days(_rv, _std, 60)
    _b = best_hour(_r) if _n >= 60 else None
    verdict("F3", "in US standard time the most volatile hour is 15 UTC", _n, 60, f"most volatile hour {_b} UTC", _b == 15)

    # F4: jumps at 08:30 New York, periodicity fitted only before SINCE
    _bars = (
        _k.group_by("ticker", pl.col("ts").dt.truncate("5m").alias("t5")).agg(pl.col("open").first(), pl.col("close").last(), pl.len().alias("n"))
        .filter(pl.col("n") == 5).rename({"t5": "ts"}).with_columns(pl.col("ts").dt.offset_by("5m").alias("close_ts")).sort("ts")
    )  # fmt: skip
    _ret = gr.timeseries.returns(_bars, kind="log")
    _f = gr.jumps.periodicity(_ret, slot="5m", fit=("2019-12-31T00:00Z", SINCE))
    _j = gr.jumps.lee_mykland(_ret, periodicity=_f).filter(pl.col("ts") >= SINCE)
    _days = sorted(_j["ts"].dt.date().unique().to_list())[:90]
    _j = _j.filter(pl.col("ts").dt.date().is_in(_days) & pl.col("jump"))
    _ny = pl.col("ts").dt.convert_time_zone("America/New_York")
    _top = _j.group_by((_ny.dt.hour().cast(pl.Int32) * 60 + _ny.dt.minute().cast(pl.Int32)).alias("m")).len().sort("len", descending=True)["m"].head(3).to_list()
    verdict("F4", "jumps cluster at 08:30 New York", len(_days), 90, f"busiest slots {[f'{m // 60:02d}:{m % 60:02d}' for m in _top]}", 510 in _top)

    # F5: the first three FOMC statements after SINCE, against ⑧'s baseline
    try:
        _ev = gr.reference.events(SINCE, "2100-01-01T00:00Z", sources="fomc").filter(pl.col("scheduled")).head(3)
        _closed = set(gr.calendar.closures("XNYS", SINCE, "2028-01-01T00:00Z")["date"].to_list())
    except gr.Refused:
        _ev, _closed = pl.DataFrame(), set()
    _absr = _ret.select("ts", pl.col("return").abs().alias("a"))
    _ratios = []
    for _e in _ev.iter_rows(named=True):
        _bin = _absr.filter(pl.col("ts") == _e["ts"])
        if _bin.is_empty():
            continue
        _local = _e["ts"].astimezone(_Zone("America/New_York"))
        _fomc_days = set(_ev["date"].to_list())
        _base = []
        for _off in range(-10, 11):
            _d2 = _e["date"] + dt.timedelta(days=_off)
            if _off == 0 or _d2.weekday() >= 5 or _d2 in _fomc_days or _d2 in _closed:
                continue
            _t = _local.replace(year=_d2.year, month=_d2.month, day=_d2.day).astimezone(dt.UTC)
            _v2 = _absr.filter(pl.col("ts") == _t)
            if _v2.height:
                _base.append(_v2["a"][0])
        if len(_base) >= 5:
            _ratios.append(_bin["a"][0] / (sum(_base) / len(_base)))
    verdict("F5", "an FOMC statement moves BTC (5m |r| ≥ 2× at two of three)", len(_ratios), 3,
            f"ratios {[round(r, 1) for r in _ratios]}", sum(r >= 2 for r in _ratios) >= 2)  # fmt: skip

    # F6: Binance before Bybit, per hour, on shared days after SINCE
    _shared = []
    try:
        _bdays = set(gr.reference.trades("BTC", SINCE, "2100-01-01T00:00Z", venues="binance-um").select(pl.col("ts").dt.date().unique()).collect().to_series().to_list())
        _ydays = set(gr.reference.trades("BTC", SINCE, "2100-01-01T00:00Z", venues="bybit-linear").select(pl.col("ts").dt.date().unique()).collect().to_series().to_list())
        _shared = sorted(_bdays & _ydays)
    except gr.Refused:
        pass
    _leads = []
    for _d3 in _shared:
        _w = (f"{_d3}T00:00Z", f"{_d3}T23:59:59.999999Z")
        _x = gr.reference.trades("BTC", *_w, venues="binance-um").select("ts", "price").collect()
        _y = gr.reference.trades("BTC", *_w, venues="bybit-linear", rpi=False).select("ts", "price").collect()
        if _x.height > 1000 and _y.height > 1000:
            _leads.extend(gr.leadlag.lead_lag(_x, _y, every="1h")["lead_ms"].to_list())
    _share = sum(v > 0 for v in _leads) / len(_leads) if _leads else None
    verdict("F6", "Binance leads Bybit in ≥ 85% of hours", len(_shared), 10, f"ahead in {_share:.1%} of {len(_leads)} hours" if _share is not None else "—",
            _share is not None and _share >= 0.85)  # fmt: skip

    # F7: the decomposition against persistence, the first 60 days of forecasts
    try:
        from galata_research.models import intraday as _i

        _res = []
        for _name, _series in (("depth", _depth), ("volume", _vol)):
            _fc = pl.concat([_i.forecast(_series, split=SINCE, model=m) for m in ("persistence", "decomposition")])
            _first = sorted(_fc["ts"].dt.date().unique().to_list())[:60]
            _fc = _fc.filter(pl.col("ts").dt.date().is_in(_first))
            _s = _i.score(_fc, benchmark="persistence").filter(pl.col("model") == "decomposition").row(0, named=True) if len(_first) >= 60 else None
            _res.append((_name, len(_first), _s))
        _held = min(n for _, n, _ in _res)
        _ok = all(s is not None and s["r2_oos"] > 0 and s["dm_p"] < 0.05 for _, _, s in _res)
        _stat = "; ".join(f"{nm} R²oos {s['r2_oos']:.3f}, DM p {s['dm_p']:.2g}" for nm, _, s in _res if s is not None)
    except gr.Refused as e:
        _held, _ok, _stat = 0, False, str(e)
    verdict("F7", "the decomposition beats persistence on depth and volume", _held, 60, _stat, _ok)

    forward = pl.DataFrame(rows)
    mo.vstack([
        mo.md(f"## The seven claims ({LABEL})"),
        mo.ui.table(forward, page_size=10, selection=None),
        mo.md(
            f"Supported **{forward.filter(pl.col('verdict') == 'supported').height}**, not supported "
            f"**{forward.filter(pl.col('verdict') == 'not supported').height}**, not yet decidable "
            f"**{forward.filter(pl.col('verdict') == 'not yet decidable').height}**, of seven."
        ),
    ])  # fmt: skip
    return (forward,)


if __name__ == "__main__":
    app.run()
