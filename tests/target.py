from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from conftest import utc

import galata_research as gr

vol = gr.models.vol
DAY = timedelta(days=1)
T0 = utc("2020-01-01T00:00")


def _bars(closes):
    n = len(closes)
    return pl.DataFrame(
        {"ticker": "BTC", "ts": [T0 + i * DAY for i in range(n)], "close_ts": [T0 + (i + 1) * DAY for i in range(n)], "close": [float(c) for c in closes]}
    )


def _forecasts(bars, first, sigmas_annual, model="m"):
    rows = [
        {"model": model, "close_ts": bars["close_ts"][first + i], "h": 1, "variance": (s**2) / 365}
        for i, s in enumerate(sigmas_annual)
    ]
    return pl.DataFrame(rows)


def _walk(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return [100 * float(np.exp(x)) for x in np.cumsum(rng.normal(0, 0.02, n))]


def a_forecast_earns_the_next_bar_once():
    # Guard: joining on target_ts instead would shift twice and earn t+2.
    closes = [100.0] * 60 + [100.0, 102.0, 102.0]
    bars = _bars(closes)
    f = _forecasts(bars, 60, [0.4, 0.4, 0.4])  # origins at bars 60, 61, 62
    got = vol.target(f, bars, split=bars["close_ts"][60], target=0.2)
    assert got["position"][60] == pytest.approx(0.5)
    r = gr.backtest.returns(got, pl.col("position"), fee=0.0)
    assert r["gross"][61] == pytest.approx(0.5 * 0.02)


def the_target_is_known_at_the_split():
    closes = _walk()
    bars = _bars(closes)
    split = bars["close_ts"][149]
    f = _forecasts(bars, 149, [0.5] * 51)
    base = vol.target(f, bars, split=split)
    changed = _bars(closes[:150] + [c * (1.5 if i % 2 else 0.7) for i, c in enumerate(closes[150:])])
    moved = vol.target(f, changed, split=split)
    assert moved["position"].to_list() == pytest.approx(base["position"].to_list(), nan_ok=True)
    assert vol.target(f, bars, split=split)["position"][149] is not None


def the_band_holds_a_small_change():
    bars = _bars([100.0] * 70)
    f = _forecasts(bars, 60, [0.2, 0.2 / 1.2, 0.2 / 1.3])  # weights 1.0, 1.2, 1.3 at target 0.2
    got = vol.target(f, bars, split=bars["close_ts"][60], target=0.2, band=0.25)
    assert got["position"][60:63].to_list() == pytest.approx([1.0, 1.0, 1.3])


def the_conditional_rule_leaves_the_middle_alone():
    bars = _bars([100.0] * 100)
    sig = [0.3 + 0.01 * (i % 10) for i in range(30)] + [0.345]
    f = _forecasts(bars, 60, sig)
    got = vol.target(f, bars, split=bars["close_ts"][60], target=0.2, rule="conditional")
    assert got["position"][90] == 1.0
    extreme = _forecasts(bars, 60, sig[:-1] + [0.9])
    assert vol.target(extreme, bars, split=bars["close_ts"][60], target=0.2, rule="conditional")["position"][90] == pytest.approx(0.2 / 0.9)


def the_count_is_everything_run():
    closes = _walk(300, 1)
    bars = _bars(closes)
    split = bars["close_ts"][199]
    f = pl.concat([_forecasts(bars, 199, list(np.full(101, s)), model=m) for m, s in (("a", 0.3), ("b", 0.4), ("c", 0.5))])
    t = vol.trials(bars, f, split=split, rules=("inverse_vol", "conditional"), bands=(0.0, 0.25))
    assert t["trial"].n_unique() == 13
    _, deflated = vol.economics(t, periods_per_year=365)
    assert deflated["trials"] == 13
