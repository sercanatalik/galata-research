"""gr.models.intraday: walk-forward hourly forecasts, on a simulated seasonal series with persistent shocks."""

import math
import random
from datetime import timedelta

import polars as pl
import pytest
from conftest import utc

pytest.importorskip("arch")
from galata_research import Refused  # noqa: E402
from galata_research.models import intraday  # noqa: E402

T0 = utc("2026-01-05T00:00")  # a Monday


def _series(days=200, phi=0.95, seed=1):
    rng = random.Random(seed)
    e, rows = 0.0, []
    for i in range(days * 24):
        e = phi * e + rng.gauss(0, 0.1)
        pattern = 0.5 * math.sin(2 * math.pi * (i % 24) / 24) + 0.2 * ((i // 24) % 7 >= 5)
        rows.append((T0 + timedelta(hours=i), math.exp(3 + pattern + e)))
    return pl.DataFrame(rows, schema={"ts": pl.Datetime("us", "UTC"), "value": pl.Float64}, orient="row")


def the_decomposition_wins_on_a_seasonal_ar_series():
    s = _series()
    split = (T0 + timedelta(days=100)).isoformat()
    f = pl.concat([intraday.forecast(s, split=split, model=m) for m in intraday.MODELS])
    got = intraday.score(f, benchmark="seasonal")
    assert got["model"][0] == "decomposition"
    mse = dict(zip(got["model"], got["mse"], strict=True))
    assert mse["decomposition"] < mse["persistence"] and mse["decomposition"] < mse["seasonal"]


def the_target_hour_is_never_an_input():
    s = _series(days=60)
    split = (T0 + timedelta(days=30)).isoformat()
    target = T0 + timedelta(days=40, hours=5)
    spiked = s.with_columns(pl.when(pl.col("ts") == target).then(1e9).otherwise(pl.col("value")).alias("value"))
    for m in intraday.MODELS:
        a = intraday.forecast(s, split=split, model=m, refit_every=24 * 20).filter(pl.col("ts") == target)["forecast"][0]
        b = intraday.forecast(spiked, split=split, model=m, refit_every=24 * 20).filter(pl.col("ts") == target)["forecast"][0]
        assert a == pytest.approx(b), m
    f = intraday.forecast(s, split=split, model="decomposition")
    assert (f["fitted_through"] < f["ts"]).all()


def the_benchmark_against_itself_scores_zero():
    s = _series(days=40)
    split = (T0 + timedelta(days=20)).isoformat()
    f = pl.concat([intraday.forecast(s, split=split, model=m) for m in ("seasonal", "persistence")])
    got = intraday.score(f, benchmark="seasonal").filter(pl.col("model") == "seasonal")
    assert got["r2_oos"][0] == pytest.approx(0)


def an_unknown_model_is_refused():
    with pytest.raises(Refused, match="seasonal, persistence, decomposition"):
        intraday.forecast(_series(days=20), split=(T0 + timedelta(days=10)).isoformat(), model="arima")


def a_missing_hour_is_left_out_of_the_score():
    s = _series(days=40)
    gone = T0 + timedelta(days=30, hours=3)
    holed = s.filter(pl.col("ts") != gone)
    split = (T0 + timedelta(days=20)).isoformat()
    f = pl.concat([intraday.forecast(holed, split=split, model=m) for m in intraday.MODELS])
    got = intraday.score(f, benchmark="seasonal")
    assert got["mse"].is_finite().all() and got["n"][0] < 20 * 24
