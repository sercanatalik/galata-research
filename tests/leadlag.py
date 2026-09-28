"""gr.leadlag: the shifted Hayashi–Yoshida correlation, against its double sum and planted leads."""

import math
import random
from datetime import timedelta

import polars as pl
import pytest
from conftest import utc

from galata_research import Refused, leadlag

T0 = utc("2026-09-25T00:00")


def _path(seconds=3600, step_ms=10, seed=1):
    """A Brownian log-price path on a fine grid: (micros, log price)."""
    rng = random.Random(seed)
    p, out = math.log(100.0), []
    for i in range(seconds * 1000 // step_ms):
        p += rng.gauss(0, 1e-5)
        out.append((i * step_ms * 1000, p))
    return out


def _sample(path, n, seed, delay_ms=0):
    """n random observation times; the price seen is the path's value delay_ms earlier."""
    rng = random.Random(seed)
    times = sorted(rng.sample(range(delay_ms * 1000 + 1, path[-1][0]), n))
    step = path[1][0]
    rows = [(T0 + timedelta(microseconds=tm), math.exp(path[max(0, (tm - delay_ms * 1000) // step)][1])) for tm in times]
    return pl.DataFrame(rows, schema={"ts": pl.Datetime("us", "UTC"), "price": pl.Float64}, orient="row")


def _double_sum(x, y, lag_ms):
    t, px = x["ts"].dt.epoch("us").to_list(), [math.log(v) for v in x["price"]]
    s, py = y["ts"].dt.epoch("us").to_list(), [math.log(v) for v in y["price"]]
    total = 0.0
    for k in range(1, len(t)):
        for j in range(1, len(s)):
            a, b = s[j - 1] - lag_ms * 1000, s[j] - lag_ms * 1000
            if t[k] > a and t[k - 1] < b:
                total += (px[k] - px[k - 1]) * (py[j] - py[j - 1])
    return total


def the_telescoped_sum_equals_the_double_sum():
    path = _path(seconds=60, step_ms=5)
    x, y = _sample(path, 300, seed=2), _sample(path, 300, seed=3)
    got = leadlag.hayashi_yoshida(x, y, [-50, 0, 50])
    for lag, hy in zip(got["lag_ms"], got["hy"], strict=True):
        assert hy == pytest.approx(_double_sum(x, y, int(lag)), abs=1e-12)


def a_copy_with_a_delay_peaks_at_the_delay():
    path = _path()
    x, y = _sample(path, 20_000, seed=4), _sample(path, 20_000, seed=5, delay_ms=200)
    got = leadlag.hayashi_yoshida(x, y)
    assert got.sort("rho", descending=True)["lag_ms"][0] == 200


def the_identical_series_correlate_fully():
    x = _sample(_path(seconds=60), 500, seed=6)
    assert leadlag.hayashi_yoshida(x, x, [0])["rho"][0] == pytest.approx(1.0)


def a_series_of_one_tick_is_refused():
    one = pl.DataFrame({"ts": [T0], "price": [100.0]})
    with pytest.raises(Refused, match="1 distinct timestamp"):
        leadlag.hayashi_yoshida(one, one, [0])


def the_leader_is_x():
    path = _path()
    x, y = _sample(path, 20_000, seed=7), _sample(path, 20_000, seed=8, delay_ms=200)
    got = leadlag.lead_lag(x, y).row(0, named=True)
    assert got["lead_ms"] == 200 and got["llr"] > 1
    back = leadlag.lead_lag(y, x).row(0, named=True)
    assert back["lead_ms"] == -200 and back["llr"] < 1


def every_bucket_finds_its_own_leader():
    path = _path(seconds=7200)
    x = _sample(path, 40_000, seed=9)
    ahead = _sample(path, 40_000, seed=10, delay_ms=200)
    first = ahead.filter(pl.col("ts") < T0 + timedelta(hours=1))
    # In the second hour y leads: x is the path 200 ms later, so y sees it 200 ms earlier.
    lagged_x = _sample(path, 40_000, seed=11, delay_ms=200).filter(pl.col("ts") >= T0 + timedelta(hours=1))
    x2 = pl.concat([x.filter(pl.col("ts") < T0 + timedelta(hours=1)), lagged_x])
    y2 = pl.concat([first, _sample(path, 40_000, seed=12).filter(pl.col("ts") >= T0 + timedelta(hours=1))])
    got = leadlag.lead_lag(x2, y2, every="1h")
    assert got["lead_ms"].to_list() == [200, -200]
