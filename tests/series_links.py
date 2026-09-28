"""timeseries: the extreme bars, and how two hourly series move together."""

from datetime import timedelta

import polars as pl
import pytest
from conftest import utc

from galata_research import Refused, timeseries


def _bars(rows):
    """(day, high, low, close) per bar, 1h wide."""
    t0 = utc("2026-01-01T00:00")
    return pl.DataFrame([
        {"ticker": "BTC", "ts": t0 + timedelta(days=d), "close_ts": t0 + timedelta(days=d, hours=1), "high": h, "low": low, "close": c}
        for d, h, low, c in rows
    ])  # fmt: skip


def a_second_extreme_too_close_is_skipped():
    import math

    bars = _bars([(0, math.exp(10), 1, 1), (1, math.exp(9), 1, 1), (5, math.exp(8), 1, 1), (9, math.exp(1), 1, 1)])
    got = timeseries.extremes(bars, 2, spacing="3d")
    assert [t.day for t in got["ts"]] == [1, 6]
    assert got["score"].to_list() == pytest.approx([10, 8])


def the_extremes_by_return():
    t0 = utc("2026-01-01T00:00")
    bars = pl.DataFrame({"ticker": "BTC", "ts": [t0 + timedelta(hours=i) for i in range(4)], "close_ts": [t0 + timedelta(hours=i + 1) for i in range(4)],
                         "high": 1.0, "low": 1.0, "close": [100.0, 101.0, 90.0, 91.0]})  # fmt: skip
    got = timeseries.extremes(bars, 1, by="return", spacing="1h")
    assert got["ts"][0] == t0 + timedelta(hours=2)


def an_unknown_score_is_refused():
    with pytest.raises(Refused, match="range, return"):
        timeseries.extremes(_bars([(0, 2, 1, 1)]), 1, by="volume")


# ---- elasticity and lagged correlation --------------------------------------------------------


def _hourly(days=120, seed=2, lead=None):
    import math
    import random

    rng = random.Random(seed)
    t0 = utc("2026-01-05T00:00")
    rows, xs = [], []
    for i in range(days * 24):
        pattern_x = 0.8 * math.sin(2 * math.pi * (i % 24) / 24)
        lx = pattern_x + rng.gauss(0, 0.3)
        xs.append(lx)
        if lead is None:
            ly = -0.5 * lx + 0.6 * math.sin(2 * math.pi * (i % 24) / 24) + rng.gauss(0, 0.1)  # the same shape as x: unadjusted, it biases the slope
        else:
            ly = (xs[i - lead] if i >= lead else 0.0) + rng.gauss(0, 0.1)
        rows.append({"ts": t0 + timedelta(hours=i), "x": math.exp(lx), "y": math.exp(ly)})
    return pl.DataFrame(rows)


def the_known_elasticity_needs_the_adjustment():
    f = _hourly()
    assert timeseries.log_elasticity(f, "y", "x")["beta"] == pytest.approx(-0.5, abs=0.05)
    assert abs(timeseries.log_elasticity(f, "y", "x", seasonal=False)["beta"] + 0.5) > 0.05


def the_first_leads_at_a_positive_lag():
    f = _hourly(lead=2)
    got = timeseries.lagged_correlation(f, "x", "y", list(range(-4, 5)))
    assert got.sort("corr", descending=True)["lag"][0] == 2


def no_non_positive_value_is_accepted():
    f = pl.DataFrame({"ts": [utc("2026-01-05T00:00"), utc("2026-01-05T01:00")], "x": [1.0, 0.0], "y": [1.0, 1.0]})
    with pytest.raises(Refused, match="must be positive"):
        timeseries.log_elasticity(f, "y", "x")
