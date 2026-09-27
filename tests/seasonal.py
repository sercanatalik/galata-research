from datetime import timedelta

import polars as pl
import pytest
from conftest import utc

from galata_research import Refused, timeseries

HOUR = timedelta(hours=1)
DAY = timedelta(days=1)
T0 = utc("2026-01-05T00:00")  # a Monday


def _hourly(values, *, start=T0):
    return pl.DataFrame(
        {
            "ticker": "BTC",
            "ts": [start + i * HOUR for i in range(len(values))],
            "close_ts": [start + (i + 1) * HOUR for i in range(len(values))],
            "return": [float(v) for v in values],
        }
    )


def _pattern(weeks, *, peak=2.0, hour=13):
    # ± alternating, so each hour's mean |r| is exact: `peak` at `hour`, 1 elsewhere.
    return [(peak if i % 24 == hour else 1.0) * (1 if (i // 24) % 2 else -1) for i in range(weeks * 168)]


FIT = ("2026-01-05T00:00Z", "2026-01-19T00:00Z")


def a_known_pattern_is_recovered():
    f = timeseries.seasonal_factors(_hourly(_pattern(2)), fit=FIT, by="hour_of_day")
    assert f.height == 168
    peak = f.filter(pl.col("hour") == 13)["factor"].unique().to_list()
    rest = f.filter(pl.col("hour") != 13)["factor"].unique().to_list()
    assert len(peak) == 1 and len(rest) == 1 and peak[0] == pytest.approx(2 * rest[0])
    assert (f["factor"] ** 2).mean() == pytest.approx(1.0)


def the_fit_window_is_all_that_is_read():
    # Guard: without the fit filter the tripled weeks after `end` would move every factor.
    two = _pattern(2)
    later = [100 * v for v in _pattern(2, peak=5.0, hour=3)]
    base = timeseries.seasonal_factors(_hourly(two), fit=FIT)
    grown = timeseries.seasonal_factors(_hourly(two + later), fit=FIT)
    assert grown["factor"].to_list() == pytest.approx(base["factor"].to_list())


def the_median_ignores_one_jump():
    values = [1.0 if i % 2 else -1.0 for i in range(3 * 168)]
    jumped = list(values)
    jumped[13] = 1000.0
    fit = ("2026-01-05T00:00Z", "2026-01-26T00:00Z")
    plain = timeseries.seasonal_factors(_hourly(values), fit=fit, by="hour_of_week", stat="median_abs")
    moved = timeseries.seasonal_factors(_hourly(jumped), fit=fit, by="hour_of_week", stat="median_abs")
    assert moved["factor"].to_list() == pytest.approx(plain["factor"].to_list())
    by_mean = timeseries.seasonal_factors(_hourly(jumped), fit=fit, by="hour_of_week")
    assert by_mean["factor"].to_list() != pytest.approx(plain["factor"].to_list())


def an_empty_slot_is_refused():
    # Monday 03:00 is missing from the two fitted weeks, present in the two after.
    frame = _hourly(_pattern(4)).filter(~((pl.col("ts") < utc("2026-01-19T00:00")) & (pl.col("ts").dt.weekday() == 1) & (pl.col("ts").dt.hour() == 3)))
    with pytest.raises(Refused, match="1 cell.*weekday 1 hour 3"):
        timeseries.seasonal_factors(frame, fit=FIT, by="hour_of_week")


def the_hour_x_weekday_factor_is_a_product():
    values = [(3.0 if (i // 24) % 7 == 5 else 1.0) * (2.0 if i % 24 == 13 else 1.0) * (1 if i % 2 else -1) for i in range(2 * 168)]
    f = timeseries.seasonal_factors(_hourly(values), fit=FIT)
    cell = {(r["weekday"], r["hour"]): r["factor"] for r in f.iter_rows(named=True)}
    assert cell[(6, 13)] / cell[(1, 0)] == pytest.approx(6.0)
    assert cell[(6, 0)] / cell[(1, 0)] == pytest.approx(3.0)


def an_unknown_layout_is_refused():
    with pytest.raises(Refused, match="by='minute'"):
        timeseries.seasonal_factors(_hourly(_pattern(1)), fit=FIT, by="minute")


def a_return_is_divided_by_its_slots_factor():
    r = _hourly(_pattern(2))
    f = timeseries.seasonal_factors(r, fit=FIT, by="hour_of_day")
    got = timeseries.deseasonalize(r, f)
    row = got.filter(pl.col("ts") == T0 + 13 * HOUR)
    assert row["deseasonalized"].item() == pytest.approx(row["return"].item() / row["factor"].item())


def a_ticker_without_factors_is_refused():
    r = _hourly(_pattern(1))
    f = timeseries.seasonal_factors(r, fit=FIT).with_columns(pl.lit("ETH").alias("ticker"))
    with pytest.raises(Refused, match="no factor"):
        timeseries.deseasonalize(r, f)


def _days(n):
    return pl.DataFrame({"ticker": "BTC", "ts": [T0 + i * DAY for i in range(n)], "close_ts": [T0 + (i + 1) * DAY for i in range(n)]})


def the_refit_every_three_schedule():
    bars = _days(10)
    got = timeseries.walk_forward_origins(bars, T0 + 5 * DAY, every=3)  # the close of bar 4
    assert got.height == 6
    assert got["refit"].to_list() == [True, False, False, True, False, False]
    assert got["fitted_through"].to_list()[:3] == [T0 + 5 * DAY] * 3
    assert got["fitted_through"].to_list()[3] == T0 + 8 * DAY


def the_rolling_and_expanding_windows():
    bars = _days(10)
    rolling = timeseries.walk_forward_origins(bars, T0 + 5 * DAY, window=3)
    assert rolling["fit_from"].to_list() == [T0 + (i + 2) * DAY for i in range(6)]
    expanding = timeseries.walk_forward_origins(bars, T0 + 5 * DAY)
    assert set(expanding["fit_from"]) == {T0}


def no_origin_sees_its_future():
    got = timeseries.walk_forward_origins(_days(40), T0 + 10 * DAY, window=5, every=7)
    assert (got["fitted_through"] <= got["close_ts"]).all()


def a_window_longer_than_the_history_is_refused():
    with pytest.raises(Refused, match="window=50.*only 4"):
        timeseries.walk_forward_origins(_days(10), T0 + 4 * DAY, window=50)
