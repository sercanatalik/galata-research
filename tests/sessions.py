"""The local clock (gr.timeseries) and exchange sessions (gr.calendar, the calendars extra)."""

import subprocess
import sys
from datetime import date

import polars as pl
import pytest
from conftest import utc

import galata_research as gr
from galata_research import Refused, timeseries


def _at(*isos):
    return pl.DataFrame({"ts": [utc(i) for i in isos]})


# ---- the local clock ------------------------------------------------------------


def the_new_york_hour_either_side_of_dst():
    got = timeseries.local_clock(_at("2025-03-07T14:30", "2025-03-10T13:30"), "America/New_York", "ny")
    assert got["ny_hour"].to_list() == [9, 9]
    assert got["ny_dst"].to_list() == [False, True]
    assert got["ny_date"].to_list() == [date(2025, 3, 7), date(2025, 3, 10)]
    assert got["ny_weekday"].to_list() == [5, 1]


def a_mismatch_week_is_dst_in_new_york_only():
    frame = timeseries.local_clock(_at("2025-03-20T12:00"), "America/New_York", "ny")
    got = timeseries.local_clock(frame, "Europe/London", "ldn").row(0, named=True)
    assert got["ny_dst"] and not got["ldn_dst"]
    assert (got["ny_hour"], got["ldn_hour"]) == (8, 12)


def an_unknown_zone_is_refused():
    with pytest.raises(Refused, match="Mars/Olympus"):
        timeseries.local_clock(_at("2025-03-20T12:00"), "Mars/Olympus", "m")


def the_local_clock_leaves_ts_alone():
    frame = _at("2025-03-20T12:00")
    assert timeseries.local_clock(frame, "Asia/Tokyo", "tk")["ts"].equals(frame["ts"])


# ---- sessions and closures -----------------------------------------------------------


def no_session_on_good_friday_or_a_day_of_mourning():
    april = gr.calendar.sessions("XNYS", "2025-04-01T00:00Z", "2025-05-01T00:00Z")["date"].to_list()
    january = gr.calendar.sessions("XNYS", "2025-01-01T00:00Z", "2025-02-01T00:00Z")["date"].to_list()
    assert date(2025, 4, 18) not in april and date(2025, 4, 17) in april
    assert date(2025, 1, 9) not in january and date(2025, 1, 8) in january


def the_day_after_thanksgiving_closes_early():
    got = gr.calendar.sessions("XNYS", "2025-11-01T00:00Z", "2025-12-01T00:00Z").filter(pl.col("date") == date(2025, 11, 28)).row(0, named=True)
    assert got["early_close"] and got["close_ts"] == utc("2025-11-28T18:00")


def an_unknown_exchange_is_refused():
    with pytest.raises(Refused, match="'NOPE'"):
        gr.calendar.sessions("NOPE", "2025-01-01T00:00Z", "2025-02-01T00:00Z")


def the_closures_are_weekdays_it_did_not_open():
    got = gr.calendar.closures("XNYS", "2025-01-01T00:00Z", "2025-05-01T00:00Z")["date"].to_list()
    assert got == [date(2025, 1, 1), date(2025, 1, 9), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18)]


def a_trade_on_good_friday_is_on_a_closed_day():
    got = gr.calendar.mark_sessions(_at("2025-04-18T15:00"), "XNYS").row(0, named=True)
    assert not got["xnys_open"] and got["xnys_closed_day"]


def the_open_is_at_nine_thirty_new_york():
    got = gr.calendar.mark_sessions(_at("2025-07-01T13:29", "2025-07-01T13:30"), "XNYS")
    assert got["xnys_open"].to_list() == [False, True]
    assert got["xnys_closed_day"].to_list() == [False, False]


def the_calendar_returns_polars():
    assert isinstance(gr.calendar.sessions("XNYS", "2025-01-01T00:00Z", "2025-01-10T00:00Z"), pl.DataFrame)


def the_calendar_without_its_extra_is_refused():
    probe = (
        "import sys; sys.modules['exchange_calendars'] = None\n"
        "import galata_research as gr\n"
        "try:\n    gr.calendar\nexcept gr.Refused as e:\n    print(e)\n"
    )
    run = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert "calendars" in run.stdout and "uv sync --extra calendars" in run.stdout


def the_plain_weekend_reopening():
    got = gr.calendar.reopenings("COMEX", "2026-09-01T00:00Z", "2026-10-01T00:00Z").filter(pl.col("open_ts") == utc("2026-09-20T22:00")).row(0, named=True)
    assert got["kind"] == "weekend" and got["closed_from"] == utc("2026-09-18T22:00") and got["closed_hours"] == pytest.approx(48)


def the_holiday_monday_reopening():
    got = gr.calendar.reopenings("XNYS", "2026-09-01T00:00Z", "2026-09-15T00:00Z")
    labor = got.filter(pl.col("open_ts").dt.date() == date(2026, 9, 8)).row(0, named=True)
    assert labor["kind"] == "weekend+holiday" and labor["closed_hours"] == pytest.approx(89.5)
