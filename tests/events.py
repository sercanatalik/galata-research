"""The event calendar: FOMC and BLS pages parsed from excerpts of the real pages, written in the test."""

from datetime import date

import polars as pl
import pytest
from conftest import utc

import galata_research as gr
from galata_research import Refused
from galata_research.reference import _events, fetch


def _calendar(blocks: dict[int, list[tuple[str, str]]]) -> str:
    """An excerpt of fomccalendars.htm: one `<h4>` per year, one month/date pair per meeting."""
    out = []
    for i, (year, meetings) in enumerate(blocks.items()):
        out.append(f'<h4><a id="{40000 + i}">{year} FOMC Meetings</a></h4><div class="panel">')
        for month, days in meetings:
            out.append(
                f'<div class="row fomc-meeting"><div class="fomc-meeting__month col-xs-5"><strong>{month}</strong></div>'
                f'<div class="fomc-meeting__date col-xs-4">{days}</div></div>'
            )
        out.append("</div>")
    return "\n".join(out)


HISTORICAL_2020 = """
<h5 class="panel-heading">January 28-29 Meeting - 2020</h5>
<h5 class="panel-heading">March 2 (unscheduled) Meeting - 2020</h5>
<h5 class="panel-heading">March 17-18 (cancelled) Meeting - 2020</h5>
<h5 class="panel-heading">April 28-29 Meeting - 2020</h5>
"""

CPI_INDEX = """
<li><a href="/news.release/archives/cpi_07142026.htm">June 2026 Consumer Price Index</a>
<li><a href="/news.release/archives/cpi_01112019.htm">December 2018 Consumer Price Index</a>
"""


def the_meeting_across_two_months_is_dated_by_its_last_day():
    got = _events.fomc_calendar(_calendar({2024: [("Apr/May", "30-1")]})).row(0, named=True)
    assert got["date"] == date(2024, 5, 1) and got["ts"] == utc("2024-05-01T18:00")


def the_projections_votes_and_unscheduled_calls():
    cal = _events.fomc_calendar(_calendar({2025: [("March", "18-19*"), ("August", "22 (notation vote)")]}))
    assert cal["date"].to_list() == [date(2025, 3, 19)] and cal["detail"].to_list() == ["sep"]
    hist = _events.fomc_historical(HISTORICAL_2020, "u")
    assert hist["date"].to_list() == [date(2020, 1, 29), date(2020, 3, 2), date(2020, 4, 29)]
    march = hist.filter(pl.col("date") == date(2020, 3, 2)).row(0, named=True)
    assert march["scheduled"] is False and march["ts"] is None
    assert hist.filter(pl.col("date") == date(2020, 1, 29))["ts"].item() == utc("2020-01-29T19:00")


def a_cpi_release_in_winter_and_in_summer():
    got = _events.bls_index(CPI_INDEX, "cpi", "u")
    assert got["date"].to_list() == [date(2019, 1, 11), date(2026, 7, 14)]
    assert got["ts"].to_list() == [utc("2019-01-11T13:30"), utc("2026-07-14T12:30")]


class _Site:
    def __init__(self, pages):
        self.pages, self.agents = pages, []

    def __call__(self, url, method, agent="galata-fetch"):
        self.agents.append((url, agent))
        body = self.pages.get(url)
        return fetch.Response(404) if body is None else fetch.Response(200, body.encode(), len(body))


@pytest.fixture(name="store")
def _store(tmp_path, monkeypatch):
    monkeypatch.setenv("GALATA_REFERENCE", str(tmp_path / "reference"))
    monkeypatch.setenv("GALATA_VAR", str(tmp_path / "record"))
    return tmp_path / "reference"


def _pages():
    return {
        _events.FOMC_CALENDAR: _calendar({2021: [("January", "26-27")]}),
        _events.FOMC_HISTORICAL.format(year=2019): "<h5>July 30-31 Meeting - 2019</h5>",
        _events.FOMC_HISTORICAL.format(year=2020): HISTORICAL_2020,
        _events.BLS_INDEX["cpi"]: CPI_INDEX,
        _events.BLS_INDEX["jobs"]: '<a href="/news.release/archives/empsit_09042026.htm">x</a>',
    }


def a_redesigned_page_is_refused(store, monkeypatch):
    pages = _pages()
    pages[_events.FOMC_CALENDAR] = "<html>a new design</html>"
    monkeypatch.setattr(fetch, "_transport", _Site(pages))
    with pytest.raises(Refused, match="fomccalendars.htm parsed to no event"):
        fetch.run(["events"], say=lambda _: None)
    assert not (store / "events" / "events.parquet").exists()


def no_contact_no_bls(store, monkeypatch):
    site = _Site(_pages())
    monkeypatch.setattr(fetch, "_transport", site)
    monkeypatch.delenv("GALATA_CONTACT", raising=False)
    said = []
    assert fetch.run(["events"], say=said.append) == 1
    assert "GALATA_CONTACT" in said[-1]
    assert not any("bls.gov" in url for url, _ in site.agents)
    assert set(pl.read_parquet(store / "events" / "events.parquet")["source"]) == {"fomc"}


def the_contact_is_not_stored(store, monkeypatch):
    site = _Site(_pages())
    monkeypatch.setattr(fetch, "_transport", site)
    monkeypatch.setenv("GALATA_CONTACT", "a@b.c")
    assert fetch.run(["events"], say=lambda _: None) == 0
    assert all(("a@b.c" in agent) == ("bls.gov" in url) for url, agent in site.agents)
    for path in store.rglob("*"):
        if path.is_file():
            assert b"a@b.c" not in path.read_bytes(), path
    got = gr.reference.events("2019-01-01T00:00Z", "2027-01-01T00:00Z")
    assert got.group_by("source").len().sort("source")["len"].to_list() == [3, 5]  # bls: 2 cpi + 1 jobs; fomc: 2019, three of 2020, 2021


def no_calendar_fetched_is_refused(store):
    store.mkdir(parents=True)
    with pytest.raises(Refused, match="galata-fetch events"):
        gr.reference.events("2020-01-01T00:00Z", "2021-01-01T00:00Z")
