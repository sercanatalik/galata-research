"""The calendar of scheduled US releases and FOMC decisions, parsed from the pages that publish them.

- **FOMC**, federalreserve.gov:
  - `fomccalendars.htm` lists 2021 on, in `<h4>YYYY FOMC Meetings</h4>` blocks;
  - `fomchistorical<year>.htm` lists earlier years as `March 2 (unscheduled) Meeting - 2020`;
  - a statement comes on a meeting's last day at 14:00 New York, the practice since 2013;
  - `*` marks a Summary of Economic Projections;
  - a notation vote is not a decision, and neither is a cancelled meeting;
  - an unscheduled action's time is not on the page, so its `ts` is null rather than a guess.
- **BLS**, bls.gov:
  - `news-release/cpi.htm` and `empsit.htm` link each release as `<name>_MMDDYYYY.htm`, so the date in the name is the release date;
  - releases are at 08:30 New York;
  - measured 2026-09-27: 12 CPI releases a year 2019–2024, and 11 in 2025, when the shutdown cancelled one.

bls.gov refuses a client whose User-Agent carries no contact; the contact is
read from `GALATA_CONTACT` and never written anywhere.
"""

import re
from datetime import date, datetime

import polars as pl

from .._errors import Refused

HOSTS = ("www.federalreserve.gov", "www.bls.gov")
FOMC_CALENDAR = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FOMC_HISTORICAL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
BLS_INDEX = {"cpi": "https://www.bls.gov/bls/news-release/cpi.htm", "jobs": "https://www.bls.gov/bls/news-release/empsit.htm"}
_BLS_NAME = {"cpi": "cpi", "jobs": "empsit"}
FIRST_YEAR = 2019
SCHEMA = {
    "source": pl.String, "event": pl.String, "date": pl.Date, "ts": pl.Datetime("us", "UTC"),
    "scheduled": pl.Boolean, "detail": pl.String, "url": pl.String,
}  # fmt: skip
_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 1
)}  # fmt: skip
_SHORT = {m[:3]: i for m, i in _MONTHS.items()}
_ET = "America/New_York"


def _instant(rows: list[dict], hour: int, minute: int) -> pl.DataFrame:
    """Rows with a `date` and `scheduled`, given `ts` = that local time in New York, in UTC; null where unscheduled."""
    frame = pl.DataFrame(rows, schema={k: v for k, v in SCHEMA.items() if k != "ts"})
    local = pl.col("date").cast(pl.Datetime("us")) + pl.duration(hours=hour, minutes=minute)
    return frame.with_columns(
        pl.when(pl.col("scheduled")).then(local.dt.replace_time_zone(_ET).dt.convert_time_zone("UTC")).alias("ts")
    ).select(list(SCHEMA))


def fomc_calendar(html: str, url: str = FOMC_CALENDAR) -> pl.DataFrame:
    """Meetings from the calendar page, each dated by its last day."""
    rows = []
    parts = re.split(r"<h4><a id=\"\d+\">(\d{4}) FOMC Meetings</a></h4>", html)
    for i in range(1, len(parts), 2):
        year = int(parts[i])
        cells = re.findall(r"fomc-meeting__month[^>]*><strong>([^<]+)</strong>.*?fomc-meeting__date[^>]*>([^<]+)<", parts[i + 1], re.S)
        for month, days in cells:
            text = days.strip()
            if "notation" in text or "cancel" in text:
                continue
            last_month = month.split("/")[-1].strip()
            m = _MONTHS.get(last_month) or _SHORT.get(last_month[:3])
            d = re.findall(r"\d+", text)
            if m is None or not d:
                raise Refused(f"{url}: cannot read the meeting {month!r} {days!r}")
            unscheduled = "unscheduled" in text
            rows.append({
                "source": "fomc", "event": "fomc", "date": date(year, m, int(d[-1])), "scheduled": not unscheduled,
                "detail": "sep" if "*" in text else ("unscheduled" if unscheduled else None), "url": url,
            })  # fmt: skip
    return _instant(rows, 14, 0)


def fomc_historical(html: str, url: str) -> pl.DataFrame:
    """Meetings from a yearly historical page: `March 17-18 (cancelled) Meeting - 2020`."""
    rows = []
    pattern = r"(January|February|March|April|May|June|July|August|September|October|November|December)(?:/([A-Za-z]+))? ([\d\-]+)(\s*\((unscheduled|cancelled)\))? (?:Meeting|Conference Call) - (\d{4})"
    for month, second, days, _, flag, year in re.findall(pattern, html):
        if flag == "cancelled":
            continue
        m = _MONTHS[second] if second in _MONTHS else _MONTHS[month]
        rows.append({
            "source": "fomc", "event": "fomc", "date": date(int(year), m, int(days.split("-")[-1])),
            "scheduled": flag != "unscheduled", "detail": "unscheduled" if flag == "unscheduled" else None, "url": url,
        })  # fmt: skip
    frame = _instant(rows, 14, 0)
    return frame.unique(subset=["date"], keep="first", maintain_order=True)


def bls_index(html: str, event: str, url: str) -> pl.DataFrame:
    """One release per archive link `<name>_MMDDYYYY.htm`, at 08:30 New York."""
    name = _BLS_NAME[event]
    dates = sorted({datetime.strptime(s, "%m%d%Y").date() for s in re.findall(rf"archives/{name}_(\d{{8}})\.htm", html)})
    rows = [{"source": "bls", "event": event, "date": d, "scheduled": True, "detail": None, "url": url} for d in dates]
    return _instant(rows, 8, 30)
