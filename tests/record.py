"""Claims about the real record under the resolved root; skipped when there is none.

Figures are the design's, measured 2026-09-25, asserted as floors or as
structure, because the record grows and retention trims the finest widths.
"""

import polars as pl
import pytest
from conftest import utc

import galata_research as gr
from galata_research import Refused

pytestmark = pytest.mark.record

EVER = utc("2020-01-01T00:00"), utc("2100-01-01T00:00")


@pytest.fixture(scope="module", autouse=True)
def _record():
    try:
        gr.frontier()
    except Refused as absent:
        pytest.skip(f"no record: {absent}")


@pytest.mark.parametrize("interval", ["1m", "1h", "4h", "1d"])
def every_series_has_at_most_one_open_bar(interval):
    got = gr.market.candles(None, interval, *EVER, traded_only=False, closed_only=False).collect()
    open_per_series = got.group_by("venue", "ticker").agg((~pl.col("closed")).sum().alias("open"))
    assert open_per_series["open"].max() <= 1


@pytest.mark.parametrize("interval", ["1m", "1h", "4h", "1d"])
def no_bar_is_returned_twice(interval):
    got = gr.market.candles(None, interval, *EVER, traded_only=False, closed_only=False).collect()
    assert got.select("venue", "ticker", "ts").is_duplicated().sum() == 0


def the_coarse_history_is_at_least_as_deep_as_measured():
    for interval, floor in [("1d", 6_001), ("4h", 19_271)]:
        got = gr.market.candles(None, interval, *EVER, traded_only=False).collect()
        assert got.height >= floor, interval


def the_first_traded_btc_daily_bar_opens_on_2023_02_26():
    got = gr.market.candles("BTC", "1d", *EVER).head(1).collect()
    assert got["ts"].item() == utc("2023-02-26T00:00")


def no_trade_less_bar_is_returned_by_default():
    got = gr.market.candles(None, "1d", *EVER).collect()
    assert got.filter(pl.col("trade_count") == 0).height == 0
