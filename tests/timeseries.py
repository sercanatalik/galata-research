import math
from datetime import timedelta

import polars as pl
import pytest
from conftest import utc

from galata_research import Refused, backtest, timeseries

DAY = timedelta(days=1)


def _bars(closes, *, ticker="BTC", skip=()):
    t0 = utc("2026-01-01T00:00")
    days = [d for d in range(len(closes) + len(skip)) if d not in skip][: len(closes)]
    return pl.DataFrame(
        {
            "ticker": ticker,
            "ts": [t0 + d * DAY for d in days],
            "close_ts": [t0 + (d + 1) * DAY for d in days],
            "close": [float(c) for c in closes],
        }
    )


def the_annualisation_factors():
    assert [timeseries.periods_per_year(i) for i in ("1m", "1h", "4h", "1d")] == [525_600, 8_760, 2_190, 365]


def an_unknown_interval_is_refused():
    with pytest.raises(Refused, match=r"'15m'.*1m, 1h, 4h, 1d"):
        timeseries.periods_per_year("15m")


def the_log_and_simple_returns_agree():
    b = _bars([100, 110])
    assert timeseries.returns(b)["return"].to_list() == [None, pytest.approx(0.1)]
    assert timeseries.returns(b, kind="log")["return"].to_list() == [None, pytest.approx(math.log(1.1))]


def a_return_never_spans_a_hole():
    # Guard: without the contiguity mask, the row after the missing day would carry 120/110 − 1.
    b = _bars([100, 110, 120, 130], skip=(2,))
    for kind in ("simple", "log"):
        got = timeseries.returns(b, kind=kind)["return"].to_list()
        assert got[0] is None and got[2] is None
        assert got[1] is not None and got[3] is not None


def no_ticker_bleeds_into_another():
    b = pl.concat([_bars([100, 110], ticker="BTC"), _bars([50, 55], ticker="ETH")])
    got = timeseries.returns(b)
    assert got.filter(pl.col("ticker") == "ETH")["return"].to_list() == [None, pytest.approx(0.1)]


def the_backtest_earns_the_same_returns():
    b = _bars([100, 110, 99, 120, 130, 125], skip=(3,))
    ours = timeseries.returns(b)["return"].to_list()
    theirs = backtest.returns(b, pl.lit(1.0), fee=0.0)["bar_return"].to_list()
    assert ours.count(None) == 2
    for a, t in zip(ours, theirs, strict=True):
        assert (a is None) == (t is None)
        if a is not None:
            assert a == pytest.approx(t)


def an_unknown_kind_is_refused():
    with pytest.raises(Refused, match="kind='pct'"):
        timeseries.returns(_bars([1, 2]), kind="pct")


def a_frame_without_a_close_is_refused():
    with pytest.raises(Refused, match="close"):
        timeseries.returns(_bars([1, 2]).drop("close"))
