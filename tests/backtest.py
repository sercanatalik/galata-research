from datetime import timedelta

import polars as pl
import pytest
from conftest import utc

from galata_research import backtest, studies

DAY = timedelta(days=1)


def _bars(closes, *, ticker="BTC", start="2026-01-01T00:00", skip=()):
    t0 = utc(start)
    days = [d for d in range(len(closes) + len(skip)) if d not in skip][: len(closes)]
    return pl.DataFrame(
        {
            "ticker": ticker,
            "ts": [t0 + d * DAY for d in days],
            "close_ts": [t0 + (d + 1) * DAY for d in days],
            "close": [float(c) for c in closes],
        }
    )


def a_signal_at_a_close_earns_the_next_bar_not_its_own():
    # Guard: without the shift, the long decided on the +10% close would earn that same +10%.
    b = _bars([100, 110, 111.1])
    position = (pl.col("close") / pl.col("close").shift(1) > 1.05).cast(pl.Float64).fill_null(0.0)
    got = backtest.returns(b, position, fee=0.0)
    assert got["gross"].to_list() == [None, pytest.approx(0.0), pytest.approx(0.01)]


def a_flip_from_long_to_short_costs_twice():
    b = _bars([100, 100, 100, 100])
    position = pl.Series([1.0, 1.0, -1.0, -1.0])
    got = backtest.returns(b, pl.lit(position), fee=0.00045)
    # Held: -, 1, 1, -1. The entry costs 1 turn, the flip 2.
    assert got["cost"].to_list() == [None, pytest.approx(0.00045), 0.0, pytest.approx(0.0009)]


def no_row_charges_funding_and_each_says_so():
    got = backtest.returns(_bars([100, 101, 102]), pl.lit(1.0))
    assert got["funding_charged"].to_list() == [False] * 3
    assert got["modelled"].all()


def a_hole_in_the_bars_is_not_spanned():
    # Day 2 is missing: the return from day 1's close to day 3's must not be earned.
    got = backtest.returns(_bars([100, 120, 121], skip=(2,)), pl.lit(1.0), fee=0.0)
    assert got["gross"].to_list() == [None, pytest.approx(0.2), None]


def every_crossover_pair_is_a_trial():
    got = studies.moving_average(_bars(range(100, 160)), fasts=[5, 20], slows=[20, 50])
    assert sorted(got["trial"].unique()) == sorted(
        f"ma {f}/{s} {side}" for f, s in [(5, 20), (5, 50), (20, 50)] for side in ("long_flat", "long_short")
    )


def a_short_trial_still_counts():
    got = studies.summary(studies.momentum(_bars(range(100, 120)), lookbacks=[5]))
    assert got.height == 1
    assert got["sharpe"].to_list() == [None]
    assert got["periods"].item() < studies.MIN_RETURNS


HOUR = timedelta(hours=1)


def _settled(start, hours, rate=0.0000125, *, ticker="BTC", skip=(), stamp=timedelta(milliseconds=121)):
    """Settled funding as gr.market.funding returns it: a few ms past each hour."""
    t0 = utc(start)
    return pl.DataFrame(
        {
            "ticker": ticker,
            "ts": [t0 + h * HOUR + stamp for h in range(hours) if h not in skip],
            "rate": [rate for h in range(hours) if h not in skip],
        }
    )


def a_long_pays_a_positive_rate_on_every_hour_held():
    b = _bars([100, 100, 100])
    # Hours due for the bar 01-02 → 01-03: 01-02 01:00 .. 01-03 00:00, all 24.
    got = backtest.returns(b, pl.lit(1.0), fee=0.0, funding=_settled("2026-01-01T00:00", 73))
    assert got["funding"].to_list()[1:] == [pytest.approx(0.0003), pytest.approx(0.0003)]
    assert got["net"].to_list()[1:] == [pytest.approx(-0.0003), pytest.approx(-0.0003)]
    assert got["funding_charged"].to_list() == [False, True, True]


def a_short_receives_it():
    b = _bars([100, 100, 100])
    got = backtest.returns(b, pl.lit(-1.0), fee=0.0, funding=_settled("2026-01-01T00:00", 73))
    assert got["net"].to_list()[1:] == [pytest.approx(0.0003), pytest.approx(0.0003)]


def a_bar_missing_an_hour_is_not_charged():
    b = _bars([100, 110, 121])
    # Hour 30 (01-02 06:00) falls in the second bar's window: that bar is not charged.
    got = backtest.returns(b, pl.lit(1.0), fee=0.0, funding=_settled("2026-01-01T00:00", 73, skip=(30,)))
    assert got["funding_charged"].to_list() == [False, False, True]
    assert got["funding"][1] is None
    assert got["net"][1] == pytest.approx(0.10), "net excludes an uncharged bar's funding"


def a_settlement_on_the_close_is_paid_by_the_bar_closing():
    t0 = utc("2026-01-01T05:00")
    b = pl.DataFrame(
        {
            "ticker": "BTC",
            "ts": [t0, t0 + HOUR, t0 + 2 * HOUR],
            "close_ts": [t0 + HOUR, t0 + 2 * HOUR, t0 + 3 * HOUR],
            "close": [100.0, 100.0, 100.0],
        }
    )
    # One settlement, stamped 06:00:00.121: the 05:00-06:00 bar's, not the 06:00-07:00 bar's.
    f = _settled("2026-01-01T06:00", 1, rate=0.001)
    got = backtest.returns(b, pl.lit(1.0), fee=0.0, funding=f)
    # The first bar has no return; the 06:00-07:00 bar owes 07:00, which is absent.
    assert got["funding_charged"].to_list() == [False, False, False]
    f = _settled("2026-01-01T06:00", 3, rate=0.001)
    got = backtest.returns(b, pl.lit(1.0), fee=0.0, funding=f)
    # 06:00-07:00 pays 07:00 only; 07:00-08:00 pays 08:00 only.
    assert got["funding"].to_list()[1:] == [pytest.approx(0.001), pytest.approx(0.001)]


def no_funding_given_changes_nothing():
    b = _bars([100, 110, 99])
    got = backtest.returns(b, pl.lit(1.0))
    assert got["funding_charged"].to_list() == [False, False, False]
    assert got["funding"].to_list() == [None, None, None]
    # Net is gross less the entry's taker fee, exactly as before funding existed.
    assert got["net"].to_list()[1:] == [pytest.approx(0.1 - 0.00045), pytest.approx(-0.1)]


def a_funding_frame_without_its_columns_is_refused():
    with pytest.raises(backtest.Refused, match="gr.market.funding"):
        backtest.returns(_bars([1, 2]), pl.lit(1.0), funding=pl.DataFrame({"ticker": ["BTC"]}))
