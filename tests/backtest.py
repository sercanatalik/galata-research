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
