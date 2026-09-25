from datetime import timedelta

import polars as pl
import pytest
from conftest import utc

from galata_research import studies
from galata_research.studies import _donchian_signal


def _bars(closes):
    t0 = utc("2026-01-01T00:00")
    return pl.DataFrame(
        {
            "ticker": "BTC",
            "ts": [t0 + timedelta(days=d) for d in range(len(closes))],
            "close_ts": [t0 + timedelta(days=d + 1) for d in range(len(closes))],
            "close": [float(c) for c in closes],
        }
    )


def a_breakout_opens_and_the_midpoint_trail_holds():
    # L=3: 12 breaks the 10s (stop 10); at 13 the stop ratchets to 11; 11.4 holds above it.
    assert _donchian_signal([10, 10, 10, 12, 13, 11.4], [3]) == [0, 0, 0, 1, 1, 1]


def a_close_below_the_stop_exits():
    # After 11.4 the midpoint of (10, 12, 13) lifts the stop to 11.5; 10.9 is below it.
    assert _donchian_signal([10, 10, 10, 12, 13, 11.4, 10.9], [3])[-1] == 0


def the_stop_a_close_sets_does_not_exit_that_close():
    # Guard: at 12.5 the stop is still 11 (set earlier), so it holds, and only then ratchets
    # to the midpoint of (12, 13, 20) = 16. The next close, 15, is below 16 and exits.
    got = _donchian_signal([10, 10, 10, 12, 13, 20, 12.5, 15], [3])
    assert got[6] == 1 and got[7] == 0


def a_lookback_without_enough_history_is_flat():
    assert _donchian_signal([10, 11, 12], [5]) == [0, 0, 0]


def the_ensemble_is_the_fraction_open():
    closes = [10] * 10 + [11]
    assert _donchian_signal(closes, [3, 5, 20])[-1] == pytest.approx(2 / 3)


def the_leverage_is_capped():
    # A steady 0.1% a day with a little noise is far below 25% a year: the scale caps at 1.
    closes = [100.0]
    for d in range(399):
        closes.append(closes[-1] * (1.001 + (0.0005 if d % 2 else -0.0005)))
    got = studies.donchian_ensemble(_bars(closes), lookbacks=[5], vol_window=90).drop_nulls("position")
    assert got["position"].max() <= 1.0
    assert got["position"].max() == pytest.approx(1.0)


def no_position_is_ever_short():
    closes = [100 + 10 * ((d % 40) - 20) ** 2 / 400 for d in range(500)]
    got = studies.donchian_ensemble(_bars(closes), vol_window=90).drop_nulls("position")
    assert got["position"].min() >= 0.0
