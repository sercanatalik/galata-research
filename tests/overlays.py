import random
from datetime import timedelta

import polars as pl
import pytest
from conftest import utc

from galata_research import Refused, overlays, studies, trades

DAY = timedelta(days=1)
FEE = 0.001


def _bars(closes, *, opens=None, highs=None, lows=None, ticker="BTC", skip=()):
    t0 = utc("2026-01-01T00:00")
    days = [d for d in range(len(closes) + len(skip)) if d not in skip][: len(closes)]
    closes = [float(c) for c in closes]
    return pl.DataFrame(
        {
            "ticker": ticker,
            "ts": [t0 + d * DAY for d in days],
            "close_ts": [t0 + (d + 1) * DAY for d in days],
            "open": [float(x) for x in opens] if opens else [closes[0]] + closes[:-1],
            "high": [float(x) for x in highs] if highs else closes,
            "low": [float(x) for x in lows] if lows else closes,
            "close": closes,
        }
    )


def _long(bars, decided=None, *, fee=FEE):
    position = pl.lit(1.0) if decided is None else pl.lit(pl.Series(decided, dtype=pl.Float64))
    return studies.trial(bars, position, "base", fee=fee)


def _walk(n=200, seed=3):
    rng = random.Random(seed)
    closes, c = [], 100.0
    for _ in range(n):
        c *= 1 + rng.gauss(0, 0.03)
        closes.append(c)
    opens = [100.0] + [c * (1 + rng.gauss(0, 0.005)) for c in closes[:-1]]
    highs = [max(o, c) * (1 + abs(rng.gauss(0, 0.01))) for o, c in zip(opens, closes)]
    lows = [min(o, c) * (1 - abs(rng.gauss(0, 0.01))) for o, c in zip(opens, closes)]
    return _bars(closes, opens=opens, highs=highs, lows=lows)


def an_overlay_with_no_level_is_the_trial_itself():
    # Guard on the re-run: the same positions, returns and costs as gr.backtest.returns, holes included.
    b = _walk()
    b = b.filter(pl.col("ts") != b["ts"][50])
    decided = [1.0 if i % 17 < 9 else (-0.5 if i % 17 < 13 else 0.0) for i in range(b.height)]
    base = _long(b, decided)
    got = overlays.apply(b, base, cooldown=3)
    for col in ("position", "bar_return", "gross", "cost", "net"):
        assert got[col].to_list() == pytest.approx(base[col].to_list(), nan_ok=True), col
    assert got["exit"].null_count() == got.height


def a_stop_touched_inside_the_bar_fills_at_its_level():
    b = _bars([100, 100, 99, 100], lows=[100, 100, 94, 100])
    got = overlays.apply(b, _long(b), stop=0.05, fee=0.0)
    # Entered at the first close (100), stopped at 95 on bar 2.
    assert got["exit"][2] == "stop" and got["exit_price"][2] == 95.0
    assert got["gross"][2] == pytest.approx(-0.05)


def a_bar_that_opens_through_the_stop_fills_at_the_open():
    b = _bars([100, 100, 88, 88], opens=[100, 100, 90, 88], lows=[100, 100, 85, 88])
    got = overlays.apply(b, _long(b), stop=0.05, fee=0.0, cooldown=5)
    assert got["exit_price"][2] == 90.0
    assert got["gross"][2] == pytest.approx(-0.10)


def the_exit_is_charged_in_the_bar_it_happens():
    b = _bars([100, 100, 99, 100], lows=[100, 100, 94, 100])
    got = overlays.apply(b, _long(b), stop=0.05, cooldown=5)
    # Row 1 enters (one turn); row 2 is stopped (one turn out); row 3 is flat and charged nothing.
    assert got["cost"].to_list()[1:] == [pytest.approx(FEE), pytest.approx(FEE), 0.0]
    assert got["position"].to_list()[1:] == [1.0, 1.0, 0.0]


def no_level_is_checked_on_the_bar_that_decided_the_position():
    # Guard: the deciding bar's low of 50 came before the entry at its close.
    b = _bars([100, 100, 100], lows=[50, 100, 100])
    got = overlays.apply(b, _long(b, [1, 1, 1]), stop=0.05)
    assert got["exit"].null_count() == 3


def a_trailing_stop_trails_the_highs_of_earlier_bars_only():
    # Highs 110 (bar 1), then bar 2 reaches 120 and falls to 105. A 10% trail from 110 sits at 99, so 105 holds.
    # Guard: trailed from bar 2's own high (120 → 108), the stop would fire on a bar whose order is unknown.
    b = _bars([100, 110, 106, 100, 100], highs=[100, 110, 120, 106, 100], lows=[100, 100, 105, 97, 100])
    got = overlays.apply(b, _long(b), stop=0.10, trailing=True, fee=0.0, cooldown=5)
    assert got["exit"][2] is None
    # Bar 3: the highest high through bar 2 is 120, so the stop is at 108; bar 3 opens at 106, through it.
    assert got["exit"][3] == "stop" and got["exit_price"][3] == 106.0


def a_take_profit_fills_at_its_level():
    b = _bars([100, 100, 110, 110], highs=[100, 100, 126, 110])
    got = overlays.apply(b, _long(b), take=0.25, fee=0.0, cooldown=5)
    assert got["exit"][2] == "take" and got["exit_price"][2] == 125.0
    assert got["gross"][2] == pytest.approx(0.25)


def a_bar_that_touches_both_levels_is_taken_as_stopped():
    b = _bars([100, 100, 100, 100], highs=[100, 100, 130, 100], lows=[100, 100, 90, 100])
    got = overlays.apply(b, _long(b), stop=0.05, take=0.25, fee=0.0, cooldown=5)
    assert got["exit"][2] == "stop"


def a_shorts_levels_are_mirrored():
    b = _bars([100, 100, 104, 100], highs=[100, 100, 106, 100])
    got = overlays.apply(b, _long(b, [-1, -1, -1, -1]), stop=0.05, fee=0.0, cooldown=5)
    assert got["exit"][2] == "stop" and got["exit_price"][2] == 105.0
    assert got["gross"][2] == pytest.approx(-0.05)


def the_cooldown_keeps_it_flat_then_the_base_returns():
    b = _bars([100, 100, 94, 94, 94, 94, 94], lows=[100, 100, 90, 94, 94, 94, 94])
    got = overlays.apply(b, _long(b), stop=0.05, cooldown=2)
    assert got["position"].to_list()[1:] == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    # Re-entry is charged, at the close before it (94), and its stop is from there.
    assert got["cost"][5] == pytest.approx(FEE)


def a_hole_earns_nothing_and_is_not_checked():
    b = _bars([100, 100, 80, 80], lows=[100, 100, 70, 80], skip=(2,))
    got = overlays.apply(b, _long(b), stop=0.05, fee=0.0)
    assert got["gross"][2] is None and got["exit"][2] is None


def the_overlays_trades_end_where_it_exited():
    b = _bars([100, 100, 94, 94, 94], lows=[100, 100, 90, 94, 94])
    got = overlays.apply(b, _long(b), stop=0.05, cooldown=10)
    t = trades.table(got, b)
    assert t.height == 1 and not t["open"][0]
    assert t["cost"][0] == pytest.approx(2 * FEE)
    assert t["gross"][0] == pytest.approx(-0.05)


def the_trial_is_named_for_its_overlay():
    b = _walk(30)
    got = overlays.apply(b, _long(b), stop=0.1, trailing=True, take=0.25, cooldown=10)
    assert set(got["trial"]) == {"base + stop 10% trail, take 25%, cool 10"}


def the_registered_grid_is_twelve_overlays():
    b = _walk(60)
    got = overlays.grid(b, _long(b))
    assert got["trial"].n_unique() == 12


def the_overlay_trades_at_the_fee_its_base_paid():
    b = _walk(40)
    assert overlays.charged_fee(_long(b, [1.0 if i % 5 else 0.0 for i in range(40)], fee=0.0007)) == pytest.approx(0.0007)
    assert overlays.charged_fee(_long(b, [0.0] * 40)) == pytest.approx(0.00045)


def a_bad_level_is_refused():
    b = _walk(10)
    with pytest.raises(Refused, match="stop=1.5"):
        overlays.apply(b, _long(b), stop=1.5)
    with pytest.raises(Refused, match="trailing"):
        overlays.apply(b, _long(b), trailing=True)


def the_shared_days_are_the_days_every_ticker_has_the_days_every_ticker_has():
    b = pl.concat([_bars([1, 2, 3]), _bars([1, 2, 3], ticker="ETH", skip=(1,))])
    got = studies.shared_days(b)
    assert got.group_by("ticker").len().sort("ticker")["len"].to_list() == [2, 2]
