import random
from datetime import timedelta

import polars as pl
import pytest
from conftest import utc

from galata_research import Refused, studies
from galata_research import indicators as ind

DAY = timedelta(days=1)

# StockCharts, "Relative Strength Index (RSI)", the cs-rsi worked example: 33 closes and the 14-day RSI.
STOCKCHARTS_CLOSES = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28,
    46.00, 46.03, 46.41, 46.22, 45.64, 46.21, 46.25, 45.71, 46.45, 45.78, 45.35, 44.03, 44.18, 44.22, 44.57,
    43.42, 42.66, 43.13,
]  # fmt: skip
STOCKCHARTS_RSI = [
    70.53, 66.32, 66.55, 69.41, 66.36, 57.97, 62.93, 63.26, 56.06, 62.38, 54.71, 50.42, 39.99, 41.46, 41.87,
    45.46, 37.30, 33.09, 37.79,
]  # fmt: skip


def _bars(closes, *, highs=None, lows=None, ticker="BTC", skip=()):
    t0 = utc("2026-01-01T00:00")
    days = [d for d in range(len(closes) + len(skip)) if d not in skip][: len(closes)]
    closes = [float(c) for c in closes]
    return pl.DataFrame(
        {
            "ticker": ticker,
            "ts": [t0 + d * DAY for d in days],
            "close_ts": [t0 + (d + 1) * DAY for d in days],
            "high": [float(h) for h in highs] if highs else [c * 1.01 for c in closes],
            "low": [float(x) for x in lows] if lows else [c * 0.99 for c in closes],
            "close": closes,
        }
    )


def _walk(n=300, seed=7, ticker="BTC"):
    rng = random.Random(seed)
    closes, c = [], 100.0
    for _ in range(n):
        c *= 1 + rng.gauss(0, 0.03)
        closes.append(c)
    highs = [c * (1 + abs(rng.gauss(0, 0.01))) for c in closes]
    lows = [c * (1 - abs(rng.gauss(0, 0.01))) for c in closes]
    return _bars(closes, highs=highs, lows=lows, ticker=ticker)


def _one(bars, expr):
    return ind.add(bars, x=expr)["x"].to_list()


def the_stockcharts_rsi_example_converges_to_its_table():
    got = _one(_bars(STOCKCHARTS_CLOSES), ind.rsi(14))
    assert got[:14] == [None] * 14
    # Their spreadsheet rounds its first averages; the gap (0.07 at the start) decays with Wilder's smoothing.
    assert got[14:] == [pytest.approx(x, abs=0.1) for x in STOCKCHARTS_RSI]
    assert got[-2:] == [pytest.approx(33.09, abs=0.005), pytest.approx(37.79, abs=0.005)]


def the_first_rsi_is_the_simple_mean_of_the_first_fourteen_changes():
    changes = [b - a for a, b in zip(STOCKCHARTS_CLOSES, STOCKCHARTS_CLOSES[1:15])]
    gain = sum(max(c, 0) for c in changes) / 14
    loss = sum(max(-c, 0) for c in changes) / 14
    assert _one(_bars(STOCKCHARTS_CLOSES), ind.rsi(14))[14] == pytest.approx(100 - 100 / (1 + gain / loss))


def a_series_that_only_rises_has_an_rsi_of_100():
    assert _one(_bars(range(100, 120)), ind.rsi(5))[-1] == 100.0


def the_ema_is_seeded_with_the_mean_of_its_first_values():
    # Span 3, alpha 1/2: seed mean(1, 2, 3) = 2, then 3, then 4.
    assert _one(_bars([1, 2, 3, 4, 5]), ind.ema(3)) == [None, None, 2.0, 3.0, 4.0]


def the_sma_waits_for_a_full_window():
    assert _one(_bars([1, 2, 3, 4]), ind.sma(3)) == [None, None, 2.0, 3.0]


def no_indicator_spans_a_hole():
    # Guard: day 3 is missing, so the 3-bar mean after it starts again rather than mixing both sides.
    b = _bars([1, 2, 3, 10, 20, 30], skip=(3,))
    assert _one(b, ind.sma(3)) == [None, None, 2.0, None, None, 20.0]
    assert _one(b, ind.ema(3)) == [None, None, 2.0, None, None, 20.0]


def every_indicator_is_known_at_its_close():
    # Guard: an indicator that read a later bar would change an earlier value when the bars are cut short.
    full = _walk()
    cut = full.head(150)
    for name, expr in {
        "sma": ind.sma(20),
        "ema": ind.ema(20),
        "rsi": ind.rsi(14),
        "macd": ind.macd(),
        "signal": ind.macd_signal(),
        "z": ind.zscore(20),
        "upper": ind.bollinger(20, 2.0, "upper"),
        "atr": ind.atr(14),
        "stochastic": ind.stochastic(),
        "supertrend": ind.supertrend(10, 3.0),
        "hold": ind.hold(ind.rsi(14) < 30, ind.rsi(14) > 70, ind.rsi(14) > 70, ind.rsi(14) < 30),
    }.items():
        a, b = _one(full, expr)[:150], _one(cut, expr)
        assert a == pytest.approx(b, nan_ok=True), name


def the_macd_is_the_fast_ema_less_the_slow():
    b = _walk(80)
    framed = ind.add(b, m=ind.macd(12, 26), fast=ind.ema(12), slow=ind.ema(26), s=ind.macd_signal(12, 26, 9))
    rows = framed.drop_nulls("slow")
    assert rows["m"].to_list() == pytest.approx((rows["fast"] - rows["slow"]).to_list())
    # The signal line starts at the mean of the first nine MACD values: bar 26 + 8.
    m = framed["m"].to_list()
    assert framed["s"][33] == pytest.approx(sum(m[25:34]) / 9)
    assert framed["s"][32] is None


def the_zscore_uses_the_population_deviation_as_bollinger_does():
    b = _bars([1, 2, 3, 4])
    framed = ind.add(b, z=ind.zscore(4), up=ind.bollinger(4, 1.0, "upper"))
    sd = (sum((x - 2.5) ** 2 for x in [1, 2, 3, 4]) / 4) ** 0.5
    assert framed["z"][3] == pytest.approx(1.5 / sd)
    assert framed["up"][3] == pytest.approx(2.5 + sd)


def a_flat_window_has_no_zscore():
    assert _one(_bars([5, 5, 5]), ind.zscore(3))[-1] is None


def the_true_range_reaches_back_to_the_previous_close():
    b = _bars([10, 12, 9], highs=[11, 13, 10], lows=[9, 11, 8])
    # Bar 0: high − low = 2. Bar 1: |13 − 10| = 3. Bar 2: |8 − 12| = 4.
    assert _one(b, ind.true_range()) == [2.0, 3.0, 4.0]


def the_atr_is_wilder_smoothed_from_the_mean_of_the_first_n():
    b = _bars([10, 12, 9, 9], highs=[11, 13, 10, 10], lows=[9, 11, 8, 8])
    # TR 2, 3, 4, 2. n = 2: seed (2 + 3)/2 = 2.5, then (2.5 + 4)/2 = 3.25, then (3.25 + 2)/2 = 2.625.
    assert _one(b, ind.atr(2)) == [None, 2.5, 3.25, 2.625]


def the_stochastic_places_the_close_in_its_range():
    b = _bars([5, 6, 10, 8], highs=[6, 7, 10, 9], lows=[4, 5, 6, 7])
    # %K over 3 bars: bar 2 (10 − 4)/(10 − 4) = 100, bar 3 (8 − 5)/(10 − 5) = 60. %D over 2: 80.
    assert _one(b, ind.stochastic(3, 2)) == [None, None, None, pytest.approx(80.0)]


def a_supertrend_turns_down_on_a_crash_and_up_on_a_recovery():
    closes = [100 + i for i in range(30)] + [129 - 6 * i for i in range(1, 11)] + [69 + 6 * i for i in range(1, 16)]
    got = _one(_bars(closes), ind.supertrend(5, 2.0))
    assert got[:4] == [None] * 4
    assert set(got[4:30]) == {1.0}
    assert -1.0 in got[30:40]
    assert got[-1] == 1.0


def an_exit_closes_only_the_side_held():
    # Guard: z ≥ 0 exits a long; while short, it must not flatten the short.
    b = _bars([0, 0, 0, 0, 0, 0])
    framed = b.with_columns(
        pl.Series("el", [False, False, False, False, True, False]),
        pl.Series("xl", [False, False, True, True, False, True]),
        pl.Series("es", [False, True, False, False, False, False]),
        pl.Series("xs", [False, False, False, True, False, False]),
    )
    got = ind.add(framed, p=ind.hold(pl.col("el"), pl.col("xl"), pl.col("es"), pl.col("xs")))["p"].to_list()
    assert got == [0.0, -1.0, -1.0, 0.0, 1.0, 0.0]


def a_held_position_waits_for_its_indicator():
    got = _one(_bars(range(100, 110)), ind.hold(ind.rsi(3) < 30, ind.rsi(3) > 70))
    assert got[:3] == [None] * 3 and got[3] == 0.0


def an_indicator_named_like_a_column_is_refused():
    with pytest.raises(Refused, match="close"):
        ind.add(_bars([1, 2]), close=ind.sma(1))


def a_bad_window_is_refused():
    with pytest.raises(Refused, match="n=0"):
        ind.ema(0)
    with pytest.raises(Refused, match="fast=26"):
        ind.macd(26, 12)


def the_registered_search_is_fifty_rules_per_ticker():
    b = pl.concat([_walk(260, seed=1), _walk(260, seed=2, ticker="ETH")])
    got = studies.indicator_signals(b)
    names = got["trial"].unique()
    assert names.len() == 50
    assert got.group_by("trial", "ticker").len().height == 100
    assert got.filter(pl.col("trial").str.ends_with("long_flat"))["position"].drop_nulls().min() >= 0


def a_rule_never_trades_before_its_indicator_exists():
    got = studies.ema_crossover(_walk(120), [5], [50], sides=["long_short"])
    # The slow EMA exists from bar 50 (index 49); the position decided there is held from index 50.
    assert got["position"][:50].is_null().all()
    assert got["position"][50] is not None


def no_position_is_carried_across_a_hole():
    # Guard: a long held before day 60 would otherwise be held straight through the restart's warm-up.
    closes = [100 + i for i in range(60)] + [200 + i for i in range(60)]
    b = _bars(closes, skip=(60,))
    got = studies.ema_crossover(b, [5], [20], sides=["long_flat"])
    assert got["position"][59] == 1.0
    # Row 60, the first bar after the hole, earns nothing: the backtest does not span it.
    assert got["bar_return"][60] is None and got["gross"][60] is None
    # Every position decided after the hole waits for the restarted EMAs (20 bars).
    assert got["position"][61:80].is_null().all()
    assert got["position"][80] is not None
