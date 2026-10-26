import inspect
import math
import random
from datetime import timedelta

import polars as pl
import pytest
from conftest import utc
from polars.testing import assert_frame_equal

from galata_research import Refused, momentum, timeseries

START = utc("2026-01-05T00:00")  # a Monday
WEEK_5M = 2016


def _bars(closes, *, minutes=5, skip=(), volume=None, ticker="BTC"):
    """Candles of `minutes` width from START, one per close, with the indices in `skip` left out."""
    width = timedelta(minutes=minutes)
    rows = [(START + i * width, c) for i, c in enumerate(closes) if i not in set(skip)]
    ts = [t for t, _ in rows]
    cs = [float(c) for _, c in rows]
    vol = [1.0] * len(rows) if volume is None else [v for i, v in enumerate(volume) if i not in set(skip)]
    return pl.DataFrame(
        {
            "ticker": ticker,
            "ts": ts,
            "close_ts": [t + width for t in ts],
            "open": cs,
            "high": cs,
            "low": cs,
            "close": cs,
            "volume": vol,
            "trade_count": pl.Series([1 if v else 0 for v in vol], dtype=pl.UInt32),
        }
    )


def _walk(n, seed=7, drift=0.0, step=0.001):
    rng = random.Random(seed)
    price, out = 100.0, []
    for _ in range(n):
        price *= math.exp(drift + step * rng.gauss(0, 1))
        out.append(price)
    return out


def a_noiseless_drift_gives_positive_z_on_every_lookback():
    closes = [100 * math.exp(1e-4 * i) for i in range(6 * WEEK_5M)]
    f = momentum.features(_bars(closes))
    last = f.drop_nulls([f"z_{n}" for n in momentum.LADDER]).tail(1)
    assert last.height == 1
    assert all(last[f"z_{n}"][0] > 0 for n in momentum.LADDER)


def no_row_moves_when_later_bars_are_appended():
    closes = _walk(7 * WEEK_5M)
    cut = 5 * WEEK_5M
    full = momentum.features(_bars(closes))
    early = momentum.features(_bars(closes[:cut]))
    t = early["close_ts"].max()
    assert_frame_equal(full.filter(pl.col("close_ts") <= t), early)


def the_weekly_factors_are_seasonal_factors_fitted_to_each_monday():
    closes = _walk(4 * WEEK_5M, seed=3)
    returns = timeseries.returns(_bars(closes), kind="log")
    weekly = momentum.weekly_factors(returns)
    monday = START + timedelta(weeks=3)
    direct = timeseries.seasonal_factors(returns, fit=(START, monday), by="hour_x_weekday", stat="mean_abs")
    ours = weekly.filter(pl.col("week") == monday).sort("weekday", "hour")
    assert ours.height == direct.height == 168
    assert ours["factor"].to_list() == pytest.approx(direct["factor"].to_list(), rel=1e-12)
    # The first week has no past to fit on.
    assert weekly.filter(pl.col("week") == START)["factor"].null_count() == 168


def a_missing_hour_nulls_the_windows_that_contain_it():
    closes = _walk(6 * WEEK_5M, seed=11)
    hole = 5 * WEEK_5M
    f = momentum.features(_bars(closes, skip=range(hole, hole + 12)))
    after = f.filter(pl.col("ts") >= START + timedelta(minutes=5 * (hole + 12)))
    before = f.filter(pl.col("ts") < START + timedelta(minutes=5 * hole))
    # Every window of the first 12 bars after the hole reaches across it.
    assert after.head(12)["z_1h"].null_count() == 12
    assert after.slice(400, 100)["z_15m"].null_count() == 0
    assert after.slice(400, 100)["z_1h"].null_count() == 0
    # The long lookbacks still reach across it, so the rows say they are after a gap.
    assert after["z_4w"].null_count() == after.height
    assert after["after_gap"].all()
    assert not before["after_gap"].any()
    # Rows before the hole are what they would be without the bars after it.
    alone = momentum.features(_bars(closes[:hole]))
    assert_frame_equal(before, alone)


def a_decision_bar_carries_the_realized_variance_of_its_minutes():
    closes = _walk(60, seed=5)
    f = momentum.features(_bars(closes, minutes=1, skip=[33]), ladder=("15m", "1d"))
    logs = [math.log(c) for c in closes]
    second = f.filter(pl.col("ts") == START + timedelta(minutes=5))
    expected = sum((logs[i] - logs[i - 1]) ** 2 for i in range(5, 10))
    assert second["rv"][0] == pytest.approx(expected, rel=1e-12)
    assert second["close"][0] == closes[9]
    # The bar missing a minute is not a bar.
    assert f.filter(pl.col("ts") == START + timedelta(minutes=30)).height == 0
    # The first bar's first minute has no previous close, so its variance is not whole.
    assert f["rv"][0] is None


def an_untraded_bar_is_dropped_and_counted():
    closes = _walk(3000, seed=2)
    volume = [0.0] * 500 + [1.0] * 2500
    bars = _bars(closes, volume=volume)
    assert momentum.untraded(bars) == 500
    f = momentum.features(bars, ladder=("15m", "1d"))
    assert f["ts"].min() == START + timedelta(minutes=5 * 500)


def a_decision_bar_that_does_not_divide_into_the_bars_is_refused():
    with pytest.raises(Refused, match="a 7m decision bar is not a whole multiple of the 5m bars"):
        momentum.features(_bars(_walk(50)), decision="7m")
    with pytest.raises(Refused, match="15m is not a whole number of 1h bars"):
        momentum.features(_bars(_walk(50)), decision="1h")


def the_encodings_of_one_z():
    frame = pl.DataFrame({"z": [0.3, -0.3, 0.0, 2.0, None]})

    def enc(how):
        return frame.select(momentum.encode(pl.col("z"), how).alias("z"))["z"].to_list()

    assert enc("binary") == [1.0, -1.0, 1.0, 1.0, None]
    assert enc("ternary") == [0.0, 0.0, 0.0, 1.0, None]
    score = enc("score")
    assert score[0] == pytest.approx(0.3 * math.exp(-0.0225) / 0.89)
    assert score[1] == pytest.approx(-score[0])
    with pytest.raises(Refused, match="not one of"):
        momentum.encode(pl.col("z"), "rank")


def the_score_is_odd_and_bounded():
    zs = [i / 100 for i in range(-800, 801)]
    s = pl.DataFrame({"z": zs}).select(momentum.encode(pl.col("z"), "score").alias("z"))["z"].to_list()
    assert max(abs(v) for v in s) == pytest.approx(math.sqrt(2) * math.exp(-0.5) / 0.89, rel=1e-3)
    assert max(abs(v) for v in s) < 0.97
    assert all(a == pytest.approx(-b) for a, b in zip(s, reversed(s), strict=True))


def a_pullback_has_a_negative_interaction():
    frame = pl.DataFrame({f"z_{n}": [-1.0] for n in momentum.SHORT} | {f"z_{n}": [2.0] for n in momentum.LONG})
    e = momentum.encoded(frame, "binary")
    assert e["interaction"][0] == -1.0
    with pytest.raises(Refused, match="needs a short"):
        momentum.encoded(frame.select("z_1d"), "binary")


def the_module_needs_no_numpy():
    assert "numpy" not in inspect.getsource(momentum)
