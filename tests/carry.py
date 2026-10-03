import random
from datetime import timedelta

import polars as pl
import pytest
from conftest import utc

from galata_research import carry, studies

DAY = timedelta(days=1)
MIN = timedelta(minutes=1)


def _premium(days, closes, ticker="BTC", drop_minute=None):
    """1m premium bars, every minute of each day at that day's close value."""
    rows = []
    for d, c in zip(days, closes):
        t0 = utc(d)
        for m in range(1440):
            if drop_minute == (d, m):
                continue
            rows.append({"ticker": ticker, "ts": t0 + m * MIN, "close": float(c)})
    return pl.DataFrame(rows)


def _settled(days, daily_rates, ticker="BTC", skip=()):
    """Three 8-hour settlements per day, closing it at 08, 16 and the next midnight; daily sum = the rate given."""
    rows = []
    for d, r in zip(days, daily_rates):
        t0 = utc(d)
        for h in (8, 16, 24):
            if (d, h) in skip:
                continue
            rows.append({"ticker": ticker, "ts": t0 + timedelta(hours=h), "rate": r / 3, "interval_hours": 8})
    return pl.DataFrame(rows)


def _days(n, start="2024-01-01T00:00"):
    t0 = utc(start)
    return [(t0 + i * DAY).isoformat() for i in range(n)]


def the_midnight_settlement_closes_the_day_before():
    days = _days(2)
    got = carry.daily(_premium(days, [0.001, 0.002]), _settled(days, [0.0003, 0.0006]))
    assert got["funding"].to_list() == [pytest.approx(0.0003), pytest.approx(0.0006)]
    assert got["premium"].to_list() == [0.001, 0.002]
    assert got["close_ts"][0] == utc("2024-01-02T00:00")


def a_day_short_of_a_minute_or_a_settlement_is_not_a_day():
    days = _days(3)
    got = carry.daily(
        _premium(days, [0.001] * 3, drop_minute=(days[0], 600)),
        _settled(days, [0.0003] * 3, skip=((days[2], 16),)),
    )
    assert got["ts"].to_list() == [utc(days[1])]


def the_hedged_return_is_the_basis_move_less_the_funding_paid():
    days = _days(3)
    day = carry.daily(_premium(days, [0.001, 0.003, 0.002]), _settled(days, [0.0003, 0.0003, -0.0006]))
    got = carry.returns(day, pl.lit(-1.0), fee=0.0015)
    # Held −1 from day 2. Day 2: basis (0.003 − 0.001)/1.001 against a short, plus 0.0003 received, less entry cost.
    assert got["position"].to_list() == [None, -1.0, -1.0]
    assert got["gross"][1] == pytest.approx(-0.002 / 1.001)
    assert got["funding"][1] == pytest.approx(-0.0003)
    assert got["net"][1] == pytest.approx(-0.002 / 1.001 + 0.0003 - 0.0015)
    # Day 3: negative funding is paid by the short.
    assert got["net"][2] == pytest.approx(0.001 / 1.003 - 0.0006)


def a_position_reads_only_the_funding_known_at_its_close():
    # Guard: the rule decided at day 2's close sees day 2's funding, which settled at that close, not day 3's.
    days = _days(4)
    day = carry.daily(_premium(days, [0.0] * 4), _settled(days, [0.0003, -0.0003, 0.0003, -0.0003]))
    got = carry.returns(day, -carry.ind.sma(1, "funding").sign(), fee=0.0)
    assert got["position"].to_list() == [None, -1.0, 1.0, -1.0]
    # Each day the held side receives what it bet against: here the sign flipped, so it pays every day.
    assert got["funding"].to_list()[1:] == [pytest.approx(0.0003)] * 3


def no_return_is_earned_across_a_missing_day():
    days = _days(4)
    prem = _premium(days, [0.0, 0.01, 0.02, 0.03]).filter(pl.col("ts").dt.truncate("1d") != utc(days[1]))
    day = carry.daily(prem, _settled(days, [0.0003] * 4))
    got = carry.returns(day, pl.lit(-1.0))
    assert got["ts"].to_list() == [utc(days[0]), utc(days[2]), utc(days[3])]
    assert got["net"][1] is None and got["net"][2] is not None


def the_registered_carry_rules_are_thirteen():
    days = _days(60)
    rng = random.Random(1)
    rates = [0.0003 + rng.gauss(0, 0.0004) for _ in days]
    day = carry.daily(_premium(days, [rng.gauss(0, 0.0005) for _ in days]), _settled(days, rates))
    got = carry.rules(day)
    assert got["trial"].n_unique() == 13
    assert set(got.columns) >= {"trial", "ticker", "ts", "net", "funding"}
    short_only = got.filter(pl.col("trial").str.ends_with("short_only"))["position"].drop_nulls()
    assert short_only.max() <= 0
    assert studies.summary(got).height == 13


def a_hurdle_keeps_the_rule_flat_on_thin_funding():
    days = _days(4)
    day = carry.daily(_premium(days, [0.0] * 4), _settled(days, [0.0001, 0.0001, 0.0006, 0.0006]))
    got = carry.rules(day, lookbacks=[1], hurdles=[0.0003], sides=["both"])
    got = got.filter(pl.col("trial") == "carry 1 0.0003 both")
    assert got["position"].to_list() == [None, 0.0, 0.0, -1.0]


def a_persistent_funding_has_a_positive_autocorrelation():
    days = _days(400)
    rng = random.Random(4)
    f, rates = 0.0, []
    for _ in days:
        f = 0.8 * f + rng.gauss(0, 0.0003)
        rates.append(f)
    got = carry.persistence(carry.daily(_premium(days, [0.0] * 400), _settled(days, rates))).row(0, named=True)
    assert got["n"] == 399
    assert got["rho"] == pytest.approx(0.8, abs=0.06)
    assert got["p"] < 0.01


def the_fade_is_charged_its_funding():
    days = _days(120)
    rng = random.Random(9)
    rates = [0.0003 + rng.gauss(0, 0.0004) for _ in days]
    rates[100] = 0.01  # an extreme day
    day = carry.daily(_premium(days, [0.0] * 120), _settled(days, rates))
    closes = [100 * (1 + 0.001 * i) for i in range(120)]
    bars = pl.DataFrame({"ticker": "BTC", "ts": [utc(d) for d in days], "close": closes}).with_columns(
        (pl.col("ts") + pl.duration(days=1)).alias("close_ts")
    )
    got = carry.fades(bars, day, _settled(days, rates))
    assert got["trial"].n_unique() == 8
    # The 30-day fade goes short after the extreme day's close, and pays nothing while flat.
    one = got.filter(pl.col("trial") == "fade 30 2 long_short")
    assert one["position"][101] == -1.0
    # gross − cost − net is the funding charged: the held position times that day's settled funding.
    held = one.filter((pl.col("position") != 0) & pl.col("net").is_not_null()).join(day.select("ts", "funding"), on="ts")
    assert held.height > 0
    charged = (held["gross"] - held["cost"] - held["net"]).to_list()
    assert charged == pytest.approx((held["position"] * held["funding"]).to_list())
