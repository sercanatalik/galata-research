import math
import random
from datetime import timedelta

import polars as pl
import pytest
from conftest import utc

from galata_research import Refused, timeseries

DAY = timedelta(days=1)
MINUTE = timedelta(minutes=1)
LN2 = math.log(2)


def _bars(rows, *, width=DAY, skip=(), in_gap=None, start="2026-01-01T00:00"):
    """`rows` are (open, high, low, close); `skip` leaves those slots empty, making holes."""
    t0 = utc(start)
    slots = [s for s in range(len(rows) + len(skip)) if s not in skip][: len(rows)]
    frame = pl.DataFrame(
        {
            "ticker": "BTC",
            "ts": [t0 + s * width for s in slots],
            "close_ts": [t0 + (s + 1) * width for s in slots],
            "open": [float(r[0]) for r in rows],
            "high": [float(r[1]) for r in rows],
            "low": [float(r[2]) for r in rows],
            "close": [float(r[3]) for r in rows],
        }
    )
    return frame if in_gap is None else frame.with_columns(pl.Series("in_gap", in_gap))


def _closes(closes, **kw):
    return _bars([(c, c, c, c) for c in closes], **kw)


def _last(frame):
    return frame["sigma"].to_list()[-1]


def the_parkinson_arithmetic():
    rows = [(100, 100, 100, 100), (100, 102, 100, 101), (101, 101 * 1.01, 101, 101), (101, 103, 100, 102)]
    got = _last(timeseries.realized(_bars(rows), "parkinson", 3))
    terms = [math.log(h / low) ** 2 / (4 * LN2) for _, h, low, _ in rows[1:]]
    assert got == pytest.approx(math.sqrt(sum(terms) / 3))


def the_garman_klass_and_rogers_satchell_arithmetic():
    rows = [(100, 100, 100, 100), (100, 110, 95, 105)]
    gk = _last(timeseries.realized(_bars(rows), "garman_klass", 2, min_periods=1))
    rs = _last(timeseries.realized(_bars(rows), "rogers_satchell", 2, min_periods=1))
    assert gk == pytest.approx(math.sqrt(0.5 * math.log(110 / 95) ** 2 - (2 * LN2 - 1) * math.log(105 / 100) ** 2))
    assert rs == pytest.approx(math.sqrt(math.log(110 / 105) * math.log(110 / 100) + math.log(95 / 105) * math.log(95 / 100)))


def the_yang_zhang_reduces_without_an_opening_jump():
    rows = [(100, 101, 99, 100), (100, 103, 99, 102), (102, 104, 100, 101), (101, 105, 100, 104), (104, 106, 101, 103)]
    got = _last(timeseries.realized(_bars(rows), "yang_zhang", 4))
    c = [math.log(r[3] / r[0]) for r in rows[1:]]
    rs = [math.log(h / cl) * math.log(h / o) + math.log(low / cl) * math.log(low / o) for o, h, low, cl in rows[1:]]
    n = 4
    mean_c = sum(c) / n
    var_c = sum((x - mean_c) ** 2 for x in c) / (n - 1)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    assert got == pytest.approx(math.sqrt(k * var_c + (1 - k) * sum(rs) / n))


def the_close_to_close_is_the_sample_deviation_of_log_returns():
    closes = [100, 101, 99, 102, 103]
    got = _last(timeseries.realized(_closes(closes), "close_to_close", 4))
    r = [math.log(b / a) for a, b in zip(closes, closes[1:])]
    m = sum(r) / 4
    assert got == pytest.approx(math.sqrt(sum((x - m) ** 2 for x in r) / 3))


def an_unknown_estimator_is_refused():
    with pytest.raises(Refused, match="hodges_tompkins.*close_to_close, parkinson"):
        timeseries.realized(_closes([1, 2, 3]), "hodges_tompkins", 2)


def the_first_bar_and_the_bar_after_a_hole_count_for_nothing():
    # Guard: without the count rule the first bar's range and the post-hole bar would both count.
    rows = [(100, 101, 99, 100)] * 5
    for est in timeseries.ESTIMATORS:
        got = timeseries.realized(_bars(rows, skip=(3,)), est, 5, min_periods=1)
        assert got["n"].to_list()[-1] == 3, est


def a_bar_in_a_gap_is_left_out():
    rows = [(100, 101, 99, 100)] * 4
    plain = timeseries.realized(_bars(rows), "parkinson", 4, min_periods=1)
    gapped = timeseries.realized(_bars(rows, in_gap=[False, False, True, False]), "parkinson", 4, min_periods=1)
    assert gapped["n"].to_list()[-1] == plain["n"].to_list()[-1] - 1


def a_partial_window_has_no_figure_by_default():
    got = timeseries.realized(_closes([100 + i for i in range(20)]), "close_to_close", 20)
    assert got["n"].to_list()[-1] == 19 and _last(got) is None


def the_estimators_measure_a_known_volatility():
    # 2,000 bars of Brownian log price, 500 steps each, σ = 0.01 per bar, zero drift.
    rng, sigma, steps = random.Random(7), 0.01, 500
    rows, price = [], 0.0
    for _ in range(2001):
        path = [price]
        for _ in range(steps):
            path.append(path[-1] + rng.gauss(0, sigma / math.sqrt(steps)))
        rows.append(tuple(math.exp(x) for x in (path[0], max(path), min(path), path[-1])))
        price = path[-1]
    bars = _bars(rows)
    per_bar = bars.select(
        timeseries.parkinson_term().alias("parkinson"),
        timeseries.garman_klass_term().alias("garman_klass"),
        timeseries.rogers_satchell_term().alias("rogers_satchell"),
        (pl.col("close") / pl.col("open")).log().pow(2).alias("r2"),
    ).slice(1)
    for est in ("parkinson", "garman_klass", "rogers_satchell"):
        mean = per_bar[est].mean()
        assert 0.9 * sigma**2 < mean < 1.02 * sigma**2, (est, mean)
    assert per_bar["r2"].var() / per_bar["parkinson"].var() > 3


def a_complete_hour_from_minutes():
    t0 = utc("2026-01-01T00:59")
    closes = [math.exp(0.001 * i) for i in range(62)]
    fine = pl.DataFrame(
        {
            "ticker": "BTC",
            "ts": [t0 + i * MINUTE for i in range(62)],
            "close_ts": [t0 + (i + 1) * MINUTE for i in range(62)],
            "open": closes,
            "high": [c * 1.0001 for c in closes],
            "low": closes,
            "close": closes,
        }
    )
    got = timeseries.realized_from(fine, "1h").filter(pl.col("ts") == utc("2026-01-01T01:00"))
    assert got["n"].item() == got["expected"].item() == 60
    assert got["rv"].item() == pytest.approx(60 * 1e-6)
    assert got["rr"].item() == pytest.approx(60 * math.log(1.0001) ** 2 / (4 * LN2))


def a_missing_minute_leaves_the_hour_blank():
    t0 = utc("2026-01-01T00:59")
    slots = [i for i in range(62) if i != 30]
    fine = pl.DataFrame(
        {
            "ticker": "BTC",
            "ts": [t0 + i * MINUTE for i in slots],
            "close_ts": [t0 + (i + 1) * MINUTE for i in slots],
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
        }
    )
    got = timeseries.realized_from(fine, "1h").filter(pl.col("ts") == utc("2026-01-01T01:00"))
    assert got["n"].item() < got["expected"].item()
    assert got["rv"].item() is None and got["rr"].item() is None


def a_finer_interval_is_refused():
    with pytest.raises(Refused, match="coarser"):
        timeseries.realized_from(_closes([1, 2, 3]), "1h")


def the_ewma_recursion():
    closes = [1.0, math.exp(0.02), math.exp(0.03), math.exp(0.06)]
    got = timeseries.ewma_vol(_closes(closes), lam=0.9, warmup=0)["sigma"].to_list()
    s1 = 4e-4
    s2 = 0.9 * s1 + 0.1 * 1e-4
    s3 = 0.9 * s2 + 0.1 * 9e-4
    assert got[0] is None
    assert [x**2 for x in got[1:]] == [pytest.approx(s1), pytest.approx(s2), pytest.approx(s3)]


def the_default_warm_up():
    got = timeseries.ewma_vol(_closes([100 * (1.01 if i % 2 else 1) for i in range(100)]))
    assert got["sigma"].head(76).null_count() == 76  # the first bar has no return, then 75 returns
    assert got["sigma"][76] is not None


def the_max_rises_fast_and_falls_slow():
    closes, p = [], 100.0
    for i in range(600):
        p *= 1.001 if i % 2 else 1 / 1.001
        if 300 <= i < 305:
            p *= 1.05 if i % 2 else 1 / 1.05
        closes.append(p)
    b = _closes(closes)
    mx = timeseries.ewma_max(b)["sigma"]
    fast = timeseries.ewma_vol(b, lam=0.94)["sigma"]
    slow = timeseries.ewma_vol(b, lam=0.97)["sigma"]
    assert mx[306] == pytest.approx(fast[306]) and fast[306] > slow[306]
    assert mx[420] == pytest.approx(slow[420]) and slow[420] > fast[420]


def a_decay_outside_the_unit_interval_is_refused():
    with pytest.raises(Refused, match="lam=1"):
        timeseries.ewma_vol(_closes([1, 2]), lam=1)


def the_signature_of_a_constant_drift():
    t0 = utc("2026-01-01T00:00")
    n = 2 * 1440 + 1
    closes = [math.exp(1e-4 * (i + 1)) for i in range(n)]
    bars = pl.DataFrame(
        {
            "ticker": "BTC",
            "ts": [t0 - MINUTE + i * MINUTE for i in range(n)],
            "close_ts": [t0 + i * MINUTE for i in range(n)],
            "close": closes,
        }
    )
    got = timeseries.signature(bars, [1, 5])
    assert got["days"].to_list() == [2, 2]
    assert got["mean_rv"].to_list() == [pytest.approx(1440 * 1e-8), pytest.approx(288 * (5e-4) ** 2)]


def the_signature_refuses_hourly_bars():
    with pytest.raises(Refused, match="1m"):
        timeseries.signature(_closes([1, 2, 3], width=timedelta(hours=1)), [5])


def the_semivariances_add_up_and_the_quarticity_is_the_fourth_moment():
    t0 = utc("2026-01-01T00:59")
    steps = [0.0, 0.01, -0.02, 0.03, -0.01] + [0.0] * 57
    logp = [sum(steps[: i + 1]) for i in range(62)]
    closes = [math.exp(x) for x in logp]
    fine = pl.DataFrame(
        {
            "ticker": "BTC",
            "ts": [t0 + i * MINUTE for i in range(62)],
            "close_ts": [t0 + (i + 1) * MINUTE for i in range(62)],
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
        }
    )
    got = timeseries.realized_from(fine, "1h").filter(pl.col("ts") == utc("2026-01-01T01:00")).row(0, named=True)
    assert got["rs_plus"] == pytest.approx(1e-4 + 9e-4)
    assert got["rs_minus"] == pytest.approx(4e-4 + 1e-4)
    assert got["rs_plus"] + got["rs_minus"] == pytest.approx(got["rv"])
    assert got["rq"] == pytest.approx(60 / 3 * (1e-8 + 16e-8 + 81e-8 + 1e-8))
