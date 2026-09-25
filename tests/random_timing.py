import random
from datetime import timedelta

import polars as pl
import pytest
from conftest import utc

from galata_research import backtest, stats, studies
from galata_research.studies import _replay, _runs


def _frame(positions, returns, trial="t", ticker="BTC"):
    t0 = utc("2026-01-01T00:00")
    return pl.DataFrame(
        {
            "trial": trial,
            "ticker": ticker,
            "ts": [t0 + timedelta(days=d) for d in range(len(positions))],
            "position": [float(p) for p in positions],
            "bar_return": [float(r) if r is not None else None for r in returns],
            "gross": None,
            "net": None,
        }
    )


def the_gross_is_position_times_bar_return():
    t0 = utc("2026-01-01T00:00")
    bars = pl.DataFrame(
        {
            "ticker": "BTC",
            "ts": [t0 + timedelta(days=d) for d in range(5)],
            "close_ts": [t0 + timedelta(days=d + 1) for d in range(5)],
            "close": [100.0, 102.0, 101.0, 105.0, 104.0],
        }
    )
    got = backtest.returns(bars, pl.lit(0.5), fee=0.0).drop_nulls("gross")
    assert got["gross"].to_list() == pytest.approx((got["position"] * got["bar_return"]).to_list())


def a_value_above_every_draw_is_one_in_n_plus_one():
    assert stats.percentile(2.0, [1.0] * 999) == pytest.approx(1 / 1000)
    assert stats.percentile(0.0, [1.0, None, -1.0]) == pytest.approx(2 / 3)


def a_shuffle_keeps_the_runs_and_the_exposure():
    positions = [1] * 5 + [0] * 3 + [-1] * 7
    runs = _runs(positions)
    assert runs == [(1, 5), (0, 3), (-1, 7)]
    for k in range(20):
        order = runs[:]
        random.Random(k).shuffle(order)
        path = [v for v, n in order for _ in range(n)]
        assert sorted(path) == sorted(positions)
        assert sum(a != b for a, b in zip(path, path[1:])) <= 2


def the_same_seed_reproduces_the_same_distribution():
    rng = random.Random(1)
    frame = _frame([1 if rng.random() > 0.5 else 0 for _ in range(200)], [rng.gauss(0, 0.01) for _ in range(200)])
    first = studies.random_timing(frame, samples=200, seed=7)
    second = studies.random_timing(frame, samples=200, seed=7)
    assert first.equals(second)


def the_perfect_timing_beats_its_twins():
    rng = random.Random(3)
    rets = [rng.gauss(0, 0.01) for _ in range(300)]
    positions = [1 if r > 0 else 0 for r in rets]
    got = studies.random_timing(_frame(positions, rets), samples=500, fee=0.0).row(0, named=True)
    assert got["percentile"] <= 2 / 501
    assert got["null"] == studies.RANDOM_TIMING_NULL


def the_observed_sharpe_is_the_replay_of_the_unshuffled_path():
    # One code path for observed and random: the observed must equal the backtest's own net Sharpe.
    t0 = utc("2026-01-01T00:00")
    rng = random.Random(9)
    closes = [100.0]
    for _ in range(199):
        closes.append(closes[-1] * (1 + rng.gauss(0, 0.02)))
    bars = pl.DataFrame(
        {"ticker": "BTC", "ts": [t0 + timedelta(days=d) for d in range(200)], "close_ts": [t0 + timedelta(days=d + 1) for d in range(200)], "close": closes}
    )
    trial = studies.momentum(bars, [5])
    got = studies.random_timing(trial, samples=50).row(0, named=True)
    assert got["observed"] == pytest.approx(stats.sharpe(trial["net"]))


def no_null_bar_is_replayed():
    got = studies.random_timing(_frame([1, 1, 0, 1, 0], [0.01, None, 0.02, -0.01, 0.03]), samples=10).row(0, named=True)
    # The null bar is dropped before the runs are cut: 1 | 0 | 1 | 0 over four bars.
    assert got["runs"] == 4


def the_replay_charges_turnover():
    assert _replay([1, 1, -1], [0.0, 0.0, 0.0], fee=0.001) == pytest.approx([-0.001, 0.0, -0.002])
