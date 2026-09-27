from math import sqrt

import polars as pl
import pytest

from galata_research import Refused, stats

# Bailey and López de Prado (2014), "A numerical example".
PAPER = {"sr": 2.5 / sqrt(250), "periods": 1250, "skew": -3.0, "kurt": 10.0, "variance": 0.5 / 250}


def the_papers_example_deflates_to_0_9004():
    assert stats.expected_max_sharpe(100, PAPER["variance"]) == pytest.approx(0.1132, abs=5e-5)
    got = stats.dsr(PAPER["sr"], PAPER["periods"], PAPER["skew"], PAPER["kurt"], 100, PAPER["variance"])
    assert got == pytest.approx(0.9004, abs=5e-5)


def the_fewer_the_trials_the_less_the_deflation():
    got = stats.dsr(PAPER["sr"], PAPER["periods"], PAPER["skew"], PAPER["kurt"], 46, PAPER["variance"])
    assert got == pytest.approx(0.9505, abs=5e-5)


def a_single_trial_has_no_selection_to_correct():
    with pytest.raises(Refused, match="trials=1"):
        stats.expected_max_sharpe(1, 0.1)


def a_constant_series_has_no_sharpe():
    assert stats.sharpe([0.001] * 50) is None
    assert stats.sharpe([0.01]) is None


def the_kurtosis_is_raw():
    skew, kurt = stats.moments([1.0, -1.0] * 50)
    assert skew == pytest.approx(0.0) and kurt == pytest.approx(1.0)


def a_failed_trial_still_counts_toward_n():
    summary = pl.DataFrame(
        {
            "trial": [f"t{i}" for i in range(10)],
            "sharpe": [0.05, 0.02, -0.01, 0.03, 0.0, 0.01, 0.04, -0.02, None, None],
            "periods": [900] * 10,
            "skew": [0.0] * 10,
            "kurt": [3.0] * 10,
        }
    )
    got = stats.deflate(summary)
    assert got["trials"] == 10
    assert got["trial"] == "t0"
    assert got["benchmark"] == pytest.approx(stats.expected_max_sharpe(10, summary["sharpe"].drop_nulls().var()))


def the_summary_and_the_statistics_agree():
    # Two routes to the same figures: the polars aggregation and the stats functions.
    import random

    from galata_research import studies

    rng = random.Random(7)
    # Skewed, fat-tailed returns, so the moments are not trivially 0 and 3.
    net = [rng.gauss(0, 0.01) + (0.05 if rng.random() < 0.03 else 0.0) for _ in range(400)]
    frame = pl.DataFrame({"trial": "x", "ticker": "BTC", "ts": range(400), "gross": net, "net": net})
    row = studies.summary(frame).row(0, named=True)
    assert row["sharpe"] == pytest.approx(stats.sharpe(net))
    assert (row["skew"], row["kurt"]) == pytest.approx(stats.moments(net))


def an_identical_return_costs_nothing():
    r = [0.01, -0.02, 0.005, 0.0]
    assert stats.performance_fee(r, r, 1.0, 365) == pytest.approx(0.0, abs=1e-9)


def the_fee_equates_utility():
    import random

    rng = random.Random(1)
    b = [rng.gauss(0.0005, 0.02) for _ in range(500)]
    r = [x * 0.8 + rng.gauss(0.0002, 0.005) for x in b]
    for gamma in (1.0, 10.0):
        fee = stats.performance_fee(r, b, gamma, 365) / (365 * 1e4)
        c = gamma / (2 * (1 + gamma))
        assert sum((x - fee) - c * (x - fee) ** 2 for x in r) == pytest.approx(sum(x - c * x * x for x in b), abs=1e-10)


def a_constant_uplift_is_its_fee_at_low_risk_aversion():
    b = [0.001, -0.002, 0.003] * 100
    r = [x + 0.0001 for x in b]
    assert stats.performance_fee(r, b, 1e-9, 365) == pytest.approx(0.0001 * 365 * 1e4, rel=1e-6)


def a_fall_and_a_partial_recovery():
    assert stats.max_drawdown([0.10, -0.20, 0.05]) == pytest.approx(0.2)
    assert stats.max_drawdown([0.01, 0.02]) == 0.0
