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
