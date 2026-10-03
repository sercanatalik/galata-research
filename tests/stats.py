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


# Rollinger and Hoffman (Red Rock Capital), "Sortino: a 'sharper' ratio", the worked example.
RED_ROCK = [0.17, 0.15, 0.23, -0.05, 0.12, 0.09, 0.13, -0.04]


def the_red_rock_example_gives_a_sortino_of_4_417():
    assert stats.sortino(RED_ROCK) == pytest.approx(4.417, abs=5e-4)


def the_downside_deviation_divides_by_every_period_not_only_the_losers():
    # Guard: over the two losers alone it would be √(41/2)% and the ratio 2.21.
    assert stats.sortino(RED_ROCK) > 4


def a_series_that_never_loses_has_no_sortino():
    assert stats.sortino([0.01, 0.02, 0.0]) is None


def the_omega_is_gains_over_losses_about_the_threshold():
    assert stats.omega([0.02, -0.01, 0.03, -0.02]) == pytest.approx(0.05 / 0.03)
    assert stats.omega([0.02, -0.01, 0.03, -0.02], threshold=0.01) == pytest.approx(0.03 / 0.05)
    assert stats.omega([0.01, 0.02]) is None


def the_cagr_compounds_to_a_year():
    assert stats.cagr([0.1, 0.1], periods_per_year=1) == pytest.approx(0.1)
    assert stats.cagr([0.1, 0.1], periods_per_year=2) == pytest.approx(0.21)
    assert stats.cagr([-1.0, 0.5], periods_per_year=2) == -1.0


def the_calmar_is_cagr_over_the_worst_drawdown():
    # 1.1, 0.55, 1.1: growth 10% over a year of three periods, a 50% drawdown.
    assert stats.calmar([0.1, -0.5, 1.0], periods_per_year=3) == pytest.approx(0.2)
    assert stats.calmar([0.1, 0.2], periods_per_year=2) is None


def the_ulcer_index_squares_each_periods_drawdown():
    # Drawdowns 0, 50%, 0.
    assert stats.ulcer_index([0.1, -0.5, 1.0]) == pytest.approx((0.25 / 3) ** 0.5)
    assert stats.ulcer_index([]) is None


def a_drawdown_lasts_until_a_new_peak():
    # Values 1.1, 0.99, 1.0395, 1.14345 (a new peak), 0.91476.
    assert stats.drawdown_duration([0.1, -0.1, 0.05, 0.1, -0.2]) == 2
    # A drawdown never recovered counts to the end.
    assert stats.drawdown_duration([0.1, -0.1, 0.05, 0.01]) == 3
    assert stats.drawdown_duration([0.01, 0.02]) == 0


def _frame(nets, *, start="2026-01-01T00:00", step_days=1, trial="t", ticker="BTC"):
    from datetime import datetime, timedelta, timezone

    t0 = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    step = timedelta(days=step_days)
    return pl.DataFrame(
        {
            "trial": trial,
            "ticker": ticker,
            "ts": [t0 + i * step for i in range(len(nets))],
            "close_ts": [t0 + (i + 1) * step for i in range(len(nets))],
            "net": pl.Series(nets, dtype=pl.Float64),
        }
    )


def a_rolling_sharpe_needs_a_full_window():
    got = stats.rolling_sharpe(_frame([0.01, 0.03, None, 0.02, 0.04, 0.0]), 2)["sharpe"].to_list()
    # A window holding the hole is null, never a window of one.
    assert got[:4] == [None, pytest.approx(0.02 / 0.0002**0.5), None, None]
    assert got[4] == pytest.approx(0.03 / 0.0002**0.5)


def a_window_with_no_variance_has_no_rolling_sharpe():
    assert stats.rolling_sharpe(_frame([0.01, 0.01, 0.01]), 2)["sharpe"].to_list() == [None, None, None]


def a_bar_belongs_to_the_month_it_opens_in():
    # Daily bars opening Jan 30, 31, Feb 1, 2: the bar closing at midnight on Feb 1 is January's.
    got = stats.period_returns(_frame([0.1, 0.1, 0.5, None], start="2026-01-30T00:00"))
    assert got["return"].to_list() == [pytest.approx(0.21), pytest.approx(0.5)]
    assert got["n"].to_list() == [2, 1]
    assert got["bars"].to_list() == [31, 28]
    assert got["full"].to_list() == [False, False]


def a_whole_month_is_full():
    got = stats.period_returns(_frame([0.0] * 28, start="2026-02-01T00:00"))
    assert got["full"].to_list() == [True]


def every_trial_and_ticker_is_described_once_one_row_per_trial_and_ticker():
    frame = pl.concat([_frame(RED_ROCK, trial="a"), _frame([0.01, -0.02, 0.03], trial="b")])
    got = stats.describe(frame, periods_per_year=1)
    assert got["trial"].to_list() == ["a", "b"]
    row = got.row(0, named=True)
    assert row["sortino"] == pytest.approx(4.417, abs=5e-4)
    assert row["periods"] == 8
    assert row["max_drawdown"] == pytest.approx(stats.max_drawdown(RED_ROCK))
