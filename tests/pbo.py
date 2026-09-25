import random
from datetime import timedelta

import polars as pl
import pytest
from conftest import utc

from galata_research import Refused, stats, studies


def _noise(columns: int, rows: int, seed: int = 11) -> pl.DataFrame:
    rng = random.Random(seed)
    return pl.DataFrame({f"s{j}": [rng.gauss(0, 0.01) for _ in range(rows)] for j in range(columns)})


def _alternating(rows: int, rng: random.Random, good: float) -> list[float]:
    return [rng.gauss(good, 0.01) for _ in range(rows)]


def the_combination_count_is_16_choose_8():
    # The paper's text says 12,780; C(16, 8) is 12,870.
    got = stats.pbo(_noise(3, 64), blocks=16)
    assert got["combinations"].height == 12_870


def a_winner_that_always_loses_out_of_sample_overfits():
    rng = random.Random(3)
    a = _alternating(50, rng, 0.02) + _alternating(50, rng, -0.02)
    b = _alternating(50, rng, -0.02) + _alternating(50, rng, 0.02)
    got = stats.pbo(pl.DataFrame({"a": a, "b": b}), blocks=2)
    assert got["pbo"] == 1.0
    assert got["prob_loss"] == 1.0


def a_consistent_winner_does_not_overfit():
    rng = random.Random(5)
    frame = _noise(9, 160).with_columns(pl.Series("edge", [rng.gauss(0.02, 0.01) for _ in range(160)]))
    assert stats.pbo(frame, blocks=8)["pbo"] == 0.0


def the_noise_overfits_about_half_the_time():
    assert 0.3 <= stats.pbo(_noise(20, 400), blocks=10)["pbo"] <= 0.7


def every_combination_reports_its_degradation():
    got = stats.pbo(_noise(6, 80), blocks=4)
    assert got["combinations"].columns == ["combination", "best", "is_sharpe", "oos_sharpe", "rank", "logit"]
    assert got["combinations"].height == 6
    assert got["slope"] is not None and got["prob_loss"] is not None


def the_pooled_sharpe_matches_a_direct_one():
    # The training Sharpe comes from block sums; it must equal the Sharpe of the rows themselves.
    frame = _noise(3, 40)
    got = stats.pbo(frame, blocks=2)["combinations"].row(0, named=True)
    first_half = frame[got["best"]].head(20)
    second_half = frame[got["best"]].tail(20)
    assert got["is_sharpe"] == pytest.approx(stats.sharpe(first_half))
    assert got["oos_sharpe"] == pytest.approx(stats.sharpe(second_half))


def an_exact_median_counts_as_overfit():
    # Guard: with three identical columns every rank is the median (λ = 0), which is not outperforming it.
    same = _noise(1, 40)["s0"]
    frame = pl.DataFrame({"a": same, "b": same, "c": same})
    got = stats.pbo(frame, blocks=2)
    assert got["pbo"] == 1.0
    # Guard: tied ranks are averaged, so a three-way tie is the median exactly, not the bottom.
    assert got["combinations"]["logit"].to_list() == [0.0, 0.0]


def the_remainder_rows_are_dropped_from_the_start():
    got = stats.pbo(_noise(3, 67), blocks=4)
    assert (got["dropped"], got["rows_per_block"]) == (3, 16)


def a_null_is_refused_naming_its_column():
    frame = _noise(3, 40).with_columns(pl.when(pl.int_range(pl.len()) == 5).then(None).otherwise(pl.col("s1")).alias("s1"))
    with pytest.raises(Refused, match="nulls in s1"):
        stats.pbo(frame, blocks=2)


def an_odd_block_count_is_refused():
    with pytest.raises(Refused, match="blocks=3"):
        stats.pbo(_noise(3, 40), blocks=3)


def a_warm_up_is_not_compared():
    t0 = utc("2026-01-01T00:00")
    rows = []
    for d in range(10):
        rows.append({"trial": "fast", "ticker": "BTC", "ts": t0 + timedelta(days=d), "net": 0.01})
        rows.append({"trial": "slow", "ticker": "BTC", "ts": t0 + timedelta(days=d), "net": 0.02 if d >= 4 else None})
    got = studies.matrix(pl.DataFrame(rows))
    assert got.columns == ["ts", "fast | BTC", "slow | BTC"]
    assert got["ts"].min() == t0 + timedelta(days=4)
