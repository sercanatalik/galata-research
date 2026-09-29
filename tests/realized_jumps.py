import math
from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from conftest import utc

import galata_research as gr
from galata_research import Refused

vol = gr.models.vol
FIVE = timedelta(minutes=5)
M = 288
SD = 0.01 / math.sqrt(M)  # a 1% day, spread evenly
T0 = utc("2020-01-01T00:00")


def _bars(returns, *, skip=()):
    """5m bars whose log returns are `returns`, from one leading bar at 23:55 the day before; `skip` drops slots."""
    closes = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(returns)]))
    slots = [i for i in range(closes.size) if i not in skip]
    start = T0 - FIVE
    return pl.DataFrame(
        {
            "ticker": "BTC",
            "ts": [start + i * FIVE for i in slots],
            "close_ts": [start + (i + 1) * FIVE for i in slots],
            "open": closes[slots],
            "high": closes[slots],
            "low": closes[slots],
            "close": closes[slots],
        }
    )


def _days(n, seed=0):
    return np.random.default_rng(seed).normal(0, SD, n * M)


def _whole(frame):
    return frame.drop_nulls("rv")


def the_rv_is_realized_from_s():
    bars = _bars(_days(5))
    ours = vol.realized_jumps(bars, "1d")
    theirs = gr.timeseries.realized_from(bars, "1d")
    joined = ours.join(theirs.select("ts", pl.col("rv").alias("rv_from")), on="ts")
    assert joined["rv"].to_list() == pytest.approx(joined["rv_from"].to_list(), nan_ok=True)
    assert _whole(ours).height == 5


def the_threshold_correction_is_cprs_constant():
    assert vol.jumps.k_gamma(1.0, 3.0) == pytest.approx(1.094, abs=5e-4)
    assert vol.jumps.k_gamma(4 / 3, 3.0) == pytest.approx(1.129, abs=5e-4)


def the_jump_free_days_are_measured_without_bias():
    m = _whole(vol.realized_jumps(_bars(_days(500)), "1d"))
    iv = M * SD**2
    for col in ("bv", "medrv", "ctbv"):
        assert (m[col] / iv).mean() == pytest.approx(1.0, abs=0.02), col


def the_published_asymptotic_variances_hold():
    m = _whole(vol.realized_jumps(_bars(_days(4000, seed=1)), "1d"))
    iv = M * SD**2
    bv = (math.sqrt(M) * (m["bv"] / iv - 1)).var()
    medrv = (math.sqrt(M) * (m["medrv"] / iv - 1)).var()
    assert bv == pytest.approx(math.pi**2 / 4 + math.pi - 3, rel=0.10)  # Barndorff-Nielsen and Shephard: ≈ 2.61
    assert medrv == pytest.approx(2.96, rel=0.10)  # Andersen, Dobrev and Schaumburg, Prop. 2


def a_jump_moves_rv_and_not_the_robust_measures():
    r = _days(1, seed=2)
    before = _whole(vol.realized_jumps(_bars(r), "1d")).row(0, named=True)
    r[140] = 10 * SD
    after = _whole(vol.realized_jumps(_bars(r), "1d")).row(0, named=True)
    rise = after["rv"] - before["rv"]
    assert rise == pytest.approx(100 * SD**2 - (_days(1, seed=2)[140]) ** 2, rel=1e-9)
    # Bipower keeps (π/2)·|J|·(|rⱼ₋₁| + |rⱼ₊₁|), about a quarter of a 10σ jump at M = 288:
    # the finite-sample bias MedRV and the threshold were built against (ADS 2012, §2.2).
    assert after["bv"] - before["bv"] < 0.35 * rise
    for col in ("medrv", "ctbv"):
        assert after[col] - before[col] < 0.1 * rise, col
    assert after["j_bns"] > 0 and after["j_tcj"] > 0


def the_tests_hold_their_size():
    m = _whole(vol.realized_jumps(_bars(_days(2000, seed=3)), "1d", alpha=0.99))
    assert m.height == 2000
    for col in ("j_bns", "j_tcj"):
        assert (m[col] > 0).mean() <= 0.03, col


def the_split_adds_up():
    r = _days(20, seed=4)
    r[::500] += 12 * SD  # some days jump
    m = _whole(vol.realized_jumps(_bars(r), "1d"))
    q = 3.090232306167813  # Φ⁻¹(0.999)
    for c, j, z in (("c_bns", "j_bns", "z_bns"), ("c_tcj", "j_tcj", "z_ctz")):
        assert (m[c] + m[j]).to_list() == pytest.approx(m["rv"].to_list())
        assert (m.filter(pl.col(z) <= q)[j] == 0).all()
        assert (m[j] > 0).any()


def no_bucket_reads_its_neighbour():
    r = _days(3, seed=5)
    base = _whole(vol.realized_jumps(_bars(r), "1d")).row(1, named=True)
    later = r.copy()
    later[2 * M :] *= 50  # every return after the second day
    moved = _whole(vol.realized_jumps(_bars(later), "1d")).row(1, named=True)
    for col, value in base.items():
        assert moved[col] == (pytest.approx(value) if isinstance(value, float) else value), col


def a_missing_bar_nulls_the_bucket():
    m = vol.realized_jumps(_bars(_days(2), skip=(M + 100,)), "1d")
    day2 = m.filter(pl.col("ts") == T0 + timedelta(days=1)).row(0, named=True)
    assert day2["n"] < day2["expected"]
    assert all(day2[c] is None for c in ("rv", "bv", "ctbv", "z_bns", "j_tcj"))


def a_bad_interval_or_alpha_is_refused():
    bars = _bars(_days(1))
    with pytest.raises(Refused, match="coarser"):
        vol.realized_jumps(bars, "1m")
    with pytest.raises(Refused, match="alpha"):
        vol.realized_jumps(bars, "1d", alpha=1.0)
