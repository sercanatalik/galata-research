"""gr.models.memory: the local Whittle estimate and Qu's test, checked on simulations before any real series.

The size, power and recovery checks are the validation that
`planning/preregistered/long-memory-or-level-shifts.md` requires before its run.
"""

import numpy as np
import pytest

from galata_research import Refused
from galata_research.models import memory

T, DRAWS = 5000, 200


def _fractional_noise(d: float, n: int, rng) -> np.ndarray:
    """(1 − L)^{−d} e_t by its MA(∞) weights, truncated at 2n, with the first n values discarded."""
    k = np.arange(1, 2 * n)
    psi = np.concatenate([[1.0], np.cumprod((k - 1 + d) / k)])
    e = rng.standard_normal(3 * n)
    return np.convolve(e, psi, mode="full")[2 * n : 3 * n]


def _level_shifts(n: int, rng, *, prob: float, sd_shift: float) -> np.ndarray:
    """White noise plus a mean that jumps by N(0, sd_shift²) with probability `prob` each step (LongMemoryTS's example)."""
    shifts = np.where(rng.random(n) < prob, rng.normal(0, sd_shift, n), 0.0)
    return rng.standard_normal(n) + np.cumsum(shifts)


def the_local_whittle_estimate_recovers_d():
    rng = np.random.default_rng(0)
    d = [memory.local_whittle(_fractional_noise(0.3, T, rng))["d"] for _ in range(50)]
    assert abs(np.mean(d) - 0.3) < 0.1


def the_qu_test_keeps_its_size_on_true_long_memory():
    rng = np.random.default_rng(1)
    rejected = sum(memory.qu_test(_fractional_noise(0.3, T, rng))["reject_5pct"] for _ in range(DRAWS))
    assert rejected / DRAWS <= 0.10


def the_qu_test_has_power_against_level_shifts():
    rng = np.random.default_rng(2)
    rejected = sum(memory.qu_test(_level_shifts(T, rng, prob=5 / T, sd_shift=0.75))["reject_5pct"] for _ in range(DRAWS))
    assert rejected / DRAWS >= 0.5


def an_unpublished_trimming_is_refused():
    with pytest.raises(Refused, match="no published critical values"):
        memory.qu_test(np.ones(100), epsilon=0.1)


def a_series_with_a_gap_is_refused():
    with pytest.raises(Refused, match="no nulls"):
        memory.local_whittle(np.array([1.0, np.nan] * 200))
