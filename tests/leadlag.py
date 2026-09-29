"""The shifted Hayashi–Yoshida lead-lag on prices simulated with a known lag."""

import numpy as np
import pytest

from galata_research import Refused
from galata_research.models import leadlag


def _pair(lag_s, n=4000, seed=1, rate_x=5.0, rate_y=2.0):
    """X a random walk on a 1 ms clock over n seconds; Y follows it `lag_s` later, both seen at Poisson times of their own."""
    rng = np.random.default_rng(seed)
    clock = np.arange(0, n, 0.001)
    walk = np.cumsum(rng.normal(0, 1e-4, clock.size))
    tx = np.sort(rng.uniform(0, n, int(n * rate_x)))
    ty = np.sort(rng.uniform(0, n, int(n * rate_y)))
    px = 100 * np.exp(walk[np.searchsorted(clock, tx, "right") - 1])
    # Y at time s shows X's path at s − lag, plus its own noise.
    shifted = np.clip(ty - lag_s, 0, n - 0.001)
    py = 50 * np.exp(walk[np.searchsorted(clock, shifted, "right") - 1] + rng.normal(0, 2e-5, ty.size))
    return tx, px, ty, py


def a_known_lag_is_found():
    tx, px, ty, py = _pair(2.0)
    found = leadlag.estimate(tx, px, ty, py)
    assert found["lag"] == 2.0 and found["llr"] > 1.5 and found["rho"] > found["rho0"]
    assert not found["edge"]


def the_leader_swapped_is_a_negative_lag():
    tx, px, ty, py = _pair(2.0)
    found = leadlag.estimate(ty, py, tx, px)
    assert found["lag"] == -2.0 and found["llr"] < 1 / 1.5


def a_synchronous_pair_peaks_at_zero_whatever_their_activity():
    # X is quoted 5x as often: a previous-tick grid would make it lead.
    tx, px, ty, py = _pair(0.0, rate_x=10.0, rate_y=2.0)
    found = leadlag.estimate(tx, px, ty, py)
    assert found["lag"] == 0.0 and 0.8 < found["llr"] < 1.25


def a_repeat_is_not_a_change():
    t = [0.0, 1.0, 2.0, 3.0]
    assert leadlag._ticks(t, [1.0, 1.0, 2.0, 2.0])[1].tolist() == [1.0, 2.0]


def no_lead_lag_from_too_few_changes():
    with pytest.raises(Refused, match="under 300"):
        leadlag.estimate(np.arange(10.0), np.arange(10.0) + 1, np.arange(10.0), np.arange(10.0) + 1)
