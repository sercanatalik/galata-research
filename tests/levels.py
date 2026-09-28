"""gr.models.levels: the random level shift model, checked on its own simulation before any real series.

The recovery and tracking checks are the validation that
`planning/preregistered/random-level-shift-forecasts.md` requires before its run.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from galata_research import Refused
from galata_research.models import levels


def _simulate(n=5000, p=0.01, s_eta=1.0, s_c=0.5, a=-4.0, seed=0):
    rng = np.random.default_rng(seed)
    shifts = np.where(rng.random(n) < p, rng.normal(0, s_eta, n), 0.0)
    tau = np.cumsum(shifts)
    return a + tau + rng.normal(0, s_c, n), a + tau


def the_parameters_are_recovered_within_a_factor_of_two():
    y, _ = _simulate()
    f = levels.fit(y)
    assert 0.005 <= f["p"] <= 0.02
    assert 0.5 <= f["sigma_eta"] <= 2.0
    assert abs(f["sigma_c"] - 0.5) < 0.1


def the_filtered_level_tracks_the_true_level():
    y, truth = _simulate(seed=1)
    f = levels.fit(y)
    _, _, filtered = levels._filter(list(y), f["a"], f["p"], f["sigma_eta"] ** 2, f["sigma_c"] ** 2)
    assert np.corrcoef(np.array(filtered) + f["a"], truth)[0, 1] > 0.9


def a_zero_return_is_predicted_through():
    y, _ = _simulate(n=600, seed=2)
    ys = list(y)
    ys[300] = None
    loglik, predicted, filtered = levels._filter(ys, -4.0, 0.01, 1.0, 0.25)
    assert np.isfinite(loglik) and filtered[300] == predicted[300]


def the_walk_never_fits_on_its_future():
    y, _ = _simulate(n=900, seed=3)
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    r = np.exp(y) * np.where(np.random.default_rng(4).random(900) < 0.5, 1, -1)
    frame = pl.DataFrame({"ticker": "X", "ts": [t0 + timedelta(days=i) for i in range(900)], "close_ts": [t0 + timedelta(days=i + 1) for i in range(900)], "return": r})
    out = levels.walk_forward(frame, split=t0 + timedelta(days=700), every=30, horizons=(1, 7))
    assert (out["fitted_through"] <= out["close_ts"]).all()
    one = out.filter(pl.col("h") == 7)
    assert np.allclose(one["cum_variance"], 7 * one["variance"])


def a_fit_on_too_few_observations_is_refused():
    with pytest.raises(Refused, match="at least 250"):
        levels.fit([0.0] * 100)
