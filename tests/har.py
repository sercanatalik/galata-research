from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from conftest import utc

import galata_research as gr
from galata_research import Refused

vol = gr.models.vol
DAY = timedelta(days=1)
T0 = utc("2020-01-01T00:00")


def _measures(rv, *, rq=None, width=DAY):
    n = len(rv)
    rv = np.asarray(rv, dtype=float)
    return pl.DataFrame(
        {
            "ticker": "BTC",
            "ts": [T0 + i * width for i in range(n)],
            "close_ts": [T0 + (i + 1) * width for i in range(n)],
            "rv": rv,
            "rs_plus": rv * 0.5,
            "rs_minus": rv * 0.5,
            "rq": np.asarray(rq if rq is not None else (rv * 1.3) ** 2, dtype=float),
        }
    )


def _har_process(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    rv = [1.0] * 30
    for _ in range(n):
        d, w, m = rv[-1], np.mean(rv[-7:]), np.mean(rv[-30:])
        rv.append(max(0.1 + 0.4 * d + 0.3 * w + 0.2 * m + rng.normal(0, 0.05), 0.01))
    return np.array(rv)


def a_har_process_is_recovered():
    rv = _har_process()
    split = T0 + 2500 * DAY
    f = vol.har(_measures(rv), split=split, every=10**6, horizons=[1])
    first = rv.size - f.height
    truth = [0.1 + 0.4 * rv[i] + 0.3 * rv[i - 6 : i + 1].mean() + 0.2 * rv[i - 29 : i + 1].mean() for i in range(first, rv.size)]
    assert f["variance"].to_numpy() == pytest.approx(np.array(truth), rel=0.05)


def the_one_step_cumulative_is_the_one_step_point():
    f = vol.har(_measures(_har_process(800)), split=T0 + 600 * DAY, every=20, horizons=[1])
    assert f["cum_variance"].to_list() == pytest.approx(f["variance"].to_list())


def a_constant_quarticity_leaves_harq_as_har():
    rv = _har_process(800)
    m = _measures(rv, rq=np.full(rv.size, 4.0))
    kw = dict(split=T0 + 600 * DAY, every=20, horizons=[1, 5])
    assert vol.har(m, model="harq", **kw)["variance"].to_list() == pytest.approx(vol.har(m, model="har", **kw)["variance"].to_list())


def no_later_measure_moves_a_refits_forecasts():
    # Guard: training on s + h > r would read RV after the refit's close and move these forecasts.
    rv = _har_process(800)
    r = 600
    later = rv.copy()
    later[r + 1 :] *= 10
    kw = dict(split=T0 + (r + 1) * DAY, every=10**6, horizons=[1, 5])
    a = vol.har(_measures(rv), **kw).filter(pl.col("close_ts") == T0 + (r + 1) * DAY)
    b = vol.har(_measures(later), **kw).filter(pl.col("close_ts") == T0 + (r + 1) * DAY)
    assert b["variance"].to_list() == pytest.approx(a["variance"].to_list())
    assert b["cum_variance"].to_list() == pytest.approx(a["cum_variance"].to_list())


def a_forecast_far_outside_the_range_is_filtered():
    rv = _har_process(800)
    rv[-1] = 500.0  # the last origin's regressors far outside any training row
    f = vol.har(_measures(rv), split=T0 + 700 * DAY, every=10**6, horizons=[1])
    last = f.row(-1, named=True)
    assert last["filtered"]
    # One refit at bucket 699, expanding from bucket 29: targets rv[30..699] (s + 1 ≤ 699).
    assert last["variance"] == pytest.approx(rv[30:700].mean(), rel=1e-9)


def no_origin_sees_its_future():
    f = vol.har(_measures(_har_process(800)), model="shar", split=T0 + 500 * DAY, every=7, horizons=[1, 7])
    assert (f["fitted_through"] <= f["close_ts"]).all()


def an_unknown_width_needs_its_lags():
    m = _measures(_har_process(800), width=timedelta(hours=1))
    with pytest.raises(Refused, match="no default lags"):
        vol.har(m, split=T0 + timedelta(hours=600))
    assert vol.har(m, split=T0 + timedelta(hours=600), lags=(1, 24, 168)).height > 0


def the_weighted_estimator_runs_and_differs():
    m = _measures(_har_process(800))
    kw = dict(split=T0 + 600 * DAY, every=50, horizons=[1])
    ols, wls = vol.har(m, **kw), vol.har(m, estimator="wls", **kw)
    assert wls["variance"].null_count() == 0
    assert wls["variance"].to_list() != pytest.approx(ols["variance"].to_list(), rel=1e-9)
