from datetime import timedelta
from math import pi, sqrt

import numpy as np
import polars as pl
import pytest
from conftest import utc

import galata_research as gr
from galata_research import Refused
from galata_research.models.vol import custom

vol = gr.models.vol
DAY = timedelta(days=1)
T0 = utc("2020-01-01T00:00")


def _returns(values):
    n = len(values)
    return pl.DataFrame(
        {"ticker": "BTC", "ts": [T0 + i * DAY for i in range(n)], "close_ts": [T0 + (i + 1) * DAY for i in range(n)], "return": values},
        schema_overrides={"return": pl.Float64},
    )


def _std_t(rng, nu, n):
    return rng.standard_t(nu, n) * sqrt((nu - 2) / nu)


def a_constant_long_run_level_is_garch():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(500) * 2
    p = {"mu": 0.0, "omega": 4.0, "alpha": 0.1, "beta": 0.8, "rho": 0.97, "phi": 0.0, "nu": 6.0}
    s, q = custom.cgarch_filter(p, y, init=4.0)
    assert np.allclose(q, 4.0)
    g = np.empty(501)
    g[0] = 4.0
    for t in range(500):
        g[t + 1] = 4.0 * (1 - 0.1 - 0.8) + 0.1 * y[t] ** 2 + 0.8 * g[t]
    assert s == pytest.approx(g)


def _simulate_cgarch(n=6000, seed=2):
    rng = np.random.default_rng(seed)
    w, a, b, rho, phi, nu = 1.0, 0.08, 0.8, 0.99, 0.03, 6.0
    z = _std_t(rng, nu, n + 500)
    s = q = w
    y = np.empty(n + 500)
    for t in range(n + 500):
        y[t] = sqrt(s) * z[t]
        q, s = w + rho * (q - w) + phi * (y[t] ** 2 - s), (w + rho * (q - w) + phi * (y[t] ** 2 - s)) + a * (y[t] ** 2 - q) + b * (s - q)
    return y[500:]


def a_simulated_component_garch_is_recovered():
    p = custom.estimate("cgarch", _simulate_cgarch())["params"]
    assert p["alpha"] + p["beta"] == pytest.approx(0.88, abs=0.06)
    assert p["rho"] == pytest.approx(0.99, abs=0.02)
    assert p["nu"] == pytest.approx(6.0, abs=2.0)


def a_non_t_component_garch_is_refused():
    with pytest.raises(Refused, match="Student-t"):
        vol.fit(_returns((_simulate_cgarch(600) / 100).tolist()), model="cgarch", dist="normal")


def an_outlier_moves_lambda_by_a_bounded_amount():
    p = {"mu": 0.0, "omega": 0.0, "phi": 0.0, "kappa": 0.1, "kappa_star": 0.05, "nu": 5.0}
    lam = custom.betat_filter(p, np.array([-1e6]))
    assert abs(lam[1] - lam[0]) <= 0.1 * 5 + 0.05 * 6 + 1e-9


def _simulate_betat(n=6000, seed=3):
    rng = np.random.default_rng(seed)
    w, phi, k, nu = 0.02, 0.95, 0.06, 5.0
    lam = w / (1 - phi)
    y = np.empty(n)
    for t in range(n):
        y[t] = np.exp(lam) * rng.standard_t(nu)
        u = (nu + 1) * y[t] ** 2 / (nu * np.exp(2 * lam) + y[t] ** 2) - 1
        lam = w + phi * lam + k * u
    return y


def a_simulated_beta_t_egarch_is_recovered():
    p = custom.estimate("betat", _simulate_betat())["params"]
    assert p["phi"] == pytest.approx(0.95, abs=0.04)
    assert p["kappa"] == pytest.approx(0.06, abs=0.03)


def a_simulated_beta_t_forecast_reproduces():
    r = _returns((_simulate_betat(900) / 100).tolist())
    kw = dict(model="betat", split=r["close_ts"][699], every=100, horizons=[1, 5], simulations=200, seed=4, min_obs=500)
    a, b = vol.walk_forward(r, **kw), vol.walk_forward(r, **kw)
    assert a.equals(b)


def the_range_constant():
    rng = np.random.default_rng(5)
    steps = rng.standard_normal((20_000, 1000)) / sqrt(1000)
    paths = np.cumsum(steps, axis=1)
    paths = np.concatenate([np.zeros((20_000, 1)), paths], axis=1)
    mean_range = float((paths.max(axis=1) - paths.min(axis=1)).mean())
    assert 0.97 * sqrt(8 / pi) < mean_range < sqrt(8 / pi)
    assert custom.RANGE_MEAN == pytest.approx(sqrt(8 / pi))


def _carr_bars(n=6000, seed=6, *, scale=1.0):
    rng = np.random.default_rng(seed)
    w, a, b = 0.05, 0.15, 0.8
    lam, prev = w / (1 - a - b), w / (1 - a - b)
    ranges = np.empty(n)
    for t in range(n):
        lam = w + a * prev + b * lam
        ranges[t] = lam * rng.exponential()
        prev = ranges[t]
    ranges = ranges * scale / 100  # the fit is on ×100
    low = np.full(n, 100.0)
    return pl.DataFrame(
        {
            "ticker": "BTC",
            "ts": [T0 + i * DAY for i in range(n)],
            "close_ts": [T0 + (i + 1) * DAY for i in range(n)],
            "high": low * np.exp(ranges),
            "low": low,
        }
    )


def a_simulated_carr_is_recovered():
    bars = _carr_bars()
    r = (bars["high"] / bars["low"]).log().to_numpy() * 100
    p = custom.estimate("carr", r)["params"]
    assert p["alpha"] == pytest.approx(0.15, abs=0.04)
    assert p["beta"] == pytest.approx(0.8, abs=0.06)


def no_carr_origin_sees_its_future():
    bars = _carr_bars(900)
    at = bars["close_ts"][699]
    f = vol.carr(bars, split=at, every=1000, horizons=[1, 3])
    assert (f["fitted_through"] <= f["close_ts"]).all()
    later = bars.with_columns(pl.when(pl.col("ts") >= at).then(pl.col("low") * (pl.col("high") / pl.col("low")) ** 2).otherwise(pl.col("high")).alias("high"))
    g = vol.carr(later, split=at, every=1000, horizons=[1, 3])
    first = lambda x: x.filter(pl.col("close_ts") == at)["variance"].to_list()  # noqa: E731
    assert first(g) == pytest.approx(first(f))


def a_high_below_low_is_refused():
    bars = _carr_bars(600).with_columns(pl.when(pl.int_range(pl.len()) == 10).then(50.0).otherwise(pl.col("high")).alias("high"))
    with pytest.raises(Refused, match="high below the low"):
        vol.carr(bars, split=bars["close_ts"][550])


def the_custom_fits_walk_forward_without_lookahead():
    r = _returns((_simulate_cgarch(900) / 100).tolist())
    for model in ("cgarch", "betat"):
        w = vol.walk_forward(r, model=model, split=r["close_ts"][699], every=100, horizons=[1, 5], simulations=100)
        assert (w["fitted_through"] <= w["close_ts"]).all(), model
        assert w["variance"].null_count() == 0 and (w["variance"] > 0).all(), model
