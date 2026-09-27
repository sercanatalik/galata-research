from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from arch import arch_model
from conftest import utc

import galata_research as gr
from galata_research import Refused

vol = gr.models.vol
HOUR = timedelta(hours=1)
DAY = timedelta(days=1)
T0 = utc("2026-01-05T00:00")  # a Monday


def _frame(values, *, width=DAY):
    n = len(values)
    return pl.DataFrame(
        {"ticker": "BTC", "ts": [T0 + i * width for i in range(n)], "close_ts": [T0 + (i + 1) * width for i in range(n)], "return": values},
        schema_overrides={"return": pl.Float64},
    )


@pytest.fixture(scope="module", name="returns")
def _returns():
    np.random.seed(21)
    sim = arch_model(None, dist="t").simulate([0.0, 0.05, 0.08, 0.9, 5.0], 900, burn=500)
    return _frame((sim["data"].to_numpy() / 100).tolist())


def _split(frame, i):
    return frame["close_ts"][i]


def no_origin_sees_its_future(returns):
    w = vol.walk_forward(returns, model="gjr", split=_split(returns, 599), every=7, horizons=[1, 5])
    assert (w["fitted_through"] <= w["close_ts"]).all()
    assert w.filter(pl.col("h") == 1).height == 301  # origins at bars 599…899


def a_shock_moves_the_forecast_from_its_own_origin(returns):
    # Guard: a one-row slip between kept rows and arch's index moves the rise to k-1 or k+1.
    values = returns["return"].to_list()
    k = 700
    shocked = list(values)
    shocked[k] = 0.25
    kw = dict(model="garch", split=_split(returns, 599), every=1000, horizons=[1])
    base = vol.walk_forward(returns, **kw)["variance"].to_list()
    moved = vol.walk_forward(_frame(shocked), **kw)["variance"].to_list()
    i = k - 599  # the origin whose close is bar k's
    assert moved[:i] == pytest.approx(base[:i])
    assert moved[i] > 10 * base[i]


def a_later_return_does_not_change_an_earlier_forecast(returns):
    values = returns["return"].to_list()
    doubled = values[:750] + [2 * v for v in values[750:]]
    kw = dict(model="gjr", split=_split(returns, 599), every=1000, horizons=[1, 5])
    a = vol.walk_forward(returns, **kw).filter(pl.col("close_ts") <= returns["close_ts"][749])
    b = vol.walk_forward(_frame(doubled), **kw).filter(pl.col("close_ts") <= returns["close_ts"][749])
    assert b["variance"].to_list() == pytest.approx(a["variance"].to_list())


def the_cumulative_is_the_running_sum(returns):
    w = vol.walk_forward(returns, model="garch", split=_split(returns, 799), every=10, horizons=[1, 2, 3, 4, 5])
    for _, g in w.group_by("close_ts"):
        g = g.sort("h")
        assert g["cum_variance"][4] == pytest.approx(g["variance"].sum())


def the_refit_every_five_schedule(returns):
    w = vol.walk_forward(returns, model="garch", split=_split(returns, 799), every=5, horizons=[1])
    assert w["refit"].to_list()[:6] == [True, False, False, False, False, True]
    assert w["fitted_through"][3] == w["close_ts"][0]


def the_analytic_models_match_arch_at_one_step(returns):
    split = _split(returns, 699)
    w = vol.walk_forward(returns, model="garch", split=split, every=1000, horizons=[1])
    y = returns["return"].to_numpy() * 100
    res = arch_model(y, dist="t").fit(disp="off", last_obs=700)
    theirs = np.asarray(res.forecast(horizon=1, start=699, reindex=False).variance)[:, 0] / 1e4
    assert w["variance"].to_list() == pytest.approx(theirs.tolist(), rel=1e-4)


def a_simulated_forecast_reproduces(returns):
    kw = dict(model="egarch", split=_split(returns, 849), every=25, horizons=[1, 5], simulations=200, seed=3)
    a, b = vol.walk_forward(returns, **kw), vol.walk_forward(returns, **kw)
    assert a.equals(b)
    assert a.filter(pl.col("h") == 5)["variance"].null_count() == 0


def _hourly(n=2000):
    np.random.seed(5)
    sim = arch_model(None, dist="t").simulate([0.0, 0.05, 0.08, 0.9, 5.0], n, burn=500)
    return _frame((sim["data"].to_numpy() / 100).tolist(), width=HOUR)


def a_factor_fitted_past_the_split_is_refused():
    r = _hourly()
    f = gr.timeseries.seasonal_factors(r, fit=(r["ts"][0], r["close_ts"][-1]))
    with pytest.raises(Refused, match="after the split"):
        vol.walk_forward(r, model="garch", split=r["close_ts"][1500], factors=f)


def the_target_cells_factor_is_applied():
    r = _hourly()
    split = r["close_ts"][1499]
    f = gr.timeseries.seasonal_factors(r, fit=(r["ts"][0], split), by="hour_of_day")
    w = vol.walk_forward(r, model="garch", split=split, every=1000, horizons=[1], factors=f)
    des = gr.timeseries.deseasonalize(r, f).rename({"deseasonalized": "d"}).select("ticker", "ts", "close_ts", pl.col("d").alias("return"))
    plain = vol.walk_forward(des, model="garch", split=split, every=1000, horizons=[1])
    cell = f.with_columns((pl.col("factor") ** 2).alias("f2")).select("weekday", "hour", "f2")
    target = w.with_columns(pl.col("target_ts").dt.weekday().alias("weekday"), pl.col("target_ts").dt.hour().alias("hour")).join(
        cell, on=["weekday", "hour"], how="left", maintain_order="left"
    )
    assert w["variance"].to_list() == pytest.approx((plain["variance"] * target["f2"]).to_list())


def the_table_says_where_its_fit_ended():
    r = _hourly(400)
    end = r["close_ts"][300]
    f = gr.timeseries.seasonal_factors(r, fit=(r["ts"][0], end))
    assert set(f["fit_end"]) == {end}


def a_first_window_under_min_obs_is_refused(returns):
    with pytest.raises(Refused, match="under min_obs=500"):
        vol.walk_forward(returns, model="garch", split=_split(returns, 299))


def every_refit_block_carries_its_nu(returns):
    w = vol.walk_forward(returns, model="garch", dist="t", split=_split(returns, 599), every=50, horizons=[1])
    blocks = w.with_columns(pl.col("refit").cast(pl.Int32).cum_sum().alias("block")).group_by("block").agg(pl.col("nu").n_unique().alias("k"), pl.col("nu").first())
    assert (blocks["k"] == 1).all() and blocks["nu"].null_count() == 0 and blocks.height == 7
    assert vol.walk_forward(returns, model="garch", dist="normal", split=_split(returns, 849), every=50, horizons=[1])["nu"].null_count() > 0
