from datetime import timedelta
from math import ceil
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from conftest import garch_t, utc

import galata_research as gr
from galata_research import Refused

corr = gr.models.corr
vol = gr.models.vol
DAY = timedelta(days=1)
T0 = utc("2026-01-05T00:00")
TRUE_R = np.array([[1.0, 0.3, 0.5], [0.3, 1.0, 0.2], [0.5, 0.2, 1.0]])


def _long(columns: dict, *, width=DAY):
    n = len(next(iter(columns.values())))
    return pl.concat(
        [
            pl.DataFrame(
                {"ticker": t, "ts": [T0 + i * width for i in range(n)], "close_ts": [T0 + (i + 1) * width for i in range(n)], "return": v},
                schema_overrides={"return": pl.Float64},
            )
            for t, v in columns.items()
        ]
    )


def _simulate_dcc(n, a, b, rbar, *, seed, omega=0.05, alpha=0.08, beta=0.9):
    """DCC(1,1) with GARCH(1,1)-normal margins, returns in units (×100 scale inside)."""
    rng = np.random.default_rng(seed)
    N = len(rbar)
    q = rbar.copy()
    s2 = np.full(N, omega / (1 - alpha - beta))
    out = np.empty((n, N))
    burn = 500
    for t in range(n + burn):
        d = np.sqrt(np.diag(q))
        r = q / np.outer(d, d)
        z = np.linalg.cholesky(r) @ rng.standard_normal(N)
        eps = np.sqrt(s2) * z
        if t >= burn:
            out[t - burn] = eps / 100
        q = (1 - a - b) * rbar + a * np.outer(z, z) + b * q
        s2 = omega + alpha * eps**2 + beta * s2
    return out


@pytest.fixture(scope="module", name="three")
def _three():
    y = _simulate_dcc(4000, 0.05, 0.90, TRUE_R, seed=7)
    return _long({"AAA": y[:, 0], "BBB": y[:, 1], "CCC": y[:, 2]})


@pytest.fixture(scope="module", name="pair")
def _pair():
    return _long({"BTC": garch_t(900, seed=21), "ETH": garch_t(900, seed=22)})


# ── the joint sample and n_eff ───────────────────────────────────────────────


def a_young_ticker_starts_the_sample():
    btc = garch_t(900, seed=1)
    gold = [None] * 600 + garch_t(300, seed=2)
    frame = _long({"BTC": btc, "GOLD": gold})
    tickers, joint, youngest = corr.joint(frame)
    assert (tickers, joint.height, youngest) == (["BTC", "GOLD"], 300, "GOLD")
    assert not joint["after_gap"].any()  # the rows before a listing are not a gap
    with pytest.raises(Refused, match=r"300.*min_obs=500.*GOLD"):
        corr.fit(frame, model="garch")


def a_hole_in_one_ticker_drops_the_row_for_all():
    eth = garch_t(50, seed=3)
    eth[20] = None
    _, joint, _ = corr.joint(_long({"BTC": garch_t(50, seed=4), "ETH": eth}))
    assert joint.height == 49
    assert T0 + 20 * DAY not in joint["ts"].to_list()
    assert joint.filter(pl.col("after_gap"))["ts"].to_list() == [T0 + 21 * DAY]


def a_single_ticker_is_not_a_correlation():
    with pytest.raises(Refused, match="BTC"):
        corr.joint(_long({"BTC": garch_t(50, seed=5)}))


def an_equal_weighting_counts_every_row():
    assert corr.n_eff([1.0] * 20) == pytest.approx(20)


def the_riskmetrics_daily_n_eff_is_32_33():
    assert corr.ewma_n_eff(0.94) == pytest.approx(32.33, abs=1e-2)


def the_closed_form_matches_the_sum():
    for lam, n in ((0.94, 10), (0.97, 250), (0.5, 3)):
        assert corr.ewma_n_eff(lam, n) == pytest.approx(corr.n_eff(lam**k for k in range(n)), rel=1e-12)


# ── DCC in sample ────────────────────────────────────────────────────────────


def a_simulated_dcc_is_recovered(three):
    f = corr.fit(three, model="garch", dist="normal")
    assert f.a == pytest.approx(0.05, abs=0.02)
    assert f.b == pytest.approx(0.90, abs=0.04)
    d = np.sqrt(np.diag(f.qbar))
    rbar = f.qbar / np.outer(d, d)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert rbar[i, j] == pytest.approx(TRUE_R[i, j], abs=0.05)
    assert f.persistence == pytest.approx(f.a + f.b)
    assert f.series.height == 3 * 4000


def no_dynamics_leaves_the_correlation_constant():
    z = np.random.default_rng(0).standard_normal((200, 3))
    qbar = z.T @ z / 200
    for kind in corr.CORRS:
        _, r = corr.filter(z, 0.0, 0.9, qbar, corr=kind)
        d = np.sqrt(np.diag(qbar))
        assert np.allclose(r, qbar / np.outer(d, d)), kind


def every_correlation_matrix_is_a_correlation_matrix(three):
    f = corr.fit(three.filter(pl.col("ts") < T0 + 1500 * DAY), model="gjr", dist="t")
    rho = f.series.sort("ts")
    for ts, group in rho.group_by("ts", maintain_order=True):
        m = np.eye(3)
        for row in group.iter_rows(named=True):
            i, j = f.tickers.index(row["ticker_i"]), f.tickers.index(row["ticker_j"])
            m[i, j] = m[j, i] = row["correlation"]
        assert np.linalg.eigvalsh(m).min() > 0


def a_model_with_no_sigma_per_bar_is_refused(pair):
    with pytest.raises(Refused, match="garch, gjr, egarch, ewma"):
        corr.fit(pair, model="har")


def the_cdcc_target_is_the_profiled_moment(three):
    f = corr.fit(three.filter(pl.col("ts") < T0 + 1500 * DAY), model="garch", dist="normal", corr="cdcc")
    z = np.column_stack([f.fits[t].series["z"].to_numpy() for t in f.tickers])
    q, _ = corr.filter(z, f.a, f.b, f.qbar, corr="cdcc")
    assert np.allclose(np.diag(f.qbar), 1.0)  # S is unit-diagonal (Aielli 2013, Def. 3.2)
    star = np.sqrt(np.einsum("tii->ti", q[:-1])) * z
    m = star.T @ star / len(z)
    d = np.sqrt(np.diag(m))
    assert np.allclose(m / np.outer(d, d), f.qbar, atol=1e-12)
    assert f.a == pytest.approx(0.05, abs=0.03) and f.b == pytest.approx(0.90, abs=0.06)


# ── rmgarch's reference fit (tests/data/README.md) ───────────────────────────


@pytest.fixture(scope="module", name="rmgarch")
def _rmgarch():
    import json
    from pathlib import Path

    return json.loads((Path(__file__).parent / "data" / "rmgarch_dcc_norm.json").read_text())


def the_filter_reproduces_rmgarchs_path(rmgarch):
    z = np.asarray(rmgarch["resid"]) / np.asarray(rmgarch["sigma"])
    _, r = corr.filter(z, rmgarch["a"], rmgarch["b"], np.asarray(rmgarch["Qbar"]))
    assert np.allclose(r[-2], rmgarch["Rlast"], atol=1e-3)  # R of the last bar
    assert np.allclose(r[-1], rmgarch["R1_forecast"], atol=1e-3)  # R of the bar after it


def the_likelihood_is_rmgarchs_at_its_parameters(rmgarch):
    # rmgarch's joint log-likelihood is the normal margins plus the correlation part over every bar,
    # Q₀ = Q̄ included; ours drops −½zᵀz from the correlation part, so it is added back here.
    e, s = np.asarray(rmgarch["resid"]), np.asarray(rmgarch["sigma"])
    z = e / s
    margins = float(np.sum(-0.5 * (np.log(2 * np.pi) + np.log(s**2) + z**2)))
    a, b, qbar = rmgarch["a"], rmgarch["b"], np.asarray(rmgarch["Qbar"])
    part = corr.loglik(z, a, b, qbar) + 0.5 * float(np.sum(z**2))
    assert margins + part == pytest.approx(rmgarch["loglik"], abs=0.05)


def the_second_step_recovers_rmgarchs_parameters_from_its_residuals(rmgarch):
    z = np.asarray(rmgarch["resid"]) / np.asarray(rmgarch["sigma"])
    est = corr.estimate(z)
    assert est["a"] == pytest.approx(rmgarch["a"], abs=1e-3)
    assert est["b"] == pytest.approx(rmgarch["b"], abs=1e-3)


def an_end_to_end_fit_lands_near_rmgarch(rmgarch):
    raw = np.loadtxt(Path(__file__).parent / "data" / "dji30ret_5.csv", delimiter=",", skiprows=1)[:1000] / 100
    f = corr.fit(_long({t: raw[:, k] for k, t in enumerate(rmgarch["meta"]["columns"])}), model="garch", dist="normal")
    # pymgarch's own tolerances. Step 1 is arch's GARCH, not rugarch's (backcast, optimiser), and the
    # likelihood is flat along a + b on 1,000 bars: measured â 0.0076 and b̂ 0.737 against R's 0.0108 and 0.787.
    assert f.converged
    assert f.a == pytest.approx(rmgarch["a"], abs=0.02)
    assert f.b == pytest.approx(rmgarch["b"], abs=0.10)


# ── EWMA ─────────────────────────────────────────────────────────────────────


def the_diagonal_is_ewmas_variance(pair):
    e = corr.ewma(pair, lam=0.94)
    w = ceil(1.94 / 0.06)
    assert e.warmup == w and not e.fitted
    r = pair.filter(pl.col("ticker") == "BTC")["return"].to_numpy()
    s2 = float(np.mean(r[:w] ** 2))
    expected = [s2]
    for x in r[w:]:
        s2 = 0.94 * s2 + 0.06 * x * x
        expected.append(s2)
    got = e.series.filter((pl.col("ticker_i") == "BTC") & (pl.col("ticker_j") == "BTC"))["covariance"].to_numpy()
    assert np.allclose(got, expected, rtol=1e-12)


def a_single_lambda_keeps_the_matrix_psd(three):
    e = corr.ewma(three.filter(pl.col("ts") < T0 + 400 * DAY), lam=0.94)
    for _, group in e.series.group_by("ts"):
        m = np.zeros((3, 3))
        for row in group.iter_rows(named=True):
            i, j = e.tickers.index(row["ticker_i"]), e.tickers.index(row["ticker_j"])
            m[i, j] = m[j, i] = row["covariance"]
        assert np.linalg.eigvalsh(m).min() >= -1e-18


# ── the walk-forward ─────────────────────────────────────────────────────────


def _split(frame, i):
    return frame["close_ts"][i]


def no_origin_sees_its_future(pair):
    w = corr.walk_forward(pair, model="gjr", split=_split(pair, 599), every=50, horizons=[1, 5])
    assert (w["fitted_through"] <= w["close_ts"]).all()
    assert w.filter((pl.col("h") == 1) & (pl.col("ticker_i") == "BTC") & (pl.col("ticker_j") == "ETH")).height == 301


def a_later_return_does_not_change_an_earlier_forecast(pair):
    # Guard: filtering the whole sample before cutting (or fitting past the refit) moves the origin's forecast.
    kw = dict(model="garch", dist="normal", split=_split(pair, 599), every=1000, horizons=[1, 3])
    base = corr.walk_forward(pair, **kw)
    doubled = pair.with_columns(pl.when(pl.col("ts") > T0 + 700 * DAY).then(pl.col("return") * 2).otherwise(pl.col("return")).alias("return"))
    moved = corr.walk_forward(doubled, **kw)
    cut = T0 + 701 * DAY  # origins whose close is at or before bar 700's close
    keep = pl.col("close_ts") <= cut
    a, b = base.filter(keep), moved.filter(keep)
    assert a.height > 0
    assert np.allclose(a["covariance"].to_numpy(), b["covariance"].to_numpy(), rtol=1e-12)
    later = pl.col("close_ts") > cut + 2 * DAY
    assert not np.allclose(base.filter(later)["covariance"].to_numpy(), moved.filter(later)["covariance"].to_numpy())


def the_diagonal_is_the_volatility_models_forecast(pair):
    kw = dict(model="gjr", dist="t", split=_split(pair, 599), every=100, horizons=[1, 4])
    w = corr.walk_forward(pair, **kw)
    for t in ("BTC", "ETH"):
        v = vol.walk_forward(pair.filter(pl.col("ticker") == t), **kw).sort("close_ts", "h")
        d = w.filter((pl.col("ticker_i") == t) & (pl.col("ticker_j") == t)).sort("close_ts", "h")
        assert np.allclose(d["covariance"].to_numpy(), v["variance"].to_numpy(), rtol=1e-10, atol=0)


def the_cumulative_covariance_is_the_running_sum(pair):
    w = corr.walk_forward(pair, model="garch", dist="normal", split=_split(pair, 800), every=100, horizons=[1, 5, 10])
    # cum at 5 is the sum of the point Σ at 1…5; only 1, 5 and 10 are emitted, so recompute the point path.
    full = corr.walk_forward(pair, model="garch", dist="normal", split=_split(pair, 800), every=100, horizons=range(1, 11))
    for h in (5, 10):
        summed = full.filter(pl.col("h") <= h).group_by("close_ts", "ticker_i", "ticker_j").agg(pl.col("covariance").sum()).sort("close_ts", "ticker_i", "ticker_j")
        cum = w.filter(pl.col("h") == h).sort("close_ts", "ticker_i", "ticker_j")
        assert np.allclose(cum["cum_covariance"].to_numpy(), summed["covariance"].to_numpy(), rtol=1e-12)


def the_correlation_reverts_to_its_long_run_level(pair):
    w = corr.walk_forward(pair, model="garch", dist="normal", split=_split(pair, 800), every=1000, horizons=[1, 40])
    f = corr.fit(pair.filter(pl.col("ts") <= pair["ts"][800]), model="garch", dist="normal")
    d = np.sqrt(np.diag(f.qbar))
    rbar = (f.qbar / np.outer(d, d))[0, 1]
    off = w.filter(pl.col("ticker_i") != pl.col("ticker_j")).sort("close_ts")
    r1 = off.filter(pl.col("h") == 1)["correlation"].to_numpy()
    r40 = off.filter(pl.col("h") == 40)["correlation"].to_numpy()
    s = f.a + f.b
    assert np.allclose(r40 - rbar, s**39 * (r1 - rbar), atol=1e-5)  # R̄ from the in-sample fit, z from the filter: ~2e-6 apart


def the_correlation_target_is_what_the_forecast_reverts_to(pair):
    w = corr.walk_forward(pair, model="garch", dist="normal", split=_split(pair, 800), every=1000, horizons=[1, 40])
    f = corr.fit(pair.filter(pl.col("ts") <= pair["ts"][800]), model="garch", dist="normal")
    d = np.sqrt(np.diag(f.qbar))
    off = w.filter(pl.col("ticker_i") != pl.col("ticker_j")).sort("close_ts")
    target = off.filter(pl.col("h") == 1)["correlation_target"].to_numpy()
    r1 = off.filter(pl.col("h") == 1)["correlation"].to_numpy()
    r40 = off.filter(pl.col("h") == 40)["correlation"].to_numpy()
    assert np.allclose(target, (f.qbar / np.outer(d, d))[0, 1], atol=1e-5)  # the in-sample fit's Q̄, normalised
    assert np.allclose(r40 - target, (f.a + f.b) ** 39 * (r1 - target), atol=1e-5)
    own = w.filter(pl.col("ticker_i") == pl.col("ticker_j"))
    assert np.allclose(own["correlation_target"].to_numpy(), 1.0)


def no_target_where_nothing_reverts(pair):
    for kind in ("ewma", "sample", "iewma"):
        w = corr.walk_forward(pair, model="garch", dist="normal", corr=kind, sample_window=50, split=_split(pair, 599), every=1000, horizons=[1])
        assert w["correlation_target"].is_null().all(), kind


def every_pair_of_an_origin_shares_one_fit_span(pair):
    w = corr.walk_forward(pair, model="garch", dist="normal", split=_split(pair, 599), every=5, horizons=[1])
    per_origin = w.group_by("close_ts").agg(pl.col("fitted_through").n_unique().alias("spans"), pl.col("refit").first()).sort("close_ts")
    assert (per_origin["spans"] == 1).all()
    assert per_origin["refit"].to_list() == [i % 5 == 0 for i in range(per_origin.height)]


def an_egarch_t_beyond_one_step_is_refused(pair):
    with pytest.raises(Refused, match="EGARCH"):
        corr.walk_forward(pair, model="egarch", dist="t", split=_split(pair, 599), horizons=[1, 5])


def the_ewma_walk_forward_is_flat_and_unfitted(pair):
    w = corr.walk_forward(pair, corr="ewma", split=_split(pair, 100), horizons=[1, 3], lam=0.94)
    assert w["a"].is_null().all() and not w["refit"].any()
    assert (w["fitted_through"] == w["close_ts"]).all()
    one, three = (w.filter(pl.col("h") == h).sort("close_ts", "ticker_i", "ticker_j") for h in (1, 3))
    assert np.allclose(one["covariance"].to_numpy(), three["covariance"].to_numpy())
    e = corr.ewma(pair, lam=0.94).series.filter(pl.col("close_ts") >= _split(pair, 100)).sort("close_ts", "ticker_i", "ticker_j")
    assert np.allclose(one["covariance"].to_numpy(), e["covariance"].to_numpy(), rtol=1e-12)


def the_seasonal_diagonal_is_the_volatility_models_forecast():
    hour = timedelta(hours=1)
    r = _long({"BTC": garch_t(1600, seed=31), "ETH": garch_t(1600, seed=32)}, width=hour)
    split = r["close_ts"][1199]
    f = gr.timeseries.seasonal_factors(r, fit=(r["ts"][0], split), by="hour_of_day")
    kw = dict(model="garch", dist="normal", split=split, every=1000, horizons=[1, 3], factors=f)
    w = corr.walk_forward(r, **kw)
    for t in ("BTC", "ETH"):
        v = vol.walk_forward(r.filter(pl.col("ticker") == t), **kw).sort("close_ts", "h")
        d = w.filter((pl.col("ticker_i") == t) & (pl.col("ticker_j") == t)).sort("close_ts", "h")
        assert np.allclose(d["covariance"].to_numpy(), v["variance"].to_numpy(), rtol=1e-10, atol=0)


def a_factor_fitted_past_the_split_is_refused(pair):
    f = gr.timeseries.seasonal_factors(pair, fit=(pair["ts"][0], pair["close_ts"][-1]), by="hour_of_day")
    with pytest.raises(Refused, match="after the split"):
        corr.walk_forward(pair, model="garch", split=_split(pair, 700), factors=f)


# ── the baselines and the scorer (compare-the-correlations) ─────────────────


def a_constant_correlation_is_the_window_moment(pair):
    w = corr.walk_forward(pair, model="garch", dist="normal", corr="ccc", split=_split(pair, 599), every=1000, horizons=[1, 5])
    rho = w.filter(pl.col("ticker_i") != pl.col("ticker_j"))
    assert rho["correlation"].round(12).n_unique() == 1  # constant across origins and horizons (through D·R·D, to rounding)
    assert (rho["a"] == 0).all() and (rho["b"] == 0).all()
    assert rho["n_eff"].unique().to_list() == [600.0]


def the_sample_covariance_is_the_last_windows_moment(pair):
    w = corr.walk_forward(pair, corr="sample", sample_window=50, split=_split(pair, 599), horizons=[1])
    btc = pair.filter(pl.col("ticker") == "BTC")["return"].to_numpy()
    first = w.filter((pl.col("ticker_i") == "BTC") & (pl.col("ticker_j") == "BTC")).sort("close_ts")["covariance"][0]
    assert first == pytest.approx(float(np.mean(btc[550:600] ** 2)), rel=1e-12)
    assert w["a"].is_null().all() and (w["n_eff"] == 50).all()


def the_losses_are_their_formulas():
    s = np.array([[2.0, 0.5], [0.5, 1.0]])
    r = np.array([1.0, -1.0])
    got = corr.losses(s, r)
    inv = np.linalg.inv(s)
    assert got["stein"] == pytest.approx(np.log(np.linalg.det(s)) + r @ inv @ r)
    assert got["frobenius"] == pytest.approx(((np.outer(r, r) - s) ** 2).sum())
    w = inv @ np.ones(2) / (np.ones(2) @ inv @ np.ones(2))
    assert got["gmv"] == pytest.approx((w @ r) ** 2)
    assert w.sum() == pytest.approx(1.0)


def every_model_is_scored_on_the_same_bars(pair):
    kw = dict(split=_split(pair, 700), horizons=[1])
    forecasts = {
        "ewma": corr.walk_forward(pair, corr="ewma", **kw),
        "sample": corr.walk_forward(pair, corr="sample", sample_window=100, **kw),
    }
    scored = corr.score(forecasts, pair)
    per = scored.group_by("model").agg(pl.col("close_ts").sort())
    assert per["close_ts"][0].to_list() == per["close_ts"][1].to_list()
    assert scored.height == 2 * 199  # origins 700…898; the last origin's bar is past the sample
    table = corr.compare(scored, loss="stein", benchmark="ewma", reps=200)
    assert set(table["model"]) == {"ewma", "sample"} and table.filter(pl.col("model") == "ewma")["dm"][0] is None


def a_dcc_beats_constant_correlation_on_its_own_process(three):
    # The scorer's sanity: on data a DCC generated, DCC's Stein loss is lower than CCC's.
    kw = dict(model="garch", dist="normal", split=three["close_ts"][2999], every=1000, horizons=[1])
    forecasts = {name: corr.walk_forward(three, corr=name, **kw) for name in ("dcc", "ccc")}
    table = corr.compare(corr.score(forecasts, three), loss="stein", benchmark="ccc", reps=200)
    assert table["model"][0] == "dcc"


def an_ewma_on_z_never_reverts(pair):
    w = corr.walk_forward(pair, model="garch", dist="normal", corr="iewma", lam=0.94, split=_split(pair, 700), every=1000, horizons=[1, 20])
    off = w.filter(pl.col("ticker_i") != pl.col("ticker_j")).sort("close_ts", "h")
    one, twenty = off.filter(pl.col("h") == 1)["correlation"].to_numpy(), off.filter(pl.col("h") == 20)["correlation"].to_numpy()
    assert np.allclose(one, twenty)  # a + b = 1: nothing to revert to
    assert (w["a"] - 0.06).abs().max() < 1e-12 and (w["b"] - 0.94).abs().max() < 1e-12


def the_hac_dm_on_white_noise_is_not_significant():
    d = np.random.default_rng(4).normal(size=2000)
    got = corr.dm_hac(d)
    assert got["lags"] == int(4 * 20 ** (2 / 9)) and got["p_value"] > 0.01
