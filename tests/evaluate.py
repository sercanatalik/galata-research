from datetime import timedelta
from math import log, sqrt

import numpy as np
import polars as pl
import pytest
from conftest import utc

import galata_research as gr
from galata_research import Refused

ev = gr.models.evaluate
DAY = timedelta(days=1)
T0 = utc("2020-01-01T00:00")


def _proxies(values, *, skip=()):
    slots = [i for i in range(len(values) + len(skip)) if i not in skip][: len(values)]
    return pl.DataFrame({"ticker": "BTC", "ts": [T0 + i * DAY for i in slots], "proxy": [float(v) for v in values]})


def _forecast(origin_bar, h, *, after_gap=False, model="m", value=1.0):
    close = T0 + (origin_bar + 1) * DAY
    return {
        "ticker": "BTC",
        "model": model,
        "close_ts": close,
        "h": h,
        "target_ts": close + (h - 1) * DAY,
        "variance": value,
        "cum_variance": value * h,
        "after_gap": after_gap,
    }


def a_cumulative_proxy_is_exactly_h_bars():
    p = _proxies([9, 9, 1, 2, 4, 9])
    got = ev.align(pl.DataFrame([_forecast(1, 3)]), p)
    assert got["proxy"].item() == 7.0


def a_missing_bar_leaves_the_target_blank():
    p = _proxies([9, 9, 1, 4, 9], skip=(3,))  # bar 3 missing
    got = ev.align(pl.DataFrame([_forecast(1, 3)]), p)
    assert got["proxy"].item() is None


def an_after_gap_row_is_dropped():
    p = _proxies([1.0] * 10)
    f = pl.DataFrame([_forecast(1, 1), _forecast(2, 1, after_gap=True)])
    assert ev.align(f, p).height == 1
    assert ev.align(f, p, drop_after_gap=False).height == 2


def the_textbook_qlike_ranks_alike():
    rng = np.random.default_rng(0)
    p = rng.exponential(2.0, 1000)
    a, b = np.full(1000, 1.5), np.full(1000, 3.0)
    ours = ev.qlike(p, a).mean() - ev.qlike(p, b).mean()
    textbook = lambda f: p / f - np.log(p / f) - 1  # noqa: E731
    assert ours == pytest.approx(textbook(a).mean() - textbook(b).mean())


def the_expected_loss_is_smallest_at_the_truth():
    p = np.random.default_rng(1).exponential(2.0, 20_000)
    losses = {f: ev.qlike(p, np.full(p.size, f)).mean() for f in (1.0, 2.0, 4.0)}
    assert min(losses, key=losses.get) == 2.0


def a_zero_proxy_has_a_finite_qlike():
    assert np.isfinite(ev.qlike(np.array([0.0]), np.array([1e-4])))


def a_constant_advantage_is_significant():
    rng = np.random.default_rng(2)
    bench = rng.normal(1.0, 0.3, 500)
    got = ev.dm(bench - 0.1 + rng.normal(0, 0.05, 500), bench, 1)
    assert got["statistic"] < 0 and got["p_value"] < 0.01


def the_hln_factor():
    rng = np.random.default_rng(3)
    got = ev.dm(rng.normal(0, 1, 50), rng.normal(0, 1, 50), 5)
    assert got["hln_factor"] == pytest.approx(sqrt((51 - 10 + 20 / 50) / 50))


def an_efficient_forecast_passes_mz():
    rng = np.random.default_rng(4)
    f = rng.uniform(0.5, 3.0, 4000)
    got = ev.mz_gls(f * rng.exponential(1.0, f.size), f)
    assert abs(got["alpha"]) < 0.1 and got["beta"] == pytest.approx(1.0, abs=0.1) and got["p_value"] > 0.05


def a_forecast_twice_too_high_fails_mz():
    rng = np.random.default_rng(4)
    f = rng.uniform(0.5, 3.0, 4000)
    got = ev.mz_gls(f * rng.exponential(1.0, f.size), 2 * f)
    assert got["beta"] == pytest.approx(0.5, abs=0.07) and got["p_value"] < 0.01


def _aligned(n=400, seed=5):
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.5, 2.0, n)
    proxy = truth * rng.exponential(1.0, n)
    rows = []
    for model, f in {"good": truth, "bad1": truth * 3, "bad2": truth / 3, "bad3": np.full(n, truth.mean() * 4)}.items():
        for i in range(n):
            rows.append({"model": model, "close_ts": T0 + i * DAY, "h": 1, "forecast": float(f[i]), "proxy": float(proxy[i])})
    return pl.DataFrame(rows)


def a_clearly_better_model_is_the_confidence_set():
    got = ev.mcs(_aligned(), 1, reps=300, seed=1)
    assert got.filter(pl.col("included"))["model"].to_list() == ["good"]


def the_seeds_reproduce():
    a, b = ev.mcs(_aligned(), 1, reps=200, seed=1), ev.mcs(_aligned(), 1, reps=200, seed=1)
    assert a.equals(b)


def the_scorecard_ratios_are_to_the_benchmark_on_the_same_rows():
    got = ev.scorecard(_aligned(), benchmark="bad1")
    assert got.filter(pl.col("model") == "bad1")["qlike_ratio"].item() == pytest.approx(1.0)
    assert got.filter(pl.col("model") == "good")["qlike_ratio"].item() < 1.0
    assert got.filter(pl.col("model") == "good")["dm"].item() < 0


def the_fluctuation_critical_values():
    got = ev.fluctuation(_aligned(), 1, model="good", benchmark="bad1", mu=0.3)
    assert got["critical_5"][0] == 3.012 and got["critical_10"][0] == 2.766


def a_late_advantage_crosses():
    rng = np.random.default_rng(6)
    n = 600
    rows = []
    for i in range(n):
        base = float(rng.normal(3.0, 0.5))
        edge = -0.5 if i >= 0.7 * n else 0.0
        rows.append({"model": "b", "close_ts": T0 + i * DAY, "h": 1, "loss": base})
        rows.append({"model": "m", "close_ts": T0 + i * DAY, "h": 1, "loss": base + edge + float(rng.normal(0, 0.3))})
    # With forecast 1, QLIKE = proxy/1 + ln 1 = proxy: the proxy column carries the loss itself.
    frame = pl.DataFrame(rows).with_columns(pl.lit(1.0).alias("forecast"), pl.col("loss").alias("proxy")).drop("loss")
    got = ev.fluctuation(frame, 1, model="m", benchmark="b", mu=0.2)
    assert got["statistic"].min() < -got["critical_5"][0]


def an_unlisted_mu_is_refused():
    with pytest.raises(Refused, match="mu=0.25"):
        ev.fluctuation(_aligned(), 1, model="good", benchmark="bad1", mu=0.25)


def the_kupiec_statistic_by_hand():
    r = np.zeros(100)
    r[[3, 20, 41, 60, 88]] = -5.0
    got = ev.var_backtest(r, np.full(100, -1.0), np.full(100, -2.0), 0.01)
    assert got["hits"] == 5
    assert got["kupiec"] == pytest.approx(2 * (95 * log(0.95 / 0.99) + 5 * log(0.05 / 0.01)))


def a_run_of_paired_hits_fails_independence():
    r = np.zeros(2000)
    for start in range(10, 2000, 100):
        r[start : start + 2] = -5.0  # 40 hits = 2%, in pairs
    got = ev.var_backtest(r, np.full(2000, -1.0), np.full(2000, -2.0), 0.02)
    assert got["independence_p"] < 0.01 and got["kupiec_p"] > 0.05


def the_dq_accepts_independent_hits_at_the_rate():
    rng = np.random.default_rng(7)
    r = rng.standard_normal(3000)
    got = ev.var_backtest(r, np.full(3000, -1.6449), np.full(3000, -2.06), 0.05)
    assert got["dq_p"] > 0.01 and got["kupiec_p"] > 0.01


def the_fz0_loss_by_hand():
    got = ev.var_backtest(np.array([-3.0]), np.array([-2.0]), np.array([-2.5]), 0.05)
    want = -(-2 + 3) / (0.05 * -2.5) + (-2) / (-2.5) + log(2.5) - 1
    assert got["fz0"] == pytest.approx(want)


def an_es_above_its_var_is_refused():
    with pytest.raises(Refused, match="ES ≤ VaR"):
        ev.var_backtest(np.zeros(10), np.full(10, -2.0), np.full(10, -1.0), 0.05)


def _multi(edges, n=400, seed=10, noise=0.5):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        for k, edge in enumerate(edges):
            base = float(rng.normal(3.0, noise))
            rows.append({"model": "b", "close_ts": T0 + i * DAY, "h": k + 1, "forecast": 1.0, "proxy": base})
            rows.append({"model": "m", "close_ts": T0 + i * DAY, "h": k + 1, "forecast": 1.0, "proxy": base - edge + float(rng.normal(0, noise))})
    return pl.DataFrame(rows)


def a_model_better_at_every_horizon_passes_both():
    a = _multi([0.3, 0.3, 0.3])
    assert ev.uspa(a, model="m", benchmark="b", reps=299)["p_value"] < 0.05
    assert ev.aspa(a, model="m", benchmark="b", reps=299)["p_value"] < 0.05


def a_model_better_on_average_passes_only_aspa():
    a = _multi([0.4, 0.4, -0.1])
    assert ev.aspa(a, model="m", benchmark="b", reps=299)["p_value"] < 0.05
    assert ev.uspa(a, model="m", benchmark="b", reps=299)["p_value"] > 0.1


def a_noise_difference_passes_neither():
    a = _multi([0.0, 0.0, 0.0], seed=11)
    assert ev.uspa(a, model="m", benchmark="b", reps=299)["p_value"] > 0.05
    assert ev.aspa(a, model="m", benchmark="b", reps=299)["p_value"] > 0.05


def the_horizon_tests_reproduce():
    a = _multi([0.1, 0.0, 0.05])
    assert ev.uspa(a, model="m", benchmark="b", reps=199, seed=4) == ev.uspa(a, model="m", benchmark="b", reps=199, seed=4)


def a_weight_vector_not_summing_to_one_is_refused():
    with pytest.raises(Refused, match="summing to 1"):
        ev.aspa(_multi([0.1, 0.1, 0.1], n=50), model="m", benchmark="b", weights=[0.5, 0.5, 0.5])


def the_block_variance_matches_the_hac_on_white_noise():
    from galata_research.models.evaluate import _block_variance, _qs_variance

    # One draw's QS estimate scatters ±5% (it sums ~55 noisy autocovariances), so both
    # estimators are checked on the average of eight independent series.
    draws = [np.random.default_rng(s).normal(0, 2.0, 30_000) for s in range(20, 28)]
    assert np.mean([_block_variance(x[:, None], 3)[0] for x in draws]) == pytest.approx(4.0, rel=0.03)
    assert np.mean([_qs_variance(x) for x in draws]) == pytest.approx(4.0, rel=0.03)


def _gw_aligned(d, log_f, h=1):
    rows = []
    for i, (di, lf) in enumerate(zip(d, log_f)):
        f = float(np.exp(lf))
        rows.append({"model": "b", "close_ts": T0 + i * DAY, "h": h, "forecast": f, "proxy": f * 2.0})
        # QLIKE(b) − QLIKE(m) = d: give m the same forecast and a proxy lower by d·f (QLIKE = p/f + ln f).
        rows.append({"model": "m", "close_ts": T0 + i * DAY, "h": h, "forecast": f, "proxy": f * 2.0 - di * f})
    return pl.DataFrame(rows)


def the_gw_statistic_by_hand():
    rng = np.random.default_rng(20)
    d, lf = rng.normal(0, 1, 300), rng.normal(0, 0.5, 300)
    got = ev.gw(_gw_aligned(d, lf), 1, model="m", benchmark="b")
    x = np.column_stack([np.ones(299), d[:-1], lf[1:]])
    z = x * d[1:, None]
    zbar = z.mean(axis=0)
    want = 299 * zbar @ np.linalg.solve(z.T @ z / 299, zbar)
    assert got["statistic"] == pytest.approx(want) and got["n"] == 299


def a_model_winning_only_in_high_vol_is_found():
    rng = np.random.default_rng(21)
    lf = rng.normal(0, 1, 800)
    d = 0.8 * (lf - lf.mean()) + rng.normal(0, 1, 800)  # wins when the forecast is high, loses when low
    a = _gw_aligned(d, lf)
    got = ev.gw(a, 1, model="m", benchmark="b")
    dm_p = ev.dm(np.zeros(800), d, 1)["p_value"]  # d averages to ~0
    assert dm_p > 0.1
    assert got["p_value"] < 0.01 and got["coefficients"]["log_forecast"] > 0


def a_noise_differential_is_not_predictable():
    rng = np.random.default_rng(22)
    got = ev.gw(_gw_aligned(rng.normal(0, 1, 800), rng.normal(0, 1, 800)), 1, model="m", benchmark="b")
    assert got["p_value"] > 0.05


def the_lagged_differential_is_known_at_the_origin():
    from galata_research.models.evaluate import _gw_frame

    d = np.arange(1.0, 41.0)
    frame = _gw_frame(_gw_aligned(d, np.zeros(40), h=3), 3, "m", "b")
    assert frame["lagged_diff"].to_numpy() == pytest.approx(frame["d"].to_numpy() - 3)


def every_gw_result_carries_the_theory_note():
    rng = np.random.default_rng(23)
    got = ev.gw(_gw_aligned(rng.normal(0, 1, 100), rng.normal(0, 1, 100)), 1, model="m", benchmark="b")
    assert "rolling" in got["theory"]


def _three(edges, n=240, seed=40):
    # Loss per (model, horizon): base + edge + noise; QLIKE = proxy with forecast 1.
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        for h in (1, 2, 3):
            base = float(rng.normal(3.0, 0.4))
            for name, edge in edges.items():
                rows.append({"model": name, "close_ts": T0 + i * DAY, "h": h, "forecast": 1.0, "proxy": base + edge + float(rng.normal(0, 0.4))})
    return pl.DataFrame(rows)


def a_model_best_at_every_horizon_is_the_set():
    got = ev.mcs_horizons(_three({"best": -0.5, "b": 0.0, "c": 0.1}), outer=59, inner=29)
    assert got.filter(pl.col("included"))["model"].to_list() == ["best"]
    assert got.filter(pl.col("model") == "best")["pvalue"].item() == 1.0


def a_pair_of_equal_models_survives_together():
    got = ev.mcs_horizons(_three({"a": 0.0, "b": 0.0, "worse": 0.6}), outer=59, inner=29)
    kept = set(got.filter(pl.col("included"))["model"].to_list())
    assert {"a", "b"} <= kept and "worse" not in kept


def the_horizon_confidence_set_reproduces():
    a = _three({"a": 0.0, "b": 0.05, "c": 0.2})
    assert ev.mcs_horizons(a, outer=39, inner=19, seed=5).equals(ev.mcs_horizons(a, outer=39, inner=19, seed=5))


def the_p_values_never_fall():
    got = ev.mcs_horizons(_three({"a": 0.0, "b": 0.1, "c": 0.3, "d": 0.5}), outer=39, inner=19).drop_nulls("eliminated").sort("eliminated")
    assert got["pvalue"].to_list() == sorted(got["pvalue"].to_list())


def _normal_pair(n, mu=(0.05, 0.02), rho=0.5, seed=1):
    return np.random.default_rng(seed).multivariate_normal(list(mu), [[1, rho], [rho, 1]], size=n)


def _vech_garch_pair(n, seed):
    # Ledoit and Wolf's null (§4.2): diagonal-vech GARCH(1,1), identical marginals, mean 16.5/52 each.
    c, a, b = np.array([0.15, 0.13, 0.15]), np.array([0.075, 0.05, 0.075]), np.array([0.90, 0.89, 0.90])
    rng = np.random.default_rng(seed)
    h = c / (1 - a - b)
    out = np.empty((n + 200, 2))
    for t in range(n + 200):
        cov = np.array([[h[0], h[1]], [h[1], h[2]]])
        r = np.linalg.cholesky(cov) @ rng.standard_normal(2)
        out[t] = r
        h = c + a * np.array([r[0] ** 2, r[0] * r[1], r[1] ** 2]) + b * h
    return out[200:] + 16.5 / 52


def the_difference_is_of_the_sample_sharpe_ratios():
    x = _normal_pair(300)
    got = ev.sharpe_difference(x[:, 0], x[:, 1], reps=99)
    assert got["delta"] == pytest.approx(x[:, 0].mean() / x[:, 0].std() - x[:, 1].mean() / x[:, 1].std(), rel=1e-12)


def a_one_period_block_gives_the_sample_covariance():
    from galata_research.models.evaluate import _block_psi, _moment_columns

    y = _moment_columns(_normal_pair(200))
    assert _block_psi(y, 1) == pytest.approx(np.cov(y.T, bias=True), abs=1e-12)


def the_hac_error_matches_memmel_under_iid_normal():
    # 100,000 pairs: the fourth-moment terms in Ψ̂ are then within about 1% of their normal values.
    n = 100_000
    x = _normal_pair(n, seed=2)
    got = ev.sharpe_difference(x[:, 0], x[:, 1], reps=19)
    sa, sb, rho = got["sharpe_a"], got["sharpe_b"], float(np.corrcoef(x.T)[0, 1])
    memmel = sqrt((2 - 2 * rho + 0.5 * (sa**2 + sb**2 - 2 * sa * sb * rho**2)) / n)
    assert got["se_hac"] == pytest.approx(memmel, rel=0.03)


def the_bootstrap_test_holds_its_size():
    # 400 null datasets: a size of 5% rejects 20 ± 4.4; three binomial sd above is 8.3%.
    rejected = sum(ev.sharpe_difference(*_vech_garch_pair(120, s).T, reps=199, seed=s)["p_boot"] <= 0.05 for s in range(400))
    assert rejected / 400 <= 0.05 + 3 * sqrt(0.05 * 0.95 / 400)


def a_real_difference_is_found():
    rng = np.random.default_rng(3)
    x = rng.normal(0.3, 1, 500)
    y = 0.5 * x + sqrt(0.75) * rng.normal(0, 1, 500) - 0.15  # Sharpe ratio 0 against 0.3
    got = ev.sharpe_difference(x, y, reps=999)
    assert got["p_boot"] < 0.01 and got["ci"][0] > 0


def a_null_pair_is_dropped_together():
    x = _normal_pair(100)
    a, b = list(x[:, 0]), list(x[:, 1])
    a[5], b[9] = None, None
    got = ev.sharpe_difference(a, b, reps=19, block=2)
    assert got["rows"] == 98


def a_constant_or_short_series_is_refused():
    x = _normal_pair(50)
    with pytest.raises(Refused, match="constant"):
        ev.sharpe_difference(np.ones(50), x[:, 1], reps=19)
    with pytest.raises(Refused, match="too few for blocks"):
        ev.sharpe_difference(x[:15, 0], x[:15, 1], reps=19, block=8)


def a_positive_elimination_p_still_builds_the_set():
    # The first-listed model eliminated with p > 0: numpy and Python p-values once mixed, and polars refused the column.
    got = ev.mcs_horizons(_three({"a": 0.03, "b": 0.0, "worse": 0.6}), outer=59, inner=29)
    assert got.schema["included"] == pl.Boolean and got.schema["pvalue"] == pl.Float64
    assert got.filter(pl.col("model") == "a")["pvalue"].item() > 0
