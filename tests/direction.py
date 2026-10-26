"""gr.models.direction: targets, Newton–IRLS against scipy, and the purged walk-forward, on hand-built frames."""

import math
import random
from datetime import timedelta

import polars as pl
import pytest
from conftest import utc

pytest.importorskip("arch")
import numpy as np  # noqa: E402
from scipy import optimize  # noqa: E402

from galata_research import Refused, momentum  # noqa: E402
from galata_research.models import direction  # noqa: E402

T0 = utc("2026-01-05T00:00")
FIVE = timedelta(minutes=5)


def _frame(closes, *, skip=(), seed=0):
    """Decision rows of 5m, with sigma and random encoded momentum columns, as targets() and walk_forward() read them."""
    rng = random.Random(seed)
    idx = [i for i in range(len(closes)) if i not in set(skip)]
    ts = [T0 + i * FIVE for i in idx]
    out = {
        "ticker": "BTC",
        "ts": ts,
        "close_ts": [t + FIVE for t in ts],
        "close": [float(closes[i]) for i in idx],
        "sigma": [0.001 * (1 + 0.1 * rng.random()) for _ in idx],
    }
    for name in momentum.LADDER:
        out[f"e_{name}"] = [rng.gauss(0, 1) for _ in idx]
    out["interaction"] = [rng.gauss(0, 1) for _ in idx]
    return pl.DataFrame(out)


def _walk(n, seed=1):
    rng = random.Random(seed)
    p, out = 100.0, []
    for _ in range(n):
        p *= math.exp(0.001 * rng.gauss(0, 1))
        out.append(p)
    return out


# ── Targets ──


def the_sign_and_the_three_class_targets():
    closes = [100.0, 100.0 * math.exp(0.0006), 100.0, 99.0, 99.0]
    t = direction.targets(_frame(closes), "5m")
    assert t["r_ahead"][0] == pytest.approx(0.0006)
    assert t["y"].to_list() == [1, 0, 0, 0, None]
    # +6 bps is inside the 9.5 bps cost: up for the sign, flat for the three classes.
    assert t["cls"].to_list() == [0, 0, -1, 0, None]
    assert t["cost"][0] == direction.COST
    assert set(t["horizon"]) == {"5m"}


def a_hole_inside_the_horizon_nulls_the_target():
    t = direction.targets(_frame(_walk(20), skip=[8]), "30m")
    assert t["h"][0] == 6
    nulls = t.filter(pl.col("r_ahead").is_null())["ts"].to_list()
    # Rows whose next 6 bars include the missing bar 8, and the last 6 rows.
    assert nulls[:6] == [T0 + i * FIVE for i in (2, 3, 4, 5, 6, 7)]
    assert t.filter(pl.col("r_ahead").is_null())["y"].null_count() == len(nulls)


def the_horizons_in_decision_bars():
    assert [momentum.bars_in(h, "5m") for h in direction.HORIZONS] == [1, 6, 60, 288]
    with pytest.raises(Refused, match="not a whole number of 1h bars"):
        direction.targets(_frame(_walk(5)), "30m", decision="1h")


# ── IRLS against scipy ──


def _data(n=3000, seed=4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3)) * [1.0, 5.0, 0.1] + [0.0, 2.0, -1.0]
    eta = 0.3 + x @ np.array([0.8, -0.1, 4.0])
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    return x, y


@pytest.mark.parametrize("ridge", [direction.RIDGE, 0.5])
def the_binary_fit_is_the_maximum_of_its_penalised_likelihood(ridge):
    x, y = _data()
    fit = direction.fit_logit(x, y, ridge=ridge)
    d = direction._design(x, fit["mean"], fit["sd"])

    def neg(b):
        return -direction.penalised_loglik(b, d, y, ridge)

    def grad(b):
        p = 1 / (1 + np.exp(-(d @ b)))
        g = d.T @ (y - p)
        g[1:] -= ridge * b[1:]
        return -g

    def hess(b):
        p = 1 / (1 + np.exp(-(d @ b)))
        return (d.T * (p * (1 - p))) @ d + np.diag(np.r_[0.0, np.full(d.shape[1] - 1, ridge)])

    ref = optimize.minimize(
        neg,
        np.zeros(d.shape[1]),
        jac=grad,
        hess=hess,
        method="trust-exact",
        options={"gtol": 1e-12},
    )
    assert np.allclose(fit["beta"], ref.x, atol=1e-6)


def a_separable_column_does_not_diverge():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.5).astype(float)
    x = np.column_stack([y + 0.01 * rng.random(500), rng.normal(size=500)])
    fit = direction.fit_logit(x, y)
    assert np.all(np.isfinite(fit["beta"]))
    p = direction.predict_logit(fit, x)
    assert np.all((p > 0) & (p < 1))
    assert np.mean((p > 0.5) == (y == 1)) == 1.0


def the_multinomial_fit_is_the_maximum_of_its_penalised_likelihood():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(2500, 2))
    eta = np.column_stack([-0.2 + x @ [1.0, -0.5], np.zeros(2500), 0.1 + x @ [-0.7, 0.9]])
    prob = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    cls = np.array([rng.choice([-1, 0, 1], p=row) for row in prob])
    ridge = 0.3
    fit = direction.fit_multinomial(x, cls, ridge=ridge)
    d = direction._design(x, fit["mean"], fit["sd"])
    ys = np.column_stack([(cls == -1), (cls == 1)]).astype(float)

    def neg(flat):
        b = flat.reshape(2, -1)
        e = d @ b.T
        norm = np.logaddexp(0, np.logaddexp(e[:, 0], e[:, 1]))
        return -(np.sum(ys * e) - np.sum(norm) - ridge / 2 * (np.sum(b[:, 1:] ** 2)))

    ref = optimize.minimize(neg, np.zeros(2 * d.shape[1]), method="BFGS", options={"gtol": 1e-9})
    assert np.allclose(fit["beta"].reshape(-1), ref.x, atol=1e-5)
    probs = direction.predict_multinomial(fit, x)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-12)


# ── Walk-forward ──


def _forecasts(target="sign", cache=None, horizon="30m"):
    t = direction.targets(_frame(_walk(3000)), horizon)
    split = (T0 + 1500 * FIVE).isoformat()
    return t, direction.walk_forward(
        t,
        horizon=horizon,
        split=split,
        encoding="score",
        target=target,
        every=400,
        cache=cache,
    )


def every_training_row_was_known_by_its_refit_and_no_row_uses_a_model_before_its_embargo():
    t, f = _forecasts()
    h = timedelta(minutes=30)
    assert (f["refit_at"] + h <= f["close_ts"]).all()
    assert (f["fitted_through"] <= f["refit_at"]).all()
    refits = sorted(f["refit_at"].unique().to_list())
    # Each row takes the latest refit whose embargo has passed.
    for row in f.filter(pl.col("set") == "B").iter_rows(named=True):
        allowed = [r for r in refits if r + h <= row["close_ts"]]
        assert row["refit_at"] == allowed[-1]
    # The training rows are exactly those whose target was known by the refit.
    known = t.drop_nulls("y").with_columns((pl.col("close_ts") + h).alias("k"))
    for r in refits:
        n = f.filter((pl.col("refit_at") == r) & (pl.col("set") == "A"))["n_train"][0]
        assert n == known.filter(pl.col("k") <= r).height


def the_probabilities_are_valid_and_a_and_b_share_rows():
    _, f = _forecasts()
    assert f["p_up"].is_between(0, 1, closed="none").all()
    a, b = f.filter(pl.col("set") == "A"), f.filter(pl.col("set") == "B")
    assert a["ts"].to_list() == b["ts"].to_list()
    assert a["refit_at"].to_list() == b["refit_at"].to_list()
    assert set(f["model"]) == {"logit", "climatology"}
    assert {
        "horizon",
        "h",
        "target",
        "set",
        "encoding",
        "fitted_through",
        "refit_at",
    } <= set(f.columns)
    _, three = _forecasts("three")
    total = three["p_down"] + three["p_flat"] + three["p_up"]
    assert ((total - 1).abs() < 1e-9).all()


def the_climatology_is_the_training_share_up():
    t, f = _forecasts()
    h = timedelta(minutes=30)
    r = f["refit_at"].min()
    known = t.drop_nulls("y").filter(pl.col("close_ts") + h <= r)
    clim = f.filter((pl.col("set") == "climatology") & (pl.col("refit_at") == r))
    assert clim["p_up"][0] == pytest.approx(known["y"].mean())


def a_cache_refits_nothing_it_already_holds(monkeypatch):
    cache = {}
    _, first = _forecasts(cache=cache)
    calls = []
    real = direction.fit_logit
    monkeypatch.setattr(direction, "fit_logit", lambda *a, **k: calls.append(1) or real(*a, **k))
    _, second = _forecasts(cache=cache)
    assert calls == []
    assert first.equals(second)


def an_unknown_target_or_encoding_is_refused():
    t = direction.targets(_frame(_walk(50)), "5m")
    with pytest.raises(Refused, match="target="):
        direction.walk_forward(t, horizon="5m", split=T0.isoformat(), encoding="score", target="up")
    with pytest.raises(Refused, match="encoding"):
        direction.walk_forward(t, horizon="5m", split=T0.isoformat(), encoding="rank")


# ── Scores ──


def _sign_forecasts(p_model, p_bench, y, h=1):
    n = len(y)
    ts = [T0 + i * FIVE for i in range(n)]
    base = {
        "ticker": "BTC",
        "ts": ts,
        "close_ts": [t + FIVE for t in ts],
        "h": h,
        "y": pl.Series(y, dtype=pl.Int8),
    }
    return pl.concat(
        [
            pl.DataFrame(base | {"set": "B", "p_up": p_model}),
            pl.DataFrame(base | {"set": "A", "p_up": p_bench}),
        ]
    )


def the_brier_skill_of_the_benchmark_is_zero_and_a_better_forecast_is_positive():
    rng = random.Random(3)
    y = [int(rng.random() < 0.7) for _ in range(400)]
    same = _sign_forecasts([0.5] * 400, [0.5] * 400, y)
    assert direction.brier_skill(same, model="B", benchmark="A") == 0.0
    better = _sign_forecasts([0.7] * 400, [0.5] * 400, y)
    assert direction.brier_skill(better, model="B", benchmark="A") > 0
    scored = direction.scored(better)
    assert scored.filter(pl.col("set") == "A")["log_loss"][0] == pytest.approx(math.log(2))


def the_comparison_reports_its_lags():
    rng = random.Random(8)
    y = [int(rng.random() < 0.5) for _ in range(3000)]
    f = _sign_forecasts([0.5 + 0.01 * rng.gauss(0, 1) for _ in y], [0.5] * 3000, y, h=60)
    out = direction.compare(f, model="B", benchmark="A")
    assert out["h"] == 60 and out["lags"] >= 59
    assert {"statistic", "p_value", "n", "mean_difference"} <= set(out)


def the_pesaran_timmermann_statistic_is_pinned_to_its_formula_on_a_table():
    # 100 rows: 40 called up and up, 20 called up and down, 10 called down and up, 30 called down and down.
    calls = [1] * 40 + [1] * 20 + [0] * 10 + [0] * 30
    outcomes = [1] * 40 + [0] * 20 + [1] * 10 + [0] * 30
    out = direction.pesaran_timmermann(calls, outcomes)
    n, p, py, px = 100, 0.70, 0.50, 0.60
    star = py * px + (1 - py) * (1 - px)
    v = star * (1 - star) / n - ((2 * py - 1) ** 2 * px * (1 - px) / n + (2 * px - 1) ** 2 * py * (1 - py) / n + 4 * py * px * (1 - py) * (1 - px) / n**2)
    assert out["statistic"] == pytest.approx((p - star) / math.sqrt(v), abs=1e-9)
    assert out["statistic"] == pytest.approx(4.103050, rel=1e-6)
    assert out["hit_rate"] == 0.70


def the_pesaran_timmermann_test_keeps_its_size_on_independent_calls():
    rng = random.Random(2026)
    rejected = 0
    for _ in range(200):
        calls = [int(rng.random() < 0.55) for _ in range(500)]
        outcomes = [int(rng.random() < 0.5) for _ in range(500)]
        rejected += direction.pesaran_timmermann(calls, outcomes)["p_value"] < 0.05
    assert rejected <= 20


def a_calibrated_forecast_lands_on_the_diagonal():
    rng = random.Random(5)
    p = [rng.random() for _ in range(20000)]
    y = [int(rng.random() < q) for q in p]
    f = _sign_forecasts(p, p, y).filter(pl.col("set") == "B")
    table = direction.reliability(f)
    assert table.height == 10
    for row in table.filter(pl.col("n") >= 100).iter_rows(named=True):
        se = math.sqrt(row["mean_forecast"] * (1 - row["mean_forecast"]) / row["n"])
        assert abs(row["share_up"] - row["mean_forecast"]) < 3 * se


def the_holm_adjustment_steps_down():
    assert direction.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def _cells(changes=None):
    rows = [
        {
            "horizon": "5m",
            "encoding": "score",
            "ticker": t,
            "mean_difference": -1e-4,
            "dm_p": 1e-6,
            "pt_p": 1e-4,
            "first_half": -1e-4,
            "second_half": -1e-4,
        }
        for t in ("BTC", "ETH")
    ]
    for (ticker, key), value in (changes or {}).items():
        next(r for r in rows if r["ticker"] == ticker)[key] = value
    return pl.DataFrame(rows)


def a_horizon_is_supported_only_on_both_tickers_and_both_halves():
    _, ok = direction.s_verdict(_cells())
    assert ok["supported"].to_list() == [True]
    _, eth_flips = direction.s_verdict(_cells({("ETH", "mean_difference"): 1e-4, ("ETH", "dm_p"): 0.5}))
    assert eth_flips["supported"].to_list() == [False]
    cells, half_flips = direction.s_verdict(_cells({("BTC", "second_half"): 2e-5}))
    assert half_flips["supported"].to_list() == [False]
    assert cells.filter(pl.col("ticker") == "BTC")["c3"].to_list() == [False]


# ── The economic test ──

DAY = timedelta(days=1)


def _daily_case(days=80, oos_from=50, p_up=None, seed=6):
    """5m decision rows over `days`, and 1d sign forecasts for set B from day `oos_from`."""
    closes = _walk(days * 288, seed=seed)
    frame = _frame(closes)
    first = T0 + oos_from * DAY
    rows = frame.filter((pl.col("close_ts") >= first) & ((pl.col("close_ts") - first).dt.total_minutes() % 1440 == 0))
    ps = p_up or [0.9 if i % 2 == 0 else 0.1 for i in range(rows.height)]
    forecasts = rows.select("ticker", "ts", "close_ts").with_columns(
        pl.lit("B").alias("set"),
        pl.lit(288).alias("h"),
        pl.lit("1d").alias("horizon"),
        pl.Series("p_up", ps[: rows.height]),
    )
    return frame, forecasts, first


def a_flip_pays_the_round_trip_beyond_its_funding():
    frame, forecasts, first = _daily_case()
    hours = pl.datetime_range(T0, T0 + 80 * DAY, "1h", eager=True, time_zone="UTC")
    funding = pl.DataFrame(
        {
            "ticker": "BTC",
            "ts": hours,
            "rate": [0.0001 if h.hour % 8 == 0 else 0.0 for h in hours],
        }
    )
    out = direction.economic(frame, forecasts, split=first.isoformat(), funding=funding)
    rule = out.filter(pl.col("trial") == "sign rule").drop_nulls("net")
    flips = rule.filter(pl.col("cost") > direction.FLIP_FEE * 1.5)
    assert flips.height >= 25
    assert flips["cost"].to_list() == pytest.approx([2 * 0.000475] * flips.height)
    # Three settlements a day, each of 1 bp, charged on the held side.
    assert rule["funding"].abs().to_list() == pytest.approx([0.0003] * rule.height)
    assert ((rule["net"] - (rule["gross"] - rule["cost"] - rule["funding"])).abs() < 1e-15).all()


def a_daily_rule_decides_once_a_day():
    frame, forecasts, first = _daily_case()
    out = direction.economic(frame, forecasts, split=first.isoformat())
    rule = out.filter(pl.col("trial") == "sign rule")
    gaps = rule["close_ts"].diff().drop_nulls().unique().to_list()
    assert gaps == [DAY]
    assert (rule["close_ts"] - rule["ts"]).unique().to_list() == [DAY]


def the_inverse_vol_book_is_vol_targets_at_one_day():
    import importlib

    vt = importlib.import_module("galata_research.models.vol.target")

    frame, forecasts, first = _daily_case()
    grid = direction._grid(frame, forecasts["close_ts"][0], 288, 300_000_000)
    ours = direction.inverse_vol(grid, split=first.isoformat())
    oos = grid.filter(pl.col("close_ts") >= first)
    variance = oos.select(
        "close_ts",
        pl.lit(1).alias("h"),
        (pl.col("sigma").pow(2) * 288).alias("variance"),
    )
    theirs = vt.target(
        variance,
        grid.select("ticker", "ts", "close_ts", "close"),
        split=first.isoformat(),
        rule="inverse_vol",
    )
    joined = ours.join(theirs.select("close_ts", pl.col("position").alias("vt")), on="close_ts").filter(pl.col("close_ts") >= first)
    assert joined.height == oos.height
    assert joined["position"].to_list() == pytest.approx(joined["vt"].to_list(), rel=1e-12)


def a_rule_that_always_holds_is_buy_and_hold():
    cells = {}
    for seed in (6, 7):
        frame, forecasts, first = _daily_case(days=150, seed=seed, p_up=[0.9] * 200)
        cells[f"BTC {seed}"] = direction.economic(frame, forecasts, split=first.isoformat())
    verdict = direction.e_verdict(cells)
    for row in verdict["cells"].iter_rows(named=True):
        assert row["sharpe_annual"] == pytest.approx(row["hold_annual"], rel=1e-12)
    assert not verdict["beats_hold_and_inverse_vol"]
    assert not verdict["supported"]


def a_warm_start_reaches_the_same_optimum():
    x, y = _data(seed=12)
    cold = direction.fit_logit(x, y)
    warm = direction.fit_logit(x, y, start=cold["beta"] + 0.3)
    assert np.allclose(cold["beta"], warm["beta"], atol=1e-9)
    rng = np.random.default_rng(1)
    cls = rng.choice([-1, 0, 1], size=len(y))
    m_cold = direction.fit_multinomial(x, cls)
    m_warm = direction.fit_multinomial(x, cls, start=m_cold["beta"] - 0.2)
    assert np.allclose(m_cold["beta"], m_warm["beta"], atol=1e-9)
