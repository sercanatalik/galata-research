import random
from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from conftest import utc

from galata_research import Refused, factors
from galata_research.models import portfolio as pf

# Two uncorrelated blocks, each a correlated pair.
BLOCKS = np.array([[0.04, 0.006, 0, 0], [0.006, 0.09, 0, 0], [0, 0, 0.01, 0.002], [0, 0, 0.002, 0.0225]])


def the_correlated_pairs_sit_together():
    assert pf.hrp_order(BLOCKS) in ([2, 3, 0, 1], [3, 2, 0, 1], [0, 1, 2, 3], [1, 0, 2, 3], [2, 3, 1, 0], [0, 1, 3, 2])


def the_hrp_bisects_by_inverse_cluster_variance_as_worked_by_hand():
    # Order [2, 3, 0, 1]. Left {2, 3}: IVP weights (0.6923, 0.3077), variance 0.007775.
    # Right {0, 1}: the same IVP shares, variance 0.030248. Left gets 1 − 0.007775/0.038023 = 0.7955,
    # and within it asset 2 gets 1 − 0.01/0.0325 = 0.6923.
    w = pf.hrp(BLOCKS)
    assert w[2] == pytest.approx(0.7955 * 0.6923, abs=2e-4)
    assert w[3] == pytest.approx(0.7955 * 0.3077, abs=2e-4)
    assert w[0] == pytest.approx(0.2045 * 0.6923, abs=2e-4)
    assert w.sum() == pytest.approx(1.0)


def the_hrp_on_two_coins_is_inverse_variance():
    c = np.array([[0.04, 0.01], [0.01, 0.09]])
    assert pf.hrp(c) == pytest.approx(pf.ivp(c))


def an_interior_minimum_variance_is_the_closed_form():
    c = np.array([[0.04, 0.006], [0.006, 0.09]])
    inv = np.linalg.solve(c, np.ones(2))
    assert pf.minvar(c) == pytest.approx(inv / inv.sum(), abs=1e-7)


def the_long_only_constraint_binds_when_a_short_would_help():
    # The third coin covaries with the first by more than the first's own variance (0.042 > 0.04):
    # unconstrained, minimum variance would short it.
    c = np.array([[0.04, 0.0, 0.042], [0.0, 0.04, 0.0], [0.042, 0.0, 0.05]])
    inv = np.linalg.solve(c, np.ones(3))
    assert (inv / inv.sum()).min() < 0
    w = pf.minvar(c)
    assert w.min() >= 0 and w[2] == pytest.approx(0.0, abs=1e-8)
    assert w.sum() == pytest.approx(1.0)


def a_covariance_with_a_zero_variance_is_refused():
    with pytest.raises(Refused, match="positive variances"):
        pf.ivp(np.array([[0.0, 0.0], [0.0, 1.0]]))


def the_ewma_is_the_riskmetrics_recursion():
    r = np.array([[0.01, 0.0], [0.0, 0.02], [0.03, -0.01]])
    s = r[:2].T @ r[:2] / 2
    s = 0.9 * s + 0.1 * np.outer(r[2], r[2])
    assert pf.ewma(r, lam=0.9, seed=2) == pytest.approx(s)


def _panel(n_days=400, delist=None, names=("BTC", "A", "B", "C"), seed=5):
    rng = random.Random(seed)
    t0 = utc("2024-01-01T00:00")
    rows = []
    common = [rng.gauss(0, 0.02) for _ in range(n_days)]
    for k, name in enumerate(names):
        c = 100.0
        last = delist[1] if delist and delist[0] == name else n_days
        for d in range(last):
            c *= 1 + 0.6 * common[d] + rng.gauss(0, 0.01 * (k + 1))
            rows.append({"ticker": name, "ts": t0 + timedelta(days=d), "close_ts": t0 + timedelta(days=d + 1), "close": c, "volume": 10.0 ** (6 - k) / c})
    return factors.panel(pl.DataFrame(rows))


def the_book_is_chosen_on_its_day_from_coins_with_history():
    p = _panel()
    m = factors.members(p, exclude=(), top=10, history=1, window=30)
    assert pf.book(p, m, "2024-06-01T00:00Z", n=2, history=100) == ["BTC", "A"]
    with pytest.raises(Refused, match="only 0 perpetuals had 500 bars"):
        pf.book(p, m, "2024-06-01T00:00Z", n=2, history=500)


def the_survivors_are_the_book_after_a_delisting():
    p = _panel(delist=("C", 250))
    on = pf.schedule("2024-05-01T00:00Z", "2024-12-01T00:00Z")
    trials, w = pf.study(p, ["BTC", "A", "B", "C"], on, "2024-12-01T00:00Z", estimators=("sample", "ewma"))
    assert set(trials["trial"]) == {f"{m} {e}" for m in pf.METHODS for e in ("sample", "ewma")} | {pf.EQUAL}
    sums = w.group_by("trial", "rebalance").agg(pl.col("w").sum())
    assert sums["w"].to_list() == pytest.approx([1.0] * sums.height)
    # C's last bar is day 249 (2024-09-06): no rebalance after it holds C.
    gone = w.filter((pl.col("ticker") == "C") & (pl.col("rebalance") > utc("2024-09-06T00:00")))
    assert gone.is_empty()
    assert w.filter((pl.col("ticker") == "C") & (pl.col("rebalance") < utc("2024-09-06T00:00"))).height > 0


def no_weight_is_set_from_a_return_after_its_rebalance():
    # Guard: a Σ estimated with the next day's return would change when the panel is cut at the rebalance.
    p = _panel()
    on = pf.schedule("2024-05-01T00:00Z", "2024-07-01T00:00Z")
    _, full = pf.study(p, ["BTC", "A", "B"], on, "2024-07-01T00:00Z", estimators=("sample", "ewma"))
    last = on[-1]
    _, cut = pf.study(p.filter(pl.col("ts") <= last), ["BTC", "A", "B"], on, "2024-07-01T00:00Z", estimators=("sample", "ewma"))
    a = full.filter(pl.col("rebalance") == last).sort("trial", "ticker")
    b = cut.filter(pl.col("rebalance") == last).sort("trial", "ticker")
    assert a["w"].to_list() == pytest.approx(b["w"].to_list())


def the_dcc_weights_are_made_and_walked_forward():
    p = _panel(n_days=620, names=("BTC", "A", "B"))
    on = pf.schedule("2025-07-01T00:00Z", "2025-08-01T00:00Z")
    trials, w = pf.study(p, ["BTC", "A", "B"], on, "2025-08-01T00:00Z", every=200)
    assert {"ivp dcc", "hrp dcc", "minvar dcc"} <= set(trials["trial"])
    assert w.filter(pl.col("trial") == "minvar dcc")["w"].min() >= 0
