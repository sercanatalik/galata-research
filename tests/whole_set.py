import random
from datetime import timedelta

import polars as pl
import pytest
from conftest import utc

from galata_research import stats, studies


def _noise(columns=20, rows=600, seed=1, edge=None):
    rng = random.Random(seed)
    data = {f"c{k}": [rng.gauss(0, 0.01) for _ in range(rows)] for k in range(columns)}
    if edge is not None:
        data["c0"] = [rng.gauss(edge * 0.01, 0.01) for _ in range(rows)]
    return pl.DataFrame(data)


def the_blocks_average_their_declared_length():
    idx = stats.stationary_bootstrap_indices(200_000, 10, random.Random(0))
    breaks = sum(1 for a, b in zip(idx, idx[1:]) if b != (a + 1) % 200_000)
    assert 9 <= len(idx) / (breaks + 1) <= 11


def the_noise_does_not_beat_the_benchmark():
    assert stats.reality_check(_noise(), reps=500)["reality_check"] > 0.1


def a_column_with_a_real_edge_does():
    got = stats.reality_check(_noise(edge=0.5), reps=500)
    assert got["best"] == "c0"
    assert got["reality_check"] < 0.01
    assert max(got["spa"].values()) < 0.01


def the_spa_p_values_are_ordered():
    spa = stats.reality_check(_noise(seed=4), reps=500)["spa"]
    assert spa["lower"] <= spa["consistent"] <= spa["upper"]


def a_result_reproduces_from_its_seed():
    frame = _noise(columns=5, rows=200)
    first, second = (stats.reality_check(frame, reps=200, seed=3) for _ in range(2))
    assert (first["reality_check"], first["spa"]) == (second["reality_check"], second["spa"])


def the_reality_check_agrees_with_arch():
    # arch's `upper` p-value (it compares raw means) is White's Reality Check.
    arch = pytest.importorskip("arch.bootstrap")
    np = pytest.importorskip("numpy")
    frame = _noise(columns=8, rows=500, seed=7, edge=0.08)
    block = int(500**0.5)
    ours = stats.reality_check(frame, reps=2000, block=block, seed=0)["reality_check"]
    losses = -np.asarray(frame.to_numpy())
    spa = arch.SPA(np.zeros(500), losses, reps=2000, block_size=block, bootstrap="stationary", seed=0)
    spa.compute()
    assert ours == pytest.approx(float(spa.pvalues["upper"]), abs=0.04)


def the_excess_is_per_ticker():
    t0 = utc("2026-01-01T00:00")
    rows = []
    for d in range(3):
        for trial, btc, eth in [("s", 0.03, 0.05), ("buy and hold", 0.01, 0.02)]:
            rows.append({"trial": trial, "ticker": "BTC", "ts": t0 + timedelta(days=d), "net": btc})
            rows.append({"trial": trial, "ticker": "ETH", "ts": t0 + timedelta(days=d), "net": eth})
    got = studies.excess(pl.DataFrame(rows), "buy and hold")
    assert got.columns == ["ts", "s | BTC", "s | ETH"]
    assert got["s | BTC"].to_list() == pytest.approx([0.02] * 3)
    assert got["s | ETH"].to_list() == pytest.approx([0.03] * 3)
