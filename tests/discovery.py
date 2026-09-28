"""gr.models.discovery: the two-venue VECM and its shares, on simulated leader and follower."""

import math
import random
from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from conftest import utc

pytest.importorskip("arch")
from galata_research.models import discovery  # noqa: E402

T0 = utc("2026-09-25T12:00")


def _prices(n=20_000, lag=3, noise=2e-5, seed=1):
    rng = random.Random(seed)
    p1, x = [], math.log(100.0)
    for _ in range(n):
        x += rng.gauss(0, 1e-4)
        p1.append(x)
    p2 = [p1[max(0, i - lag)] + rng.gauss(0, noise) for i in range(n)]
    return pl.DataFrame({"p1": p1, "p2": p2})


def the_grid_carries_forward_never_back():
    x = pl.DataFrame({"ts": [T0 + timedelta(milliseconds=200), T0 + timedelta(milliseconds=2500)], "price": [100.0, 101.0]})
    y = pl.DataFrame({"ts": [T0 + timedelta(milliseconds=700)], "price": [200.0]})
    got = discovery.grid(x, y, "1s")
    assert got["ts"][0] == T0 + timedelta(seconds=1)
    step = got.filter(pl.col("ts") == T0 + timedelta(seconds=2)).row(0, named=True)
    assert step["p1"] == pytest.approx(math.log(100.0)) and step["p2"] == pytest.approx(math.log(200.0))


def a_leader_and_a_follower():
    got = discovery.shares(discovery.vecm(_prices(), lags=10)).row(0, named=True)
    assert got["cs"] > 0.8 and got["is_mid"] > 0.8 and got["ils"] > 0.8


def the_swapped_venues_swap_the_shares():
    p = _prices()
    one = discovery.shares(discovery.vecm(p, lags=10))
    two = discovery.shares(discovery.vecm(p.select(pl.col("p2").alias("p1"), pl.col("p1").alias("p2")), lags=10))
    for col in ("cs", "is_low", "is_mid", "is_high", "ils"):
        assert one[col].to_list() == pytest.approx(two[col].reverse().to_list(), abs=1e-9)


def the_shares_sum_to_one():
    got = discovery.shares(discovery.vecm(_prices(seed=4, noise=1e-4), lags=10))
    assert got["cs"].sum() == pytest.approx(1) and got["ils"].sum() == pytest.approx(1)
    assert all(lo <= mid <= hi for lo, mid, hi in zip(got["is_low"], got["is_mid"], got["is_high"], strict=True))
    assert np.isfinite(got["is_mid"].to_numpy()).all()
