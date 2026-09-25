import random

import polars as pl
import pytest

from galata_research import stats


def _ar(phi, n, seed):
    rng, x, out = random.Random(seed), 0.0, []
    for _ in range(n):
        x = phi * x + rng.gauss(0, 1)
        out.append(x)
    return out


def the_block_agrees_with_arch():
    arch = pytest.importorskip("arch.bootstrap")
    np = pytest.importorskip("numpy")
    for phi, seed in [(0.6, 1), (0.2, 2), (0.9, 3)]:
        series = _ar(phi, 2000, seed)
        theirs = float(arch.optimal_block_length(np.asarray(series))["stationary"].iloc[0])
        assert stats.optimal_block(series) == pytest.approx(theirs, abs=1e-6)


def the_persistence_lengthens_the_block():
    assert stats.optimal_block(_ar(0.8, 2000, 4)) > stats.optimal_block(_ar(0.1, 2000, 4))


def the_auto_block_is_reported():
    frame = pl.DataFrame({f"c{k}": _ar(0.3 + 0.2 * k, 600, k) for k in range(3)})
    got = stats.reality_check(frame, reps=50, block="auto")
    blocks = sorted(stats.optimal_block(frame[c]) for c in frame.columns)
    assert got["block"] == pytest.approx(max(1.0, blocks[1]))
