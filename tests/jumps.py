"""gr.jumps: Boudt–Croux–Laurent's robust periodicity and Lee–Mykland's test, on simulated returns."""

import math
import random
from datetime import timedelta

import polars as pl
import pytest
from conftest import utc

from galata_research import Refused, jumps

FIVE = timedelta(minutes=5)


def _returns(values, start="2026-01-05T00:00", step=FIVE):
    """5m returns from a list; None is a hole."""
    t0 = utc(start)
    return pl.DataFrame(
        {"ticker": "BTC", "ts": [t0 + i * step for i in range(len(values))], "close_ts": [t0 + (i + 1) * step for i in range(len(values))],
         "return": values},
        schema={"ticker": pl.String, "ts": pl.Datetime("us", "UTC"), "close_ts": pl.Datetime("us", "UTC"), "return": pl.Float64},
    )  # fmt: skip


def _gauss(n, seed=11, scale=lambda i: 1.0):
    rng = random.Random(seed)
    return [rng.gauss(0, 1e-3) * scale(i) for i in range(n)]


# ---- the periodicity ----------------------------------------------------------------


def the_consistency_factor_is_1_081():
    c = math.sqrt(jumps.CHI2_99)
    phi = math.exp(-c * c / 2) / math.sqrt(2 * math.pi)
    truncated = 1 - 2 * c * phi / math.erf(c / math.sqrt(2))
    assert jumps.WSD_CONSISTENCY == pytest.approx(1 / truncated, abs=1e-3)


def a_known_pattern_is_recovered():
    # 60 days of 1h slots; slot 13 twice as volatile.
    r = _returns(_gauss(24 * 60, scale=lambda i: 2.0 if i % 24 == 13 else 1.0), step=timedelta(hours=1))
    f = jumps.periodicity(r, slot="1h", by="time_of_day")
    busy = f.filter(pl.col("slot") == 13)["factor"].item()
    rest = f.filter(pl.col("slot") != 13)["factor"].mean()
    assert busy / rest == pytest.approx(2.0, rel=0.1)
    assert (f["factor"] ** 2).mean() == pytest.approx(1.0)


def a_jump_does_not_move_the_factor():
    # 300 days, so one return is 1/300 of its slot: dropping it (weight 0) is what may move the factor.
    values = _gauss(24 * 300, seed=3)
    before = jumps.periodicity(_returns(values, step=timedelta(hours=1)), slot="1h", by="time_of_day")
    values[24 * 150 + 5] *= 1000
    after = jumps.periodicity(_returns(values, step=timedelta(hours=1)), slot="1h", by="time_of_day")
    slot5 = [t.filter(pl.col("slot") == 5)["factor"].item() for t in (before, after)]
    assert slot5[1] == pytest.approx(slot5[0], rel=0.01)


def an_unknown_layout_is_refused():
    with pytest.raises(Refused, match="time_of_week, time_of_day"):
        jumps.periodicity(_returns([0.001] * 10), by="time_of_month")


# ---- Lee–Mykland ----------------------------------------------------------------------


def the_published_window_at_five_minutes():
    got = jumps.lee_mykland(_returns(_gauss(400)))
    first = got.with_row_index().filter(pl.col("sigma_local").is_not_null())["index"][0]
    # σ̂ needs K − 2 = 268 products; the first product is at row 1, so the first σ̂ is at row 269.
    assert first == 269


def the_threshold_at_five_minutes_and_one_percent():
    n = 288
    root = math.sqrt(2 * math.log(n))
    c = math.sqrt(2 / math.pi)
    c_n = root / c - (math.log(math.pi) + math.log(math.log(n))) / (2 * c * root)
    s_n = 1 / (c * root)
    assert jumps.gumbel_threshold(n, 0.01) == pytest.approx(c_n + s_n * 4.6001, abs=1e-3)
    assert jumps.gumbel_threshold(n, 0.01) == pytest.approx(5.40, abs=0.01)


def a_planted_jump_is_found():
    values = _gauss(3000, seed=5)
    values[2000] = 20e-3
    got = jumps.lee_mykland(_returns(values))
    found = got.with_row_index().filter(pl.col("jump"))["index"].to_list()
    assert 2000 in found and len(found) <= 2


def the_scale_does_not_see_its_own_return():
    values = _gauss(1000, seed=9)
    base = jumps.lee_mykland(_returns(values))["sigma_local"][700]
    values[700] = 50e-3  # the return judged at 700 changes; its own scale must not
    assert jumps.lee_mykland(_returns(values))["sigma_local"][700] == pytest.approx(base)


def a_hole_restarts_the_scale():
    values = _gauss(1000, seed=13)
    values[500] = None
    got = jumps.lee_mykland(_returns(values), window=20)["sigma_local"]
    # Products at 500 and 501 touch the hole; σ̂ at i needs products i−18 … i−1, so it is back at 520.
    assert got[480:500].null_count() == 0
    assert got[501:520].null_count() == 19 and got[520] is not None


def a_periodicity_keeps_a_busy_slot_from_jumping():
    # 40 days of 5m returns; the 13:30 slot three times as volatile, fat-tailed.
    rng = random.Random(21)
    values = [rng.gauss(0, 1e-3) * (3.0 if (i % 288) == 162 else 1.0) * (4.0 if rng.random() < 0.02 else 1.0) for i in range(288 * 40)]
    r = _returns(values)
    raw = jumps.lee_mykland(r)
    adjusted = jumps.lee_mykland(r, periodicity=jumps.periodicity(r, slot="5m", by="time_of_day"))
    in_slot = pl.col("ts").dt.hour().cast(pl.Int32) * 60 + pl.col("ts").dt.minute().cast(pl.Int32) == 13 * 60 + 30
    assert raw.filter(in_slot)["jump"].sum() > adjusted.filter(in_slot)["jump"].sum()


def the_bh_rule_finds_the_planted_jump():
    values = _gauss(3000, seed=5)
    values[2000] = 20e-3
    got = jumps.lee_mykland(_returns(values), rule="bh", q=0.05)
    assert got["jump"][2000] and got["jump"].sum() <= 3


def an_unknown_rule_is_refused():
    with pytest.raises(Refused, match="gumbel, bh"):
        jumps.lee_mykland(_returns([0.001] * 10), rule="bonferroni")
