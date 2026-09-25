import random
from datetime import timedelta
from math import exp, log

import polars as pl
import pytest
from conftest import utc

from galata_research import studies


def _bars(returns_by_ticker, seed=0):
    rng = random.Random(seed)
    t0 = utc("2024-01-01T00:00")
    rows = []
    for ticker, rets in returns_by_ticker.items():
        close = 100.0
        for d, r in enumerate(rets):
            open_ = close * exp(rng.gauss(0, 0.001))
            close = open_ * exp(r)
            high = max(open_, close) * exp(abs(rng.gauss(0, 0.005)))
            low = min(open_, close) * exp(-abs(rng.gauss(0, 0.005)))
            rows.append({"ticker": ticker, "ts": t0 + timedelta(days=d), "close_ts": t0 + timedelta(days=d + 1),
                         "open": open_, "high": high, "low": low, "close": close})  # fmt: skip
    return pl.DataFrame(rows)


def _shapes(frame):
    lo, lh, ll, lc = (frame[c].log().to_list() for c in ("open", "high", "low", "close"))
    return sorted((round(h - o, 12), round(l - o, 12), round(c - o, 12)) for o, h, l, c in list(zip(lo, lh, ll, lc))[1:])


def _gaps(frame):
    lo, lc = frame["open"].log().to_list(), frame["close"].log().to_list()
    return sorted(round(lo[t] - lc[t - 1], 12) for t in range(1, len(lo)))


def _walk(n, seed):
    rng = random.Random(seed)
    return [rng.gauss(0, 0.02) for _ in range(n)]


def the_last_close_is_kept():
    bars = _bars({"BTC": _walk(300, 1)})
    got = studies.permute_bars(bars, random.Random(0))
    assert got["close"][-1] == pytest.approx(bars["close"][-1])
    assert got.row(0) == bars.row(0)


def the_shapes_and_gaps_are_kept_as_multisets():
    bars = _bars({"BTC": _walk(200, 2)})
    got = studies.permute_bars(bars, random.Random(1))
    assert _shapes(got) == pytest.approx(_shapes(bars))
    assert _gaps(got) == pytest.approx(_gaps(bars))
    assert got["close"].to_list() != pytest.approx(bars["close"].to_list())


def every_permuted_bar_is_a_valid_bar():
    got = studies.permute_bars(_bars({"BTC": _walk(300, 3)}), random.Random(2))
    ok = got.select(
        (pl.col("low") <= pl.min_horizontal("open", "close") + 1e-9).all().alias("low"),
        (pl.col("high") >= pl.max_horizontal("open", "close") - 1e-9).all().alias("high"),
    )
    assert ok.row(0) == (True, True)


def the_same_permutation_serves_every_ticker():
    # Guard: a permutation per ticker would part a day's BTC bar from that day's ETH bar.
    shared = _walk(200, 4)
    bars = _bars({"BTC": shared, "ETH": [2 * r for r in shared]})
    got = studies.permute_bars(bars, random.Random(3))
    btc = got.filter(pl.col("ticker") == "BTC")["close"].log().diff().drop_nulls()
    eth = got.filter(pl.col("ticker") == "ETH")["close"].log().diff().drop_nulls()
    assert pl.DataFrame({"b": btc, "e": eth}).select(pl.corr("b", "e")).item() > 0.95


def a_ticker_on_another_calendar_is_refused():
    bars = pl.concat([_bars({"BTC": _walk(50, 5)}), _bars({"ETH": _walk(40, 6)})])
    with pytest.raises(ValueError, match="another calendar"):
        studies.permute_bars(bars, random.Random(0))


def _search(bars):
    return studies.momentum(bars, [3, 10])


def a_market_with_real_persistence_is_found():
    # AR(1) returns with coefficient 0.5: momentum has something to find, and permuting destroys it.
    rng, r, rets = random.Random(7), 0.0, []
    for _ in range(800):
        r = 0.5 * r + rng.gauss(0, 0.01)
        rets.append(r)
    got = studies.permutation_test(_bars({"BTC": rets}), _search, samples=60, seed=0)
    assert got["p_best"] < 0.05
    assert got["null"] == studies.PERMUTATION_NULL


def a_random_walk_is_not():
    got = studies.permutation_test(_bars({"BTC": _walk(800, 8)}), _search, samples=60, seed=0)
    assert got["p_best"] > 0.05


def a_result_reproduces_from_its_seed():
    bars = _bars({"BTC": _walk(200, 9)})
    first, second = (studies.permutation_test(bars, _search, samples=10, seed=5) for _ in range(2))
    assert (first["p_best"], first["permuted_bests"]) == (second["p_best"], second["permuted_bests"])
