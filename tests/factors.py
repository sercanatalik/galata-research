from datetime import timedelta

import polars as pl
import pytest
from conftest import utc

from galata_research import Refused, factors, studies

DAY = timedelta(days=1)


def _daily(closes: dict[str, list[float]], start="2024-01-01T00:00", skip: dict[str, set[int]] | None = None, volume=None):
    t0 = utc(start)
    rows = []
    for t, cs in closes.items():
        for i, c in enumerate(cs):
            if c is None or (skip and i in skip.get(t, set())):
                continue
            rows.append({"ticker": t, "ts": t0 + i * DAY, "close_ts": t0 + (i + 1) * DAY, "close": float(c), "volume": float((volume or {}).get(t, 1.0))})
    return pl.DataFrame(rows)


def a_delisted_coin_is_a_member_only_while_it_trades():
    d = _daily({"A": [1.0] * 10, "GONE": [1.0] * 5 + [None] * 5})
    m = factors.members(factors.panel(d), exclude=(), top=5, history=2, window=2)
    gone = m.filter(pl.col("ticker") == "GONE")
    assert gone["ts"].max() == utc("2024-01-05T00:00")
    assert gone["traded"].to_list() == [False, True, True, True, True]


def the_traded_set_is_the_top_by_trailing_dollar_volume():
    d = _daily({"BIG": [1.0] * 5, "MID": [1.0] * 5, "SMALL": [1.0] * 5}, volume={"BIG": 100, "MID": 10, "SMALL": 1})
    m = factors.members(factors.panel(d), exclude=(), top=2, history=1, window=1)
    last = m.filter(pl.col("ts") == utc("2024-01-05T00:00")).sort("ticker")
    assert dict(zip(last["ticker"], last["traded"])) == {"BIG": True, "MID": True, "SMALL": False}


def an_excluded_name_is_never_traded():
    d = _daily({"USDC": [1.0] * 5, "BTC": [1.0] * 5})
    m = factors.members(factors.panel(d), top=5, history=1, window=1)
    assert not m.filter(pl.col("ticker") == "USDC")["traded"].any()


def a_coin_needs_its_history_before_it_is_eligible():
    d = _daily({"NEW": [1.0] * 4})
    m = factors.members(factors.panel(d), exclude=(), top=5, history=3, window=1)
    assert m["eligible"].to_list() == [False, False, True, True]


def the_momentum_is_the_return_over_the_window_and_a_hole_voids_it():
    d = _daily({"BTC": [100, 110, 121, 133.1, 146.41, 161.051, 177.1561, 194.87171, 214.358881]})
    s = factors.scores(factors.panel(d), factors.members(factors.panel(d), exclude=(), top=5, history=1, window=1))
    assert s["mom 7"][7] == pytest.approx(1.1**7 - 1)
    assert s["mom 7"][6] is None
    holed = _daily({"BTC": [100.0 * 1.1**i for i in range(10)]}, skip={"BTC": {5}})
    hs = factors.scores(factors.panel(holed), factors.members(factors.panel(holed), exclude=(), top=5, history=1, window=1))
    # The 7-day window ending after the hole holds the null return: no figure.
    assert hs["mom 7"].drop_nulls().len() == 0


def every_score_is_known_at_its_close():
    # Guard: a score that read a later bar would change an earlier value when the bars are cut short.
    import random

    rng = random.Random(1)
    closes = {t: [100.0] for t in ("BTC", "A", "B", "C")}
    for t in closes:
        for _ in range(140):
            closes[t].append(closes[t][-1] * (1 + rng.gauss(0, 0.03)))
    full = _daily(closes)
    cut = full.filter(pl.col("ts") < utc("2024-04-15T00:00"))
    sf = factors.scores(factors.panel(full), factors.members(factors.panel(full), exclude=(), top=4, history=1, window=30))
    sc = factors.scores(factors.panel(cut), factors.members(factors.panel(cut), exclude=(), top=4, history=1, window=30))
    a = sf.join(sc, on=["ticker", "ts"], how="inner", suffix="_cut")
    for col in factors.SCORES:
        assert a[col].to_list() == pytest.approx(a[f"{col}_cut"].to_list(), nan_ok=True), col


def the_long_short_holds_the_top_fifth_long_and_the_bottom_short():
    names = [f"C{i}" for i in range(10)]
    d = _daily({n: [100.0, 100.0 * (1 + i / 100)] + [100.0 * (1 + i / 100)] * 8 for i, n in enumerate(names)} | {"BTC": [100.0] * 10})
    p = factors.panel(d)
    m = factors.members(p, exclude=(), top=11, history=2, window=1)
    s = factors.scores(p, m)
    on = [utc("2024-01-02T00:00")]
    # A known ranking stands in for a score: coin i scores i.
    s1 = s.with_columns(pl.col("ticker").str.strip_chars("C").cast(pl.Int64, strict=False).cast(pl.Float64).alias("mom 7"))
    w = factors.weights(m, s1, "mom 7", on, "long_short")
    # 10 scored coins (BTC has no number): round(10 × 0.2) = 2 a side, ±0.25 each.
    got = dict(zip(w["ticker"], w["w"]))
    assert got == {"C9": 0.25, "C8": 0.25, "C0": -0.25, "C1": -0.25}


def the_portfolio_is_held_from_the_next_day_and_pays_its_turnover_and_funding():
    d = _daily({"A": [100, 100, 110, 121, 121], "B": [100, 100, 100, 100, 100]})
    settled = pl.DataFrame({"ticker": "A", "ts": [utc("2024-01-04T00:00")], "rate": [0.001], "interval_hours": [8]})
    p = factors.panel(d, settled)
    w = pl.DataFrame({"rebalance": [utc("2024-01-02T00:00")], "ticker": ["A"], "w": [1.0]})
    got = factors.portfolio(p, w, [utc("2024-01-02T00:00")], "2024-01-05T00:00Z", "t", fee=0.001)
    # Held on the 3rd and 4th: A's +10% then +10%. The 3rd pays one unit of turnover; the 3rd's funding settled at midnight on the 4th.
    assert got["ts"].to_list() == [utc("2024-01-03T00:00"), utc("2024-01-04T00:00")]
    assert got["gross"].to_list() == [pytest.approx(0.1), pytest.approx(0.1)]
    assert got["cost"].to_list() == [pytest.approx(0.001), 0.0]
    assert got["funding"].to_list() == [pytest.approx(0.001), 0.0]
    assert got["funding_missing"].to_list() == [0, 1]
    assert got["net"][0] == pytest.approx(0.1 - 0.001 - 0.001)


def a_delisted_holding_earns_nothing_after_its_last_bar():
    d = _daily({"GONE": [100, 100, 50, None, None], "B": [100] * 5})
    p = factors.panel(d)
    w = pl.DataFrame({"rebalance": [utc("2024-01-01T00:00")], "ticker": ["GONE"], "w": [-1.0]})
    got = factors.portfolio(p, w, [utc("2024-01-01T00:00")], "2024-01-05T00:00Z", "t", fee=0.0)
    # Held the 2nd to the 4th: −1 × 0, then −1 × −50% on the 3rd, then nothing on the 4th, after its last bar.
    assert got["gross"].to_list() == [0.0, pytest.approx(0.5), 0.0]


def the_turnover_is_against_the_previous_weights():
    d = _daily({"A": [100] * 20, "B": [100] * 20})
    p = factors.panel(d)
    on = [utc("2024-01-01T00:00"), utc("2024-01-08T00:00")]
    w = pl.DataFrame({"rebalance": [on[0], on[1], on[1]], "ticker": ["A", "A", "B"], "w": [1.0, 0.5, 0.5]})
    got = factors.portfolio(p, w, on, "2024-01-15T00:00Z", "t", fee=1.0)
    # First: 0 → A 1 (one unit). Second: A 1 → 0.5, B 0 → 0.5 (one unit).
    assert got.filter(pl.col("cost") > 0)["ts"].to_list() == [utc("2024-01-02T00:00"), utc("2024-01-09T00:00")]
    assert got["cost"].sum() == pytest.approx(2.0)


def the_rebalances_start_at_the_first_full_set_and_step_a_week():
    m = pl.DataFrame({"ticker": ["A", "B"] * 20, "ts": [utc("2024-01-01T00:00") + (i // 2) * DAY for i in range(40)], "traded": [i >= 10 for i in range(40)]})
    on = factors.rebalances(m, "2024-01-01T00:00Z", "2024-01-20T00:00Z", top=2)
    assert on == [utc("2024-01-06T00:00"), utc("2024-01-13T00:00")]
    with pytest.raises(Refused, match="traded set of 3"):
        factors.rebalances(m, "2024-01-01T00:00Z", "2024-01-20T00:00Z", top=3)


def the_registered_search_is_twelve_rules_and_a_benchmark():
    import random

    rng = random.Random(4)
    names = ["BTC"] + [f"C{i}" for i in range(24)]
    closes = {}
    for n in names:
        c, xs = 100.0, []
        for _ in range(200):
            c *= 1 + rng.gauss(0, 0.04)
            xs.append(c)
        closes[n] = xs
    d = _daily(closes, volume={n: rng.uniform(1, 100) for n in names})
    p = factors.panel(d)
    m = factors.members(p, exclude=(), top=20, history=90, window=30)
    got = factors.rules(p, m, "2024-01-01T00:00Z", "2024-07-15T00:00Z", top=20)
    assert got["trial"].n_unique() == 13
    assert studies.summary(got).height == 13
    ls = got.filter(pl.col("trial").str.ends_with("long_short") & (pl.col("position") > 0))
    assert ls["position"].max() == pytest.approx(1.0)
