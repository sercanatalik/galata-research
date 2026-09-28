"""gr.liquidity: the spread measured from quotes and trades, and estimated from bars.

EDGE is pinned to its authors' test values on their own series (`tests/data/`).
Roll, Corwin–Schultz and Abdi–Ranaldo are checked against an independent
computation on a few bars each.
"""

import math
import random
import statistics
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest
from conftest import utc

from galata_research import Refused, liquidity

DATA = Path(__file__).parent / "data"
MINUTE = timedelta(minutes=1)


def _bars(ohlc, start="2026-09-20T12:00", ticker="BTC", skip=(), gap=()):
    """1m bars from (open, high, low, close) rows; minutes in `skip` are left out, those in `gap` marked in_gap."""
    t0 = utc(start)
    rows = [
        {"ticker": ticker, "ts": t0 + i * MINUTE, "close_ts": t0 + (i + 1) * MINUTE, "open": o, "high": h, "low": low,
         "close": c, "in_gap": i in gap}
        for i, (o, h, low, c) in enumerate(ohlc)
        if i not in skip
    ]  # fmt: skip
    return pl.DataFrame(rows)


# ---- EDGE, as its authors compute it ------------------------------------------------


def the_authors_edge_values():
    full = pl.read_csv(DATA / "bidask-ohlc.csv")
    miss = pl.read_csv(DATA / "bidask-ohlc-miss.csv", null_values=["NA", ""])
    cols = ["Open", "High", "Low", "Close"]
    assert liquidity.edge(*[full[c].to_list() for c in cols]) == pytest.approx(0.0101849034905478, abs=1e-12)
    assert liquidity.edge(*[full[c][:10].to_list() for c in cols], sign=True) == pytest.approx(-0.016889917516422, abs=1e-12)
    assert liquidity.edge(*[miss[c].to_list() for c in cols]) == pytest.approx(0.01013284969780197, abs=1e-12)


def a_flat_series_has_no_edge():
    assert liquidity.edge([18.21, 17.61, 17.61], [18.21, 17.61, 17.61], [17.61] * 3, [17.61] * 3) is None


def every_bucket_agrees_with_the_reference_edge():
    rng = random.Random(7)
    price, ohlc = 100.0, []
    for _ in range(180):  # three hours of 1m bars with a bid-ask bounce
        path = [price]
        for _ in range(10):
            price *= math.exp(rng.gauss(0, 2e-4))
            path.append(price * (1 + rng.choice((-1, 1)) * 5e-5))
        ohlc.append((path[0], max(path), min(path), path[-1]))
    got = liquidity.from_bars(_bars(ohlc), "edge", "1h")
    assert got.height == 3
    for k, row in enumerate(got.iter_rows(named=True)):
        lo = 0 if k == 0 else 60 * k - 1  # the bucket's bars and the bar before its first
        want = liquidity.edge(*[[b[i] for b in ohlc[lo : 60 * (k + 1)]] for i in range(4)])
        assert row["spread"] == pytest.approx(want, rel=1e-9)
        assert row["pairs"] == (59 if k == 0 else 60)


# ---- the other estimators, by hand ----------------------------------------------------


def the_roll_arithmetic():
    closes = [100 * math.exp(x) for x in (0, 0.001, 0, 0.001, 0)]
    got = liquidity.from_bars(_bars([(c, c, c, c) for c in closes]), "roll", "1h").row(0, named=True)
    d = [math.log(b) - math.log(a) for a, b in zip(closes, closes[1:], strict=False)]
    cov = statistics.covariance(d[1:], d[:-1])
    assert got["pairs"] == 3
    assert got["spread"] == pytest.approx(2 * math.sqrt(-cov))


def a_roll_without_negative_covariance_is_null():
    got = liquidity.from_bars(_bars([(100, 100, 100, 100)] * 4), "roll", "1h").row(0, named=True)
    assert got["spread"] is None


def the_corwin_schultz_arithmetic():
    got = liquidity.from_bars(_bars([(100, 101, 99, 100), (101, 102, 100, 101)]), "corwin_schultz", "1h").row(0, named=True)
    beta = math.log(101 / 99) ** 2 + math.log(102 / 100) ** 2
    gamma = math.log(102 / 99) ** 2
    k = 3 - 2 * math.sqrt(2)
    alpha = (math.sqrt(2 * beta) - math.sqrt(beta)) / k - math.sqrt(gamma / k)
    want = max(2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha)), 0.0)
    assert got["pairs"] == 1 and got["spread"] == pytest.approx(want)


def the_abdi_ranaldo_arithmetic():
    got = liquidity.from_bars(_bars([(100, 101, 99, 100.5), (100.5, 100.8, 99.8, 100.2)]), "abdi_ranaldo", "1h").row(0, named=True)
    c1, m1, m = math.log(100.5), (math.log(101) + math.log(99)) / 2, (math.log(100.8) + math.log(99.8)) / 2
    assert got["spread"] == pytest.approx(math.sqrt(max(4 * (c1 - m1) * (c1 - m), 0)))
    assert got["spread_bps"] == pytest.approx(got["spread"] * 1e4)


def a_hole_breaks_the_pair():
    bars = _bars([(100, 101, 99, 100.5), (100.5, 100.8, 99.8, 100.2), (100.2, 100.9, 99.9, 100.4)], skip={1})
    assert liquidity.from_bars(bars, "abdi_ranaldo", "1h")["pairs"].to_list() == [0]


def a_bar_in_a_gap_counts_for_nothing():
    bars = _bars([(100, 101, 99, 100.5), (100.5, 100.8, 99.8, 100.2), (100.2, 100.9, 99.9, 100.4)], gap={1})
    assert liquidity.from_bars(bars, "corwin_schultz", "1h")["pairs"].to_list() == [0]


def an_unknown_estimator_is_refused():
    with pytest.raises(Refused, match="roll, corwin_schultz, abdi_ranaldo, edge"):
        liquidity.from_bars(_bars([(1, 1, 1, 1)] * 3), "hasbrouck", "1h")


# ---- measured spreads -------------------------------------------------------------------


def _quotes(rows):
    return pl.DataFrame(
        [{"venue": "v", "ticker": "BTC", "ts": utc(t), "bid_px": b, "ask_px": a} for t, b, a in rows]
    )


def _trades(rows):
    return pl.DataFrame(
        [{"venue": "v", "ticker": "BTC", "ts": utc(t), "price": p, "size": 1.0, "aggressor": s} for t, p, s in rows]
    )


def the_quoted_spread_of_one_tick():
    got = liquidity.quoted(_quotes([("2026-09-26T00:00:00", 84059.9, 84060.0)])).row(0, named=True)
    assert got["mid"] == pytest.approx(84059.95)
    assert got["spread_bps"] == pytest.approx(0.1 / 84059.95 * 1e4)


def the_prevailing_quote_is_strictly_earlier():
    quotes = _quotes([("2026-09-26T12:00:00.000", 99, 101), ("2026-09-26T12:00:00.005", 101, 103)])
    got = liquidity.effective(_trades([("2026-09-26T12:00:00.005", 101, "bid")]), quotes, horizons=()).row(0, named=True)
    assert got["mid"] == 100 and got["effective_bps"] == pytest.approx(200)


def a_stale_quote_is_not_used():
    quotes = _quotes([("2026-09-26T12:00:00", 99, 101)])
    got = liquidity.effective(_trades([("2026-09-26T12:00:03", 101, "bid")]), quotes, horizons=()).row(0, named=True)
    assert got["effective_bps"] is None


def the_realized_spread_is_known_later():
    quotes = _quotes([("2026-09-26T12:00:00", 99, 101), ("2026-09-26T12:00:04", 100, 102)])
    got = liquidity.effective(_trades([("2026-09-26T12:00:01", 101, "bid")]), quotes, horizons=("5s",), tolerance="10s").row(0, named=True)
    assert got["realized_bps_5s"] == pytest.approx(0)
    assert got["impact_bps_5s"] == pytest.approx(200)
    assert got["known_ts_5s"] == utc("2026-09-26T12:00:06")


def a_sell_crossing_pays_the_spread_too():
    quotes = _quotes([("2026-09-26T12:00:00", 99, 101)])
    got = liquidity.effective(_trades([("2026-09-26T12:00:01", 99, "ask")]), quotes, horizons=()).row(0, named=True)
    assert got["effective_bps"] == pytest.approx(200)


def an_aggressor_of_another_word_is_refused():
    with pytest.raises(Refused, match="'buy' is neither bid nor ask"):
        liquidity.effective(_trades([("2026-09-26T12:00:01", 99, "buy")]), _quotes([("2026-09-26T12:00:00", 99, 101)]))


# ---- bars from trades ----------------------------------------------------------------------


def a_quiet_minute_has_no_bar():
    got = liquidity.bars(_trades([("2026-09-26T12:00:10", 100, "bid"), ("2026-09-26T12:02:30", 101, "ask")]), "1m")
    assert got["ts"].to_list() == [utc("2026-09-26T12:00"), utc("2026-09-26T12:02")]
    assert got["close_ts"][0] == utc("2026-09-26T12:01")


def the_bars_keep_the_trade_order():
    same = "2026-09-26T12:00:10"
    got = liquidity.bars(_trades([(same, 100, "bid"), (same, 102, "bid"), (same, 99, "ask"), (same, 101, "ask")]), "1m").row(0, named=True)
    assert (got["open"], got["high"], got["low"], got["close"], got["trade_count"]) == (100, 102, 99, 101, 4)


# ---- the cost of a size ---------------------------------------------------------------------


def _depth(points):
    return pl.DataFrame({"ts": [utc("2026-09-26T12:00")] * len(points), "band_pct": [p for p, _ in points], "notional": [n for _, n in points]})


def the_cost_within_the_first_band():
    got = liquidity.cost_of_size(_depth([(1.0, 10e6)]), [1e6]).row(0, named=True)
    assert got["side"] == "buy" and got["reach_bps"] == pytest.approx(10) and got["cost_bps"] == pytest.approx(5)


def the_cost_across_two_bands():
    got = liquidity.cost_of_size(_depth([(1.0, 1e6), (2.0, 3e6)]), [2e6]).row(0, named=True)
    assert got["reach_bps"] == pytest.approx(150) and got["cost_bps"] == pytest.approx(87.5)


def a_size_beyond_the_book_has_no_cost():
    got = liquidity.cost_of_size(_depth([(1.0, 1e6), (5.0, 2e6)]), [3e6]).row(0, named=True)
    assert got["beyond"] and got["cost_bps"] is None and got["reach_bps"] is None


def the_sides_follow_the_sign():
    got = liquidity.cost_of_size(_depth([(-1.0, 2e6), (1.0, 1e6)]), [1e6]).sort("side")
    assert got["side"].to_list() == ["buy", "sell"]
    assert got["cost_bps"].to_list() == pytest.approx([50, 25])


def a_band_past_the_reach_is_left_out():
    book = pl.DataFrame({
        "ts": [utc("2026-09-26T12:00")], "bid_px": [99.99], "ask_px": [100.01], "bid_sz": [1.0], "ask_sz": [2.0],
        "bid_depth_2bps": [5.0], "ask_depth_2bps": [6.0], "bid_depth_10bps": [9.0], "ask_depth_10bps": [None],
    }, schema_overrides={"ask_depth_10bps": pl.Float64})  # fmt: skip
    got = liquidity.book_points(book)
    asks = got.filter(pl.col("band_pct") > 0).sort("band_pct")
    assert asks["band_pct"].to_list() == pytest.approx([0.01, 0.02]) and asks["notional"].to_list() == pytest.approx([200, 600])
    assert got.filter(pl.col("band_pct") < 0).height == 3


# ---- signed flow, Kyle's lambda, Amihud -------------------------------------------------------


def the_buys_and_sells_in_a_minute():
    got = liquidity.flow(_trades([("2026-09-26T12:00:10", 100, "bid"), ("2026-09-26T12:00:20", 100, "ask")]).with_columns(
        pl.Series("size", [3.0, 1.0])), "1m").row(0, named=True)  # fmt: skip
    assert got["signed_notional"] == 200 and got["imbalance"] == pytest.approx(0.5)


def an_empty_minute_breaks_the_return():
    got = liquidity.flow(_trades([("2026-09-26T12:00:10", 100, "bid"), ("2026-09-26T12:01:10", 101, "bid"), ("2026-09-26T12:03:10", 102, "bid")]), "1m")
    assert got["return_bps"][1] == pytest.approx(1e4 * math.log(101 / 100))
    assert got["return_bps"][2] is None and got["return_bps"][0] is None


def a_known_slope_is_recovered():
    flows = pl.DataFrame({"signed_notional": [-2e6, -1e6, 0.0, 1e6, 3e6], "return_bps": [-6.0, -3.0, 0.0, 3.0, 9.0]})
    got = liquidity.kyle_lambda(flows).row(0, named=True)
    assert got["lambda_bps_per_m"] == pytest.approx(3) and got["r2"] == pytest.approx(1) and got["n"] == 5


def the_lambda_per_group():
    flows = pl.DataFrame({"hod": [1, 1, 1, 2, 2, 2], "signed_notional": [1e6, 2e6, 3e6, 1e6, 2e6, 3e6], "return_bps": [1.0, 2.0, 3.0, 5.0, 10.0, 15.0]})
    got = liquidity.kyle_lambda(flows, by="hod")
    assert got["lambda_bps_per_m"].to_list() == pytest.approx([1, 5])


def the_amihud_ratio_by_hand():
    flows = pl.DataFrame({"buy_notional": [1e6, 1e6], "sell_notional": [0.0, 1e6], "return_bps": [2.0, -4.0]})
    assert liquidity.amihud(flows)["amihud_bps_per_m"][0] == pytest.approx(2)


def the_flow_refuses_another_word():
    with pytest.raises(Refused, match="'buy' is neither bid nor ask"):
        liquidity.flow(_trades([("2026-09-26T12:00:10", 100, "buy")]), "1m")


# ---- shocks and resilience -------------------------------------------------------------------


def _book_rows(t0, asks):
    return pl.DataFrame({
        "venue": "v", "ticker": "BTC", "ts": [utc(t0) + timedelta(seconds=i) for i in range(len(asks))],
        "bid_px": 99.99, "ask_px": 100.01, "ask_depth_2bps": asks, "bid_depth_2bps": 10.0,
    })  # fmt: skip


def the_largest_second_is_a_shock():
    flows = pl.DataFrame({"venue": "v", "ticker": "BTC", "ts": [utc("2026-09-26T12:00:00") + timedelta(seconds=i) for i in range(1000)],
                          "signed_notional": [5e6 if i == 500 else 1e3 for i in range(1000)]})  # fmt: skip
    got = liquidity.shocks(flows, share=0.001)
    assert got.height == 1 and got["side"][0] == "ask" and got["ts"][0] == utc("2026-09-26T12:08:20")


def a_book_that_refills_in_four_seconds():
    book = _book_rows("2026-09-26T12:00:00", [10.0, 2.0, 5.0, 8.0, 9.5, 10.0])
    events = pl.DataFrame({"venue": ["v"], "ticker": ["BTC"], "ts": [utc("2026-09-26T12:00:00")], "side": ["ask"]})
    got = liquidity.resilience(book, events, horizon=5).row(0, named=True)
    assert got["depth_min_ratio"] == pytest.approx(0.2) and got["recovery_s"] == 4


def a_book_that_does_not_refill():
    book = _book_rows("2026-09-26T12:00:00", [10.0] + [5.0] * 10)
    events = pl.DataFrame({"venue": ["v"], "ticker": ["BTC"], "ts": [utc("2026-09-26T12:00:00")], "side": ["ask"]})
    got = liquidity.resilience(book, events, horizon=10).row(0, named=True)
    assert got["depth_min_ratio"] == pytest.approx(0.5) and got["recovery_s"] is None


def every_day_gets_as_many_quiet_placebo_seconds_as_shocks():
    flows = pl.DataFrame({"venue": "v", "ticker": "BTC", "ts": [utc("2026-09-26T12:00:00") + timedelta(seconds=i) for i in range(5000)],
                          "signed_notional": [float((i * 7919) % 1000) - 500 for i in range(5000)]})  # fmt: skip
    shock = liquidity.shocks(flows, share=0.001)
    got = liquidity.placebo(flows, shock, seed=3)
    median = flows["signed_notional"].abs().median()
    assert got.height == shock.height and (got["signed_notional"].abs() < median).all()
    assert got["ts"].to_list() == liquidity.placebo(flows, shock, seed=3)["ts"].to_list()


# ---- schedules -------------------------------------------------------------------------------


def _hours(values):
    t0 = utc("2026-09-26T00:00")
    return [t0 + timedelta(hours=i) for i in range(len(values))]


def the_allocation_two_to_one():
    assert liquidity.allocate(300, [2, 1]) == pytest.approx([200, 100])


def no_negative_weight():
    with pytest.raises(Refused, match="non-negative"):
        liquidity.allocate(100, [1, -1])


def a_uniform_plan_on_a_flat_book():
    ts = _hours([0] * 24)
    plan = pl.DataFrame({"ts": ts, "notional": [1e6] * 24})
    depth = pl.DataFrame({"ts": ts, "depth": [100e6] * 24})
    got = liquidity.schedule_cost(plan, depth, slices=60)
    assert got["cost_bps"] == pytest.approx((1e6 / 60) / 100e6 * 50) and got["over_depth"] == 0


def the_depth_weighting_beats_uniform_on_an_uneven_book():
    ts = _hours([0] * 24)
    d = [100e6] * 12 + [50e6] * 12
    depth = pl.DataFrame({"ts": ts, "depth": d})
    uniform = liquidity.schedule_cost(pl.DataFrame({"ts": ts, "notional": liquidity.allocate(24e6, [1] * 24)}), depth)
    weighted = liquidity.schedule_cost(pl.DataFrame({"ts": ts, "notional": liquidity.allocate(24e6, d)}), depth)
    assert weighted["cost_bps"] < uniform["cost_bps"]


def a_plan_hour_without_depth_is_refused():
    ts = _hours([0, 0])
    with pytest.raises(Refused, match="no depth for the plan hour"):
        liquidity.schedule_cost(pl.DataFrame({"ts": ts, "notional": [1.0, 1.0]}), pl.DataFrame({"ts": ts[:1], "depth": [1.0]}))


# ── order-flow imbalance (measure-the-order-flow) ────────────────────────────


def _book(rows, start="2026-09-28T07:00:00"):
    """(seconds, bid_px, bid_sz, ask_px, ask_sz) as BTC quotes."""
    t0 = utc(start)
    return pl.DataFrame(
        [{"venue": "hyperliquid", "ticker": "BTC", "ts": t0 + timedelta(seconds=s), "bid_px": bp, "bid_sz": bs, "ask_px": ap, "ask_sz": az} for s, bp, bs, ap, az in rows],
        schema={"venue": pl.String, "ticker": pl.String, "ts": pl.Datetime("us", "UTC"), "bid_px": pl.Float64, "bid_sz": pl.Float64, "ask_px": pl.Float64, "ask_sz": pl.Float64},
    )


def the_order_flow_is_cont_kukanov_and_stoikovs():
    b = _book([
        (0, 100, 5, 101, 5),
        (1, 100, 7, 101, 5),   # bid queue grows by 2: +2
        (2, 100.5, 3, 101, 5),  # bid steps up: +3 (the new queue)
        (3, 100.5, 3, 100.8, 4),  # ask steps down: −4 (the new queue)
        (4, 100.5, 3, 100.8, 1),  # ask queue shrinks by 3: +3
        (5, 100.2, 6, 100.8, 1),  # bid steps down: −3 (the old queue)
    ])  # fmt: skip
    out = liquidity.ofi(b, "10s")
    assert out["ofi"].to_list() == [2 + 3 - 4 + 3 - 3]
    assert out["events"].to_list() == [6]
    assert out["depth"][0] == pytest.approx(statistics.mean([(5 + 5) / 2, (7 + 5) / 2, (3 + 5) / 2, (3 + 4) / 2, (3 + 1) / 2, (6 + 1) / 2]))


def no_order_flow_across_a_crossed_book_or_a_gap():
    b = _book([
        (0, 100, 5, 101, 5),
        (1, 101, 5, 101, 5),   # locked: dropped
        (2, 100, 9, 101, 5),   # after the locked state: no term, not +4
        (20, 100, 1, 101, 5),  # after a 18 s gap: no term, not −8
    ])  # fmt: skip
    out = liquidity.ofi(b, "10s", max_gap_us=5_000_000)
    assert out["ofi"].to_list() == [0.0, 0.0]


def the_mid_return_spans_consecutive_buckets_only():
    b = _book([(0, 100, 1, 101, 1), (12, 101, 1, 102, 1), (45, 99, 1, 100, 1)])
    out = liquidity.ofi(b, "10s")
    assert out["return_bps"][1] == pytest.approx(1e4 * math.log(101.5 / 100.5))
    assert out["return_bps"][0] is None and out["return_bps"][2] is None  # the bucket before 40 s held no update


def the_impact_regression_recovers_its_slope():
    rng = random.Random(9)
    xs = [rng.gauss(0, 1) for _ in range(500)]
    frame = pl.DataFrame({"ofi_norm": xs, "return_bps": [0.3 + 2.0 * x + rng.gauss(0, 0.5) for x in xs]})
    fit = liquidity.impact(frame)
    assert fit["beta"] == pytest.approx(2.0, abs=0.08)
    assert fit["r2"] == pytest.approx(4 / 4.25, abs=0.02)  # var(βx)/var(y) = 4/(4 + 0.25)
    assert fit["t"] > 30 and fit["n"] == 500


def no_impact_from_too_few_buckets_or_a_flat_regressor():
    with pytest.raises(Refused, match="under 100"):
        liquidity.impact(pl.DataFrame({"ofi_norm": [1.0] * 50, "return_bps": [1.0] * 50}))
    with pytest.raises(Refused, match="does not vary"):
        liquidity.impact(pl.DataFrame({"ofi_norm": [1.0] * 200, "return_bps": [float(i) for i in range(200)]}))
