import random
from datetime import timedelta

import polars as pl
import pytest
from conftest import utc

from galata_research import Refused, flow

SEC = timedelta(seconds=1)


def _book(rows, ticker="BTC", start="2026-01-07T00:00:00"):
    """rows: (bid, ask, bid_sz, ask_sz) per consecutive second."""
    t0 = utc(start)
    return pl.DataFrame(
        {
            "venue": "bybit-linear",
            "ticker": ticker,
            "ts": [t0 + i * SEC for i in range(len(rows))],
            "bid_px": [float(r[0]) for r in rows],
            "ask_px": [float(r[1]) for r in rows],
            "bid_sz": [float(r[2]) for r in rows],
            "ask_sz": [float(r[3]) for r in rows],
        }
    )


def the_student_tail_matches_the_tables():
    # t-table critical values: 11 df at 0.025 and 0.005; 1 df at 0.025.
    assert flow.t_sf(2.201, 11) == pytest.approx(0.025, abs=2e-5)
    assert flow.t_sf(3.106, 11) == pytest.approx(0.005, abs=2e-5)
    assert flow.t_sf(12.706, 1) == pytest.approx(0.025, abs=2e-5)
    assert flow.t_sf(-2.201, 11) == pytest.approx(0.975, abs=2e-5)


def the_weighted_mid_leans_toward_the_thinner_side():
    g = flow.seconds(_book([(100, 101, 9, 1)]))
    # Nine on the bid, one on the ask: the price is likelier to rise, so the weighted mid sits near the ask.
    assert g["wmid"][0] == pytest.approx((100 * 1 + 101 * 9) / 10)
    assert g["mid"][0] == 100.5


def a_crossed_second_is_not_a_second():
    assert flow.seconds(_book([(100, 101, 1, 1), (101, 101, 1, 1), (100, 101, 1, 1)])).height == 2


def the_leaders_price_is_its_last_trade_known_at_the_second():
    b = _book([(100, 101, 1, 1)] * 10)
    t0 = utc("2026-01-07T00:00:00")
    trades = pl.DataFrame({"ticker": "BTC", "ts": [t0 + timedelta(milliseconds=500), t0 + 3 * SEC, t0 + 3 * SEC + timedelta(milliseconds=1)], "price": [50.0, 51.0, 52.0]})
    g = flow.seconds(b, trades, stale="5s")
    # Second 0 precedes the first trade; second 3 sees the trade stamped exactly then; second 4 the one just after.
    # Guard: a forward join would give second 0 the price at 0.5 s.
    assert g["leader_px"].to_list()[:5] == [None, 50.0, 50.0, 51.0, 52.0]
    # More than 5 s after the last trade, the price is stale and missing.
    assert g["leader_px"][9] is None


def a_slope_and_its_newey_west_t():
    rng = random.Random(2)
    xs = [rng.gauss(0, 1) for _ in range(2000)]
    frame = pl.DataFrame({"day": [i // 1000 for i in range(2000)], "x": xs, "y": [2 * x + rng.gauss(0, 1) for x in xs]})
    got = flow.slope(frame, "x", "y")
    assert got["slope"] == pytest.approx(2.0, abs=0.06)
    assert got["t"] > 30 and got["p"] < 1e-10
    noise = frame.with_columns(pl.Series("y", [rng.gauss(0, 1) for _ in range(2000)]))
    assert abs(flow.slope(noise, "x", "y")["t"]) < 3


def a_newey_west_sum_does_not_reach_across_days():
    # Guard: two days each [1, -1]; within a day the lag-one cross is −1, across the boundary it would add −1 more.
    z = pl.DataFrame({"day": [0, 0, 1, 1], "z": [1.0, -1.0, 1.0, -1.0]})
    assert flow._nw_sum(z, 1) == pytest.approx(4 + 2 * 0.5 * (-2))


def a_taker_pays_the_spread_twice_and_the_fee_both_ways():
    g = flow.seconds(_book([(100, 100.1, 1, 1), (100, 100.1, 1, 1), (100.2, 100.3, 1, 1)]))
    signals = pl.DataFrame({"ticker": "BTC", "ts": [utc("2026-01-07T00:00:00")], "side": [1]})
    got = flow.taker(g, signals, 2)
    # In at the ask 100.1, out at the bid 100.2: +9.99 bps gross, less 2 × 5.5 bps.
    assert got["gross_bps"][0] == pytest.approx(1e4 * (100.2 / 100.1 - 1))
    assert got["net_bps"][0] == pytest.approx(1e4 * (100.2 / 100.1 - 1) - 11)
    short = flow.taker(g, signals.with_columns(pl.lit(-1).alias("side")), 2)
    assert short["gross_bps"][0] == pytest.approx(-1e4 * (100.3 / 100.0 - 1))


def no_trade_closes_after_midnight_or_on_a_missing_second():
    g = flow.seconds(_book([(100, 101, 1, 1)] * 3, start="2026-01-07T23:59:58"))
    signals = pl.DataFrame({"ticker": "BTC", "ts": [utc("2026-01-07T23:59:58"), utc("2026-01-07T23:59:59")], "side": [1, 1]})
    got = flow.taker(g, signals, 1)
    # The first closes at 23:59:59. The second would close at 00:00:00 on the next day's replay.
    assert got["ts"].to_list() == [utc("2026-01-07T23:59:58")]


def a_rule_is_scored_on_its_daily_means():
    trades = pl.DataFrame(
        {"rule": "r", "ticker": "BTC", "day": [d for d in range(4) for _ in range(10)], "net_bps": [1.0 + 0.1 * d for d in range(4) for _ in range(10)]}
    )
    got = flow.score(trades).row(0, named=True)
    means = [1.0, 1.1, 1.2, 1.3]
    sd = (sum((m - 1.15) ** 2 for m in means) / 3) ** 0.5
    assert got["t"] == pytest.approx(1.15 / (sd / 2))
    assert got["p"] == pytest.approx(flow.t_sf(got["t"], 3))
    few = flow.score(trades.head(20)).row(0, named=True)
    assert few["mean_bps"] is None and few["trades"] == 20


def the_ofi_bucket_is_decided_at_its_last_second_and_scored_on_the_next():
    rows = [(100, 101, 1, 1)] * 10 + [(100, 101, 5, 1)] * 10 + [(102, 103, 1, 1)] * 10
    got = flow.ofi_frame(_book(rows), every=10)
    assert got["decide_ts"].to_list() == [utc("2026-01-07T00:00:09"), utc("2026-01-07T00:00:19"), utc("2026-01-07T00:00:29")]
    # Bucket 1's OFI is the bid growing from 1 to 5; bucket 2's mid return is ln(102.5/100.5).
    assert got["x"][1] > 0
    assert got["y"][1] == pytest.approx(1e4 * __import__("math").log(102.5 / 100.5))
    assert got["y"][2] is None


def the_ofi_z_score_leaves_the_bucket_itself_out():
    # Guard: with the bucket in its own window, the z of a lone spike after a flat window would be capped near √n.
    frame = pl.DataFrame(
        {"ticker": "BTC", "day": utc("2026-01-07"), "ts": [utc("2026-01-07T00:00:00") + i * 10 * SEC for i in range(6)], "x": [0.0, 1.0, 0.0, 1.0, 0.0, 9.0]}
    ).with_columns((pl.col("ts") + 9 * SEC).alias("decide_ts"), pl.lit(None, pl.Float64).alias("y"))
    g = flow.seconds(_book([(100, 101, 1, 1)] * 200))
    got = flow.ofi_rules(frame, g, thresholds=[10.0], horizons=[10], window=5)
    # The previous 5 have mean 0.4, sd 0.548: z = 15.7, above 10. Counted in its own window, z would be 1.8.
    assert got.height == 1


def the_lead_is_the_leaders_last_second_against_bybits_next():
    b = _book([(100, 101, 1, 1), (100, 101, 1, 1), (101, 102, 1, 1), (101, 102, 1, 1)])
    t0 = utc("2026-01-07T00:00:00")
    trades = pl.DataFrame({"ticker": "BTC", "ts": [t0, t0 + SEC, t0 + 2 * SEC, t0 + 3 * SEC], "price": [50.0, 51.0, 51.0, 51.0]})
    got = flow.lead_frame(flow.seconds(b, trades))
    # Second 1: the leader rose 50 → 51; Bybit's mid rises over second 2.
    assert got["x"][1] == pytest.approx(1e4 * __import__("math").log(51 / 50))
    assert got["y"][1] == pytest.approx(1e4 * __import__("math").log(101.5 / 100.5))
    # Guard: y is the next second's move, not the same second's (which is 0 here).
    assert got["y"][0] == 0.0


def the_weighted_mid_beats_the_mid_when_the_imbalance_predicts():
    # Thin ask, then the price ticks up: the weighted mid was nearer the later mid.
    rows = [(100, 101, 9, 1), (101, 102, 1, 1)] * 20
    got = flow.microprice_frame(flow.seconds(_book(rows)), 1)
    assert got.filter(pl.col("ts").dt.second() % 2 == 0)["d"].min() > 0


def a_slope_with_too_few_rows_is_refused():
    with pytest.raises(Refused, match="at least 30"):
        flow.slope(pl.DataFrame({"day": [0] * 5, "x": [1.0] * 5, "y": [1.0] * 5}), "x", "y")
