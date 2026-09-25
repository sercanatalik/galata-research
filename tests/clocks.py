import polars as pl
import pytest
from conftest import MARKS, utc

import galata_research as gr
from galata_research import Refused, market

DAY = utc("2026-09-25T00:00"), utc("2026-09-26T00:00")


# Settled and live funding


def a_live_rate_is_not_a_settled_one(tape):
    # Guard: without the at_micros split, the live pushes are read as settlements.
    tape.settled_rate("BTC", "2026-09-25T06:00:00.121", "2026-09-25T06:27:34", "0.0000125")
    tape.live_rate("BTC", "2026-09-25T06:30:00", "0.0000131")
    tape.write()
    assert market.funding("BTC", *DAY).collect().select("ts", "rate").rows() == [(utc("2026-09-25T06:00:00.121"), 0.0000125)]
    assert market.funding_live("BTC", *DAY).collect()["rate"].to_list() == [0.0000131]


def a_settlement_after_as_of_is_not_known(tape):
    tape.settled_rate("BTC", "2026-09-25T06:00:00.121", "2026-09-25T09:00", "0.0000125")
    tape.settled_rate("BTC", "2026-09-25T07:00:00.040", "2026-09-25T09:00", "0.0000126")
    tape.write()
    got = market.funding("BTC", *DAY, as_of=utc("2026-09-25T07:00")).collect()
    assert got["rate"].to_list() == [0.0000125]


def a_settlement_is_counted_once(tape):
    tape.settled_rate("BTC", "2026-09-25T06:00:00.121", "2026-09-25T06:27:34", "0.0000125")
    tape.settled_rate("BTC", "2026-09-25T06:00:00.121", "2026-09-25T08:10:00", "0.0000125")
    tape.write()
    assert market.funding("BTC", *DAY).collect()["recv_ts"].to_list() == [utc("2026-09-25T06:27:34")]


# Marks


def no_marks_row_has_a_venue_time(tape):
    tape.mark("BTC", "2026-09-25T10:00:00.2").write()
    got = market.marks("BTC", *DAY).collect()
    assert "ts" not in got.columns
    assert got.columns == list(gr.clocks.MARK_SCHEMA)


def a_segment_before_the_corrected_schema_is_refused(tape):
    import pyarrow as pa

    old = pa.schema([f for f in MARKS if f.name not in ("mid", "premium")])
    tape.mark("BTC", "2026-09-25T10:00:00.2")
    for c in ("mid", "premium"):
        tape.rows[-1].pop(c)
    tape.write(schema=old)
    with pytest.raises(Refused, match="lacks mid, premium — it predates datawatch de22b3c; rebuild the tape"):
        market.marks("BTC", *DAY)


def a_repeated_mark_is_dropped_only_on_request(tape):
    for second in range(3):
        tape.mark("BTC", f"2026-09-25T10:00:0{second}.2")
    tape.write()
    assert market.marks("BTC", *DAY).collect().height == 3
    collapsed = market.marks("BTC", *DAY, collapse=True).collect()
    assert collapsed["recv_ts"].to_list() == [utc("2026-09-25T10:00:00.2")]


def a_return_to_an_earlier_state_is_kept(tape):
    for second, rate in enumerate(["1", "2", "1"]):
        tape.live_rate("BTC", f"2026-09-25T10:00:0{second}.2", rate)
    tape.write()
    assert market.funding_live("BTC", *DAY, collapse=True).collect()["rate"].to_list() == [1.0, 2.0, 1.0]


# join_recv


def _trade_and_marks(tape):
    tape.trade("BTC", "2026-09-25T10:00:01.000", "2026-09-25T10:00:01.35", "1")
    tape.mark("BTC", "2026-09-25T10:00:00.2", mark="100")
    tape.mark("BTC", "2026-09-25T10:00:01.2", mark="101")
    tape.write()


def the_mark_at_a_trade_is_the_last_received_by_then(tape):
    _trade_and_marks(tape)
    got = gr.join_recv(market.trades("BTC", *DAY), market.marks("BTC", *DAY)).collect()
    assert got.select("mark_recv", "matched_recv_ts").row(0) == (100.0, utc("2026-09-25T10:00:00.2"))


def a_mark_received_after_the_trade_is_never_matched(tape):
    # Guard: a forward or nearest join would hand the trade the 10:00:01.2 mark.
    tape.trade("BTC", "2026-09-25T10:00:01.000", "2026-09-25T10:00:01.35", "1")
    tape.mark("BTC", "2026-09-25T10:00:01.2", mark="101")
    tape.write()
    got = gr.join_recv(market.trades("BTC", *DAY), market.marks("BTC", *DAY)).collect()
    assert got["mark_recv"].to_list() == [None]


def a_mark_from_before_a_gap_is_not_carried_across_it(tape):
    # Guard: without the tolerance, a 71-hour-old mark reads as current.
    tape.mark("BTC", "2026-09-22T07:00:00", mark="99")
    tape.trade("BTC", "2026-09-25T06:00:00", "2026-09-25T06:00:00.3", "1")
    tape.write()
    trades, marks = market.trades("BTC", *DAY), market.marks("BTC", utc("2026-09-20T00:00"), DAY[1])
    assert gr.join_recv(trades, marks).collect()["mark_recv"].to_list() == [None]
    assert gr.join_recv(trades, marks, tolerance=None).collect()["mark_recv"].to_list() == [99.0]


def a_second_venue_timed_frame_is_refused(tape):
    tape.trade("BTC", "2026-09-25T10:00", "2026-09-25T10:00:00.3", "1").write()
    trades = market.trades("BTC", *DAY)
    with pytest.raises(Refused, match="right has ts, a venue clock"):
        gr.join_recv(trades, trades)


def a_name_collision_is_refused(tape):
    _trade_and_marks(tape)
    left = market.trades("BTC", *DAY).with_columns(pl.lit(0.0).alias("mark_recv"))
    with pytest.raises(Refused, match="left already has mark_recv"):
        gr.join_recv(left, market.marks("BTC", *DAY))


def every_left_row_is_kept_in_order(tape):
    for n, at in enumerate(["10:00:03", "10:00:01", "10:00:02"]):
        tape.trade("BTC", f"2026-09-25T{at}", f"2026-09-25T{at}.3", str(n))
    tape.mark("BTC", "2026-09-25T10:00:00.5")
    tape.write()
    left = market.trades("BTC", *DAY).collect().sort("trade_id", descending=True)
    got = gr.join_recv(left, market.marks("BTC", *DAY))
    assert isinstance(got, pl.DataFrame)
    assert got["trade_id"].to_list() == ["2", "1", "0"]
    assert got["mark_recv"].is_not_null().all()
