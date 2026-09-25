import polars as pl
import pytest
from conftest import load, utc

import galata_research as gr
from galata_research import Refused, market

DAY = utc("2026-09-25T00:00"), utc("2026-09-26T00:00")


def _masked_trades(tape, **kwargs) -> pl.DataFrame:
    return gr.mask_gaps(market.trades("BTC", *DAY), "trades", **kwargs).collect()


# Loading


def a_gap_starting_before_the_window_is_found(tape):
    tape.gap("BTC", "trades", "2026-09-20T21:52:19.190783", "2026-09-22T06:27:27.399168").write()
    got = market.gaps(None, utc("2026-09-21T00:00"), utc("2026-09-22T00:00")).collect()
    assert got["from_recv_ts"].to_list() == [utc("2026-09-20T21:52:19.190783")]


def a_gap_has_no_venue_time(tape):
    tape.gap("BTC", "trades", "2026-09-25T10:00", "2026-09-25T10:01").write()
    got = market.gaps("BTC", *DAY).collect()
    assert "ts" not in got.columns
    assert got.columns == list(gr.gaps.GAP_SCHEMA)


def the_series_filter_keeps_one_series(tape):
    for series in ["trades", "quotes", "candles"]:
        tape.gap("BTC", series, "2026-09-25T10:00", "2026-09-25T10:01")
    tape.write()
    assert market.gaps("BTC", *DAY, series="quotes").collect()["series"].to_list() == ["quotes"]


# Ticks


def a_replayed_trade_inside_a_gap_is_marked(tape):
    tape.trade("BTC", "2026-09-25T10:00:30", "2026-09-25T10:01:05", "replayed")
    tape.trade("BTC", "2026-09-25T10:05:00", "2026-09-25T10:05:00.3", "after")
    tape.gap("BTC", "trades", "2026-09-25T10:00", "2026-09-25T10:01")
    tape.write()
    got = _masked_trades(tape)
    assert got.select("trade_id", "in_gap", "gap_cause").rows() == [("replayed", True, "downtime"), ("after", False, None)]


def the_gap_end_is_exclusive(tape):
    tape.quote("BTC", "2026-09-25T10:01", "2026-09-25T10:01:00.3")
    tape.gap("BTC", "quotes", "2026-09-25T10:00", "2026-09-25T10:01")
    tape.write()
    got = gr.mask_gaps(market.quotes("BTC", *DAY), "quotes").collect()
    assert got["in_gap"].to_list() == [False]


def a_venue_wide_gap_covers_every_ticker(tape):
    for n, ticker in enumerate(["BTC", "ETH"]):
        tape.trade(ticker, "2026-09-25T10:00:30", "2026-09-25T10:00:30.3", str(n))
    tape.gap(None, "trades", "2026-09-25T10:00", "2026-09-25T10:01")
    tape.write()
    got = gr.mask_gaps(market.trades(None, *DAY), "trades").collect()
    assert got["in_gap"].to_list() == [True, True]


def a_tick_just_before_a_gaps_receipt_bound_is_marked(tape):
    # Guard: venue time trails receipt, so the loss starts before `from` on the venue clock.
    tape.quote("BTC", "2026-09-25T09:59:59.500", "2026-09-25T09:59:59.9")
    tape.gap("BTC", "quotes", "2026-09-25T10:00", "2026-09-25T10:01")
    tape.write()
    quotes = market.quotes("BTC", *DAY)
    assert gr.mask_gaps(quotes, "quotes").collect()["in_gap"].to_list() == [True]
    assert gr.mask_gaps(quotes, "quotes", margin="0s").collect()["in_gap"].to_list() == [False]


def a_negative_margin_is_refused(tape):
    tape.trade("BTC", "2026-09-25T10:00", "2026-09-25T10:00:00.3", "1").write()
    with pytest.raises(Refused, match="margin='-1s' is negative"):
        gr.mask_gaps(market.trades("BTC", *DAY), "trades", margin="-1s")


def a_row_inside_two_overlapping_gaps_is_marked_once(tape):
    # Guard: without the merge, the later-starting gap hides the one still covering 11:30.
    tape.trade("BTC", "2026-09-25T11:30", "2026-09-25T12:00:01", "1")
    tape.gap("BTC", "trades", "2026-09-25T10:00", "2026-09-25T12:00", cause="downtime")
    tape.gap("BTC", "trades", "2026-09-25T10:30", "2026-09-25T11:00", cause="crash_unflushed")
    tape.write()
    got = _masked_trades(tape)
    assert got.select("in_gap", "gap_cause").rows() == [(True, "crash_unflushed,downtime")]


def no_row_is_dropped_by_the_mask(tape):
    for minute in range(10):
        tape.trade("BTC", f"2026-09-25T10:0{minute}:30", f"2026-09-25T10:0{minute}:31", str(minute))
    tape.gap("BTC", "trades", "2026-09-25T10:02", "2026-09-25T10:05")
    tape.write()
    got = _masked_trades(tape)
    assert got.drop("in_gap", "gap_cause").equals(market.trades("BTC", *DAY).collect())
    assert got["in_gap"].sum() == 3


# Bars


def a_gap_inside_an_hour_marks_the_hour(tape):
    # Guard: masking a bar by its open alone misses a gap that starts inside it.
    # Each bar's last receipt came before the gap ended, so nothing restated it.
    tape.bar("BTC", "1h", "2026-09-25T12:00", "2026-09-25T13:00:01")
    tape.bar("BTC", "1h", "2026-09-25T13:00", "2026-09-25T13:19:59")
    tape.bar("BTC", "1h", "2026-09-25T14:00", "2026-09-25T15:00:01")
    tape.gap("BTC", "candles", "2026-09-25T13:20", "2026-09-25T13:20:40")
    tape.write()
    got = gr.mask_gaps(market.candles("BTC", "1h", *DAY), "candles").collect()
    assert got["in_gap"].to_list() == [False, True, False]


def a_bar_restated_after_the_gap_is_not_marked(tape):
    # Guard: a candle is the venue's aggregate, so a receipt after the gap is the whole bar.
    tape.bar("BTC", "1m", "2026-09-25T10:00", "2026-09-25T10:00:15")
    tape.bar("BTC", "1m", "2026-09-25T10:00", "2026-09-25T11:30:00", is_final=True)
    tape.bar("BTC", "1m", "2026-09-25T10:01", "2026-09-25T11:30:00", is_final=True)
    tape.gap("BTC", "candles", "2026-09-25T10:00:20", "2026-09-25T11:29:50")
    tape.write()
    got = gr.mask_gaps(market.candles("BTC", "1m", *DAY), "candles").collect()
    assert got["in_gap"].to_list() == [False, False]


def a_missing_close_is_refused_for_candles(tape):
    tape.bar("BTC", "1h", "2026-09-25T12:00", "2026-09-25T13:00:01").write()
    with pytest.raises(Refused, match="needs close_ts"):
        gr.mask_gaps(market.candles("BTC", "1h", *DAY).drop("close_ts"), "candles")


def a_missing_receipt_is_refused_for_candles(tape):
    tape.bar("BTC", "1h", "2026-09-25T12:00", "2026-09-25T13:00:01").write()
    with pytest.raises(Refused, match="needs recv_ts"):
        gr.mask_gaps(market.candles("BTC", "1h", *DAY).drop("recv_ts"), "candles")


def an_unknown_dataset_is_refused(tape):
    with pytest.raises(Refused, match="dataset='marks' is not one of candles, trades, quotes"):
        gr.mask_gaps(pl.LazyFrame(), "marks")


def a_frame_masked_twice_is_refused(tape):
    tape.trade("BTC", "2026-09-25T10:00", "2026-09-25T10:00:00.3", "1").write()
    tape.gap("BTC", "trades", "2026-09-25T11:00", "2026-09-25T11:01").write()
    once = gr.mask_gaps(market.trades("BTC", *DAY), "trades")
    with pytest.raises(Refused, match="already has in_gap, gap_cause"):
        gr.mask_gaps(once, "trades")


def a_lazy_frame_stays_lazy(tape):
    tape.bar("BTC", "1h", "2026-09-25T12:00", "2026-09-25T13:00:01").write()
    tape.gap("BTC", "candles", "2026-09-25T20:00", "2026-09-25T20:01").write()
    lazy = market.candles("BTC", "1h", *DAY)
    assert isinstance(gr.mask_gaps(lazy, "candles"), pl.LazyFrame)
    assert isinstance(gr.mask_gaps(load("BTC", "1h", *DAY), "candles"), pl.DataFrame)
