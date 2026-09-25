import pytest
from conftest import load, utc

from galata_research import Refused

DAY = utc("2026-09-25T00:00"), utc("2026-09-26T00:00")


# One row per bar, the latest receipt


def a_refetched_candle_is_counted_once(tape):
    for received in ["2026-09-25T02:00", "2026-09-25T09:00", "2026-09-25T14:50"]:
        tape.bar("BTC", "1h", "2026-09-25T01:00", received)
    tape.write()
    assert load("BTC", "1h", *DAY).height == 1


def the_latest_receipt_wins(tape):
    tape.bar("CL", "4h", "2026-09-25T12:00", "2026-09-25T16:50:07", close="93.881")
    tape.bar("CL", "4h", "2026-09-25T12:00", "2026-09-25T16:50:03", close="93.819")
    tape.write()
    assert load("CL", "4h", *DAY)["close"].item() == 93.881


def a_receipt_tie_is_broken_by_stream_seq(tape):
    tape.bar("BTC", "1h", "2026-09-25T01:00", "2026-09-25T03:00", close="2", seq=9)
    tape.bar("BTC", "1h", "2026-09-25T01:00", "2026-09-25T03:00", close="1", seq=5)
    tape.write()
    assert load("BTC", "1h", *DAY)["close"].item() == 2.0


# Closed by the record, not by the flag


def a_forming_walked_bar_is_open_despite_is_final(tape):
    # The walk stamps its page's last bar final while it is still forming.
    tape.bar("CL", "4h", "2026-09-25T12:00", "2026-09-25T14:50", is_final=True).write()
    assert load("CL", "4h", *DAY).height == 0
    assert load("CL", "4h", *DAY, closed_only=False)["closed"].to_list() == [False]


def a_later_bar_closes_a_bar_with_no_receipt_after_close(tape):
    # Nothing traded in 09:33's last seconds, so its last push came before its close.
    tape.bar("BTC", "1m", "2026-09-25T09:33", "2026-09-25T09:33:58", is_final=False)
    tape.bar("BTC", "1m", "2026-09-25T09:34", "2026-09-25T09:34:10", is_final=False)
    tape.write()
    got = load("BTC", "1m", *DAY)
    assert got["ts"].to_list() == [utc("2026-09-25T09:33")]


def an_own_receipt_after_close_closes_the_last_bar(tape):
    tape.bar("BTC", "1h", "2026-09-25T09:00", "2026-09-25T10:00:00.4").write()
    assert load("BTC", "1h", *DAY).height == 1


def a_windows_last_bar_is_closed_by_a_bar_outside_it(tape):
    # Lookahead guard: closure must be judged against the record, not the window.
    # Judging it only within the scanned window leaves 23:00 open.
    tape.bar("BTC", "1h", "2026-09-25T23:00", "2026-09-25T23:30")
    tape.bar("BTC", "1h", "2026-09-26T00:00", "2026-09-26T00:30")
    tape.write()
    got = load("BTC", "1h", *DAY)
    assert got["ts"].to_list() == [utc("2026-09-25T23:00")]


def a_bar_is_closed_by_a_later_bar_across_a_missing_day(tape):
    tape.bar("BTC", "1d", "2026-09-25T00:00", "2026-09-25T12:00")
    tape.bar("BTC", "1d", "2026-09-28T00:00", "2026-09-28T12:00")
    tape.write()
    assert load("BTC", "1d", *DAY).height == 1


def an_open_bar_is_marked_when_asked_for(tape):
    tape.bar("BTC", "1h", "2026-09-25T13:00", "2026-09-25T13:10")
    tape.bar("BTC", "1h", "2026-09-25T14:00", "2026-09-25T14:10")
    tape.write()
    got = load("BTC", "1h", *DAY, closed_only=False)
    assert got["closed"].to_list() == [True, False]
    assert "closed" not in load("BTC", "1h", *DAY).columns


# Known at its close


def _two_hours(tape):
    tape.bar("BTC", "1h", "2026-09-25T13:00", "2026-09-25T13:59")
    tape.bar("BTC", "1h", "2026-09-25T14:00", "2026-09-25T15:00:01")
    tape.write()


def a_bar_open_at_as_of_is_not_known(tape):
    # Lookahead guard: filtering as_of on ts (the open) returns 13:00 at 13:30.
    _two_hours(tape)
    got = load("BTC", "1h", *DAY, as_of=utc("2026-09-25T13:30"))
    assert got.height == 0


def a_bar_closing_at_as_of_is_known(tape):
    _two_hours(tape)
    got = load("BTC", "1h", *DAY, as_of=utc("2026-09-25T14:00"))
    assert got["ts"].to_list() == [utc("2026-09-25T13:00")]


def the_close_ts_is_the_exclusive_end(tape):
    tape.bar("GOLD", "4h", "2026-09-25T12:00", "2026-09-25T16:00").write()
    got = load("GOLD", "4h", *DAY)
    assert got["close_ts"].item() == utc("2026-09-25T16:00")


def the_window_is_half_open(tape):
    for hour in ["23", "00"]:
        day = "2026-09-24" if hour == "23" else "2026-09-25"
        tape.bar("BTC", "1h", f"{day}T{hour}:00", "2026-09-26T01:00")
    tape.bar("BTC", "1h", "2026-09-26T00:00", "2026-09-26T01:00")
    tape.write()
    got = load("BTC", "1h", *DAY)
    assert got["ts"].to_list() == [utc("2026-09-25T00:00")]


# Trade-less bars


def _pre_trading(tape):
    tape.bar("BTC", "1d", "2023-02-25T00:00", "2026-09-25T14:51", trade_count=0)
    tape.bar("BTC", "1d", "2023-02-26T00:00", "2026-09-25T14:51", trade_count=2514)
    tape.write()


def no_trade_less_bar_is_returned_by_default(tape):
    _pre_trading(tape)
    got = load("BTC", "1d", utc("2023-01-01T00:00"), utc("2023-03-01T00:00"))
    assert got["ts"].to_list() == [utc("2023-02-26T00:00")]


def the_trade_less_filter_can_be_turned_off(tape):
    _pre_trading(tape)
    got = load("BTC", "1d", utc("2023-01-01T00:00"), utc("2023-03-01T00:00"), traded_only=False)
    assert got["trade_count"].to_list() == [0, 2514]


# Refusals, and true empty answers


def an_unknown_interval_is_refused_with_the_list(tape):
    tape.bar("BTC", "1h", "2026-09-25T00:00", "2026-09-25T02:00").write()
    with pytest.raises(Refused, match="interval='15m' is not one of 1m, 1h, 4h, 1d"):
        load("BTC", "15m", *DAY)


def an_unknown_ticker_is_refused_with_the_list(tape):
    tape.bar("BTC", "1h", "2026-09-25T00:00", "2026-09-25T02:00")
    tape.bar("ETH", "1h", "2026-09-25T00:00", "2026-09-25T02:00")
    tape.write()
    with pytest.raises(Refused, match="no 1h candles for SOL; it holds BTC, ETH"):
        load(["SOL"], "1h", *DAY)


def every_ticker_is_loaded_when_none_is_named(tape):
    for ticker in ["ETH", "BTC"]:
        tape.bar(ticker, "1h", "2026-09-25T00:00", "2026-09-25T02:00")
    tape.bar("HYPE", "4h", "2026-09-25T00:00", "2026-09-25T05:00")
    tape.write()
    assert load(None, "1h", *DAY)["ticker"].to_list() == ["BTC", "ETH"]


def a_known_ticker_outside_coverage_is_empty(tape):
    tape.bar("BTC", "1h", "2026-09-25T00:00", "2026-09-25T02:00").write()
    got = load("BTC", "1h", utc("2025-01-01T00:00"), utc("2025-02-01T00:00"))
    assert got.height == 0
    assert got.columns == load("BTC", "1h", *DAY).columns


def a_record_without_candles_is_refused(tape):
    with pytest.raises(Refused, match="no candles"):
        load("BTC", "1h", *DAY)


def no_bookkeeping_column_is_returned(tape):
    tape.bar("BTC", "1h", "2026-09-25T00:00", "2026-09-25T02:00").write()
    assert load("BTC", "1h", *DAY).columns == [
        "venue", "ticker", "interval", "ts", "close_ts",
        "open", "high", "low", "close", "volume", "trade_count", "recv_ts",
    ]  # fmt: skip
