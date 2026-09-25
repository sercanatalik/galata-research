import polars as pl
import pytest
from conftest import utc
from polars.testing import assert_frame_equal

from galata_research import Refused, _root, _scan, market

DAY = utc("2026-09-25T00:00"), utc("2026-09-26T00:00")


def _trades(*args, **kwargs) -> pl.DataFrame:
    return market.trades(*args, **kwargs).collect()


def _quotes(*args, **kwargs) -> pl.DataFrame:
    return market.quotes(*args, **kwargs).collect()


# Known tickers


def a_mixed_row_group_still_yields_every_ticker(tape):
    for ticker in ["HYPE", "BTC", "ETH"]:
        tape.quote(ticker, "2026-09-25T09:00", "2026-09-25T09:00:00.3")
    tape.write()
    files = _scan.segments(_root.root() / "tape" / "kind=quotes")
    assert _scan.tickers(files) == {"BTC", "ETH", "HYPE"}


def the_single_ticker_groups_are_known_from_footers(tape, monkeypatch):
    for n, ticker in enumerate(["BTC", "ETH"]):
        tape.quote(ticker, "2026-09-25T09:00", "2026-09-25T09:00:00.3", seq=n + 1).write()
    import pyarrow.parquet as pq

    def unread(*_, **__):
        raise AssertionError("a single-ticker row group was read")

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", unread)
    files = _scan.segments(_root.root() / "tape" / "kind=quotes")
    assert _scan.tickers(files) == {"BTC", "ETH"}


def a_record_without_quotes_refuses_the_call(tape):
    tape.trade("BTC", "2026-09-25T09:00", "2026-09-25T09:00:00.3", "1").write()
    with pytest.raises(Refused, match="no quotes under"):
        _quotes("BTC", *DAY)


def an_unknown_ticker_is_refused_with_the_list(tape):
    tape.trade("BTC", "2026-09-25T09:00", "2026-09-25T09:00:00.3", "1")
    tape.trade("ETH", "2026-09-25T09:00", "2026-09-25T09:00:00.3", "2")
    tape.write()
    with pytest.raises(Refused, match="no trades for SOL; it holds BTC, ETH"):
        _trades(["SOL"], *DAY)


# Trades


def a_replayed_execution_is_counted_once(tape):
    tape.trade("BTC", "2026-09-25T08:59:59.9", "2026-09-25T09:00:00", "42")
    tape.trade("BTC", "2026-09-25T08:59:59.9", "2026-09-25T09:00:17", "42")
    tape.write()
    got = _trades("BTC", *DAY)
    assert got.height == 1
    assert got["recv_ts"].item() == utc("2026-09-25T09:00")


def an_id_is_the_venues(tape):
    tape.trade("BTC", "2026-09-25T09:00", "2026-09-25T09:00:01", "42", venue="hyperliquid")
    tape.trade("BTC", "2026-09-25T09:00", "2026-09-25T09:00:01", "42", venue="rh_crypto")
    tape.write()
    assert _trades("BTC", *DAY)["venue"].to_list() == ["hyperliquid", "rh_crypto"]


def the_trades_in_one_millisecond_keep_arrival_order(tape):
    for seq, trade_id in [(7, "b"), (5, "c"), (9, "a")]:
        tape.trade("BTC", "2026-09-25T09:00:00.001", "2026-09-25T09:00:01", trade_id, seq=seq)
    tape.write()
    assert _trades("BTC", *DAY)["trade_id"].to_list() == ["c", "b", "a"]


def the_trades_in_one_message_keep_the_messages_order(tape):
    # One message, one stream_seq, many trades: only the row order is the venue's.
    for trade_id in ["z", "a", "m"]:
        tape.trade("BTC", "2026-09-25T09:00:00.001", "2026-09-25T09:00:01", trade_id, seq=3)
    tape.write()
    assert _trades("BTC", *DAY)["trade_id"].to_list() == ["z", "a", "m"]


def a_tick_after_as_of_is_not_known(tape):
    # Lookahead guard: without the as_of bound, 12:00:00.001 is returned at 12:00:00.
    tape.trade("BTC", "2026-09-25T11:59:59.999", "2026-09-25T12:00:00.2", "1")
    tape.trade("BTC", "2026-09-25T12:00:00.001", "2026-09-25T12:00:00.3", "2")
    tape.write()
    got = _trades("BTC", *DAY, as_of=utc("2026-09-25T12:00"))
    assert got["trade_id"].to_list() == ["1"]


def every_trade_column_is_stated(tape):
    tape.trade("BTC", "2026-09-25T09:00", "2026-09-25T09:00:01", "1", price="84708.5", size="0.0619").write()
    got = _trades("BTC", *DAY)
    assert got.columns == ["venue", "ticker", "ts", "price", "size", "aggressor", "trade_id", "recv_ts"]
    assert (got["price"].item(), got["size"].item()) == (84708.5, 0.0619)


# Quotes


def every_quote_stands_even_when_it_repeats(tape):
    tape.quote("BTC", "2026-09-25T09:00:00.100", "2026-09-25T09:00:00.4")
    tape.quote("BTC", "2026-09-25T09:00:00.200", "2026-09-25T09:00:00.5")
    tape.write()
    assert _quotes("BTC", *DAY).height == 2


def an_untimed_quote_is_not_placed_on_the_clock(tape):
    tape.quote("BTC", "2026-09-25T09:00", "2026-09-25T09:00:00.4", at_micros=0)
    tape.rows[-1]["at_micros"] = None
    tape.quote("BTC", "2026-09-25T09:00:01", "2026-09-25T09:00:01.4")
    tape.write()
    got = _quotes("BTC", *DAY)
    assert got["ts"].to_list() == [utc("2026-09-25T09:00:01")]


def a_venue_that_states_no_spread_gives_nulls(tape):
    tape.quote("BTC", "2026-09-25T09:00", "2026-09-25T09:00:00.4").write()
    got = _quotes("BTC", *DAY)
    assert got.schema["bid_spread"] == pl.Float64 and got.schema["ask_spread"] == pl.Float64
    assert got["bid_spread"].is_null().all() and got["ask_spread"].is_null().all()


def the_tick_engines_agree_row_for_row(tape):
    for n in range(3):
        tape.quote("BTC", f"2026-09-25T09:00:0{n}", f"2026-09-25T09:00:0{n}.4", bid=str(99 + n))
        tape.trade("BTC", f"2026-09-25T09:00:0{n}", f"2026-09-25T09:00:0{n}.4", str(n))
    tape.write()
    assert_frame_equal(_quotes("BTC", *DAY), market.quotes("BTC", *DAY, engine="duckdb").pl())
    assert_frame_equal(_trades("BTC", *DAY), market.trades("BTC", *DAY, engine="duckdb").pl())


def a_known_ticker_outside_coverage_is_empty_with_the_columns(tape):
    tape.trade("BTC", "2026-09-25T09:00", "2026-09-25T09:00:01", "1").write()
    got = _trades("BTC", utc("2025-01-01T00:00"), utc("2025-01-02T00:00"))
    assert got.height == 0
    assert got.schema == pl.Schema(market.TRADE_SCHEMA)
