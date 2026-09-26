from datetime import UTC, datetime

import polars as pl
import pytest
from conftest import load, us, utc
from polars.testing import assert_frame_equal

from galata_research import Refused, _scan, market

START, END = utc("2026-09-25T00:00"), utc("2026-09-26T00:00")


def a_naive_datetime_is_refused(tape):
    tape.bar("BTC", "1h", "2026-09-25T00:00", "2026-09-25T02:00").write()
    with pytest.raises(Refused, match=r"start=.*has no zone"):
        load("BTC", "1h", datetime(2026, 9, 25), END)


def a_string_with_an_offset_is_accepted(tape):
    tape.bar("BTC", "1h", "2026-09-25T00:00", "2026-09-25T02:00").write()
    got = load("BTC", "1h", "2026-09-25T00:00:00+00:00", "2026-09-26T00:00:00Z")
    assert got.height == 1


def an_empty_window_is_refused(tape):
    tape.bar("BTC", "1h", "2026-09-25T00:00", "2026-09-25T02:00").write()
    with pytest.raises(Refused, match="must be before end"):
        load("BTC", "1h", START, START)


def the_microseconds_survive_into_ts(tape):
    at = us("2026-09-25T00:00:00.554321")
    tape.bar("BTC", "1m", "2026-09-25T00:00", "2026-09-25T00:02", at_micros=at).write()
    got = load("BTC", "1m", START, END)
    assert got["ts"].item() == datetime(2026, 9, 25, 0, 0, 0, 554321, tzinfo=UTC)


def every_price_is_a_float(tape):
    tape.bar("BTC", "1h", "2026-09-25T00:00", "2026-09-25T02:00", close="84708.123456789012345678").write()
    got = load("BTC", "1h", START, END)
    for column in ["open", "high", "low", "close", "volume"]:
        assert got.schema[column] == pl.Float64
    assert got.schema["trade_count"] == pl.UInt32
    assert got["close"].item() == pytest.approx(84708.123456789)


def no_ts_is_filled_from_recv(tape):
    # A row the venue didn't time is not placed on the venue clock at its receipt.
    tape.bar("BTC", "1h", "2026-09-25T00:00", "2026-09-25T05:00", at_micros=None)
    tape.rows[-1]["at_micros"] = None
    tape.bar("BTC", "1h", "2026-09-25T01:00", "2026-09-25T03:00").write()
    got = load("BTC", "1h", START, END, closed_only=False)
    assert got["ts"].to_list() == [utc("2026-09-25T01:00")]
    assert utc("2026-09-25T05:00") not in got["ts"].to_list()


def the_engines_agree_row_for_row(tape):
    for hour in range(4):
        tape.bar("BTC", "1h", f"2026-09-25T0{hour}:00", "2026-09-25T09:00", close=str(100 + hour))
    tape.write()
    polars = load("BTC", "1h", START, END)
    duck = market.candles("BTC", "1h", START, END, engine="duckdb").pl()
    assert_frame_equal(polars, duck)


def an_unknown_engine_is_refused(tape):
    tape.bar("BTC", "1h", "2026-09-25T00:00", "2026-09-25T02:00").write()
    with pytest.raises(Refused, match="engine='pandas' is not one of polars, duckdb"):
        market.candles("BTC", "1h", START, END, engine="pandas")


def a_segment_with_another_schema_is_refused_by_column_name(tape):
    import pyarrow as pa
    from conftest import CANDLES

    renamed = pa.schema([f.with_name("px_close") if f.name == "close" else f for f in CANDLES])
    tape.bar("BTC", "1h", "2026-09-25T00:00", "2026-09-25T02:00")
    tape.rows[-1]["px_close"] = tape.rows[-1].pop("close")
    tape.write(schema=renamed)
    with pytest.raises(Refused, match="lacks close"):
        load("BTC", "1h", START, END)


def a_column_appended_to_the_tape_does_not_break_a_scan(tmp_path):
    # The tape's schemas are additive-only: datawatch appended `premium` to
    # funding, and files written before it lack it until a rebuild. Sorted by
    # date, the old file comes first, which polars refused (SchemaError).
    old, new = tmp_path / "old.parquet", tmp_path / "new.parquet"
    pl.DataFrame({"rate": [1.0], "next_micros": [None]}, schema={"rate": pl.Float64, "next_micros": pl.Int64}).write_parquet(old)
    pl.DataFrame(
        {"rate": [2.0], "next_micros": [None], "premium": [0.5]},
        schema={"rate": pl.Float64, "next_micros": pl.Int64, "premium": pl.Float64},
    ).write_parquet(new)
    assert _scan.scan([old, new], ["rate"]).collect()["rate"].to_list() == [1.0, 2.0]
    assert _scan.scan([new, old], ["rate"]).collect()["rate"].to_list() == [2.0, 1.0]
