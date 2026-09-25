from conftest import utc

import galata_research as gr


def the_frontier_comes_from_names_and_footers(tape):
    tape.bar("BTC", "1h", "2026-09-24T22:00", "2026-09-25T09:00", seq=100)
    tape.bar("BTC", "1h", "2026-09-25T08:00", "2026-09-25T09:00:00.5", seq=200)
    tape.bar("ETH", "1h", "2026-09-25T09:00", "2026-09-25T09:30", seq=150)
    tape.write()
    row = gr.frontier().row(0, named=True)
    assert row["dataset"] == "candles"
    assert row["last_stream_seq"] == 200
    assert row["max_ts"] == utc("2026-09-25T09:00")
    assert row["max_recv_ts"] == utc("2026-09-25T09:30")
    assert row["days"] == 2


def the_frontier_does_not_scan_older_days(tape):
    tape.bar("BTC", "1h", "2026-09-25T08:00", "2026-09-25T09:00", seq=7).write()
    # An older day whose bytes are not Parquet at all: only its name may be read.
    old = tape.root / "tape" / "kind=candles" / "date=2026-09-20"
    old.mkdir()
    (old / "s-1_5.parquet").write_bytes(b"not parquet")
    row = gr.frontier().row(0, named=True)
    assert row["last_stream_seq"] == 7
    assert row["days"] == 2


def the_frontier_has_a_row_only_for_loadable_datasets(tape):
    tape.bar("BTC", "1h", "2026-09-25T08:00", "2026-09-25T09:00", seq=1)
    tape.quote("BTC", "2026-09-25T08:00", "2026-09-25T08:00:00.3", seq=2)
    tape.trade("BTC", "2026-09-25T08:00", "2026-09-25T08:00:00.3", "1", seq=3)
    tape.gap("BTC", "trades", "2026-09-25T07:00", "2026-09-25T07:01", seq=5)
    tape.write()
    tape.quote("BTC", "2026-09-25T08:00", "2026-09-25T08:00:00.3", seq=4).write(kind="mints")
    assert gr.frontier()["dataset"].to_list() == ["candles", "gaps", "quotes", "trades"]


def a_dataset_with_no_venue_time_has_no_max_ts(tape):
    tape.mark("BTC", "2026-09-25T08:00:00.3", seq=9).write()
    row = gr.frontier().row(0, named=True)
    assert row["dataset"] == "marks"
    assert row["max_ts"] is None
    assert row["max_recv_ts"] == utc("2026-09-25T08:00:00.3")


def a_dataset_the_record_lacks_has_no_row(tape):
    tape.bar("BTC", "1h", "2026-09-25T08:00", "2026-09-25T09:00").write()
    assert gr.frontier()["dataset"].to_list() == ["candles"]
