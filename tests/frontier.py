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
    tape.bar("BTC", "1h", "2026-09-25T08:00", "2026-09-25T09:00").write()
    tape.bar("BTC", "1h", "2026-09-25T08:00", "2026-09-25T09:00").write(kind="quotes")
    assert gr.frontier()["dataset"].to_list() == ["candles"]
