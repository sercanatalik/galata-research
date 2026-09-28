"""Signals read point-in-time, on a fixture tape written in galata-datawatch's signals schema."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from conftest import us, utc

import galata_research as gr
from galata_research import Refused

HOUR = 3_600_000_000
# galata-datawatch `signals::schema()`, field for field.
SIGNALS = pa.schema(
    [
        pa.field("signal", pa.string(), False), pa.field("horizon", pa.string(), False), pa.field("measure", pa.string(), False),
        pa.field("ticker_i", pa.string(), False), pa.field("ticker_j", pa.string(), True), pa.field("h", pa.int64(), False),
        pa.field("value", pa.float64(), True), pa.field("absent", pa.string(), True), pa.field("n_eff", pa.float64(), True),
        pa.field("asof_micros", pa.int64(), False), pa.field("target_micros", pa.int64(), False), pa.field("computed_micros", pa.int64(), False),
        pa.field("fitted_through_micros", pa.int64(), True), pa.field("fit_from_micros", pa.int64(), True),
        pa.field("model", pa.string(), False), pa.field("params", pa.string(), False), pa.field("fitted", pa.bool_(), False),
        pa.field("after_gap", pa.bool_(), False), pa.field("code", pa.string(), False), pa.field("run_id", pa.string(), False),
    ]
)  # fmt: skip


@pytest.fixture(name="record")
def _record(tmp_path, monkeypatch):
    monkeypatch.setenv("GALATA_VAR", str(tmp_path))
    (tmp_path / "tape").mkdir()
    return tmp_path


def _run(root: Path, horizon: str, asof: str, computed: str, *, rho: float = 0.8, signal: str = "varcov", absent: str | None = None) -> None:
    """One run's rows for BTC and ETH: two variances and a correlation."""
    a, c = us(asof), us(computed)
    cells = [("covariance", "BTC", "BTC", 1e-4), ("covariance", "ETH", "ETH", 2e-4), ("covariance", "BTC", "ETH", 1.1e-4), ("correlation", "BTC", "ETH", rho)]
    rows = [
        {
            "signal": signal, "horizon": horizon, "measure": m, "ticker_i": i, "ticker_j": j, "h": 1,
            "value": None if absent else v, "absent": absent, "n_eff": 32.3, "asof_micros": a, "target_micros": a,
            "computed_micros": c, "fitted_through_micros": a, "fit_from_micros": a - 1000 * HOUR,
            "model": "gjr-t/dcc", "params": "{}", "fitted": True, "after_gap": False, "code": "abc", "run_id": f"run-{c}",
        }
        for m, i, j, v in cells
    ]  # fmt: skip
    directory = root / "tape" / "kind=signals" / f"date={asof[:10]}"
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SIGNALS), directory / f"t-{c}_{c}_1_0.parquet")


def a_figure_is_on_two_clocks(record):
    _run(record, "4h", "2026-09-28T04:00", "2026-09-28T04:15")
    rows = gr.signals.history("varcov", "2026-09-28T00:00Z", "2026-09-29T00:00Z").collect()
    assert rows.height == 4
    assert set(rows["ts"]) == {utc("2026-09-28T04:00")}
    assert set(rows["computed_ts"]) == {utc("2026-09-28T04:15")}
    assert rows.schema == gr.signals.SCHEMA


def an_as_of_hides_what_was_not_yet_computed(record):
    _run(record, "4h", "2026-09-28T00:00", "2026-09-28T00:15")
    _run(record, "4h", "2026-09-28T04:00", "2026-09-28T04:15")
    before = gr.signals.history("varcov", "2026-09-28T00:00Z", "2026-09-29T00:00Z", as_of="2026-09-28T04:10Z").collect()
    assert set(before["ts"]) == {utc("2026-09-28T00:00")}  # the 04:00 bar had closed, but its figure did not exist yet


def a_figure_computed_twice_is_its_latest_within_the_bound(record):
    _run(record, "4h", "2026-09-28T04:00", "2026-09-28T04:15", rho=0.8)
    _run(record, "4h", "2026-09-28T04:00", "2026-09-28T04:45", rho=0.7)  # a repeated run
    rho = lambda **kw: gr.signals.history("varcov", "2026-09-28T00:00Z", "2026-09-29T00:00Z", measure="correlation", **kw).collect()["value"].to_list()  # noqa: E731
    assert rho() == [0.7]
    assert rho(as_of="2026-09-28T04:30Z") == [0.8]


def the_view_known_at_an_instant_is_each_horizons_newest(record):
    _run(record, "4h", "2026-09-28T00:00", "2026-09-28T00:15")
    _run(record, "4h", "2026-09-28T04:00", "2026-09-28T04:15")
    _run(record, "1w", "2026-09-21T00:00", "2026-09-21T00:15")
    held = gr.signals.known_at("varcov", "2026-09-28T04:10Z")
    by = {h: set(g["ts"]) for (h,), g in held.group_by("horizon")}
    assert by == {"4h": {utc("2026-09-28T00:00")}, "1w": {utc("2026-09-21T00:00")}}


def an_absent_figure_keeps_its_reason(record):
    _run(record, "1d", "2026-09-28T00:00", "2026-09-28T00:15", absent="265 returns, under min_obs=500")
    rows = gr.signals.history("varcov", "2026-09-28T00:00Z", "2026-09-29T00:00Z").collect()
    assert rows["value"].null_count() == rows.height
    assert set(rows["absent"]) == {"265 returns, under min_obs=500"}


def the_matrix_is_square_and_symmetric(record):
    _run(record, "4h", "2026-09-28T04:00", "2026-09-28T04:15")
    rows = gr.signals.history("varcov", "2026-09-28T00:00Z", "2026-09-29T00:00Z").collect()
    m = gr.signals.matrix(rows, "covariance")
    assert m.columns == ["ticker", "BTC", "ETH"]
    assert m.filter(pl.col("ticker") == "BTC")["ETH"][0] == m.filter(pl.col("ticker") == "ETH")["BTC"][0] == pytest.approx(1.1e-4)


def a_matrix_of_two_asofs_is_refused(record):
    _run(record, "4h", "2026-09-28T00:00", "2026-09-28T00:15")
    _run(record, "4h", "2026-09-28T04:00", "2026-09-28T04:15")
    rows = gr.signals.history("varcov", "2026-09-28T00:00Z", "2026-09-29T00:00Z").collect()
    with pytest.raises(Refused, match="one horizon at one asof"):
        gr.signals.matrix(rows, "covariance")


def no_signals_is_refused_by_name(record):
    with pytest.raises(Refused, match="derive-the-signals"):
        gr.signals.history("varcov", "2026-09-28T00:00Z", "2026-09-29T00:00Z")


def the_duckdb_engine_holds_the_same_rows(record):
    _run(record, "4h", "2026-09-28T04:00", "2026-09-28T04:15")
    rel = gr.signals.history("varcov", "2026-09-28T00:00Z", "2026-09-29T00:00Z", engine="duckdb")
    assert rel.count("*").fetchone()[0] == 4
