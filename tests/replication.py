import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest

NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"


@pytest.fixture(scope="module", name="table")
def _table():
    sys.path.insert(0, str(NOTEBOOKS))
    spec = importlib.util.spec_from_file_location("replication_notebook", NOTEBOOKS / "replication.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.survival_table


def _replayed(rows):
    return pl.DataFrame([{"ticker": t, "bars": b, "claim": c, "verdict": v} for t, b, c, v in rows])


def a_contradicting_verdict_reads_as_not_holding(table):
    md = table(_replayed([("BTC", "1d", "t beats normal", "consistent"), ("ETH", "1d", "t beats normal", "contradicts")]), pl)
    assert "| t beats normal (" in md and "| yes | no | 0 of 1 |" in md


def a_cell_no_one_decided_is_a_dash_and_not_counted(table):
    rows = [
        ("BTC", "1d", "HAR beats GARCH", "consistent"), ("BTC", "4h", "HAR beats GARCH", "can't tell"),
        ("ETH", "1d", "HAR beats GARCH", "consistent"), ("ETH", "4h", "HAR beats GARCH", "consistent"),
    ]  # fmt: skip
    md = table(_replayed(rows), pl)
    assert "| yes · — | yes · yes | 1 of 1 |" in md


def the_persistence_claim_is_one_row_across_bars(table):
    rows = [("BTC", "4h", "α+β≈1 at 4h is the daily cycle", "contradicts"), ("BTC", "1h", "α+β≈1 at 1h is the daily cycle", "consistent")]
    md = table(_replayed(rows), pl)
    assert md.count("α+β≈1 intraday is the daily cycle") == 1 and "| no · yes | 0 of 0 |" in md
