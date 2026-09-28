import ast
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from conftest import utc

import galata_research as gr
from galata_research import Refused, studies, utils

NOTEBOOKS = Path(__file__).parent.parent / "notebooks"
_MODULES = {"gr", "galata_research", *gr.__all__}


def _private_calls(source: str) -> list[tuple[int, str]]:
    """Every `x._name` (not dunder) where `x` is the package or one of its public modules."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("_")
            and not node.attr.startswith("__")
            and isinstance(node.value, ast.Name)
            and node.value.id in _MODULES
        ):
            found.append((node.lineno, node.attr))
    return found


def the_root_is_public(tape):
    assert gr.root() == tape.root


def the_missing_columns_are_named():
    with pytest.raises(Refused, match=r"needs close; load bars"):
        utils.require(pl.DataFrame({"ts": [1]}), ["ts", "close"], "load bars")


def a_naive_time_is_refused():
    with pytest.raises(Refused, match="start="):
        utils.window(datetime(2026, 1, 1), "2026-01-02T00:00Z")


def a_buy_and_hold_is_a_trial():
    t0 = utc("2026-01-01T00:00")
    day = timedelta(days=1)
    bars = pl.DataFrame(
        {
            "ticker": "BTC",
            "ts": [t0, t0 + day],
            "close_ts": [t0 + day, t0 + 2 * day],
            "close": [100.0, 110.0],
        }
    )
    got = studies.trial(bars, pl.lit(1.0), "buy and hold")
    assert got.columns == ["trial", "ticker", "ts", "position", "bar_return", "gross", "net"]
    assert set(got["trial"]) == {"buy and hold"}


def the_private_name_guard_catches_a_planted_call():
    # Guard on the guard: it must see what a notebook would write.
    assert _private_calls("x = studies._trial(bars, p, 'n')\ny = gr._root.root()") == [(1, "_trial"), (2, "_root")]
    assert _private_calls("studies.trial(bars)\nobj._cache\n__name__") == []


def no_notebook_calls_a_private_name():
    found = [
        f"{path.name}:{line} {name}"
        for path in sorted(NOTEBOOKS.glob("*.py"))
        for line, name in _private_calls(path.read_text())
    ]
    assert not found, "private names in notebooks: " + ", ".join(found)


def the_fetcher_is_not_imported_by_the_library():
    import subprocess
    import sys

    probe = "import sys, galata_research as gr; gr.reference.root; print('galata_research.reference.fetch' in sys.modules)"
    run = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert run.stdout.strip() == "False"


def no_loader_opens_a_socket(tmp_path, monkeypatch):
    import socket

    from galata_research.reference import _manifest

    store = tmp_path / "reference"
    path = _manifest.day_path(store, "depth", "binance-um", "BTC", utc("2026-09-20T00:00").date())
    path.parent.mkdir(parents=True)
    ts = utc("2026-09-20T00:00:04")
    pl.DataFrame({"venue": ["binance-um"], "ticker": "BTC", "symbol": "BTCUSDT", "ts": [ts], "band_pct": -0.2, "depth": 1.0, "notional": 1.0}).write_parquet(path)
    _manifest.upsert(store, [{"kind": "depth", "venue": "binance-um", "ticker": "BTC", "date": ts.date(), "url": "u", "bytes": 1,
                              "sha256": "s", "published_sha256": None, "fetched_at_recv": ts, "rows": 1, "status": "ok"}])  # fmt: skip
    monkeypatch.setenv("GALATA_REFERENCE", str(store))

    def refuse(*a, **k):
        raise AssertionError("a loader opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    assert gr.reference.depth("BTC", "2026-09-20T00:00Z", "2026-09-21T00:00Z").collect().height == 1
    assert gr.reference.coverage()["days_ok"].to_list() == [1]


def the_fetch_command_lists_its_kinds():
    import subprocess
    import sys

    run = subprocess.run([sys.executable, "-m", "galata_research.reference.fetch", "--help"], capture_output=True, text=True, check=False)
    assert run.returncode == 0
    assert "trades, depth, book, candles" in " ".join(run.stdout.split())
