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
