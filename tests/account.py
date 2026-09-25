import pytest
from conftest import ledger, position, snapshot, utc

import galata_research as gr
from galata_research import Refused

SEPT = utc("2026-09-01T00:00"), utc("2026-10-01T00:00")


def _margin(**kwargs):
    return gr.account.margin(kwargs.pop("accounts", None), *SEPT, **kwargs).collect()


def _positions(**kwargs):
    return gr.account.positions(kwargs.pop("accounts", None), *SEPT, **kwargs).collect()


# Margin


def the_venues_time_is_ts(tape):
    # Guard: the receipt is 90 ms later, and must not stand in for the venue's clock.
    ledger(tape.root, [{"payload": snapshot(time_ms=1790347182376), "recv": 1790347182465855}])
    got = _margin()
    assert got["ts"].item() == utc("2026-09-25T14:39:42.376")
    assert got["recv_ts"].item() == utc("2026-09-25T14:39:42.465855")


def every_dex_is_its_own_snapshot(tape):
    ledger(tape.root, [{"payload": snapshot(dex="")}, {"payload": snapshot(dex="xyz")}])
    assert _margin()["dex"].to_list() == ["", "xyz"]


def a_unified_account_does_not_hold_equity(tape):
    ledger(tape.root, [{"payload": snapshot(mode="unifiedAccount")}, {"payload": snapshot(dex="xyz", mode="default")}])
    assert _margin()["equity_held"].to_list() == [False, True]


def the_window_selects_on_ts(tape):
    ledger(tape.root, [{"payload": snapshot(time_ms=1790347182376)}, {"payload": snapshot(dex="xyz", time_ms=1759276800000)}])
    got = gr.account.margin(None, utc("2026-09-25T00:00"), utc("2026-09-26T00:00")).collect()
    assert got["dex"].to_list() == [""]


# Positions


def a_short_position_is_signed_and_has_a_mark(tape):
    ledger(tape.root, [{"payload": snapshot(positions=[position(szi="-0.83", value="53120.0")])}])
    got = _positions()
    assert got["size"].item() == -0.83
    assert got["mark_px"].item() == pytest.approx(64000.0)
    assert got.select("leverage_type", "leverage", "cum_funding_all_time").row(0) == ("cross", 20.0, 3.21)


def a_null_liquidation_price_is_absent(tape):
    ledger(tape.root, [{"payload": snapshot(positions=[position(liquidation=None)])}])
    assert _positions()["liquidation_px"].to_list() == [None]


def a_flat_position_has_no_mark(tape):
    ledger(tape.root, [{"payload": snapshot(positions=[position(szi="0.0", value="0.0")])}])
    assert _positions()["mark_px"].to_list() == [None]


def a_flat_account_has_no_position_rows(tape):
    ledger(tape.root, [{"payload": snapshot()}])
    got = _positions()
    assert got.height == 0
    assert got.schema == gr.account.POSITION_SCHEMA


# Refusals


def a_missing_liquidation_key_is_refused_by_path(tape):
    # Guard: a missing key must not decode as null, which would read as "no liquidation".
    p = position()
    del p["position"]["liquidationPx"]
    ledger(tape.root, [{"payload": snapshot(positions=[p])}])
    with pytest.raises(Refused, match=r"assetPositions\[0\]\.position \(BTC\) carries no liquidationPx"):
        _positions()


def a_snapshot_without_time_is_refused(tape):
    s = snapshot()
    del s["clearinghouseState"]["time"]
    ledger(tape.root, [{"payload": s}])
    with pytest.raises(Refused, match="clearinghouseState carries no time"):
        _margin()


def an_unknown_channel_is_refused(tape):
    ledger(tape.root, [{"payload": {}, "channel": "spotClearinghouseState"}])
    with pytest.raises(Refused, match="channel 'spotClearinghouseState' is not decoded here"):
        _margin()


def the_mode_rows_are_skipped(tape):
    ledger(tape.root, [{"payload": "unifiedAccount", "channel": "userAbstraction"}, {"payload": snapshot()}])
    assert _margin().height == 1


def a_different_schema_version_is_refused(tape):
    ledger(tape.root, [{"payload": snapshot(), "schema_version": 2}])
    with pytest.raises(Refused, match="schema_version 2 is not 1"):
        _margin()


def an_unknown_alias_is_refused_with_the_list(tape):
    ledger(tape.root, [{"payload": snapshot()}])
    with pytest.raises(Refused, match="no account savings; it holds main"):
        _margin(accounts="savings")


def a_ledger_without_snapshots_is_refused(tape):
    with pytest.raises(Refused, match="no margin snapshots"):
        _margin()
