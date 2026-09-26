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


# The ledger projection: fills, funding payments, ledger updates

from decimal import Decimal  # noqa: E402

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from conftest import DECIMAL, us  # noqa: E402

_BASE = [
    ("venue", pa.string()),
    ("account", pa.string()),
    ("dex", pa.string()),
    ("ticker", pa.string()),
    ("at_micros", pa.int64()),
    ("recv_micros", pa.int64()),
    ("schema_version", pa.uint16()),
]
PROJECTED = {
    # As datawatch's ledger/project.rs writes them.
    "fills": pa.schema([*_BASE, ("side", pa.string()), ("price", DECIMAL), ("size", DECIMAL), ("start_position", DECIMAL),
                        ("direction", pa.string()), ("closed_pnl", DECIMAL), ("fee", DECIMAL), ("fee_token", pa.string()),
                        ("builder_fee", DECIMAL), ("crossed", pa.bool_()), ("order_id", pa.uint64()), ("trade_id", pa.uint64()),
                        ("twap_id", pa.uint64())]),
    "funding_payments": pa.schema([*_BASE, ("usdc", DECIMAL), ("size", DECIMAL), ("rate", DECIMAL), ("samples", pa.uint32())]),
    "ledger_updates": pa.schema([*_BASE, ("update_kind", pa.string()), ("effect_known", pa.bool_()), ("effect_usdc", DECIMAL),
                                 ("counterparty_kind", pa.string()), ("counterparty", pa.string()), ("token", pa.string()),
                                 ("amount", DECIMAL), ("fee", DECIMAL)]),
}


def _project(root, kind, rows, account="main"):
    path = root / "ledger-tape" / "venue=hyperliquid" / f"account={account}" / f"kind={kind}" / "rows.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    base = {"venue": "hyperliquid", "account": account, "dex": None, "ticker": None, "schema_version": 1}
    pq.write_table(pa.Table.from_pylist([{**base, **r} for r in rows], schema=PROJECTED[kind]), path)


def _fill(at, trade_id, order_id):
    return {"ticker": "BTC", "at_micros": us(at), "recv_micros": us(at) + 1, "side": "bid", "price": Decimal("81213.5"),
            "size": Decimal("0.01"), "order_id": order_id, "trade_id": trade_id, "crossed": True}


def the_fills_in_a_window_are_read_as_projected(tape):
    _project(tape.root, "fills", [_fill("2026-09-10T12:00", 7, 100), _fill("2026-10-02T12:00", 8, 101)])
    got = gr.account.fills("main", *SEPT).collect()
    assert got.select("trade_id", "order_id", "side").rows() == [(7, 100, "bid")]
    # Float64 on load, as every money column in research is.
    assert got.select("price", "size").rows() == [(pytest.approx(81213.5), pytest.approx(0.01))]
    assert got["ts"].to_list() == [utc("2026-09-10T12:00")]


def no_projection_is_refused_by_name(tape):
    with pytest.raises(Refused, match="ledger.tape"):
        gr.account.fills("main", *SEPT)


def an_empty_kind_is_an_empty_frame(tape):
    _project(tape.root, "funding_payments", [])
    got = gr.account.funding_payments("main", *SEPT).collect()
    assert got.height == 0
    assert got.columns == list(gr.account.FUNDING_PAYMENT_SCHEMA)


def a_ledger_update_moving_two_dexes_is_two_rows(tape):
    at = us("2026-09-12T09:00")
    moved = {"at_micros": at, "recv_micros": at, "update_kind": "accountClassTransfer", "effect_known": True,
             "counterparty_kind": None, "counterparty": None, "amount": Decimal("10")}
    _project(tape.root, "ledger_updates", [{**moved, "effect_usdc": Decimal("-10")},
                                           {**moved, "dex": "xyz", "effect_usdc": Decimal("10")}])
    got = gr.account.ledger_updates(None, *SEPT).collect()
    assert got.select("dex", "effect_usdc").rows() == [(None, -10.0), ("xyz", 10.0)]
    assert got["update_kind"].unique().to_list() == ["accountClassTransfer"]


def an_account_the_projection_does_not_hold_is_refused_with_the_list(tape):
    _project(tape.root, "fills", [])
    with pytest.raises(Refused, match=r"savings.*main"):
        gr.account.fills("savings", *SEPT)
