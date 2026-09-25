"""My accounts, from the ledger: margin and positions per snapshot.

The ledger keeps each `clearinghouseState` answer verbatim, as a payload.
This decodes that one channel, strictly, until datawatch projects the ledger
into typed datasets as it does the tape: a second parser, kept small and
temporary on purpose.

A shape the venue does not document is refused by path. A statement quietly
missing a row would read as flat, and a field quietly missing would read as
zero (galata-legacy `crates/hyperliquid/src/wire.rs`).
"""

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import polars as pl

from . import _root, _scan
from ._errors import Refused

DECODED = "clearinghouseState"
# Its answer is embedded in every snapshot as `mode`, so its own rows add nothing.
SKIPPED = {"userAbstraction"}
SCHEMA_VERSION = 1

_SUMMARY = {"accountValue": "account_value", "totalNtlPos": "total_ntl_pos", "totalRawUsd": "total_raw_usd", "totalMarginUsed": "total_margin_used"}

MARGIN_SCHEMA = {
    "venue": pl.String,
    "account": pl.String,
    "dex": pl.String,
    "ts": _scan.UTC_US,
    "recv_ts": _scan.UTC_US,
    "mode": pl.String,
    "equity_held": pl.Boolean,
    **{name: pl.Float64 for name in _SUMMARY.values()},
    **{f"cross_{name}": pl.Float64 for name in _SUMMARY.values()},
    "cross_maintenance_margin_used": pl.Float64,
    "withdrawable": pl.Float64,
    "positions": pl.UInt32,
}

POSITION_SCHEMA = {
    "venue": pl.String,
    "account": pl.String,
    "dex": pl.String,
    "ts": _scan.UTC_US,
    "recv_ts": _scan.UTC_US,
    "coin": pl.String,
    "size": pl.Float64,
    "entry_px": pl.Float64,
    "mark_px": pl.Float64,
    "position_value": pl.Float64,
    "unrealized_pnl": pl.Float64,
    "liquidation_px": pl.Float64,
    "margin_used": pl.Float64,
    "leverage_type": pl.String,
    "leverage": pl.Float64,
    "max_leverage": pl.Float64,
    "return_on_equity": pl.Float64,
    "cum_funding_all_time": pl.Float64,
    "cum_funding_since_open": pl.Float64,
    "cum_funding_since_change": pl.Float64,
}


def margin(
    accounts: Sequence[str] | str | None,
    start: datetime | str,
    end: datetime | str,
    *,
    venue: str | None = None,
    engine: str = "polars",
):
    """One row per snapshot per `(venue, account, dex)`, `ts` in `[start, end)`.

    `ts` is the venue's own `clearinghouseState.time` (whole milliseconds).
    `equity_held` is false on a unified account: there, perps `account_value`
    is margin plus unrealised, 0 while flat, and the cash is in spot USDC.
    """
    margins, _ = _decoded(accounts, start, end, venue, engine)
    return _scan.finish(margins, engine)


def positions(
    accounts: Sequence[str] | str | None,
    start: datetime | str,
    end: datetime | str,
    *,
    venue: str | None = None,
    engine: str = "polars",
):
    """One row per position per snapshot. `size` is signed; `mark_px` is `position_value / |size|`.

    `coin` is the venue's name for the market (`xyz:GOLD`), not the tape's
    ticker. `liquidation_px` is the venue's forward-looking estimate, null
    where it printed none. `cum_funding_*` are as printed; their sign is not
    reinterpreted.
    """
    _, held = _decoded(accounts, start, end, venue, engine)
    return _scan.finish(held, engine)


def _decoded(accounts, start, end, venue, engine) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    lo, hi = _scan.window(start, end)
    if engine not in _scan.ENGINES:
        raise Refused(f"engine={engine!r} is not one of {', '.join(_scan.ENGINES)}")
    margins, held = [], []
    for row in _rows(accounts, venue):
        if row["channel"] in SKIPPED:
            continue
        m, ps = _snapshot(row)
        if lo <= m["_ts"] < hi:
            margins.append(m)
            held.extend(ps)
    return _frame(margins, MARGIN_SCHEMA), _frame(held, POSITION_SCHEMA)


def _rows(accounts, venue) -> list[dict]:
    ledger = _root.root() / "ledger"
    kinds = sorted(ledger.glob("venue=*/account=*/kind=margin"))
    if venue is not None:
        kinds = [k for k in kinds if k.parent.parent.name == f"venue={venue}"]
    if not kinds:
        raise Refused(f"the ledger has no margin snapshots under {ledger}" + (f" for venue {venue}" if venue else ""))
    held = sorted({k.parent.name.removeprefix("account=") for k in kinds})
    if accounts is not None:
        wanted = [accounts] if isinstance(accounts, str) else list(accounts)
        unknown = [a for a in wanted if a not in held]
        if unknown:
            raise Refused(f"the ledger holds no account {', '.join(unknown)}; it holds {', '.join(held)}")
        kinds = [k for k in kinds if k.parent.name.removeprefix("account=") in wanted]
    # The alias is a directory level only (`account=<alias>`), never a column.
    read = ["venue", "channel", "recv_micros", "seq", "schema_version", "payload"]
    frames = []
    for kind in kinds:
        files: list[Path] = sorted(kind.glob("date=*/*.parquet"))
        if files:
            alias = kind.parent.name.removeprefix("account=")
            frames.append(
                pl.scan_parquet(files, hive_partitioning=False).select(read).with_columns(pl.lit(alias).alias("account"))
            )
    return pl.concat(frames).collect().to_dicts() if frames else []


def _snapshot(row: dict) -> tuple[dict, list[dict]]:
    where = f"{row['venue']}/{row['account']} seq {row['seq']}"
    if row["channel"] != DECODED:
        raise Refused(f"{where}: channel {row['channel']!r} is not decoded here; only {DECODED} is")
    if row["schema_version"] != SCHEMA_VERSION:
        raise Refused(f"{where}: schema_version {row['schema_version']} is not {SCHEMA_VERSION}")
    try:
        payload = json.loads(bytes(row["payload"]))
    except ValueError as e:
        raise Refused(f"{where}: the payload is not JSON ({e})") from None

    state = _obj(payload, "clearinghouseState", where)
    dex = _key(payload, "dex", where)
    mode = _key(_obj(payload, "mode", where), "answer", f"{where} mode")
    time = _key(state, "time", f"{where} clearinghouseState")
    if not isinstance(time, int):
        raise Refused(f"{where} clearinghouseState.time {time!r} is not integer milliseconds")
    base = {"venue": row["venue"], "account": row["account"], "dex": dex, "_ts": time * 1_000, "_recv": row["recv_micros"]}

    m = {**base, "mode": mode, "equity_held": mode != "unifiedAccount"}
    for block, prefix in (("marginSummary", ""), ("crossMarginSummary", "cross_")):
        summary = _obj(state, block, f"{where} clearinghouseState")
        for key, name in _SUMMARY.items():
            m[prefix + name] = _num(summary, key, f"{where} {block}")
    m["cross_maintenance_margin_used"] = _num(state, "crossMaintenanceMarginUsed", f"{where} clearinghouseState")
    m["withdrawable"] = _num(state, "withdrawable", f"{where} clearinghouseState")

    entries = _key(state, "assetPositions", f"{where} clearinghouseState")
    if not isinstance(entries, list):
        raise Refused(f"{where} clearinghouseState.assetPositions is not an array")
    m["positions"] = len(entries)
    return m, [_position(base, entry, i, where) for i, entry in enumerate(entries)]


def _position(base: dict, entry, index: int, where: str) -> dict:
    position = _obj(entry, "position", f"{where} assetPositions[{index}]") if isinstance(entry, dict) else None
    if position is None:
        raise Refused(f"{where} assetPositions[{index}] carries no position object")
    coin = _key(position, "coin", f"{where} assetPositions[{index}].position")
    path = f"{where} assetPositions[{index}].position ({coin})"
    size = _num(position, "szi", path)
    value = _num(position, "positionValue", path)
    leverage = _obj(position, "leverage", path)
    funding = _obj(position, "cumFunding", path)
    return {
        **base,
        "coin": coin,
        "size": size,
        "entry_px": _num(position, "entryPx", path),
        # A flat position has no mark: there is nothing to divide by, and a zero would be a price.
        "mark_px": value / abs(size) if size else None,
        "position_value": value,
        "unrealized_pnl": _num(position, "unrealizedPnl", path),
        "liquidation_px": _num(position, "liquidationPx", path, nullable=True),
        "margin_used": _num(position, "marginUsed", path),
        "leverage_type": _key(leverage, "type", f"{path}.leverage"),
        "leverage": _num(leverage, "value", f"{path}.leverage"),
        "max_leverage": _num(position, "maxLeverage", path, nullable=True),
        "return_on_equity": _num(position, "returnOnEquity", path, nullable=True),
        "cum_funding_all_time": _num(funding, "allTime", f"{path}.cumFunding"),
        "cum_funding_since_open": _num(funding, "sinceOpen", f"{path}.cumFunding"),
        "cum_funding_since_change": _num(funding, "sinceChange", f"{path}.cumFunding"),
    }


def _key(obj: dict, key: str, path: str):
    if key not in obj:
        raise Refused(f"{path} carries no {key}")
    return obj[key]


def _obj(obj: dict, key: str, path: str) -> dict:
    value = _key(obj, key, path)
    if not isinstance(value, dict):
        raise Refused(f"{path}.{key} is not an object")
    return value


def _num(obj: dict, key: str, path: str, *, nullable: bool = False) -> float | None:
    """A decimal the venue printed as a string (or a bare number), as a float.

    The key missing is refused. `null` is absence, and allowed only where the
    venue documents it.
    """
    value = _key(obj, key, path)
    if value is None:
        if nullable:
            return None
        raise Refused(f"{path}.{key} is null")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise Refused(f"{path}.{key} {value!r} is not a decimal") from None


def _frame(rows: list[dict], schema: dict) -> pl.LazyFrame:
    frame = pl.DataFrame(rows, schema={**{k: v for k, v in schema.items() if k not in ("ts", "recv_ts")}, "_ts": pl.Int64, "_recv": pl.Int64}) if rows else None
    if frame is None:
        return pl.LazyFrame(schema=schema)
    return (
        frame.lazy()
        .with_columns(_scan.clock("_ts", "ts"), _scan.clock("_recv", "recv_ts"))
        .select(list(schema))
        .sort(["venue", "account", "dex", "ts"])
    )
