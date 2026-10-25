"""hyperliquid-s3: Hyperliquid's published fills, from a fixture bucket written in each test, never the network.

The bucket is served by replacing `fetch._transport`, the one place a socket
opens: a listing answers from the keys it holds, a GET with a key's bytes and
ETag, anything else 404. `fetch._signer` is replaced by one that signs
nothing, except where a test checks botocore's own signature. The three
trimmed hours under `data/hyperliquid_s3/` are the bucket's real formats
(`data/README.md`).
"""

import hashlib
import json
import urllib.parse
from datetime import date
from pathlib import Path

import lz4.frame
import polars as pl
import pytest
from conftest import Tape, utc

import galata_research as gr
from galata_research import Refused
from galata_research.reference import _hyperliquid, _instruments, _manifest, fetch

DATA = Path(__file__).parent / "data" / "hyperliquid_s3"
HOST = _hyperliquid.HOST
VENUE = "hyperliquid-s3"
EMPTY = lz4.frame.compress(b"")
COINS = {"BTC": "BTC", "xyz:CL": "CL"}
SYMBOLS = {"BTC": "BTC", "CL": "xyz:CL"}
D = date(2026, 9, 24)
SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
TOKEN = "IQoJb3JpZ2luX2VjEXAMPLESESSIONTOKEN"


def _ms(iso: str) -> int:
    return int(utc(iso).timestamp() * 1000)


def _pair(coin, tid, at, px, sz="1", taker="B", liquidation=False, crossed=(True, False)):
    """One trade as the fills formats write it: the taker's fill, crossed, then the maker's."""

    def fill(side, cross, user):
        f = {"coin": coin, "px": str(px), "sz": str(sz), "side": side, "time": _ms(at), "crossed": cross, "tid": tid,
             "dir": "Open Long", "oid": 1, "hash": "0x0", "fee": "0.0"}  # fmt: skip
        if liquidation:
            f["liquidation"] = {"liquidatedUser": "0xdead", "markPx": str(px), "method": "market"}
        return [user, f]

    return [fill(taker, crossed[0], "0xtaker"), fill("A" if taker == "B" else "B", crossed[1], "0xmaker")]


def _lz4(lines: list[str]) -> bytes:
    return lz4.frame.compress(("\n".join(lines) + "\n").encode())


def _blocks(*trades) -> bytes:
    """node_fills_by_block: each trade in a block of its own."""
    return _lz4(
        [json.dumps({"local_time": "", "block_time": "", "block_number": n, "events": t}, separators=(",", ":")) for n, t in enumerate(trades)]
    )


def _fills(*trades) -> bytes:
    """node_fills: one [address, fill] a line."""
    return _lz4([json.dumps(f, separators=(",", ":")) for t in trades for f in t])


def _node_trade(coin, at, px, sz="1", side="B", override="Na") -> dict:
    """node_trades: `at` to the nanosecond, as the archive writes it."""
    return {"coin": coin, "side": side, "time": at, "px": str(px), "sz": str(sz), "hash": "0x0",
            "trade_dir_override": override, "side_info": [{"user": "0xb"}, {"user": "0xs"}]}  # fmt: skip


def _trades(*rows: dict) -> bytes:
    return _lz4([json.dumps(r, separators=(",", ":")) for r in rows])


def _key(prefix: str, hour_iso: str) -> str:
    return _hyperliquid.key(prefix, utc(hour_iso))


BLOCK = "node_fills_by_block"
K14 = _key(BLOCK, "2026-09-24T14:00:00")


# ---- the bucket ----------------------------------------------------------------


class Bucket:
    """A fixture bucket: `objects[key] = bytes`, with every request and its headers recorded."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.asked: list[tuple[str, str, dict | None]] = []

    def etag(self, key: str) -> str:
        return hashlib.md5(self.objects[key]).hexdigest()

    def __call__(self, url, method, agent="galata-fetch", headers=None):
        self.asked.append((method, url, headers))
        parts = urllib.parse.urlsplit(url)
        assert parts.hostname == HOST
        if parts.query:
            prefix = urllib.parse.parse_qs(parts.query)["prefix"][0]
            listed = "".join(
                f"<Contents><Key>{k}</Key><Size>{len(v)}</Size><ETag>&quot;{self.etag(k)}&quot;</ETag></Contents>"
                for k, v in sorted(self.objects.items())
                if k.startswith(prefix)
            )
            body = f"<ListBucketResult xmlns='http://s3.amazonaws.com/doc/2006-03-01/'>{listed}<IsTruncated>false</IsTruncated></ListBucketResult>"
            return fetch.Response(200, body.encode(), len(body))
        key = urllib.parse.unquote(parts.path[1:])
        if key not in self.objects:
            return fetch.Response(404)
        body = self.objects[key]
        return fetch.Response(200, body if method == "GET" else b"", len(body), {"etag": f'"{self.etag(key)}"'})

    def gets(self) -> list[str]:
        return [urllib.parse.urlsplit(u).path[1:] for m, u, _ in self.asked if m == "GET" and "?" not in u]

    def lists(self) -> int:
        return sum("?" in u for _, u, _ in self.asked)

    def day(self, day: date, bodies: dict[str, bytes] | None = None, missing=()) -> "Bucket":
        """Every file a day needs, empty unless given, except `missing`."""
        for k in _hyperliquid.keys(day):
            if k not in missing:
                self.objects[k] = (bodies or {}).get(k, EMPTY)
        return self


class _Unsigned:
    profile = "test"

    def __call__(self, method, url):
        return {"x-amz-request-payer": "requester"}


@pytest.fixture(name="bucket")
def _bucket(monkeypatch):
    b = Bucket()
    monkeypatch.setattr(fetch, "_transport", b)
    monkeypatch.setattr(fetch, "_signer", _Unsigned)
    return b


@pytest.fixture(name="store")
def _store(tmp_path, monkeypatch):
    path = tmp_path / "reference"
    monkeypatch.setenv("GALATA_REFERENCE", str(path))
    monkeypatch.setenv("GALATA_VAR", str(tmp_path / "record"))
    return path


def _run(*argv) -> tuple[int, list[str]]:
    said: list[str] = []
    return fetch.run(list(argv), say=said.append), said


def _fetch(*tickers, day="2026-09-24", extra=()):
    return _run("candles", VENUE, *tickers, "--from", day, "--to", day, "--days", day, *extra)


def _no_aws(monkeypatch, tmp_path):
    for v in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_SECURITY_TOKEN"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


def _keys(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET)
    monkeypatch.setenv("AWS_SESSION_TOKEN", TOKEN)


def _assemble(day, bodies, missing=(), coins=COINS, symbols=SYMBOLS):
    read = {}
    for k in _hyperliquid.keys(day):
        if k in missing:
            read[k] = None
            continue
        try:
            read[k] = _hyperliquid.read(bodies.get(k, EMPTY), k, coins)
        except _hyperliquid.HourFailed as e:
            read[k] = e
    return _hyperliquid.assemble(day, read, symbols)


def _bar(bars: pl.DataFrame, ticker: str, minute: str) -> dict:
    rows = bars.filter((pl.col("ticker") == ticker) & (pl.col("ts") == utc(minute))).to_dicts()
    assert len(rows) == 1, rows
    return rows[0]


# ---- the instruments -----------------------------------------------------------


def a_ticker_the_venue_lacks_is_refused_with_the_held_list(store, bucket):
    held = "BRENTOIL, BTC, CL, ETH, GOLD, HYPE, NVDA, SILVER, SKHX, SP500, XYZ100"
    with pytest.raises(Refused, match=f"hyperliquid-s3 lists no WTIOIL; it lists {held}"):
        _fetch("WTIOIL")
    assert bucket.asked == []


def no_xyz_ticker_is_another_venues_symbol():
    ours = {t: i.symbol for (v, t), i in _instruments.INSTRUMENTS.items() if v == VENUE}
    assert {t: s for t, s in ours.items() if t not in ("BTC", "ETH", "HYPE")} == {
        t: f"xyz:{t}" for t in ("BRENTOIL", "CL", "GOLD", "NVDA", "SILVER", "SKHX", "SP500", "XYZ100")
    }
    assert {ours[t] for t in ("BTC", "ETH", "HYPE")} == {"BTC", "ETH", "HYPE"}


def the_handovers_read_each_hour_from_one_format():
    assert _instruments.prefix_at(utc("2025-03-22T09:00:00")) is None
    assert _instruments.prefix_at(utc("2025-05-25T13:00:00")) == "node_trades"
    assert _instruments.prefix_at(utc("2025-05-25T14:00:00")) == "node_fills"
    assert _instruments.prefix_at(utc("2025-07-27T07:00:00")) == "node_fills"
    assert _instruments.prefix_at(utc("2025-07-27T08:00:00")) == BLOCK


def an_ordinary_day_is_twenty_five_files():
    keys = _hyperliquid.keys(D)
    assert len(keys) == 25
    assert "node_fills_by_block/hourly/20260925/0.lz4" in keys  # its last second's spill
    assert "node_fills_by_block/hourly/20260923/23.lz4" not in keys  # no spill runs backwards


# ---- signing and the bucket's bounds -------------------------------------------


def the_bucket_is_asked_with_a_signature_and_requester_pays(monkeypatch, tmp_path):
    _no_aws(monkeypatch, tmp_path)
    _keys(monkeypatch)
    headers = fetch.Signer()("GET", f"https://{HOST}/{K14}")
    assert headers["x-amz-request-payer"] == "requester"
    auth = headers["Authorization"]
    assert auth.startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/") and "/ap-northeast-1/s3/aws4_request" in auth
    assert "x-amz-request-payer" in auth.split("SignedHeaders=")[1]
    assert headers["X-Amz-Security-Token"] == TOKEN  # sent as the protocol needs, never kept
    assert SECRET not in json.dumps(headers)


def no_request_is_sent_without_credentials(store, monkeypatch, tmp_path):
    _no_aws(monkeypatch, tmp_path)
    asked = []
    monkeypatch.setattr(fetch, "_transport", lambda *a, **k: asked.append(a))
    with pytest.raises(Refused, match=r"no AWS credentials.*requester-pays.*AWS_PROFILE.*~/.aws/config"):
        _fetch("BTC")
    assert asked == []


def no_secret_reaches_the_store_or_the_output(store, bucket, monkeypatch, tmp_path):
    _no_aws(monkeypatch, tmp_path)
    _keys(monkeypatch)
    monkeypatch.setattr(fetch, "_signer", fetch.Signer)
    bucket.day(D, {K14: _blocks(_pair("BTC", 1, "2026-09-24T14:05:00", 100))})
    code, said = _fetch("BTC")
    assert code == 0
    assert all(h and "Authorization" in h for _, _, h in bucket.asked)
    written = "\n".join(said)
    for path in store.rglob("*.parquet"):
        frame = pl.read_parquet(path)
        written += "\n".join(str(v) for c in frame.columns if frame[c].dtype == pl.String for v in frame[c].to_list())
    for secret in (SECRET, TOKEN, "AKIDEXAMPLE"):
        assert secret not in written


def a_key_outside_the_three_prefixes_is_refused_before_sending(bucket):
    sign = _Unsigned()
    for asked in ("replica_cmds/hourly/20260924/14.lz4", "node_trades/hourly/../../replica_cmds/x", "explorer_blocks/0"):
        with pytest.raises(Refused, match=r"outside what hyperliquid-s3 reads: s3://hl-mainnet-node-data/ under node_trades/hourly/, node_fills/hourly/, node_fills_by_block/hourly/"):
            fetch.s3(asked, sign)
    with pytest.raises(Refused, match="outside"):
        fetch.s3("?list-type=2&prefix=misc_events_by_block%2F", sign)
    assert bucket.asked == []


def the_bucket_is_never_asked_without_a_signature(bucket):
    with pytest.raises(Refused, match="is not an archive this command fetches from"):
        fetch.get(f"https://{HOST}/{K14}")
    assert bucket.asked == []


def an_expired_login_is_named_not_shown(store, monkeypatch):
    def answer(url, method, agent="galata-fetch", headers=None):
        return fetch.Response(403, b"<Error><Code>ExpiredToken</Code><Message>x</Message></Error>")

    monkeypatch.setattr(fetch, "_transport", answer)
    monkeypatch.setattr(fetch, "_signer", _Unsigned)
    with pytest.raises(Refused, match=r"HTTP 403 \(ExpiredToken\).*profile test logged in"):
        _fetch("BTC")


# ---- reading the three formats -------------------------------------------------


def _raw_trades(key: str, coins) -> set:
    found = set()
    for line in lz4.frame.decompress((DATA / key).read_bytes()).splitlines():
        record = json.loads(line)
        events = record["events"] if isinstance(record, dict) else [record]
        found |= {(f["coin"], f["tid"]) for _, f in events if f["coin"] in coins}
    return found


def every_fills_format_reads_to_one_row_per_trade():
    coins = {"BTC": "BTC", "ETH": "ETH", "xyz:CL": "CL", "xyz:SILVER": "SILVER"}
    for key in ("node_fills/hourly/20250701/12.lz4", f"{BLOCK}/hourly/20260924/14.lz4"):
        got = _hyperliquid.read((DATA / key).read_bytes(), key, coins)
        assert got.height == len(_raw_trades(key, coins)) > 0
        assert got["liquidation"].any()


def the_xyz_coins_are_read_by_their_venue_names():
    key = f"{BLOCK}/hourly/20260924/14.lz4"
    got = _hyperliquid.read((DATA / key).read_bytes(), key, {"xyz:CL": "CL", "xyz:SILVER": "SILVER"})
    assert set(got["ticker"]) == {"CL", "SILVER"}


def a_spot_coin_beside_the_perps_is_left_out():
    key = f"{BLOCK}/hourly/20260924/14.lz4"
    assert b'"coin":"@107"' in lz4.frame.decompress((DATA / key).read_bytes())
    got = _hyperliquid.read((DATA / key).read_bytes(), key, {"BTC": "BTC"})
    assert set(got["ticker"]) == {"BTC"}


def the_node_trades_fixture_keeps_every_row_but_the_vaults():
    key = "node_trades/hourly/20250401/12.lz4"
    rows = [json.loads(line) for line in lz4.frame.decompress((DATA / key).read_bytes()).splitlines()]
    coins = {r["coin"]: r["coin"] for r in rows if not r["coin"].startswith("@")}
    got = _hyperliquid.read((DATA / key).read_bytes(), key, coins)
    kept = [r for r in rows if r["coin"] in coins and r["trade_dir_override"] != "NetChildVaultPositions"]
    assert got.height == len(kept)
    assert got["liquidation"].sum() == sum(r["trade_dir_override"] == "LiquidatedMarket" for r in kept)


def a_vault_row_is_not_a_trade_and_a_liquidation_is():
    blob = _trades(
        _node_trade("BTC", "2025-04-01T12:00:00.000000001", 100),
        _node_trade("BTC", "2025-04-01T12:00:01.5", 99, override="LiquidatedMarket"),
        _node_trade("BTC", "2025-04-01T12:00:02", 98, override="NetChildVaultPositions"),
    )
    got = _hyperliquid.read(blob, "t", {"BTC": "BTC"})
    assert got["price"].to_list() == [100.0, 99.0]
    assert got["liquidation"].to_list() == [False, True]
    assert got["micros"].to_list() == [utc("2025-04-01T12:00:00").timestamp() * 1e6, utc("2025-04-01T12:00:01.5").timestamp() * 1e6]


def an_override_of_another_word_fails_the_hour():
    with pytest.raises(_hyperliquid.HourFailed, match="trade_dir_override 'Other'") as e:
        _hyperliquid.read(_trades(_node_trade("BTC", "2025-04-01T12:00:00", 1, override="Other")), "t", {"BTC": "BTC"})
    assert e.value.cause == "unparsed"


def every_identical_node_trades_row_counts():
    row = _node_trade("BTC", "2025-04-01T12:00:00.5", 100)
    assert _hyperliquid.read(_trades(row, row), "t", {"BTC": "BTC"}).height == 2


def a_maker_and_taker_pair_counts_once():
    got = _hyperliquid.read(_blocks(_pair("BTC", 7, "2026-09-24T14:00:01", 100, sz="2", taker="A")), "b", COINS)
    assert got.select("size", "taker_buy").rows() == [(2.0, False)]


def a_trade_with_two_takers_fails_its_hour():
    with pytest.raises(_hyperliquid.HourFailed, match="BTC trade 7 has 2 taker fills") as e:
        _hyperliquid.read(_fills(_pair("BTC", 7, "2026-09-24T14:00:01", 100, crossed=(True, True))), "f", COINS)
    assert e.value.cause == "unparsed"


def a_line_of_no_known_shape_fails_the_hour():
    for blob in (_lz4(["not json"]), _lz4(['{"something":"else"}'])):
        with pytest.raises(_hyperliquid.HourFailed) as e:
            _hyperliquid.read(blob, "x", COINS)
        assert e.value.cause == "unparsed"


def a_file_cut_short_is_unreadable():
    with pytest.raises(_hyperliquid.HourFailed) as e:
        _hyperliquid.read(_blocks(_pair("BTC", 1, "2026-09-24T14:00:01", 100))[:-6], "b", COINS)
    assert e.value.cause == "unreadable"


# ---- bars ----------------------------------------------------------------------


def the_prices_follow_the_trades_own_order():
    trades = [_pair("BTC", n, f"2026-09-24T14:05:0{n}", px) for n, px in enumerate([100, 102, 99, 101], start=1)]
    bars, gaps = _assemble(D, {K14: _blocks(*trades)})
    bar = _bar(bars, "BTC", "2026-09-24T14:05:00")
    assert (bar["open"], bar["high"], bar["low"], bar["close"]) == (100, 102, 99, 101)
    assert (bar["high_all"], bar["low_all"], bar["trade_count"], bar["volume"]) == (102, 99, 4, 4)
    assert bar["close_ts"] == utc("2026-09-24T14:06:00")
    assert (bar["venue"], bar["symbol"], bar["interval"]) == (VENUE, "BTC", "1m")


def a_liquidation_wick_is_low_all_not_low():
    bars, _ = _assemble(D, {K14: _blocks(
        _pair("BTC", 1, "2026-09-24T14:05:01", 83400),
        _pair("BTC", 2, "2026-09-24T14:05:02", 83350, liquidation=True),
        _pair("BTC", 3, "2026-09-24T14:05:03", 83351),
    )})  # fmt: skip
    bar = _bar(bars, "BTC", "2026-09-24T14:05:00")
    assert (bar["low"], bar["low_all"], bar["high"], bar["high_all"]) == (83351, 83350, 83400, 83400)
    assert (bar["volume"], bar["trade_count"], bar["liquidations"]) == (3, 3, 1)


def a_minute_of_liquidations_only_takes_their_prices():
    bars, _ = _assemble(D, {K14: _blocks(
        _pair("BTC", 1, "2026-09-24T14:05:01", 100, liquidation=True),
        _pair("BTC", 2, "2026-09-24T14:05:02", 98, liquidation=True),
    )})  # fmt: skip
    bar = _bar(bars, "BTC", "2026-09-24T14:05:00")
    assert (bar["high"], bar["high_all"], bar["low"], bar["low_all"]) == (100, 100, 98, 98)


def the_bar_sums_notional_and_the_takers_buys():
    bars, _ = _assemble(D, {K14: _blocks(
        _pair("BTC", 1, "2026-09-24T14:05:01", 100, sz="2", taker="B"),
        _pair("BTC", 2, "2026-09-24T14:05:02", 101, sz="1", taker="A"),
    )})  # fmt: skip
    bar = _bar(bars, "BTC", "2026-09-24T14:05:00")
    assert (bar["notional"], bar["taker_buy_volume"]) == (301, 2)


def a_trade_carried_in_the_next_hours_file_is_placed_by_its_time():
    k13 = _key(BLOCK, "2026-09-24T13:00:00")
    bars, gaps = _assemble(D, {k13: _blocks(_pair("BTC", 1, "2026-09-24T13:10:00", 50)),
                               K14: _blocks(_pair("BTC", 2, "2026-09-24T13:59:59.900", 99), _pair("BTC", 3, "2026-09-24T14:00:01", 100))})  # fmt: skip
    assert _bar(bars, "BTC", "2026-09-24T13:59:00")["close"] == 99
    assert _bar(bars, "BTC", "2026-09-24T14:00:00")["close"] == 100
    assert gaps.is_empty()


def the_days_last_second_comes_from_the_next_days_file():
    nxt = _key(BLOCK, "2026-09-25T00:00:00")
    spill = _blocks(_pair("BTC", 1, "2026-09-24T23:59:59.900", 99))
    bars, _ = _assemble(D, {nxt: spill})
    assert _bar(bars, "BTC", "2026-09-24T23:59:00")["close"] == 99
    following, _ = _assemble(date(2026, 9, 25), {nxt: spill})
    assert following.is_empty()  # it is the 24th's, not the 25th's


def a_quiet_minute_has_no_bar_and_no_gap():
    bars, gaps = _assemble(D, {K14: _blocks(_pair("BTC", 1, "2026-09-24T14:05:01", 100), _pair("BTC", 2, "2026-09-24T14:07:01", 100))})
    assert bars.filter(pl.col("ticker") == "BTC")["ts"].to_list() == [utc("2026-09-24T14:05:00"), utc("2026-09-24T14:07:00")]
    assert gaps.is_empty()


def a_trade_stamped_after_its_files_hour_fails_the_hour():
    bars, gaps = _assemble(D, {K14: _blocks(_pair("BTC", 1, "2026-09-24T14:05:01", 100), _pair("BTC", 2, "2026-09-24T15:00:01", 1))})
    assert gaps.filter(pl.col("from_ts") == utc("2026-09-24T14:00:00"))["cause"].to_list() == ["unparsed"]
    assert bars.filter(pl.col("ts") < utc("2026-09-24T15:00:00")).is_empty()


# ---- gaps ----------------------------------------------------------------------


def an_absent_hour_is_a_gap_for_every_ticker():
    day = date(2025, 11, 3)
    k6, k7 = _key(BLOCK, "2025-11-03T06:00:00"), _key(BLOCK, "2025-11-03T07:00:00")
    bars, gaps = _assemble(day, {k6: _blocks(_pair("BTC", 1, "2025-11-03T06:30:00", 1), _pair("BTC", 2, "2025-11-03T06:59:59.5", 2))}, missing=(k7,))
    hour = gaps.filter(pl.col("from_ts") == utc("2025-11-03T07:00:00")).to_dicts()
    assert hour == [{"venue": VENUE, "ticker": None, "series": "candles", "cause": "archive-missing",
                     "from_ts": utc("2025-11-03T07:00:00"), "to_ts": utc("2025-11-03T08:00:00")}]  # fmt: skip
    # Hour 6's last second could be in the missing file too: it is a gap, and its minute has no bar.
    assert gaps.filter(pl.col("to_ts") == utc("2025-11-03T07:00:00"))["from_ts"].to_list() == [utc("2025-11-03T06:59:59")]
    assert bars["ts"].to_list() == [utc("2025-11-03T06:30:00")]


def an_unreadable_or_unparsed_hour_is_a_gap_with_its_cause():
    k15 = _key(BLOCK, "2026-09-24T15:00:00")
    bars, gaps = _assemble(D, {K14: _blocks(_pair("BTC", 1, "2026-09-24T14:05:00", 1))[:-6], k15: _lz4(["not json"])})
    causes = dict(gaps.filter(pl.col("to_ts") - pl.col("from_ts") == pl.duration(hours=1)).select("from_ts", "cause").iter_rows())
    assert causes == {utc("2026-09-24T14:00:00"): "unreadable", utc("2026-09-24T15:00:00"): "unparsed"}
    assert bars.is_empty()


def the_archives_first_hours_are_gaps():
    _, gaps = _assemble(date(2025, 3, 22), {})
    early = gaps.filter(pl.col("from_ts") < utc("2025-03-22T10:00:00"))
    assert early.height == 10 and set(early["cause"]) == {"archive-missing"}


# ---- the handovers -------------------------------------------------------------


def the_handover_hour_comes_from_the_later_format_only():
    day = date(2025, 5, 25)
    bodies = {
        _key("node_trades", "2025-05-25T13:00:00"): _trades(_node_trade("BTC", "2025-05-25T13:10:00.000000001", 3)),
        _key("node_trades", "2025-05-25T14:00:00"): _trades(_node_trade("BTC", "2025-05-25T14:10:00", 1)),
        _key("node_fills", "2025-05-25T14:00:00"): _fills(_pair("BTC", 9, "2025-05-25T14:10:00", 2)),
    }
    bars, _ = _assemble(day, bodies)
    assert _bar(bars, "BTC", "2025-05-25T13:10:00")["close"] == 3
    hand = _bar(bars, "BTC", "2025-05-25T14:10:00")
    assert (hand["close"], hand["trade_count"]) == (2, 1)


def the_second_handover_reads_by_block_from_its_cut():
    day = date(2025, 7, 27)
    bodies = {
        _key("node_fills", "2025-07-27T08:00:00"): _fills(_pair("BTC", 1, "2025-07-27T08:10:00", 1)),
        _key(BLOCK, "2025-07-27T08:00:00"): _blocks(_pair("BTC", 1, "2025-07-27T08:10:00", 2)),
    }
    bars, _ = _assemble(day, bodies)
    hand = _bar(bars, "BTC", "2025-07-27T08:10:00")
    assert (hand["close"], hand["trade_count"]) == (2, 1)


# ---- fetching ------------------------------------------------------------------


def _trading_day(bucket, day=D):
    return bucket.day(day, {K14: _blocks(
        _pair("BTC", 1, "2026-09-24T14:05:01", 100), _pair("xyz:CL", 2, "2026-09-24T14:05:02", 95),
        _pair("BTC", 3, "2026-09-24T14:06:01", 101),
    )})  # fmt: skip


def a_single_pass_writes_every_ticker_and_asks_each_file_once(store, bucket):
    _trading_day(bucket)
    code, said = _fetch("BTC", "CL")
    assert code == 0, said
    assert sorted(bucket.gets()) == sorted(_hyperliquid.keys(D))
    m = _manifest.read(store).filter(pl.col("venue") == VENUE)
    assert sorted(m.select("ticker", "status", "rows").rows()) == [("BTC", "ok", 2), ("CL", "ok", 1)]
    got = gr.reference.candles(["BTC", "CL"], "2026-09-24T00:00Z", "2026-09-25T00:00Z", venues=[VENUE]).collect()
    assert got.select("ticker", "close").rows() == [("BTC", 100.0), ("BTC", 101.0), ("CL", 95.0)]
    ledger = _manifest.read_objects(store)
    assert ledger.height == 25 and set(ledger["status"]) == {"ok"}
    assert ledger.filter(pl.col("key") == K14)["etag"][0] == bucket.etag(K14)


def a_second_fetch_downloads_nothing(store, bucket):
    _trading_day(bucket)
    _fetch("BTC")
    before, manifest = len(bucket.asked), _manifest.read(store)
    code, said = _fetch("BTC")
    assert code == 0 and len(bucket.asked) == before
    assert _manifest.read(store).equals(manifest)
    assert "1 ticker-day(s) were already in the manifest" in said[-1]


def a_dry_run_states_files_bytes_and_cost_and_downloads_nothing(store, bucket):
    _trading_day(bucket)
    code, said = _run("candles", VENUE, "BTC", "--from", "2026-09-24", "--to", "2026-09-24", "--days", "2026-09-24", "--dry-run")
    assert code == 0
    assert bucket.gets() == [] and bucket.lists() == 2  # the day's directory and the next day's
    assert "25 files" in said[0] and "about $" in said[-1] and "an estimate" in said[-1]
    assert not store.exists() or not list(store.rglob("*.parquet"))


def the_documented_dry_run_runs(store, bucket):
    for d in range(21, 28):
        bucket.day(date(2026, 9, d))
    code, said = _run("candles", VENUE, "BTC", "CL", "--from", "2026-09-21", "--to", "2026-09-27", "--sample", "every:1", "--dry-run")
    assert code == 0 and said[-1].startswith("7 day(s) to fetch: 175 files")
    assert bucket.gets() == []


def a_range_without_declared_days_is_refused_with_its_cost(store, bucket):
    _trading_day(bucket)
    with pytest.raises(Refused, match=r"declared days only: 1 days need 25 hour files, 0\.0 GB, about \$0\.00.*--days or --sample"):
        _run("candles", VENUE, "BTC", "--from", "2026-09-24", "--to", "2026-09-24")
    assert bucket.gets() == []


def a_changed_hour_on_refetch_keeps_the_held_day(store, bucket):
    _trading_day(bucket)
    _fetch("BTC")
    bucket.objects[K14] = _blocks(_pair("BTC", 1, "2026-09-24T14:05:01", 999))
    code, said = _fetch("BTC", extra=("--refetch",))
    assert code == 1 and "MISMATCH" in said[0] and K14 in said[0]
    assert _manifest.read(store)["status"].to_list() == ["mismatch"]
    assert _manifest.read_objects(store).filter(pl.col("key") == K14)["etag"][0] != bucket.etag(K14)
    path = _manifest.day_path(store, "candles", VENUE, "BTC", D)
    assert pl.read_parquet(path)["close"].to_list() == [100.0, 101.0]


def a_day_with_a_gap_is_partial_and_asked_again_only_when_told(store, bucket):
    k7 = _key(BLOCK, "2026-09-24T07:00:00")
    _trading_day(bucket)
    del bucket.objects[k7]
    code, said = _fetch("BTC")
    assert code == 0 and " partial " in said[0] and "archive-missing" in said[0]
    assert gr.reference.coverage().select("days_ok", "days_partial").row(0) == (0, 1)
    asked = len(bucket.asked)
    _fetch("BTC")
    assert len(bucket.asked) == asked
    bucket.objects[k7] = EMPTY
    _fetch("BTC", extra=("--refetch-absent",))
    assert _manifest.read(store)["status"].to_list() == ["ok"]
    assert gr.reference.gaps(["BTC"], "2026-09-24T00:00Z", "2026-09-25T00:00Z", venues=[VENUE]).is_empty()


def a_day_the_archive_lacks_is_absent(store, bucket):
    code, said = _fetch("BTC", day="2026-09-30")
    assert code == 0 and "absent" in said[0]
    assert _manifest.read(store)["status"].to_list() == ["absent"]


def no_day_before_a_tickers_listing_is_asked(store, bucket):
    code, said = _run("candles", VENUE, "SP500", "--from", "2026-03-16", "--to", "2026-03-18", "--days", "2026-03-16,2026-03-17,2026-03-18")
    assert "before 2026-03-18; 2 earlier day(s) not requested" in said[0]
    assert {r["date"] for r in _manifest.read(store).iter_rows(named=True)} == {date(2026, 3, 18)}


def the_record_is_untouched_by_a_fetch(store, bucket, tmp_path, monkeypatch):
    record = tmp_path / "record"
    Tape(record).bar("BTC", "1m", "2026-09-24T14:05:00", "2026-09-24T14:06:01").write()
    before = sorted((p, p.stat().st_mtime_ns) for p in record.rglob("*"))
    _trading_day(bucket)
    assert _fetch("BTC")[0] == 0
    assert sorted((p, p.stat().st_mtime_ns) for p in record.rglob("*")) == before


def a_kind_the_venue_lacks_is_refused(store, bucket):
    with pytest.raises(Refused, match="built as candles, its gaps beside them; it publishes no trades"):
        _run("trades", VENUE, "BTC", "--from", "2026-09-24", "--to", "2026-09-24", "--days", "2026-09-24")


# ---- reading from research -----------------------------------------------------


def a_bar_is_not_known_before_its_close(store, bucket):
    _trading_day(bucket)
    _fetch("BTC")
    got = gr.reference.candles("BTC", "2026-09-24T00:00Z", "2026-09-25T00:00Z", venues=[VENUE], as_of="2026-09-24T14:06:30Z").collect()
    assert got["ts"].to_list() == [utc("2026-09-24T14:05:00")]


def the_one_minute_bars_resample_to_five(store, bucket):
    trades = [_pair("BTC", n, f"2026-09-24T14:0{n}:01", 100 + n, liquidation=(n == 2)) for n in range(5)]
    bucket.day(D, {K14: _blocks(*trades)})
    _fetch("BTC")
    fine = gr.reference.candles("BTC", "2026-09-24T00:00Z", "2026-09-25T00:00Z", venues=[VENUE]).collect()
    five = gr.timeseries.resample(fine, "5m")
    assert five.select("open", "close", "high_all", "low_all", "volume", "trade_count", "n").rows() == [(100.0, 104.0, 104.0, 100.0, 5.0, 5, 5)]


def a_read_without_the_venue_keeps_binances_columns(store, bucket):
    _trading_day(bucket)
    _fetch("BTC")
    got = gr.reference.candles("BTC", "2026-09-24T00:00Z", "2026-09-25T00:00Z", venues=[VENUE]).collect()
    assert {"high_all", "low_all", "notional", "taker_buy_volume", "liquidations"} <= set(got.columns)
    assert "high_all" not in gr.reference.candles("BTC", "2026-09-24T00:00Z", "2026-09-25T00:00Z").collect().columns


def the_gaps_overlapping_a_window_are_returned(store, bucket):
    day = date(2025, 11, 3)
    bucket.day(day, missing=(_key(BLOCK, "2025-11-03T07:00:00"),))
    _fetch("BTC", day="2025-11-03")
    got = gr.reference.gaps(["BTC"], "2025-11-03T00:00Z", "2025-11-04T00:00Z", venues=[VENUE])
    hour = got.filter(pl.col("from_ts") == utc("2025-11-03T07:00:00"))
    assert hour.select("ticker", "cause", "to_ts").rows() == [(None, "archive-missing", utc("2025-11-03T08:00:00"))]
    assert gr.reference.gaps(None, "2025-11-03T12:00Z", "2025-11-04T00:00Z", venues=[VENUE]).is_empty()


# ---- against the record --------------------------------------------------------


@pytest.fixture(name="record")
def _record(tmp_path, monkeypatch, bucket):
    monkeypatch.setenv("GALATA_VAR", str(tmp_path / "record"))
    monkeypatch.setenv("GALATA_REFERENCE", str(tmp_path / "reference"))
    return Tape(tmp_path / "record")


def _both(record, bucket, ours, theirs):
    """`ours`: trades in 14:05; `theirs`: the record's bar for 14:05 as (close, trade_count)."""
    bucket.day(D, {K14: _blocks(*ours)})
    assert _fetch("BTC")[0] == 0
    close, count = theirs
    record.bar("BTC", "1m", "2026-09-24T14:05:00", "2026-09-24T14:06:01", close=close, trade_count=count).write()
    return gr.reference.compare_record("BTC", "2026-09-24T00:00Z", "2026-09-25T00:00Z")


def the_same_bars_match(record, bucket):
    got = _both(record, bucket, [_pair("BTC", 1, "2026-09-24T14:05:10", 100, sz="1.5")], ("100", 1))
    assert (got["both"], got["reference_only"], got["record_only"], got["differences"].height) == (1, 0, 0, 0)
    assert got["statement"] is None


def a_shifted_price_is_a_difference(record, bucket):
    got = _both(record, bucket, [_pair("BTC", 1, "2026-09-24T14:05:10", 100, sz="1.5")], ("101", 1))
    diff = got["differences"].row(0, named=True)
    assert (got["both"], diff["close"], diff["close_record"]) == (1, 100.0, 101.0)


def a_liquidation_wick_is_not_a_difference(record, bucket):
    ours = [_pair("BTC", 1, "2026-09-24T14:05:10", 100, sz="0.5"), _pair("BTC", 2, "2026-09-24T14:05:11", 99, sz="0.5", liquidation=True),
            _pair("BTC", 3, "2026-09-24T14:05:12", 100, sz="0.5")]  # fmt: skip
    got = _both(record, bucket, ours, ("100", 3))
    assert (got["both"], got["differences"].height) == (1, 0)


def a_ticker_the_record_lacks_is_stated_not_matched(record, bucket):
    bucket.day(D, {K14: _blocks(_pair("xyz:SILVER", 1, "2026-09-24T14:05:10", 60))})
    assert _fetch("SILVER")[0] == 0
    record.bar("BTC", "1m", "2026-09-24T14:05:00", "2026-09-24T14:06:01").write()
    got = gr.reference.compare_record("SILVER", "2026-09-24T00:00Z", "2026-09-25T00:00Z")
    assert got["statement"] == "the record holds no 1m bars of SILVER to compare"
    assert got["both"] == 0
