"""The reference store: fetched from fixture archives written in the test, never the network.

Each archive is served by replacing `fetch._transport`, the one place a
socket opens, with a dict of URL → bytes. A URL the dict lacks answers 404,
as the archives do for a day they do not hold.
"""

import gzip
import hashlib
import io
import json
import zipfile
from datetime import date

import polars as pl
import pytest
from conftest import utc
from polars.testing import assert_frame_equal

import galata_research as gr
from galata_research import Refused
from galata_research.reference import _manifest, _sources, fetch

BINANCE = "https://data.binance.vision/data/futures/um/daily"


def _zip(name: str, text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, text)
    return buf.getvalue()


def _depth_file(day: str, rows: list[str] | None = None) -> bytes:
    rows = rows or [
        f"{day} 00:00:04,-0.20,1.5,120000.0",
        f"{day} 00:00:04,0.20,1.2,96000.0",
    ]
    return _zip(
        f"BTCUSDT-bookDepth-{day}.csv",
        "timestamp,percentage,depth,notional\n" + "\n".join(rows) + "\n",
    )


def _agg_file(day: str, rows: list[str]) -> bytes:
    header = "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
    return _zip(f"BTCUSDT-aggTrades-{day}.csv", header + "\n".join(rows) + "\n")


def _kline_file(day: str, rows: list[str]) -> bytes:
    return _zip(f"BTCUSDT-1m-{day}.csv", "\n".join(rows) + "\n")  # headerless, as 2020's files are


def _bybit_trades_file(rows: list[str], rpi: bool = True) -> bytes:
    header = "timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional"
    return gzip.compress(((header + (",RPI" if rpi else "")) + "\n" + "\n".join(rows) + "\n").encode())


def _book_file(messages: list[dict]) -> bytes:
    return _zip("book.data", "\n".join(json.dumps(m) for m in messages) + "\n")


class Archive:
    """A fixture archive: `files[url] = bytes`, with every request recorded."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.asked: list[tuple[str, str]] = []

    def __call__(self, url, method, agent="galata-fetch"):
        self.asked.append((method, url))
        if url not in self.files:
            return fetch.Response(404)
        body = self.files[url]
        return fetch.Response(200, body if method == "GET" else b"", len(body))

    def publish(self, url: str, body: bytes, checksum: bool = True):
        self.files[url] = body
        if checksum and "binance" in url:
            self.files[url + ".CHECKSUM"] = f"{hashlib.sha256(body).hexdigest()}  x.zip\n".encode()


@pytest.fixture(name="archive")
def _archive(monkeypatch):
    a = Archive()
    monkeypatch.setattr(fetch, "_transport", a)
    return a


@pytest.fixture(name="store")
def _store(tmp_path, monkeypatch):
    path = tmp_path / "reference"
    monkeypatch.setenv("GALATA_REFERENCE", str(path))
    monkeypatch.setenv("GALATA_VAR", str(tmp_path / "record"))  # no record: nothing to be inside
    return path


def _run(*argv) -> tuple[int, list[str]]:
    said: list[str] = []
    return fetch.run(list(argv), say=said.append), said


def _depth_url(day: str, symbol: str = "BTCUSDT") -> str:
    return f"{BINANCE}/bookDepth/{symbol}/{symbol}-bookDepth-{day}.zip"


# ---- the store ----------------------------------------------------------------


def the_variable_names_the_store(tmp_path, monkeypatch):
    env, declared = tmp_path / "env", tmp_path / "file"
    env.mkdir(), declared.mkdir()
    (tmp_path / "galata-research.toml").write_text(f'reference_root = "{declared}"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GALATA_REFERENCE", str(env))
    assert gr.reference.root() == env


def a_missing_store_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("GALATA_REFERENCE", str(tmp_path / "nowhere"))
    with pytest.raises(Refused, match=r"nowhere.*does not exist.*GALATA_REFERENCE"):
        gr.reference.trades(["BTC"], "2026-09-20T00:00Z", "2026-09-21T00:00Z")


def no_reference_write_lands_under_the_record(tmp_path, monkeypatch, archive):
    record = tmp_path / "record"
    (record / "tape").mkdir(parents=True)
    monkeypatch.setenv("GALATA_VAR", str(record))
    monkeypatch.setenv("GALATA_REFERENCE", str(record / "reference"))
    with pytest.raises(Refused, match="inside the record"):
        _run("depth", "binance-um", "BTC", "--from", "2026-09-20", "--to", "2026-09-20")
    assert not (record / "reference").exists()


# ---- the manifest and the fetch -------------------------------------------------


def a_second_fetch_downloads_nothing(store, archive):
    for d in ("2026-09-20", "2026-09-21"):
        archive.publish(_depth_url(d), _depth_file(d))
    code, _ = _run("depth", "binance-um", "BTC", "--from", "2026-09-20", "--to", "2026-09-21")
    assert code == 0
    before = _manifest.read(store)
    archive.asked.clear()
    code, said = _run("depth", "binance-um", "BTC", "--from", "2026-09-20", "--to", "2026-09-21")
    assert code == 0 and archive.asked == []
    assert _manifest.read(store).drop("fetched_at_recv").equals(before.drop("fetched_at_recv"))
    assert "2 were already in the manifest" in said[-1]


def a_dry_run_writes_nothing(store, archive):
    for d in range(20, 27):
        archive.publish(
            f"https://public.bybit.com/trading/BTCUSDT/BTCUSDT2026-09-{d}.csv.gz",
            b"x" * 1000,
        )
    code, said = _run(
        "trades",
        "bybit-linear",
        "BTC",
        "--from",
        "2026-09-20",
        "--to",
        "2026-09-26",
        "--dry-run",
    )
    assert code == 0
    assert said[-1].startswith("7 day(s) to fetch, 0.0 MB")
    assert all(method == "HEAD" for method, _ in archive.asked)
    assert not store.exists()


def an_absent_day_is_not_asked_again(store, archive):
    code, said = _run("depth", "binance-um", "BTC", "--from", "2026-09-20", "--to", "2026-09-20")
    assert code == 0 and "absent" in said[0]
    row = _manifest.read(store).row(0, named=True)
    assert (
        row["status"] == "absent"
        and not _manifest.day_path(store, "depth", "binance-um", "BTC", date(2026, 9, 20)).exists()
    )
    archive.asked.clear()
    _run("depth", "binance-um", "BTC", "--from", "2026-09-20", "--to", "2026-09-20")
    assert archive.asked == []
    _run(
        "depth",
        "binance-um",
        "BTC",
        "--from",
        "2026-09-20",
        "--to",
        "2026-09-20",
        "--refetch-absent",
    )
    assert archive.asked


def a_published_checksum_that_disagrees_is_refused(store, archive):
    url = _depth_url("2026-09-20")
    archive.publish(url, _depth_file("2026-09-20"), checksum=False)
    archive.files[url + ".CHECKSUM"] = b"0" * 64 + b"  x.zip\n"
    code, said = _run("depth", "binance-um", "BTC", "--from", "2026-09-20", "--to", "2026-09-20")
    assert code != 0 and "2026-09-20" in said[0] and "MISMATCH" in said[0]
    assert _manifest.read(store)["status"].to_list() == ["mismatch"]
    assert not _manifest.day_path(store, "depth", "binance-um", "BTC", date(2026, 9, 20)).exists()


def a_changed_refetch_keeps_the_old_day(store, archive):
    url = _depth_url("2026-09-20")
    archive.publish(url, _depth_file("2026-09-20"))
    _run("depth", "binance-um", "BTC", "--from", "2026-09-20", "--to", "2026-09-20")
    path = _manifest.day_path(store, "depth", "binance-um", "BTC", date(2026, 9, 20))
    kept = path.read_bytes()
    archive.publish(url, _depth_file("2026-09-20", ["2026-09-20 00:00:34,-0.20,9.0,1.0"]))
    code, said = _run(
        "depth",
        "binance-um",
        "BTC",
        "--from",
        "2026-09-20",
        "--to",
        "2026-09-20",
        "--refetch",
    )
    assert code != 0 and "MISMATCH" in said[0] and "2026-09-20" in said[0]
    assert path.read_bytes() == kept
    assert _manifest.read(store)["status"].to_list() == ["mismatch"]


def a_host_outside_the_archives_is_refused(archive):
    with pytest.raises(Refused, match="example.com is not an archive"):
        fetch.get("https://example.com/BTCUSDT.zip")
    assert archive.asked == []


# ---- the instrument map ---------------------------------------------------------


def a_hip3_ticker_on_binance_is_refused(store, archive):
    with pytest.raises(Refused, match=r"binance-um lists no CL; it lists BTC, ETH, GOLD, HYPE"):
        _run("depth", "binance-um", "CL", "--from", "2026-01-01", "--to", "2026-01-02")


def no_day_before_the_listing_is_requested(store, archive):
    _, said = _run(
        "depth",
        "binance-um",
        "HYPE",
        "--from",
        "2025-05-28",
        "--to",
        "2025-05-31",
        "--dry-run",
    )
    assert "archived from 2025-05-30" in said[0] and "2 earlier day(s)" in said[0]
    days = sorted(url.rsplit("-bookDepth-", 1)[1][:10] for _, url in archive.asked)
    assert days == ["2025-05-30", "2025-05-31"]


def a_full_range_book_without_a_sample_is_refused(store, archive):
    archive.publish(
        "https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT/2026-09-26_BTCUSDT_ob200.data.zip",
        b"x" * 93_000_000,
    )
    with pytest.raises(
        Refused,
        match=r"declared days only: 1348 days of BTC would be about 125\.4 GB.*--sample",
    ):
        _run("book", "bybit-linear", "BTC", "--from", "2023-01-18", "--to", "2026-09-26")


def every_third_day_turns_through_the_week():
    days = fetch.days_between(date(2026, 1, 1), date(2026, 3, 31))
    chosen = fetch.declared(days, None, "every:3")
    weekdays = [d.weekday() for d in chosen]
    assert len(chosen) == 30 and {weekdays.count(w) for w in range(7)} <= {4, 5}
    # A later --from picks the same days, not a shifted set.
    assert fetch.declared(days[5:], None, "every:3") == [d for d in chosen if d >= days[5]]


def the_monthly_sample_is_the_first_weekday_of_the_month():
    days = fetch.days_between(date(2026, 1, 1), date(2026, 3, 31))
    assert fetch.declared(days, None, "monthly:wed") == [date(2026, 1, 7), date(2026, 2, 4), date(2026, 3, 4)]


def a_sample_of_another_shape_is_refused():
    with pytest.raises(Refused, match="every:<n>"):
        fetch.declared([date(2026, 1, 1)], None, "monthly:1")


def the_book_archive_is_ob500_before_the_switch():
    assert _sources.url("book", "bybit-linear", "BTC", date(2025, 8, 20)).endswith("2025-08-20_BTCUSDT_ob500.data.zip")
    assert _sources.url("book", "bybit-linear", "BTC", date(2025, 8, 21)).endswith("2025-08-21_BTCUSDT_ob200.data.zip")


# ---- normalization ----------------------------------------------------------------


def _parse(kind, venue, blob, day="2026-09-20", ticker="BTC"):
    return _sources.parse(kind, venue, ticker, date.fromisoformat(day), blob, "fixture")


def the_maker_flag_becomes_the_aggressor():
    got = _parse("trades", "binance-um", _agg_file("2026-09-20", [
        "1,81225.7,0.007,8,8,1789862400002,true",
        "2,81225.8,0.401,9,12,1789862400140,false",
    ]))  # fmt: skip
    assert got["aggressor"].to_list() == ["ask", "bid"]
    assert got["trade_id"].to_list() == ["1", "2"]
    assert got["ts"][0] == utc("2026-09-20T00:00:00.002")


def the_bybit_side_is_the_takers():
    got = _parse("trades", "bybit-linear", _bybit_trades_file([
        "1790380800.0232,BTCUSDT,Buy,0.035,84059.90,ZeroMinusTick,a,1,0.035,2942.1,0",
        "1790380800.1176,BTCUSDT,Sell,0.002,84059.90,ZeroMinusTick,b,1,0.002,168.1,1",
    ]), day="2026-09-26")  # fmt: skip
    assert got["aggressor"].to_list() == ["bid", "ask"]
    assert got["rpi"].to_list() == [False, True]


def a_sub_millisecond_bybit_time_is_kept():
    got = _parse(
        "trades",
        "bybit-linear",
        _bybit_trades_file(["1790380800.0232,BTCUSDT,Buy,1,1,T,a,1,1,1,0"]),
        day="2026-09-26",
    )
    assert got["ts"][0] == utc("2026-09-26T00:00:00.023200")


def an_old_bybit_file_is_read_forwards_in_time():
    got = _parse("trades", "bybit-linear", _bybit_trades_file([
        "1585180700.0647,BTCUSDT,Buy,1,6698.5,T,late,1,1,1",
        "1585132572.9822,BTCUSDT,Sell,1,6500.0,T,early,1,1,1",
    ], rpi=False), day="2020-03-25")  # fmt: skip
    assert got["trade_id"].to_list() == ["early", "late"]
    assert got["rpi"].to_list() == [
        None,
        None,
    ]  # the file predates RPI and states nothing


def the_reference_aggressor_is_the_tapes_word():
    binance = _parse("trades", "binance-um", _agg_file("2026-09-20", ["1,1,1,1,1,1789862400002,true", "2,1,1,2,2,1789862400003,False"]))
    bybit = _parse("trades", "bybit-linear", _bybit_trades_file(["1790380800.0232,BTCUSDT,Sell,1,1,T,a,1,1,1,0"]), day="2026-09-26")
    assert set(binance["aggressor"]) | set(bybit["aggressor"]) == {"bid", "ask"}


def a_side_of_another_word_is_refused():
    blob = _bybit_trades_file(["1790380800.0232,BTCUSDT,Hold,1,1,T,a,1,1,1,0"])
    with pytest.raises(Refused, match="'Hold' is neither Buy nor Sell"):
        _parse("trades", "bybit-linear", blob, day="2026-09-26")


def the_bid_side_of_depth_is_negative():
    got = _parse(
        "depth",
        "binance-um",
        _depth_file(
            "2026-09-20",
            [
                "2026-09-20 00:00:04,-1.00,2083.3,168315514.0",
                "2026-09-20 00:00:04,-5,1,1",
            ],
        ),
    )
    assert got["band_pct"].to_list() == [-1.0, -5.0]


def a_file_in_microseconds_is_refused():
    blob = _agg_file("2026-09-20", ["1,81225.7,0.007,8,8,1789862400002000,true"])
    with pytest.raises(Refused, match=r"fixture: transact_time implies the year \d{5}"):
        _parse("trades", "binance-um", blob)


def a_depth_file_outside_its_day_is_refused():
    blob = _depth_file("2026-09-20", ["2026-09-21 00:00:04,-0.20,1.5,1.0"])
    with pytest.raises(Refused, match="outside 2026-09-20 UTC"):
        _parse("depth", "binance-um", blob)


def a_headerless_kline_is_read():
    got = _parse("candles", "binance-um", _kline_file("2020-01-01", [
        "1577836800000,7189.43,7190.52,7177,7182.44,246.092,1577836859999,1767430.16,336,46.63,334813.19,0",
    ]), day="2020-01-01")  # fmt: skip
    assert got.row(0, named=True) | {} == got.row(0, named=True)
    assert got["close_ts"][0] == utc("2020-01-01T00:01") and got["trade_count"][0] == 336 and got["open"][0] == 7189.43


# ---- the book ---------------------------------------------------------------------

T0 = 1790380800000  # 2026-09-26 00:00:00 UTC, in ms


def _snapshot(ms, b, a):
    return {"type": "snapshot", "ts": ms, "data": {"b": b, "a": a}}


def _delta(ms, b=(), a=()):
    return {"type": "delta", "ts": ms, "data": {"b": list(b), "a": list(a)}}


def _book(messages):
    return _sources.book([json.dumps(m) for m in messages], date(2026, 9, 26))


def a_delta_of_zero_removes_the_level():
    got = _book([
        _snapshot(T0 + 100, [["100.0", "2"], ["99.99", "1"], ["99.0", "5"]], [["100.01", "3"], ["101.0", "4"]]),
        _delta(T0 + 1500, b=[["100.0", "0"]]),
    ])  # fmt: skip
    assert got["ts"][0] == utc("2026-09-26T00:00:01")
    assert got["bid_px"][:2].to_list() == [100.0, 99.99]
    assert got["bid_sz"][1] == 1.0
    assert got.height == 86_399


def a_band_past_the_reach_is_null():
    # Mid 100.005; the deepest bid, 99.965, is 4.0 bps below it.
    got = _book(
        [
            _snapshot(
                T0,
                [["100.0", "2"], ["99.99", "1"], ["99.965", "5"]],
                [["100.01", "3"], ["100.2", "4"]],
            )
        ]
    )
    row = got.row(0, named=True)
    assert row["bid_reach_bps"] == pytest.approx(0.04 / 100.005 * 1e4)
    assert row["bid_depth_10bps"] is None
    assert row["bid_depth_2bps"] == 3.0  # 100.0 and 99.99; 99.965 is past 2 bps
    assert row["ask_depth_10bps"] == 3.0  # 100.2 is 19.5 bps above the mid


def no_book_row_before_the_first_snapshot():
    got = _book(
        [
            _delta(T0 + 10, b=[["100.0", "1"]]),
            _snapshot(T0 + 5_000, [["100.0", "2"]], [["100.1", "1"]]),
        ]
    )
    assert got["ts"][0] == utc("2026-09-26T00:00:05")


def no_book_row_while_a_side_is_empty():
    # HYPE's first book day (2024-12-04) holds only empty snapshots.
    assert _book([_snapshot(T0, [], []), _snapshot(T0 + 3_000, [], [])]).is_empty()


# ---- OKX --------------------------------------------------------------------------

_OKX_HEADER = "instrument_name,trade_id,side,price,size,created_time"
_OKX_0926 = 1790380800000  # 2026-09-26 00:00 UTC, in the file dated 2026-09-26 (16:00 to 16:00 UTC)


def _okx_file(day: str, rows: list[str], source: bool = True, symbol: str = "BTC-USDT-SWAP") -> bytes:
    header = _OKX_HEADER + (",source" if source else "")
    return _zip(f"{symbol}-trades-{day}.csv", header + "\n" + "\n".join(rows) + "\n")


def an_okx_contract_becomes_btc():
    got = _parse("trades", "okx-swap", _okx_file("2026-09-26", [
        f"BTC-USDT-SWAP,11,buy,83760.0,3.58,{_OKX_0926},0",
        f"BTC-USDT-SWAP,12,sell,83759.9,1,{_OKX_0926 + 5},1",
    ]), day="2026-09-26")  # fmt: skip
    assert got["size"].to_list() == pytest.approx([0.0358, 0.01])
    assert got["aggressor"].to_list() == ["bid", "ask"]
    assert got["rpi"].to_list() == [False, True]
    assert got["ts"][0] == utc("2026-09-26T00:00")


def an_eth_contract_is_a_tenth():
    got = _parse("trades", "okx-swap", _okx_file("2026-09-26", [f"ETH-USDT-SWAP,1,BUY,3000,169,{_OKX_0926}"], source=False,
                 symbol="ETH-USDT-SWAP"), day="2026-09-26", ticker="ETH")  # fmt: skip
    assert got["size"].to_list() == pytest.approx([16.9])
    assert got["aggressor"].to_list() == ["bid"]
    assert got["rpi"].to_list() == [None]  # the file predates ELP and states nothing


def the_doubled_okx_format_is_refused():
    blob = _okx_file("2021-10-01", ["BTC-USDT-SWAP,118368467,SELL,43135.1,4.0,1633017600564",
                                    "BTC-USDT-SWAP,118368467,BUY,43135.1,4.0,1633017600564"], source=False)  # fmt: skip
    with pytest.raises(Refused, match=r"1 trade_id\(s\) repeat, the first 118368467"):
        _parse("trades", "okx-swap", blob, day="2021-10-01")


def an_okx_row_outside_its_beijing_day_is_refused():
    blob = _okx_file("2026-09-26", [f"BTC-USDT-SWAP,1,buy,1,1,{_OKX_0926 + 16 * 3_600_000 + 60_000}"])
    with pytest.raises(Refused, match="more than a minute outside the archive's day, 16:00 to 16:00 UTC"):
        _parse("trades", "okx-swap", blob, day="2026-09-26")


def an_okx_file_cut_seconds_early_is_read():
    # BTC 2023-01-20 begins at 2023-01-19 15:59:45.526, 15 s before its day.
    got = _parse("trades", "okx-swap", _okx_file("2023-01-20", ["BTC-USDT-SWAP,468659344,buy,20875.8,3.0,1674143985526"], source=False),
                 day="2023-01-20")  # fmt: skip
    assert got["ts"].to_list() == [utc("2023-01-19T15:59:45.526")]


def a_ticker_okx_is_not_mapped_for_is_refused(store, archive):
    with pytest.raises(Refused, match=r"okx-swap lists no HYPE; it lists BTC, ETH"):
        _run("trades", "okx-swap", "HYPE", "--from", "2026-09-26", "--to", "2026-09-26", "--days", "2026-09-26")


def a_utc_day_comes_from_two_okx_files(store, archive):
    hour = 3_600_000
    files = {
        "2026-09-26": [f"BTC-USDT-SWAP,1,buy,1,1,{_OKX_0926 - hour}", f"BTC-USDT-SWAP,2,buy,1,1,{_OKX_0926 + hour}"],
        "2026-09-27": [f"BTC-USDT-SWAP,3,sell,1,1,{_OKX_0926 + 23 * hour}", f"BTC-USDT-SWAP,4,sell,1,1,{_OKX_0926 + 25 * hour}"],
    }
    for d, rows in files.items():
        archive.publish(_sources.url("trades", "okx-swap", "BTC", date.fromisoformat(d)), _okx_file(d, rows))
    code, said = _run("trades", "okx-swap", "BTC", "--from", "2026-09-26", "--to", "2026-09-26", "--days", "2026-09-26")
    assert code == 0, said
    assert [u.rsplit("-trades-", 1)[1] for m, u in archive.asked if m == "GET"] == ["2026-09-26.zip", "2026-09-27.zip"]
    got = gr.reference.trades("BTC", "2026-09-26T00:00Z", "2026-09-27T00:00Z").collect()
    assert got["trade_id"].to_list() == ["2", "3"]


def an_elp_fill_is_dropped_with_rpi(store, archive):
    rows = [f"BTC-USDT-SWAP,1,buy,1,1,{_OKX_0926},0", f"BTC-USDT-SWAP,2,buy,1,1,{_OKX_0926 + 1},1"]
    archive.publish(_sources.url("trades", "okx-swap", "BTC", date(2026, 9, 26)), _okx_file("2026-09-26", rows))
    assert _run("trades", "okx-swap", "BTC", "--from", "2026-09-25", "--to", "2026-09-25", "--days", "2026-09-25")[0] == 0
    got = gr.reference.trades("BTC", "2026-09-26T00:00Z", "2026-09-26T01:00Z", rpi=False).collect()
    assert got["trade_id"].to_list() == ["1"]


def the_okx_archive_is_the_only_okx_host():
    assert "static.okx.com" in fetch.HOSTS
    assert ("okx-swap", "trades") in fetch.HEAVY
    with pytest.raises(Refused, match="www.okx.com is not an archive"):
        fetch.get("https://www.okx.com/api/v5/market/trades?instId=BTC-USDT-SWAP")


# ---- the loaders --------------------------------------------------------------------


def _fetched(store, archive, kind, venue, days, blob_for, ticker="BTC"):
    for d in days:
        archive.publish(_sources.url(kind, venue, ticker, date.fromisoformat(d)), blob_for(d))
    code, said = _run(
        kind,
        venue,
        ticker,
        "--from",
        days[0],
        "--to",
        days[-1],
        "--days",
        ",".join(days),
    )
    assert code == 0, said


def a_kline_is_not_known_at_its_open(store, archive):
    rows = [
        "1789862400000,1,2,0.5,1.5,10,1789862459999,15,3,5,7,0",
        "1789862460000,1.5,2,1,1.8,10,1789862519999,15,3,5,7,0",
    ]
    _fetched(
        store,
        archive,
        "candles",
        "binance-um",
        ["2026-09-20"],
        lambda d: _kline_file(d, rows),
    )
    got = gr.reference.candles("BTC", "2026-09-20T00:00Z", "2026-09-21T00:00Z", as_of="2026-09-20T00:01:30Z").collect()
    assert got["ts"].to_list() == [utc("2026-09-20T00:00")]


def an_unknown_ticker_in_the_store_is_refused(store, archive):
    _fetched(store, archive, "depth", "binance-um", ["2026-09-20"], _depth_file)
    _fetched(
        store,
        archive,
        "depth",
        "binance-um",
        ["2026-09-20"],
        lambda d: _depth_file(d).replace(b"BTC", b"HYP"),
        ticker="HYPE",
    )
    with pytest.raises(Refused, match=r"holds no depth for SOL; it holds BTC, HYPE"):
        gr.reference.depth(["SOL"], "2026-09-20T00:00Z", "2026-09-21T00:00Z")


def a_hole_in_the_range_is_counted_missing(store, archive):
    rows = lambda d: _bybit_trades_file([f"{int(utc(d + 'T01:00').timestamp())}.5,BTCUSDT,Buy,1,1,T,{d},1,1,1,0"])
    _fetched(store, archive, "trades", "bybit-linear", ["2026-09-20", "2026-09-22"], rows)
    cover = gr.reference.coverage().row(0, named=True)
    assert (cover["days_ok"], cover["days_absent"], cover["days_missing"]) == (2, 0, 1)
    assert (cover["first"], cover["last"]) == (date(2026, 9, 20), date(2026, 9, 22))
    got = gr.reference.trades("BTC", "2026-09-20T00:00Z", "2026-09-23T00:00Z").collect()
    assert got["trade_id"].to_list() == ["2026-09-20", "2026-09-22"]


def no_reference_frame_carries_recv_ts(store, archive):
    _fetched(store, archive, "depth", "binance-um", ["2026-09-20"], _depth_file)
    for loader in (
        gr.reference.trades,
        gr.reference.depth,
        gr.reference.book,
        gr.reference.candles,
    ):
        try:
            frame = loader(None, "2026-09-20T00:00Z", "2026-09-21T00:00Z")
        except Refused:
            continue  # a kind not fetched here
        assert "recv_ts" not in frame.collect_schema().names()
    assert all("recv_ts" not in s for s in _sources.SCHEMAS.values())


def an_rpi_filter_leaves_binance_alone(store, archive):
    _fetched(store, archive, "trades", "bybit-linear", ["2026-09-26"], lambda d: _bybit_trades_file([
        "1790380800.0232,BTCUSDT,Buy,1,1,T,public,1,1,1,0", "1790380800.1176,BTCUSDT,Sell,1,1,T,retail,1,1,1,1",
    ]))  # fmt: skip
    _fetched(
        store,
        archive,
        "trades",
        "binance-um",
        ["2026-09-26"],
        lambda d: _agg_file(d, ["7,1,1,1,1,1790380800500,true"]),
    )
    got = gr.reference.trades("BTC", "2026-09-26T00:00Z", "2026-09-27T00:00Z", rpi=False).collect()
    assert sorted(got["trade_id"].to_list()) == ["7", "public"]
    assert "rpi" not in got.columns


def a_duckdb_relation_has_the_same_rows(store, archive):
    _fetched(store, archive, "depth", "binance-um", ["2026-09-20"], _depth_file)
    duck = gr.reference.depth("BTC", "2026-09-20T00:00Z", "2026-09-21T00:00Z", engine="duckdb").pl()
    assert_frame_equal(
        duck,
        gr.reference.depth("BTC", "2026-09-20T00:00Z", "2026-09-21T00:00Z").collect(),
    )


# ---- update ------------------------------------------------------------------------


def _one_series(monkeypatch, series=(("depth", "binance-um", "BTC", None),)):
    monkeypatch.setattr(fetch, "UPDATES", series)
    monkeypatch.setattr(fetch, "run_events", lambda say=print: 0)


def an_update_runs_from_the_last_day_to_yesterday(store, archive, monkeypatch):
    _one_series(monkeypatch)
    _fetched(store, archive, "depth", "binance-um", ["2026-09-20"], _depth_file)
    archive.publish(_depth_url("2026-09-21"), _depth_file("2026-09-21"))  # 09-22 not yet published
    assert fetch.run_update(say=lambda _: None, today=date(2026, 9, 23)) == 0
    status = {r["date"].isoformat(): r["status"] for r in _manifest.read(store).iter_rows(named=True)}
    assert status == {"2026-09-20": "ok", "2026-09-21": "ok", "2026-09-22": "absent"}
    archive.publish(_depth_url("2026-09-22"), _depth_file("2026-09-22"))  # published a day late
    assert fetch.run_update(say=lambda _: None, today=date(2026, 9, 23)) == 0
    assert _manifest.read(store).filter(pl.col("date") == date(2026, 9, 22))["status"].to_list() == ["ok"]


def an_update_of_a_series_never_fetched_is_refused(store, archive, monkeypatch):
    _one_series(monkeypatch)
    with pytest.raises(Refused, match="depth binance-um BTC was never fetched"):
        fetch.run_update(say=lambda _: None, today=date(2026, 9, 23))


def an_update_dry_run_downloads_nothing(store, archive, monkeypatch):
    _one_series(monkeypatch)
    _fetched(store, archive, "depth", "binance-um", ["2026-09-20"], _depth_file)
    before = len(archive.asked)
    said = []
    assert fetch.run(["update", "--dry-run"], say=said.append) == 0
    assert all(method == "HEAD" for method, _ in archive.asked[before:])
    with pytest.raises(Refused, match="takes only --dry-run"):
        fetch.run(["update", "--jobs", "2"], say=said.append)


def the_update_keeps_the_forward_claims_series():
    assert {(k, v, t) for k, v, t, _ in fetch.UPDATES} >= {("depth", "binance-um", "BTC"), ("candles", "binance-um", "BTC"),
                                                            ("trades", "binance-um", "BTC"), ("trades", "bybit-linear", "BTC")}  # fmt: skip


# ---- funding and the premium --------------------------------------------------------

MONTHLY = "https://data.binance.vision/data/futures/um/monthly"


def _funding_url(month: str, symbol: str = "BTCUSDT") -> str:
    return f"{MONTHLY}/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"


def _funding_file(month: str, rows: list[str]) -> bytes:
    return _zip(f"BTCUSDT-fundingRate-{month}.csv", "calc_time,funding_interval_hours,last_funding_rate\n" + "\n".join(rows) + "\n")


JAN = ["1704067200000,8,0.00037409", "1704096000003,8,0.00027213", "1704124800000,8,-0.0001"]  # 2024-01-01 00, 08, 16 UTC
FEB = ["1706745600000,8,0.0002"]  # 2024-02-01 00 UTC


def the_funding_is_fetched_a_month_at_a_time(store, archive):
    archive.publish(_funding_url("2024-01"), _funding_file("2024-01", JAN))
    archive.publish(_funding_url("2024-02"), _funding_file("2024-02", FEB))
    code, _ = _run("funding", "binance-um", "BTC", "--from", "2024-01-20", "--to", "2024-02-03")
    assert code == 0
    m = _manifest.read(store)
    assert m["date"].to_list() == [date(2024, 1, 1), date(2024, 2, 1)]
    assert set(m["status"]) == {"ok"}


def a_month_is_read_from_a_window_inside_it(store, archive):
    # Guard: the file is held under the month's first day, which a window from the 1st at 08:00 does not reach.
    archive.publish(_funding_url("2024-01"), _funding_file("2024-01", JAN))
    _run("funding", "binance-um", "BTC", "--from", "2024-01-01", "--to", "2024-01-31")
    got = gr.reference.funding("BTC", "2024-01-01T08:00Z", "2024-02-01T00:00Z").collect()
    assert got["rate"].to_list() == [0.00027213, -0.0001]
    assert got["interval_hours"].to_list() == [8, 8]
    assert got["ts"][0] == utc("2024-01-01T08:00:00.003")


def a_settlement_outside_its_month_is_refused():
    with pytest.raises(Refused, match="outside 2024-02"):
        _sources.parse("funding", "binance-um", "BTC", date(2024, 2, 1), _funding_file("2024-02", JAN), "x.zip")


def every_settlement_covers_the_hours_up_to_it():
    settled = pl.DataFrame(
        {
            "ticker": "BTC",
            "ts": [utc("2024-01-01T00:00"), utc("2024-01-01T08:00:00.003")],
            "rate": [0.0003, 0.0002],
            "interval_hours": [8, 8],
        }
    )
    got = gr.reference.funding_hours(settled)
    assert got.height == 16
    assert got["ts"][0] == utc("2023-12-31T17:00") and got["ts"][-1] == utc("2024-01-01T08:00")
    assert got.filter(pl.col("rate") != 0)["ts"].to_list() == [utc("2024-01-01T00:00"), utc("2024-01-01T08:00")]


def a_shorter_interval_keeps_the_earlier_settlement():
    # Guard: an 8h window reaching back over a 4h settlement must not zero its rate.
    settled = pl.DataFrame(
        {"ticker": "BTC", "ts": [utc("2024-01-01T04:00"), utc("2024-01-01T08:00")], "rate": [0.0001, 0.0002], "interval_hours": [4, 8]}
    )
    got = gr.reference.funding_hours(settled)
    assert got.filter(pl.col("ts") == utc("2024-01-01T04:00"))["rate"].to_list() == [0.0001]
    assert got["rate"].sum() == pytest.approx(0.0003)


def the_settled_funding_charges_a_daily_bar_through_the_backtest():
    from galata_research import backtest

    day = utc("2024-01-01T00:00")
    bars = pl.DataFrame({"ticker": "BTC", "ts": [day, day + (day - utc("2023-12-31T00:00"))], "close": [100.0, 100.0]}).with_columns(
        (pl.col("ts") + pl.duration(days=1)).alias("close_ts")
    )
    # Settlements at 08 and 16 on the 2nd and 00 on the 3rd: the second bar's three.
    settled = pl.DataFrame(
        {
            "ticker": "BTC",
            "ts": [utc("2024-01-02T08:00"), utc("2024-01-02T16:00"), utc("2024-01-03T00:00")],
            "rate": [0.0001, 0.0002, 0.0003],
            "interval_hours": [8, 8, 8],
        }
    )
    got = backtest.returns(bars, pl.lit(1.0), fee=0.0, funding=gr.reference.funding_hours(settled))
    assert got["funding"].to_list() == [None, pytest.approx(0.0006)]


def a_premium_kline_is_read():
    got = _parse("premium", "binance-um", _zip("BTCUSDT-1m-2024-01-01.csv", (
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
        "1704067200000,0.00075030,0.00090957,0.00059476,0.00089825,0,1704067259999,0,12,0,0,0\n"
    )), day="2024-01-01")  # fmt: skip
    assert got.columns == list(_sources.PREMIUM)
    assert got["close"][0] == 0.00089825 and got["close_ts"][0] == utc("2024-01-01T00:01")


def a_body_cut_off_mid_transfer_is_asked_again(monkeypatch):
    import http.client

    calls = []

    def flaky(url, method, agent="galata-fetch"):
        calls.append(url)
        if len(calls) == 1:
            raise http.client.IncompleteRead(b"x" * 10, 5)
        return fetch.Response(200, b"whole", 5)

    monkeypatch.setattr(fetch, "_transport", flaky)
    monkeypatch.setattr(fetch.time, "sleep", lambda _: None)
    got = fetch.get(_depth_url("2026-09-20"))
    assert got.body == b"whole" and len(calls) == 2


# ---- the whole Binance list -------------------------------------------------------------

LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"


def _listing_page(prefixes=(), keys=(), truncated=False, next_marker=None) -> bytes:
    body = "".join(f"<CommonPrefixes><Prefix>{p}</Prefix></CommonPrefixes>" for p in prefixes)
    body += "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys)
    more = f"<NextMarker>{next_marker}</NextMarker>" if next_marker else ""
    return f'<?xml version="1.0"?><ListBucketResult><IsTruncated>{"true" if truncated else "false"}</IsTruncated>{more}{body}</ListBucketResult>'.encode()


def _daily_file(month: str, symbol: str, rows: list[str]) -> bytes:
    return _zip(f"{symbol}-1d-{month}.csv", "\n".join(rows) + "\n")


def every_listed_perpetual_is_fetched_for_the_months_it_has(store, archive):
    from galata_research.reference import _sources as src

    base = "data/futures/um/monthly/klines/"
    archive.publish(src.listing_url(base), _listing_page([base + "AAAUSDT/", base + "GONEUSDT/"], truncated=True, next_marker=base + "GONEUSDT/"), checksum=False)
    archive.publish(src.listing_url(base, base + "GONEUSDT/"), _listing_page([base + "BBBUSDC/"]), checksum=False)
    for sym, months in (("AAAUSDT", ["2024-01", "2024-02"]), ("GONEUSDT", ["2024-01"])):
        keys = [f"{base}{sym}/1d/{sym}-1d-{m}.zip" for m in months] + [f"{base}{sym}/1d/{sym}-1d-{m}.zip.CHECKSUM" for m in months]
        archive.publish(src.listing_url(f"{base}{sym}/1d/"), _listing_page(keys=keys), checksum=False)
    archive.publish(f"https://data.binance.vision/{base}AAAUSDT/1d/AAAUSDT-1d-2024-01.zip", _daily_file("2024-01", "AAAUSDT", ["1704067200000,1,2,0.5,1.5,10,1704153599999,15,3,5,7,0"]))
    archive.publish(f"https://data.binance.vision/{base}AAAUSDT/1d/AAAUSDT-1d-2024-02.zip", _daily_file("2024-02", "AAAUSDT", ["1706745600000,1.5,2,1,1.8,10,1706831999999,18,3,5,7,0"]))
    archive.publish(f"https://data.binance.vision/{base}GONEUSDT/1d/GONEUSDT-1d-2024-01.zip", _daily_file("2024-01", "GONEUSDT", ["1704067200000,9,9,8,8.5,1,1704153599999,8.5,1,0,0,0"]))
    code, said = _run("daily", "binance-um", "--all", "--from", "2024-01-01", "--to", "2024-03-31")
    assert code == 0, said
    assert said[0] == "the archive lists 2 USDT perpetuals with daily"  # the USDC-margined one is not a USDT perpetual
    m = _manifest.read(store)
    # GONE's listing stops in January: February and March are never asked.
    assert sorted(zip(m["ticker"], m["date"])) == [("AAA", date(2024, 1, 1)), ("AAA", date(2024, 2, 1)), ("GONE", date(2024, 1, 1))]
    got = gr.reference.daily(None, "2024-01-01T00:00Z", "2024-03-01T00:00Z").collect()
    assert got.sort("ticker", "ts")["close"].to_list() == [1.5, 1.8, 8.5]
    assert got["close_ts"][0] == utc("2024-01-02T00:00")


def an_all_with_tickers_is_refused(store, archive):
    with pytest.raises(Refused, match="no tickers"):
        _run("daily", "binance-um", "BTC", "--all", "--from", "2024-01-01", "--to", "2024-01-31")


def a_daily_bar_outside_its_month_is_refused():
    blob = _daily_file("2024-02", "BTCUSDT", ["1704067200000,1,2,0.5,1.5,10,1704153599999,15,3,5,7,0"])
    with pytest.raises(Refused, match="outside 2024-02"):
        _sources.parse("daily", "binance-um", "BTC", date(2024, 2, 1), blob, "x.zip")


def every_binance_base_asset_is_its_usdt_perpetual():
    from galata_research.reference import _instruments

    assert _instruments.instrument("binance-um", "1000PEPE").symbol == "1000PEPEUSDT"
    assert _instruments.instrument("binance-um", "GOLD").symbol == "XAUUSDT"  # the record's own mapping is kept
    assert _instruments.instrument("binance-um", "币安人生").symbol == "币安人生USDT"
    with pytest.raises(Refused, match="base asset"):
        _instruments.instrument("binance-um", "btc/usdt")
