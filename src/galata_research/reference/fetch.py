"""`galata-fetch`: a venue's published historical archive, one day at a time, into the reference store.

Research reads the record and never captures it. This command is the one
exception, stated as settled point 8: it downloads **published archives**
only, from the hosts in `HOSTS`. It never calls a venue's live API and
never my account. Nothing in `galata_research` imports it, so loading never
opens a socket.

    galata-fetch depth binance-um BTC ETH --from 2023-01-01 --to 2026-09-26 --dry-run
    galata-fetch book bybit-linear BTC --from 2023-01-18 --to 2026-09-26 --sample weekly:wed
    galata-fetch trades okx-swap BTC ETH --from 2023-01-02 --to 2026-09-26 --days 2026-09-26  # files of 09-26 and 09-27
    galata-fetch update --dry-run   # what the forward claims need, from each series' last day to yesterday

- **Idempotent**: a day whose manifest row is `ok` is skipped; `--refetch`
  asks again and records `mismatch` if the bytes changed, keeping the old day.
- **Checked**: Binance's published `.CHECKSUM` must match the bytes, or the
  day is `mismatch` and nothing is written.
- **Absence stated**: HTTP 404 is `absent`, not asked again without
  `--refetch-absent`.
- **Declared days** for the heavy kinds: Bybit's book is 93 MB zipped a day
  for BTC and Binance's aggTrades 10 MB (2026-09); without `--days` or
  `--sample` the command refuses, stating what the whole range would cost.
"""

import argparse
import hashlib
import http.client
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

import polars as pl

from .. import _root as _record_root
from .._errors import Refused
from . import _events, _instruments, _manifest, _root, _sources

# s3-ap-northeast-1.amazonaws.com is the bucket behind data.binance.vision, asked only for its listing.
HOSTS = ("data.binance.vision", "s3-ap-northeast-1.amazonaws.com", "public.bybit.com", "quote-saver.bycsi.com", "static.okx.com")
HEAVY = {("bybit-linear", "book"), ("binance-um", "trades"), ("okx-swap", "trades")}
MAX_JOBS = 4
MAX_JOBS_LIGHT = 16  # for the kinds of a few kilobytes a file
_FLUSH = 500  # manifest rows written together
_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_RETRIES = (1.0, 2.0, 4.0)


class Response:
    def __init__(self, status: int, body: bytes = b"", size: int | None = None):
        self.status, self.body, self.size = status, body, size


def _transport(url: str, method: str, agent: str = "galata-fetch") -> Response:
    """The one place a socket opens. Tests replace it with a fixture archive."""
    request = urllib.request.Request(url, method=method, headers={"User-Agent": agent})
    try:
        with urllib.request.urlopen(request, timeout=120) as r:
            body = r.read() if method == "GET" else b""
            length = r.headers.get("Content-Length")
            return Response(r.status, body, int(length) if length else len(body))
    except urllib.error.HTTPError as e:
        return Response(e.code)


def get(url: str, method: str = "GET", *, hosts: tuple[str, ...] = HOSTS, agent: str = "galata-fetch") -> Response:
    """A listed host's answer, retried on a network error, a truncated body or a 5xx; any other host is refused."""
    host = urlsplit(url).hostname
    if host not in hosts:
        raise Refused(f"{host} is not an archive this command fetches from; it fetches from {', '.join(hosts)}")
    for pause in (*_RETRIES, None):
        try:
            response = _transport(url, method, agent)
        # A body cut off mid-transfer (IncompleteRead) is retried like any network error.
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead):
            if pause is None:
                raise
            time.sleep(pause)
            continue
        if response.status < 500 or pause is None:
            return response
        time.sleep(pause)
    raise AssertionError("unreachable")


def listing(prefix: str) -> tuple[list[str], list[str]]:
    """The archive's sub-prefixes and keys one level under `prefix`, every page of them."""
    import re as _re

    prefixes: list[str] = []
    keys: list[str] = []
    marker = ""
    while True:
        response = get(_sources.listing_url(prefix, marker))
        if response.status != 200:
            raise Refused(f"the archive's listing of {prefix} answered HTTP {response.status}")
        text = response.body.decode()
        prefixes += [p for p in _re.findall(r"<Prefix>([^<]*)</Prefix>", text) if p != prefix]
        keys += _re.findall(r"<Key>([^<]*)</Key>", text)
        if "<IsTruncated>true</IsTruncated>" not in text:
            return prefixes, keys
        found = _re.search(r"<NextMarker>([^<]*)</NextMarker>", text)
        marker = found.group(1) if found else (keys or prefixes)[-1]


def universe(kind: str) -> list[str]:
    """Every USDT perpetual the archive holds a kind for, as base-asset tickers, delisted ones included."""
    prefixes, _ = listing(_sources.monthly_prefix(kind))
    symbols = sorted(p.rstrip("/").rsplit("/", 1)[1] for p in prefixes)
    return [s.removesuffix("USDT") for s in symbols if s.endswith("USDT") and len(s) > 4]


def held_months(kind: str, ticker: str) -> list[date]:
    """The months the archive holds a monthly kind for one ticker, from its listing: no month is guessed."""
    import re as _re

    symbol = _instruments.instrument("binance-um", ticker).symbol
    _, keys = listing(_sources.monthly_prefix(kind, symbol))
    months = {_re.search(r"-(\d{4})-(\d{2})\.zip$", k) for k in keys if k.endswith(".zip")}
    return sorted(date(int(m.group(1)), int(m.group(2)), 1) for m in months if m)


def days_between(first: date, last: date) -> list[date]:
    return [first + timedelta(days=d) for d in range((last - first).days + 1)]


def declared(days: list[date], listed: str | None, sample: str | None) -> list[date] | None:
    """The days asked for by `--days` or `--sample`; None when neither is given.

    - `weekly:<dow>`: one weekday, every week.
    - `monthly:<dow>`: the first such weekday of each month.
    - `every:<n>`: every n-th day counted from 0001-01-01, so the same days
      whatever `--from` is. With n = 3 (or any n coprime to 7) the days turn
      through the week evenly, which an hour-of-week profile needs and
      `weekly:wed` cannot give.
    """
    if listed:
        named = {date.fromisoformat(d.strip()) for d in listed.split(",") if d.strip()}
        return [d for d in days if d in named]
    if sample:
        kind, _, arg = sample.partition(":")
        if kind == "weekly" and arg in _WEEKDAYS:
            return [d for d in days if d.weekday() == _WEEKDAYS.index(arg)]
        if kind == "monthly" and arg in _WEEKDAYS:
            return [d for d in days if d.weekday() == _WEEKDAYS.index(arg) and d.day <= 7]
        if kind == "every" and arg.isdigit() and int(arg) >= 1:
            return [d for d in days if d.toordinal() % int(arg) == 0]
        raise Refused(f"--sample {sample!r} is not weekly:<dow>, monthly:<dow> or every:<n>, with dow one of {', '.join(_WEEKDAYS)}")
    return None


def _refuse_inside_the_record(store: Path) -> None:
    try:
        record = _record_root.root().resolve()
    except Refused:
        return
    if store.resolve().is_relative_to(record):
        raise Refused(f"the reference store {store} is inside the record {record}; research never writes to the record")


def plan(
    kind: str,
    venue: str,
    ticker: str,
    first: date,
    last: date,
    listed: str | None,
    sample: str | None,
    say,
) -> list[date]:
    """The archive dates to consider for one series, after the listing clip and the declared-days rule.

    Days are asked as UTC days. Where a venue's archive day ends before the
    UTC day (OKX's ends at 16:00 UTC), a UTC day brings the two archive dates
    that hold it.
    """
    start = _instruments.first_day(venue, kind, ticker)
    if kind in _instruments.MONTHLY:
        # One file a month, held under its first day: every month the range touches.
        start = start.replace(day=1)
        first = first.replace(day=1)
    if first < start:
        say(f"{ticker} {kind} on {venue} is archived from {start}; {(start - first).days} earlier day(s) not requested")
    days = days_between(max(first, start), last)
    if kind in _instruments.MONTHLY:
        return sorted({d.replace(day=1) for d in days})
    chosen = declared(days, listed, sample)
    if chosen is None:
        if (venue, kind) in HEAVY and days:
            probe = get(_sources.url(kind, venue, ticker, days[-1]), "HEAD")
            each = probe.size or 0
            raise Refused(
                f"{kind} on {venue} is fetched on declared days only: {len(days)} days of {ticker} would be about "
                f"{each * len(days) / 1e9:.1f} GB ({each / 1e6:.1f} MB on {days[-1]}); name them with --days or --sample weekly:wed"
            )
        return days
    return sorted({a for d in chosen for a in _instruments.archive_days(venue, d)})


def _fetch_day(store: Path, kind: str, venue: str, ticker: str, day: date, held: dict | None) -> tuple[dict, str]:
    """Download, check, parse and write one day; the manifest row and a line to print."""
    archive = _sources.url(kind, venue, ticker, day)
    began = time.perf_counter()
    row = {
        "kind": kind, "venue": venue, "ticker": ticker, "date": day, "url": archive, "bytes": None,
        "sha256": None, "published_sha256": None, "fetched_at_recv": datetime.now(UTC), "rows": None,
    }  # fmt: skip
    response = get(archive)
    if response.status == 404:
        return {
            **row,
            "status": "absent",
        }, f"{day} {kind} {venue} {ticker} absent (404)"
    if response.status != 200:
        raise Refused(f"{archive} answered HTTP {response.status}")
    digest = hashlib.sha256(response.body).hexdigest()
    row.update(bytes=len(response.body), sha256=digest)
    if (checksum := _sources.checksum_url(venue, archive)) is not None:
        published = get(checksum)
        if published.status == 200:
            row["published_sha256"] = published.body.split()[0].decode()
            if row["published_sha256"] != digest:
                return (
                    {**row, "status": "mismatch"},
                    f"{day} {kind} {venue} {ticker} MISMATCH against the published checksum",
                )
    if held is not None and held["status"] == "ok" and held["sha256"] != digest:
        # The archive rewrote a day already held: keep what was loaded, say so.
        return {
            **held,
            "fetched_at_recv": row["fetched_at_recv"],
            "status": "mismatch",
        }, (
            f"{day} {kind} {venue} {ticker} MISMATCH: the archive's bytes changed since {held['fetched_at_recv']:%Y-%m-%d}; the held day is kept"
        )
    frame = _sources.parse(kind, venue, ticker, day, response.body, archive.rsplit("/", 1)[1])
    path = _manifest.day_path(store, kind, venue, ticker, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)
    row.update(rows=frame.height, status="ok")
    return (
        row,
        f"{day} {kind} {venue} {ticker} ok {frame.height:,} rows {len(response.body) / 1e6:.2f} MB {time.perf_counter() - began:.1f}s",
    )


def run_events(say=print) -> int:
    """`galata-fetch events`: the FOMC and BLS calendars, whole, into `events/` beside the day store.

    bls.gov is asked only with `GALATA_CONTACT` set, and the contact rides in
    the User-Agent alone: no file, log or manifest holds it.
    """
    store, _ = _root.resolve()
    _refuse_inside_the_record(store)
    frames, sources, failed = [], [], []

    def page(url: str, agent: str = "galata-fetch") -> str:
        response = get(url, hosts=_events.HOSTS, agent=agent)
        if response.status != 200:
            raise Refused(f"{url} answered HTTP {response.status}")
        sources.append({"url": url, "bytes": len(response.body), "sha256": hashlib.sha256(response.body).hexdigest(),
                        "fetched_at_recv": datetime.now(UTC)})  # fmt: skip
        return response.body.decode("utf-8", "replace")

    def keep(frame, url):
        if frame.is_empty():
            raise Refused(f"{url} parsed to no event: the page has changed shape; nothing is written")
        sources[-1]["rows"] = frame.height
        frames.append(frame)
        say(f"{url}: {frame.height} events")

    calendar = _events.fomc_calendar(page(_events.FOMC_CALENDAR))
    keep(calendar, _events.FOMC_CALENDAR)
    for year in range(_events.FIRST_YEAR, calendar["date"].min().year):
        url = _events.FOMC_HISTORICAL.format(year=year)
        keep(_events.fomc_historical(page(url), url), url)
    contact = os.environ.get("GALATA_CONTACT", "").strip()
    held = store / "events" / "events.parquet"
    if not contact:
        # BLS rows fetched earlier with a contact are kept, not dropped by a run that may not ask bls.gov.
        kept = pl.read_parquet(held).filter(pl.col("source") == "bls") if held.is_file() else None
        if kept is not None and kept.height:
            frames.append(kept)
            say(f"bls.gov not asked (GALATA_CONTACT unset): {kept.height} BLS events from the last fetch kept")
        else:
            failed.append("bls.gov is asked only with GALATA_CONTACT set: its policy refuses a client without a contact")
    else:
        for event, url in _events.BLS_INDEX.items():
            keep(_events.bls_index(page(url, f"galata-research (contact: {contact})"), event, url), url)

    folder = store / "events"
    folder.mkdir(parents=True, exist_ok=True)
    events = pl.concat(frames).sort("date", "source", "event")
    for name, frame in (("events", events), ("sources", pl.DataFrame(sources))):
        tmp = folder / f".{name}.tmp"
        frame.write_parquet(tmp)
        tmp.replace(folder / f"{name}.parquet")
    say(f"{events.height} events from {len(sources)} pages" + "".join(f"; REFUSED: {f}" for f in failed))
    return 1 if failed else 0


# What `galata-fetch update` keeps current: the forward claims' series (liquidity-forward.md), sampled
# trades on the same every-9th-day phase as the study's.
UPDATES = (
    ("depth", "binance-um", "BTC", None),
    ("candles", "binance-um", "BTC", None),
    ("candles", "binance-um", "ETH", None),
    ("trades", "binance-um", "BTC", "every:9"),
    ("trades", "bybit-linear", "BTC", "every:9"),
)


def run_update(dry_run: bool = False, say=print, today: date | None = None) -> int:
    """`galata-fetch update`: each series in `UPDATES` from the day after its last `ok` day through yesterday, UTC.

    A day asked before the archive publishes it is recorded `absent`, so the
    range's absent days are asked again. A series never fetched is refused:
    its start is a choice, made once by hand. The event calendar is refetched
    whole unless `dry_run`.
    """
    store, _ = _root.resolve()
    manifest = _manifest.read(store).filter(pl.col("status") == "ok")
    yesterday = (today or datetime.now(UTC).date()) - timedelta(days=1)
    code = 0
    for kind, venue, ticker, sample in UPDATES:
        held = manifest.filter((pl.col("kind") == kind) & (pl.col("venue") == venue) & (pl.col("ticker") == ticker))
        if held.is_empty():
            raise Refused(f"{kind} {venue} {ticker} was never fetched; seed it once with galata-fetch {kind} {venue} {ticker} --from ...")
        first = held["date"].max() + timedelta(days=1)
        if first > yesterday:
            say(f"{kind} {venue} {ticker}: current through {first - timedelta(days=1)}")
            continue
        argv = [kind, venue, ticker, "--from", first.isoformat(), "--to", yesterday.isoformat(), "--refetch-absent"]
        argv += ["--sample", sample] if sample else ["--days", ",".join(d.isoformat() for d in days_between(first, yesterday))]
        code |= run(argv + (["--dry-run"] if dry_run else []), say)
    if not dry_run:
        code |= run_events(say)
    return code


def run(argv: list[str] | None = None, say=print) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["events"]:
        return run_events(say)
    if argv[:1] == ["update"]:
        extra = set(argv[1:]) - {"--dry-run"}
        if extra:
            raise Refused(f"galata-fetch update takes only --dry-run, not {', '.join(sorted(extra))}")
        return run_update("--dry-run" in argv, say)
    args = _parser().parse_args(argv)
    if args.to < getattr(args, "from"):
        raise Refused("--to is before --from")
    most = MAX_JOBS_LIGHT if args.kind in _instruments.LIGHT else MAX_JOBS
    if args.jobs is None:
        args.jobs = most
    if not 1 <= args.jobs <= most:
        raise Refused(f"--jobs {args.jobs} is outside 1..{most} for {args.kind}")
    months: dict[str, list[date]] | None = None
    if args.all:
        if args.tickers:
            raise Refused("--all names every ticker the archive holds; give no tickers with it")
        if args.venue != _instruments.OPEN_VENUE or args.kind not in _instruments.MONTHLY:
            raise Refused(f"--all lists {_instruments.OPEN_VENUE}'s monthly kinds ({', '.join(sorted(_instruments.MONTHLY))}) only")
        args.tickers = universe(args.kind)
        say(f"the archive lists {len(args.tickers)} USDT perpetuals with {args.kind}")
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            months = dict(zip(args.tickers, pool.map(lambda t: held_months(args.kind, t), args.tickers), strict=True))
    elif not args.tickers:
        raise Refused("name the tickers, or give --all")
    for ticker in args.tickers:
        _instruments.instrument(args.venue, ticker)
    _instruments.first_day(args.venue, args.kind, args.tickers[0])

    store, _ = _root.resolve()
    _refuse_inside_the_record(store)
    held = {(r["kind"], r["venue"], r["ticker"], r["date"]): r for r in _manifest.read(store).iter_rows(named=True)}

    work: list[tuple[str, date]] = []
    skipped = 0
    for ticker in args.tickers:
        asked = plan(args.kind, args.venue, ticker, getattr(args, "from"), args.to, args.days, args.sample, say)
        if months is not None:
            # Only the months the listing shows: a delisted coin's months stop where its archive does.
            listed = set(months[ticker])
            asked = [d for d in asked if d in listed]
        for day in asked:
            row = held.get((args.kind, args.venue, ticker, day))
            if (
                row is None
                or (row["status"] == "absent" and args.refetch_absent)
                or (row["status"] == "ok" and args.refetch)
            ):
                work.append((ticker, day))
            else:
                skipped += 1

    if args.dry_run:
        total = 0
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            sizes = pool.map(lambda w: get(_sources.url(args.kind, args.venue, w[0], w[1]), "HEAD"), work)
            for (ticker, day), size in zip(work, sizes, strict=True):
                shown = "absent" if size.status == 404 else f"{(size.size or 0) / 1e6:.2f} MB"
                total += (size.size or 0) if size.status == 200 else 0
                say(f"{day} {args.kind} {args.venue} {ticker} {shown}")
        say(f"{len(work)} day(s) to fetch, {total / 1e6:,.1f} MB; {skipped} already in the manifest")
        return 0

    failed = 0
    pending: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                _fetch_day,
                store,
                args.kind,
                args.venue,
                t,
                d,
                held.get((args.kind, args.venue, t, d)),
            ): (t, d)
            for t, d in work
        }
        for future in as_completed(futures):
            ticker, day = futures[future]
            try:
                row, line = future.result()
            except Exception as e:  # noqa: BLE001 -- one day's failure is named, and the others still land
                failed += 1
                say(f"{day} {args.kind} {args.venue} {ticker} FAILED: {e}")
                continue
            pending.append(row)
            if len(pending) >= _FLUSH:
                _manifest.upsert(store, pending)
                pending = []
            failed += row["status"] == "mismatch"
            say(line)
    if pending:
        _manifest.upsert(store, pending)
    say(f"{len(work)} day(s) asked, {failed} failed or mismatched; {skipped} were already in the manifest")
    return 1 if failed else 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="galata-fetch",
        description="Fetch a venue's published historical archive into the reference store. "
        f"Kinds: {', '.join(_instruments.KINDS)}. Venues: {', '.join(_instruments.VENUES)}. "
        "`galata-fetch events` fetches the FOMC and BLS calendars instead (BLS needs GALATA_CONTACT); "
        "`galata-fetch update [--dry-run]` brings the declared series up to yesterday.",
    )
    p.add_argument("kind", choices=_instruments.KINDS, help=", ".join(_instruments.KINDS))
    p.add_argument("venue", choices=_instruments.VENUES)
    p.add_argument("tickers", nargs="*", metavar="TICKER")
    p.add_argument("--all", action="store_true", help="every USDT perpetual the archive lists (binance-um, monthly kinds)")
    p.add_argument(
        "--from",
        required=True,
        type=date.fromisoformat,
        help="first UTC day, inclusive",
    )
    p.add_argument("--to", required=True, type=date.fromisoformat, help="last UTC day, inclusive")
    p.add_argument("--days", help="comma-separated UTC days, instead of every day")
    p.add_argument("--sample", help="weekly:<mon..sun>, monthly:<mon..sun> (the first of the month) or every:<n>, instead of every day")
    p.add_argument("--dry-run", action="store_true", help="state days and bytes; download nothing")
    p.add_argument("--jobs", type=int, default=None, help=f"parallel requests: at most {MAX_JOBS}, or {MAX_JOBS_LIGHT} for the light kinds")
    p.add_argument("--refetch", action="store_true", help="ask again for days held ok, and compare")
    p.add_argument(
        "--refetch-absent",
        action="store_true",
        help="ask again for days the archive lacked",
    )
    return p


def main() -> None:
    try:
        sys.exit(run())
    except Refused as e:
        print(f"galata-fetch: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
