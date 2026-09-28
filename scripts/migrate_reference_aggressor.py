"""Rewrite the reference store's trades from `buy`/`sell` to the tape's `bid`/`ask`, once.

`fetch-reference-market-data` normalised the aggressor as `buy`/`sell`; the
tape, and so every other frame in this library, says `bid` (a buy crossing)
and `ask` (a sell crossing). `estimate-the-spread` fixed the parser. This
rewrites what was already stored. It is idempotent, refuses a file holding
any other word, and `--reverse` undoes it. The manifest is untouched: its
sha256 is of the archive's bytes, not of the Parquet.

    uv run python scripts/migrate_reference_aggressor.py [--reverse]
"""

import sys
import time

import polars as pl

from galata_research.reference import _root

FORWARD = {"buy": "bid", "sell": "ask"}


def main(reverse: bool = False) -> int:
    mapping = {v: k for k, v in FORWARD.items()} if reverse else FORWARD
    target = set(mapping.values())
    began = time.perf_counter()
    files = sorted((_root.root() / "kind=trades").rglob("date=*.parquet"))
    rewritten = rows = 0
    for path in files:
        frame = pl.read_parquet(path)
        words = set(frame["aggressor"].drop_nulls().unique())
        unknown = words - set(mapping) - target
        if unknown:
            print(f"{path}: aggressor holds {sorted(unknown)}, neither {sorted(mapping)} nor {sorted(target)}", file=sys.stderr)
            return 2
        if words <= target:
            continue  # already migrated
        out = frame.with_columns(pl.col("aggressor").replace(mapping))
        tmp = path.with_suffix(".tmp")
        out.write_parquet(tmp)
        tmp.replace(path)
        rewritten += 1
        rows += out.height
    print(f"{len(files)} files, {rewritten} rewritten, {rows:,} rows, {time.perf_counter() - began:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main("--reverse" in sys.argv[1:]))
