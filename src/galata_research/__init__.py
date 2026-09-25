"""galata-research: the record galata-datawatch keeps, loaded as polars or DuckDB.

    import galata_research as gr

    gr.market.candles(["BTC"], "4h", start, end, as_of=t)   # pl.LazyFrame
    gr.mask_gaps(bars, "candles")                            # in_gap, gap_cause
    gr.account.margin("main", start, end)                   # my margin, per snapshot
    gr.frontier()                                           # how far the record goes

The library owns the record's semantics, not its I/O: dedupe, closure, the
clock and the cast are applied once, here, so a notebook never reads a
re-fetched bar twice or a bar's open as its close.
"""

from . import account, market
from ._errors import Refused
from ._frontier import frontier
from .gaps import mask_gaps

__all__ = ["Refused", "account", "frontier", "market", "mask_gaps"]
