"""galata-research: the record galata-datawatch keeps, loaded as polars or DuckDB.

    import galata_research as gr

    gr.market.candles(["BTC"], "4h", start, end, as_of=t)   # pl.LazyFrame
    gr.mask_gaps(bars, "candles")                            # in_gap, gap_cause
    gr.account.margin("main", start, end)                   # my margin, per snapshot
    gr.join_recv(trades, gr.market.marks(["BTC"], start, end))  # mark_recv at each trade
    gr.frontier()                                           # how far the record goes
    gr.root()                                               # where the record is
    gr.reference.depth(["BTC"], start, end)                  # other venues' archives, fetched by galata-fetch
    gr.timeseries.returns(bars, kind="log")                 # null across a hole
    gr.timeseries.periods_per_year("1h")                    # 8760
    gr.models.vol.fit(returns, model="gjr", dist="t")       # the [models] extra, loaded on use

The library owns the record's semantics, not its I/O: dedupe, closure, the
clock and the cast are applied once, here, so a notebook never reads a
re-fetched bar twice or a bar's open as its close.
"""

from . import account, backtest, jumps, leadlag, liquidity, market, reference, stats, studies, timeseries, utils
from ._errors import Refused
from ._frontier import frontier
from ._root import root
from .clocks import join_recv
from .gaps import mask_gaps

__all__ = [
    "Refused",
    "account",
    "backtest",
    "frontier",
    "join_recv",
    "jumps",
    "leadlag",
    "liquidity",
    "market",
    "reference",
    "mask_gaps",
    "root",
    "stats",
    "studies",
    "timeseries",
    "utils",
    "models",
    "calendar",
]


def __getattr__(name: str):
    # gr.models needs numpy, scipy and arch, so it is imported on first use
    # (PEP 562), and `import galata_research` never needs them.
    # gr.calendar needs exchange_calendars (and its pandas): the calendars extra, loaded the same way.
    if name in ("models", "calendar"):
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
