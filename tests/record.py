"""Claims about the real record under the resolved root; skipped when there is none.

Figures are the design's, measured 2026-09-25, asserted as floors or as
structure, because the record grows and retention trims the finest widths.
"""

import polars as pl
import pytest
from conftest import utc

import galata_research as gr
from galata_research import Refused

pytestmark = pytest.mark.record

EVER = utc("2020-01-01T00:00"), utc("2100-01-01T00:00")


@pytest.fixture(scope="module", autouse=True)
def _record():
    try:
        gr.frontier()
    except Refused as absent:
        pytest.skip(f"no record: {absent}")


@pytest.mark.parametrize("interval", ["1m", "1h", "4h", "1d"])
def every_series_has_at_most_one_open_bar(interval):
    got = gr.market.candles(None, interval, *EVER, traded_only=False, closed_only=False).collect()
    open_per_series = got.group_by("venue", "ticker").agg((~pl.col("closed")).sum().alias("open"))
    assert open_per_series["open"].max() <= 1


@pytest.mark.parametrize("interval", ["1m", "1h", "4h", "1d"])
def no_bar_is_returned_twice(interval):
    got = gr.market.candles(None, interval, *EVER, traded_only=False, closed_only=False).collect()
    assert got.select("venue", "ticker", "ts").is_duplicated().sum() == 0


def the_coarse_history_is_at_least_as_deep_as_measured():
    for interval, floor in [("1d", 6_001), ("4h", 19_271)]:
        got = gr.market.candles(None, interval, *EVER, traded_only=False).collect()
        assert got.height >= floor, interval


def the_first_traded_btc_daily_bar_opens_on_2023_02_26():
    got = gr.market.candles("BTC", "1d", *EVER).head(1).collect()
    assert got["ts"].item() == utc("2023-02-26T00:00")


def no_trade_less_bar_is_returned_by_default():
    got = gr.market.candles(None, "1d", *EVER).collect()
    assert got.filter(pl.col("trade_count") == 0).height == 0


# Ticks


def no_execution_is_returned_twice():
    got = gr.market.trades(None, *EVER).collect()
    assert got.select("venue", "ticker", "trade_id").is_duplicated().sum() == 0


def no_trade_id_carries_two_contents():
    # The loader keeps the first receipt without comparing; this holds that
    # every replay is the same execution, so nothing is lost by it.
    raw = pl.scan_parquet(gr._root.root() / "tape" / "kind=trades" / "**" / "*.parquet", hive_partitioning=False)
    contents = raw.group_by("venue", "ticker", "trade_id").agg(
        pl.struct("at_micros", "price", "size", "aggressor").n_unique().alias("contents")
    )
    assert contents.filter(pl.col("contents") > 1).collect().height == 0


def no_quote_repeats_a_venue_ticker_ts():
    got = gr.market.quotes(None, *EVER).collect()
    assert got.select("venue", "ticker", "ts").is_duplicated().sum() == 0


def every_tick_is_a_whole_millisecond_on_hyperliquid():
    for loader in (gr.market.trades, gr.market.quotes):
        got = loader(None, *EVER).filter(pl.col("venue") == "hyperliquid").collect()
        assert (got["ts"].dt.microsecond() % 1000 == 0).all(), loader.__name__


# Gaps


def every_gap_is_on_the_receipt_clock():
    got = gr.market.gaps(None, *EVER).collect()
    assert "ts" not in got.columns
    assert (got["from_recv_ts"] < got["to_recv_ts"]).all()


def every_masked_trade_count_matches_a_direct_count():
    # An independent count, in SQL over the raw tape, of executions whose
    # venue time falls in [from - 1 s, to) of a trades gap.
    import duckdb

    tape = gr._root.root() / "tape"
    direct = duckdb.sql(f"""
        with t as (select distinct venue, ticker, trade_id, at_micros
                   from read_parquet('{tape}/kind=trades/**/*.parquet', hive_partitioning=false)),
             g as (select * from read_parquet('{tape}/kind=gaps/**/*.parquet', hive_partitioning=false)
                   where series = 'trades')
        select count(distinct (t.venue, t.ticker, t.trade_id)) from t join g
          on t.venue = g.venue and (g.ticker is null or g.ticker = t.ticker)
         and t.at_micros >= g.from_micros - 1000000 and t.at_micros < g.to_micros
    """).fetchone()[0]
    masked = gr.mask_gaps(gr.market.trades(None, *EVER), "trades").filter(pl.col("in_gap")).collect()
    assert masked.height == direct
