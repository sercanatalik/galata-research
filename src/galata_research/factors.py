"""Cross-sectional factors: a point-in-time universe, rankings, and the weekly portfolios they make.

    p = gr.factors.panel(gr.reference.daily(None, ...), gr.reference.funding(None, ...))
    m = gr.factors.members(p, exclude=gr.factors.INDEX_AND_STABLE)        # who could be traded, each day
    trials = gr.factors.rules(p, m, start="2020-01-01", end="2025-01-01")  # the registered 12, and the benchmark

**Point in time.** A coin is a candidate only on days it has a bar, so one
delisted later is in the universe until it was delisted, and a coin listed
later is not in it before. The traded set on day t is ranked on dollar
volume up to t. Every score at t uses bars up to t only. The portfolio
decided at t's close is held from the next day.
(`planning/preregistered/rank-the-universe.md`)

**The portfolio.** Weights are set at each rebalance and held fixed until the
next; drift is not traded. Turnover is charged the fee on the first day held.
Funding is charged per day held, weight × that day's settled funding (a long
pays positive funding), and nothing on a coin-day with no funding archive;
`funding_missing` counts those. A coin with no bar on a day earns nothing
that day: delisted, it exited at its last close.

The frames these return are `gr.studies` trial frames with the whole
portfolio under the ticker `"universe"`, so `studies.summary`, `matrix`,
`excess` and `gr.stats` read them as they read any trial.
"""

from collections.abc import Iterable
from datetime import datetime, timedelta

import polars as pl

from . import timeseries, utils
from ._errors import Refused

BINANCE_TAKER = 0.0005
UNIVERSE = "universe"

# Binance's index perpetuals and its stablecoin: never a coin's own return.
INDEX_AND_STABLE = frozenset({"BTCDOM", "DEFI", "FOOTBALL", "BLUEBIRD", "USDC"})
# The non-crypto perpetuals Binance listed from 2025, as registered (best effort, by hand, 2026-10-04).
NON_CRYPTO = frozenset(
    """AAOI AAPL ACN ADBE ALAB AMAT AMC AMD AMZN ANET ANTHROPIC APLD APP ARM ASML ASTS AVGO AXTI BABA
    BITO BMNR BRKB BX BYD BZ CBRS CIEN CL COHR COIN COPPER COST CRCL CRDO CRM CRWD CRWV CSCO
    CSOPSAMSUNG2L CSOPSKHYNIX2L CVNA CVX CXMT DDOG DELL DIA DIS DJT DKNG EBAY EWJ EWT EWY EWZ GDX
    GEV GLW GME GOOGL GS GTLB HANMI HD HIMS HK0625 HK0700 HK0992 HK1810 HOOD HPE HUT HYUNDAI IBM
    INTC IONQ IREN IWM JPM KO KODEX200 KORU KUAISHOU LGELECTRONICS LLY LRCX MARA MDB MEITUAN META
    MINIMAX MP MRK MRNA MRVL MSFT MSTR MU NATGAS NAVER NBIS NFLX NKE NOK NOW NVDA NVDL NVO OKLO
    OPENAI ORCL PANW PAXG PDD PLTR POPMART PYPL QCOM QQQ RDDT RIVN RKLB SAMSUNG SAMSUNGEM SHOP
    SKHYNIX SLX SMCI SMH SNDK SNOW SOFI SONY SOXL SOXS SPCX SPY SQQQ STRC TBT TEM TENCENT TMF
    TQQQ TSLA TSLL TSM TTWO TXN TZA UBER UNH UNITREE URNM USDBRL UVXY VRT VST WDC WMT XAG XAU
    XAUT XBI XLE XOM XPD XPT ZHIPU ZM ZS""".split()
)


def panel(daily: pl.LazyFrame | pl.DataFrame, settled: pl.LazyFrame | pl.DataFrame | None = None) -> pl.DataFrame:
    """Per (ticker, day): `close, ret` (null after a hole), `dollar_volume`, `bars` (held so far), `funding` (the day's settlements, or null).

    `daily` is `gr.reference.daily`; `settled` is `gr.reference.funding`. A
    settlement at midnight closes the day before, as in `gr.carry.daily`.
    """
    utils.require(daily, ("ticker", "ts", "close_ts", "close", "volume"), "load daily bars with gr.reference.daily")
    p = (
        utils.lazy(daily)
        .select("ticker", "ts", "close_ts", pl.col("close").cast(pl.Float64), pl.col("volume").cast(pl.Float64))
        .unique(["ticker", "ts"], keep="last")
        .sort("ticker", "ts")
        .with_columns(
            pl.when(timeseries.contiguous()).then(timeseries.simple_return()).alias("ret"),
            (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
            (pl.int_range(pl.len()) + 1).over("ticker").alias("bars"),
        )
        .drop("volume")
    )
    if settled is None:
        p = p.with_columns(pl.lit(None, pl.Float64).alias("funding"))
    else:
        utils.require(settled, ("ticker", "ts", "rate"), "load funding with gr.reference.funding")
        settles = pl.col("ts").dt.truncate("1h")
        f = (
            utils.lazy(settled)
            .unique(["ticker", "ts"])
            .group_by("ticker", (settles - pl.duration(microseconds=1)).dt.truncate("1d").alias("ts"))
            .agg(pl.col("rate").sum().alias("funding"))
        )
        p = p.join(f, on=["ticker", "ts"], how="left")
    return p.sort("ticker", "ts").collect()


def members(panel_: pl.DataFrame, *, exclude: Iterable[str] = INDEX_AND_STABLE, top: int = 50, history: int = 90, window: int = 30) -> pl.DataFrame:
    """Per (ticker, day): `dv` (mean dollar volume over `window` days), `eligible`, `traded` (in the day's top `top` by dv)."""
    utils.require(panel_, ("ticker", "ts", "dollar_volume", "bars"), "build the panel with gr.factors.panel")
    out = set(exclude)
    m = (
        panel_.select("ticker", "ts", "dollar_volume", "bars")
        .sort("ticker", "ts")
        .with_columns(pl.col("dollar_volume").rolling_mean(window).over("ticker").alias("dv"))
        .with_columns(((pl.col("bars") >= history) & pl.col("dv").is_not_null() & ~pl.col("ticker").is_in(list(out))).alias("eligible"))
        .with_columns(
            pl.when(pl.col("eligible")).then(pl.col("dv")).rank("ordinal", descending=True).over("ts").alias("_rank")
        )
        .with_columns((pl.col("eligible") & (pl.col("_rank") <= top)).fill_null(False).alias("traded"))
        .select("ticker", "ts", "dv", "eligible", "traded")
    )
    return m


def scores(panel_: pl.DataFrame, members_: pl.DataFrame, *, market: str = "BTC") -> pl.DataFrame:
    """Per (ticker, day), every registered score, higher meaning long: `mom 7`, `mom 30`, `mom 90`, `lowvol`, `resmom`, `small`.

    A window that holds a hole (a null return) is null, never shorter.
    `resmom` is the 30-day return less β × `market`'s, β being the 90-day
    covariance over the variance of `market`'s returns.
    """
    lr = (1 + pl.col("ret")).log()
    mkt = panel_.filter(pl.col("ticker") == market).select("ts", pl.col("ret").alias("_m"))
    if mkt.is_empty():
        raise Refused(f"the panel holds no {market} for the residual momentum's market")
    s = (
        panel_.sort("ticker", "ts")
        .join(mkt, on="ts", how="left")
        .with_columns(
            *[(lr.rolling_sum(L).over("ticker").exp() - 1).alias(f"mom {L}") for L in (7, 30, 90)],
            (-pl.col("ret").rolling_std(30).over("ticker")).alias("lowvol"),
            ((1 + pl.col("_m")).log().rolling_sum(30).over("ticker").exp() - 1).alias("_m30"),
            ((pl.col("ret") * pl.col("_m")).rolling_mean(90).over("ticker")
             - pl.col("ret").rolling_mean(90).over("ticker") * pl.col("_m").rolling_mean(90).over("ticker")).alias("_cov"),
            pl.col("_m").rolling_var(90, ddof=0).over("ticker").alias("_var"),
        )  # fmt: skip
        .with_columns(
            (pl.col("mom 30") - pl.when(pl.col("_var") > 0).then(pl.col("_cov") / pl.col("_var")) * pl.col("_m30")).alias("resmom"),
        )
        .join(members_.select("ticker", "ts", "dv"), on=["ticker", "ts"], how="left")
        .with_columns((-pl.col("dv")).alias("small"))
        .select("ticker", "ts", "mom 7", "mom 30", "mom 90", "lowvol", "resmom", "small")
    )
    return s


def _day(value) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise Refused(f"{value} has no time zone; give UTC")
    return value


def rebalances(members_: pl.DataFrame, start, end, *, top: int = 50, every: int = 7) -> list[datetime]:
    """The rebalance days in `[start, end)`: from the first with a full traded set, every `every` days."""
    lo, hi = _day(start), _day(end)
    full = (
        members_.filter(pl.col("traded") & (pl.col("ts") >= lo) & (pl.col("ts") < hi))
        .group_by("ts")
        .len()
        .filter(pl.col("len") >= top)
        .sort("ts")
    )
    if full.is_empty():
        raise Refused(f"no day in [{lo:%Y-%m-%d}, {hi:%Y-%m-%d}) has a traded set of {top}")
    first = full["ts"][0]
    out, day = [], first
    while day < hi:
        out.append(day)
        day += timedelta(days=every)
    return out


def weights(members_: pl.DataFrame, scores_: pl.DataFrame, score: str | None, on: list[datetime], side: str, *, fifth: float = 0.2) -> pl.DataFrame:
    """Per rebalance day and ticker, the weight decided at its close: `rebalance, ticker, w`.

    `score=None` is the equal-weight traded set. Ranked among the traded
    coins with a score; the top and bottom fifth are round(n × `fifth`) names.
    """
    if side not in ("long_short", "long_only"):
        raise Refused(f"side={side!r} is not one of long_short, long_only")
    days = pl.DataFrame({"ts": on}, schema={"ts": members_.schema["ts"]})
    traded = members_.filter(pl.col("traded")).join(days, on="ts", how="inner")
    if score is None:
        return traded.select(pl.col("ts").alias("rebalance"), "ticker", (1.0 / pl.len().over("ts")).alias("w"))
    ranked = (
        traded.join(scores_.select("ticker", "ts", pl.col(score).alias("_s")), on=["ticker", "ts"], how="left")
        .filter(pl.col("_s").is_not_null() & pl.col("_s").is_finite())
        .with_columns(
            pl.col("_s").rank("ordinal", descending=True).over("ts").alias("_r"),
            pl.len().over("ts").alias("_n"),
        )
        .with_columns((pl.col("_n") * fifth).round().cast(pl.Int64).clip(lower_bound=1).alias("_k"))
    )
    top = pl.col("_r") <= pl.col("_k")
    bottom = pl.col("_r") > pl.col("_n") - pl.col("_k")
    if side == "long_only":
        w = pl.when(top).then(1.0 / pl.col("_k"))
    else:
        w = pl.when(top).then(0.5 / pl.col("_k")).when(bottom).then(-0.5 / pl.col("_k"))
    return ranked.select(pl.col("ts").alias("rebalance"), "ticker", w.alias("w")).filter(pl.col("w").is_not_null())


def portfolio(panel_: pl.DataFrame, w: pl.DataFrame, on: list[datetime], end, name: str, *, fee: float = BINANCE_TAKER) -> pl.DataFrame:
    """The daily returns of holding each rebalance's weights from the next day to the next rebalance, as a trial frame.

    Columns: `trial, ticker ("universe"), ts, close_ts, position` (gross
    exposure), `bar_return` (null), `gross, cost, funding, net`, and
    `funding_missing` (position-days held with no funding archive).
    """
    hi = _day(end)
    schema_ts = panel_.schema["ts"]
    held = []
    for i, r in enumerate(on):
        until = on[i + 1] if i + 1 < len(on) else hi
        held.append(pl.DataFrame({"ts": pl.datetime_range(r + timedelta(days=1), until, "1d", eager=True, time_zone="UTC")}).with_columns(
            pl.lit(r).cast(schema_ts).alias("rebalance")
        ))  # fmt: skip
    days = pl.concat(held).with_columns(pl.col("ts").cast(schema_ts)).filter(pl.col("ts") < hi)
    days = days.with_columns(pl.col("rebalance").cast(schema_ts))
    book = days.join(w.with_columns(pl.col("rebalance").cast(schema_ts)), on="rebalance", how="inner")
    rows = book.join(panel_.select("ticker", "ts", "ret", "funding"), on=["ticker", "ts"], how="left")
    daily = rows.group_by("ts", maintain_order=False).agg(
        (pl.col("w") * pl.col("ret").fill_null(0.0)).sum().alias("gross"),
        (pl.col("w") * pl.col("funding").fill_null(0.0)).sum().alias("funding"),
        pl.col("w").abs().sum().alias("position"),
        (pl.col("funding").is_null() & (pl.col("w") != 0)).sum().alias("funding_missing"),
    )
    # Turnover at each rebalance, against the weights held before it; charged on its first day held.
    wide = w.with_columns(pl.col("rebalance").cast(schema_ts))
    follows = pl.DataFrame({"rebalance": on[1:], "_prev": on[:-1]}, schema={"rebalance": schema_ts, "_prev": schema_ts})
    before = follows.join(wide.rename({"rebalance": "_prev", "w": "_old"}), on="_prev", how="inner").drop("_prev")
    turn = (
        wide.join(before, on=["rebalance", "ticker"], how="full", coalesce=True)
        .with_columns((pl.col("w").fill_null(0.0) - pl.col("_old").fill_null(0.0)).abs().alias("_t"))
        .group_by("rebalance")
        .agg(pl.col("_t").sum().alias("turnover"))
        .with_columns((pl.col("rebalance") + pl.duration(days=1)).alias("ts"))
    )
    out = (
        days.select("ts")
        .join(daily, on="ts", how="left")
        .join(turn.select("ts", "turnover"), on="ts", how="left")
        .with_columns(pl.col("gross", "funding", "position").fill_null(0.0), pl.col("turnover").fill_null(0.0), pl.col("funding_missing").fill_null(0))
        .with_columns((pl.col("turnover") * fee).alias("cost"))
        .with_columns((pl.col("gross") - pl.col("cost") - pl.col("funding")).alias("net"))
        .sort("ts")
    )
    return out.select(
        pl.lit(name).alias("trial"),
        pl.lit(UNIVERSE).alias("ticker"),
        "ts",
        (pl.col("ts") + pl.duration(days=1)).alias("close_ts"),
        "position",
        pl.lit(None, pl.Float64).alias("bar_return"),
        "gross",
        "cost",
        "funding",
        "net",
        "turnover",
        "funding_missing",
    )


SCORES = ("mom 7", "mom 30", "mom 90", "lowvol", "resmom", "small")
BENCHMARK = "ew universe"


def rules(panel_: pl.DataFrame, members_: pl.DataFrame, start, end, *, fee: float = BINANCE_TAKER, every: int = 7, top: int = 50) -> pl.DataFrame:
    """The registered 12 (each score, `long_short` and `long_only`) and the equal-weight traded set, over `[start, end)`."""
    on = rebalances(members_, start, end, top=top, every=every)
    s = scores(panel_, members_)
    frames = [portfolio(panel_, weights(members_, s, None, on, "long_only"), on, end, BENCHMARK, fee=fee)]
    for score in SCORES:
        for side in ("long_short", "long_only"):
            frames.append(portfolio(panel_, weights(members_, s, score, on, side), on, end, f"{score} {side}", fee=fee))
    return pl.concat(frames)
