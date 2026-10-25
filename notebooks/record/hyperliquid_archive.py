import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from datetime import UTC, datetime

    import marimo as mo
    import polars as pl

    import galata_research as gr

    return UTC, datetime, gr, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # Hyperliquid's archive against the record

    `hyperliquid-s3` is Hyperliquid's own history, built from the fills it
    publishes on S3 (`galata-fetch candles hyperliquid-s3 ...`), one 1m bar
    per traded minute, apart from the record. Where the record also holds a
    ticker, from 2026-09-20, the two are compared minute by minute outside
    both sides' gaps: prices exact, volume to 1e-9, trade counts equal.
    `high` and `low` are compared; `high_all` and `low_all`, which keep
    liquidation prices, are not, since the venue's candles leave them out.
    """)
    return


@app.cell
def _(gr, mo, pl):
    VENUE = "hyperliquid-s3"
    try:
        cover = gr.reference.coverage().filter(pl.col("venue") == VENUE)
    except gr.Refused as _refused:
        cover = pl.DataFrame()
        mo.output.replace(mo.md(f"No reference store: {_refused}"))
    mo.stop(cover.is_empty(), mo.md(f"The reference store holds no `{VENUE}` days; fetch them with `galata-fetch candles {VENUE} ...`."))
    mo.vstack([mo.md("## What the store holds"), mo.md("`days_partial` are days with a gap; `gaps` below says where."), cover])
    return VENUE, cover


@app.cell
def _(UTC, VENUE, cover, datetime, gr, mo, pl):
    first = datetime.combine(cover["first"].min(), datetime.min.time(), UTC)
    last = datetime.combine(cover["last"].max(), datetime.max.time(), UTC)
    holes = gr.reference.gaps(None, first, last, venues=[VENUE])
    by_cause = holes.group_by("cause").agg(pl.len().alias("gaps"), (pl.col("to_ts") - pl.col("from_ts")).sum().alias("span")).sort("cause")
    mo.vstack([mo.md(f"## Gaps, {first:%Y-%m-%d} to {last:%Y-%m-%d}"), by_cause if holes.height else mo.md("None: every hour was read."), holes.head(50)])
    return first, last


@app.cell
def _(UTC, cover, datetime, first, gr, last, mo, pl):
    # The record's live 1m bars begin 2026-09-20; the six tickers it captures are compared, the rest stated.
    since = max(first, datetime(2026, 9, 20, tzinfo=UTC))
    rows, stated = [], []
    for _ticker in sorted(cover["ticker"]):
        _got = gr.reference.compare_record(_ticker, since, last)
        if _got["statement"]:
            stated.append(f"- **{_ticker}**: {_got['statement']}")
            continue
        rows.append({"ticker": _ticker, "both": _got["both"], "reference_only": _got["reference_only"],
                     "record_only": _got["record_only"], "differ": _got["differences"].height})  # fmt: skip
    table = pl.DataFrame(rows, schema={"ticker": pl.String, "both": pl.Int64, "reference_only": pl.Int64, "record_only": pl.Int64, "differ": pl.Int64})
    mo.vstack(
        [
            mo.md(f"## Against the record, from {since:%Y-%m-%d}"),
            mo.md("`differ` counts minutes held by both whose open, high, low, close, volume or trade count disagree."),
            table,
            mo.md("\n".join(stated)) if stated else mo.md(""),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
