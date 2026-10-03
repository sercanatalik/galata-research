import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from datetime import timedelta

    import marimo as mo
    import polars as pl

    import galata_research as gr
    from galata_research import flow

    return flow, gr, mo, pl, timedelta


@app.cell
def _(mo):
    mo.md("""
    # A pre-registered test: does the order flow pay a taker?

    Three claims, frozen in `planning/preregistered/trade-the-flow.md` and
    committed alone (`5c4e7c2`) before `gr.flow` or this notebook existed.
    Each is tested on Bybit's book, one row per second, on the first
    Wednesday of each month from October 2025 to September 2026.

    - **H1, order-flow imbalance.** Does a 10-second bucket's OFI predict the
      next bucket's return (S)? Does a taker rule on it pay (E, N = 12)?
    - **H2, the micro-price.** Does the size-weighted mid forecast the mid
      1 s and 10 s on better than the mid itself (S)?
    - **H3, Binance's lead.** Does Binance's last second predict Bybit's next
      one (S)? Does a taker on Bybit get paid for it (E, N = 18)?

    E fills at the decision second's own touch, which flatters every rule,
    and pays Bybit's 0.055% taker fee each way. Expected: H2 S supported,
    H1 and H3 S likely supported, every E not supported.
    """)
    return


@app.cell
def _(flow, gr, pl, timedelta):
    TICKERS = ["BTC", "ETH"]
    SPAN = ("2025-10-01T00:00Z", "2026-10-01T00:00Z")
    _days = (
        gr.reference.book(TICKERS, *SPAN, venues="bybit-linear")
        .select(pl.col("ts").dt.truncate("1d").alias("day"))
        .unique()
        .sort("day")
        .collect()["day"]
        .to_list()
    )

    def one_day(day):
        _lo, _hi = day.isoformat(), (day + timedelta(days=1)).isoformat()
        book = gr.reference.book(TICKERS, _lo, _hi, venues="bybit-linear").collect()
        leader = gr.reference.trades(TICKERS, _lo, _hi, venues="binance-um").select("ticker", "ts", "price").collect()
        grid = flow.seconds(book, leader)
        ofi = flow.ofi_frame(book)
        lead = flow.lead_frame(grid)
        return {
            "ofi": ofi,
            "ofi_trades": flow.ofi_rules(ofi, grid),
            "micro_1": flow.microprice_frame(grid, 1),
            "micro_10": flow.microprice_frame(grid, 10),
            "lead": lead,
            "lead_trades": flow.lead_rules(lead, grid),
            "seconds": grid.group_by("ticker").agg(pl.len().alias("seconds"), pl.col("leader_px").is_not_null().sum().alias("with_leader")).with_columns(pl.lit(day).alias("day")),
        }

    _parts = [one_day(d) for d in _days]
    pieces = {k: pl.concat([p[k] for p in _parts]) for k in _parts[0]} if _parts else {}
    days = _days
    return TICKERS, days, pieces


@app.cell
def _(days, mo, pieces, pl):
    mo.vstack(
        [
            mo.md(f"## {len(days)} days held"),
            pieces["seconds"].group_by("ticker").agg(pl.len().alias("days"), pl.col("seconds").sum(), pl.col("with_leader").sum()).sort("ticker"),
            mo.md("A second with a crossed or locked book is not a second. One with no Binance trade in the last 5 s has no leader price."),
        ]
    )
    return


@app.cell
def _(TICKERS, flow, mo, pieces, pl):
    def per_ticker(fn, frame):
        return pl.DataFrame([{"ticker": t, **fn(frame.filter(pl.col("ticker") == t))} for t in TICKERS])

    h1s = per_ticker(lambda f: flow.slope(f, "x", "y", lags=6), pieces["ofi"])
    h3s = per_ticker(lambda f: flow.slope(f, "x", "y", lags=6), pieces["lead"])
    h2 = pl.concat(
        [
            per_ticker(lambda f: flow.diebold_mariano(f, "d", lags=1 + 5), pieces["micro_1"]).with_columns(pl.lit(1).alias("horizon_s")),
            per_ticker(lambda f: flow.diebold_mariano(f, "d", lags=10 + 5), pieces["micro_10"]).with_columns(pl.lit(10).alias("horizon_s")),
        ]
    )
    s_verdicts = {
        "H1 S": bool(((h1s["slope"] > 0) & (h1s["t"] > 3)).all()),
        "H2 S": bool(((h2["mean"] > 0) & (h2["p"] < 0.01)).all()),
        "H3 S": bool(((h3s["slope"] > 0) & (h3s["t"] > 3)).all()),
    }
    mo.vstack(
        [
            mo.md("## The statistical parts"),
            mo.md("**H1 S**: the next 10 s mid return (bps) on this bucket's OFI over depth. Newey–West t within days."),
            h1s,
            mo.md("**H2 S**: the mid's squared error less the weighted mid's, in bps². Positive means the weighted mid forecast better."),
            h2,
            mo.md("**H3 S**: Bybit's next-second mid return on Binance's last-second return, both in bps."),
            h3s,
            mo.md(" · ".join(f"**{k}: {'supported' if v else 'not supported'}**" for k, v in s_verdicts.items())),
        ]
    )
    return


@app.cell
def _(flow, mo, pieces, pl):
    def verdict(trades, n_rules):
        scored = flow.score(trades)
        best = scored.drop_nulls("mean_bps").group_by("rule").agg(pl.col("mean_bps").mean().alias("_avg")).sort("_avg", descending=True)
        if best.is_empty():
            return scored, None, False
        rule = best["rule"][0]
        both = scored.filter(pl.col("rule") == rule)
        ok = both.height == 2 and bool(((both["mean_bps"] > 0) & (both["p"] < 0.05 / n_rules)).all())
        return scored, rule, ok

    h1e, h1_rule, h1_ok = verdict(pieces["ofi_trades"], 12)
    h3e, h3_rule, h3_ok = verdict(pieces["lead_trades"], 18)
    mo.vstack(
        [
            mo.md("## The economic parts: a taker at the touch, 0.055% each way"),
            mo.md("The best rule is the one with the highest mean net return per trade averaged over the two tickers."),
            mo.md(f"**H1 E** (N = 12). Best rule: `{h1_rule}`. **{'Supported' if h1_ok else 'Not supported'}.**"),
            h1e,
            mo.md(f"**H3 E** (N = 18). Best rule: `{h3_rule}`. **{'Supported' if h3_ok else 'Not supported'}.**"),
            h3e,
            mo.md(
                "Mean net bps per trade, with the t over the days' mean returns (days − 1 degrees of freedom). "
                "Support needs the best rule positive on both tickers at p < 0.05/N. `gross` is visible by "
                "adding 11 bps: what the signal earned before the fee."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
