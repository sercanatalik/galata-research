import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from datetime import UTC, datetime

    import altair as alt
    import marimo as mo
    import polars as pl

    import galata_research as gr

    return UTC, alt, datetime, gr, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # My account

    Margin and positions per snapshot, decoded from the ledger's
    `clearinghouseState` answers, one per dex. `ts` is the venue's own time.
    On a **unified account** (`equity_held = false`), perps `account_value`
    is margin plus unrealised, 0 while flat, and the cash sits in spot USDC,
    which the ledger does not read.
    """)
    return


@app.cell
def _(UTC, datetime, gr, mo):
    EVER = datetime(2020, 1, 1, tzinfo=UTC), datetime(2100, 1, 1, tzinfo=UTC)
    margin = gr.account.margin(None, *EVER).collect()
    held = margin.group_by("venue", "account", "dex", "mode", "equity_held").len().sort("venue", "account", "dex")
    mo.vstack([mo.md(f"## {margin.height} snapshots"), held])
    return EVER, margin


@app.cell
def _(alt, margin, mo):
    figures = ["account_value", "total_margin_used", "cross_maintenance_margin_used", "withdrawable"]
    long = margin.unpivot(index=["ts", "dex"], on=figures, variable_name="figure", value_name="usd")
    chart = (
        alt.Chart(long)
        .mark_line(point=True, strokeWidth=1)
        .encode(x="ts:T", y="usd:Q", color="figure:N", row=alt.Row("dex:N", title="dex"))
        .properties(height=140, width="container")
    )
    mo.vstack([mo.md("## Margin through time, per dex"), chart])
    return


@app.cell
def _(EVER, gr, mo):
    held_positions = gr.account.positions(None, *EVER).collect()
    mo.vstack(
        [
            mo.md("## Positions"),
            held_positions if held_positions.height else mo.md("Flat in every snapshot: no position rows."),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
