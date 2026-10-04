import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import polars as pl

    import galata_research as gr

    sys.path.insert(0, str(Path(__file__).parent))
    return gr, mo, pl


@app.cell
def _(mo):
    mo.md(r"""
    # The liquidity study: what survives

    The study's other notebooks each answered a question. This one sets
    their answers against what the literature and the data vendors claim,
    claim by claim.

    **Every verdict below is computed from those notebooks' own results**
    (run here with `app.embed()`, BTC unless stated). None is written by
    hand, and each verdict's rule is stated beside it:

    - **consistent**: this record agrees;
    - **contradicts**: it disagrees;
    - **mixed**: it agrees on one ticker or measure and not another;
    - **can't tell**: the record cannot test it.

    A study that reports only the claims it confirms is the flattering result
    this repository exists to remove. The claims that go against the study's
    own framing are kept.
    """)
    return


@app.cell
async def _():
    import liquidity as _week
    import liquidity_clock as _clock
    import liquidity_costs as _costs
    import liquidity_stress as _stress
    import liquidity_venues as _venues

    week = (await _week.app.embed()).defs
    costs = (await _costs.app.embed()).defs
    clock = (await _clock.app.embed()).defs
    venues = (await _venues.app.embed()).defs
    stress = (await _stress.app.embed()).defs
    return clock, costs, stress, venues, week


@app.cell
def _(gr, pl):
    def _eth_jumps():
        # ⑦ for ETH, which the embedded notebook (on BTC) does not run: jump counts only.
        k = gr.reference.candles("ETH", "2019-01-01T00:00Z", "2100-01-01T00:00Z", venues="binance-um").collect()
        bars = (
            k.group_by("ticker", pl.col("ts").dt.truncate("5m").alias("t5"))
            .agg(pl.col("open").first(), pl.col("close").last(), pl.len().alias("n"))
            .filter(pl.col("n") == 5).rename({"t5": "ts"})
            .with_columns(pl.col("ts").dt.offset_by("5m").alias("close_ts")).sort("ts")
        )  # fmt: skip
        r = gr.timeseries.returns(bars, kind="log")
        return int(gr.jumps.lee_mykland(r, periodicity=gr.jumps.periodicity(r, slot="5m"))["jump"].sum())

    def _eth_elasticity():
        # ⑱ for ETH: ±1% depth against realized variance, the day's shape off.
        depth = (
            gr.reference.depth("ETH", "2023-01-01T00:00Z", "2100-01-01T00:00Z", venues="binance-um")
            .filter(pl.col("band_pct").abs() == 1.0).group_by("ts").agg(pl.col("notional").sum())
            .group_by(pl.col("ts").dt.truncate("1h")).agg(pl.col("notional").median().alias("depth"), pl.len().alias("n"))
            .filter(pl.col("n") >= 100).drop("n").collect()
        )  # fmt: skip
        k = gr.reference.candles("ETH", "2023-01-01T00:00Z", "2100-01-01T00:00Z", venues="binance-um").collect()
        rv = (
            gr.timeseries.returns(k, kind="log").group_by(pl.col("ts").dt.truncate("1h"))
            .agg((pl.col("return") ** 2).sum().alias("rv"), pl.col("return").is_not_null().sum().alias("n"))
            .filter((pl.col("n") >= 58) & (pl.col("rv") > 0)).drop("n")
        )  # fmt: skip
        return gr.timeseries.log_elasticity(depth.join(rv, on="ts").filter(pl.col("depth") > 0), "depth", "rv")

    eth_jumps, eth_elasticity = _eth_jumps(), _eth_elasticity()
    return eth_elasticity, eth_jumps


@app.cell
def _(clock, costs, eth_elasticity, eth_jumps, pl, stress, venues, week):
    rows = []

    def claim(n, source, says, rule, measured, verdict):
        rows.append({"#": n, "source": source, "the claim": says, "rule": rule, "this record": measured, "verdict": verdict})

    # 1–3: the day's shape (②, ⑩)
    _peaks = week["peaks"]
    _vol_hr = _peaks.filter((pl.col("venue") == "binance-um") & (pl.col("measure") == "volume $"))["best hour"][0]
    claim(1, "Brauneis, Mestel & Theissen 2025 (online 2024)", "activity peaks at 16–17 UTC; US-hours peak 13:30–17 UTC", "Binance volume's best hour in 13–17 UTC",
          f"best hour {_vol_hr} UTC", "consistent" if 13 <= _vol_hr <= 17 else "contradicts")  # fmt: skip
    _depth_hr = _peaks.filter((pl.col("venue") == "binance-um") & (pl.col("measure") == "depth ±1.0%"))["best hour"][0]
    claim(2, "Amberdata 2025; Talos", "depth best, cost lowest around 11–13 UTC", "Binance ±1% depth's best hour in 9–13 UTC",
          f"best hour {_depth_hr} UTC", "consistent" if 9 <= _depth_hr <= 13 else "contradicts")  # fmt: skip
    _wk = costs["size_costs"].filter(pl.col("size") == 1e6).group_by("weekend").agg(pl.col("cost_bps").median())
    _we, _wd = (_wk.filter(pl.col("weekend") == v)["cost_bps"][0] for v in (True, False))
    claim(3, "Amberdata 2025", "weekend depth slightly above weekday depth", "the $1M cost on weekends at or below weekdays",
          f"weekend {_we:.3f} bps, weekday {_wd:.3f} bps", "consistent" if _we <= _wd else "contradicts")  # fmt: skip

    # 4–5: whose clock (⑥)
    _l = clock["lineup"].filter((pl.col("measure") == "variance ÷ day") & (pl.col("year") >= 2023))
    _shift = _l.filter((pl.col("peak UTC, US daylight") == 14) & (pl.col("peak UTC, US standard") == 15)).height
    claim(4, "CryptoSlate (Kraken, 2022–25)", "the most volatile hour is 14 UTC in US daylight time, 15 UTC in standard time",
          "years since 2023 with peaks 14 → 15", f"{_shift} of {_l.height} years", "consistent" if _shift >= _l.height / 2 else "mixed")  # fmt: skip
    _h = clock["holidays"].filter((pl.col("sample") == "2022 on") & (pl.col("measure") == "variance")).row(0, named=True)
    _hi = float(_h["90% band"].split("…")[1])
    claim(5, "CryptoSlate (Kraken, 2022–25)", "US-hours variance share drops on US holidays (55.7% → 41.9%)", "NYSE-closed weekdays lower, band below 0",
          f"{_h['share, NYSE open']:.1%} → {_h['share, NYSE closed']:.1%} ({_h['90% band']})", "consistent" if _hi < 0 else "contradicts")  # fmt: skip

    # 6–9: jumps and releases (⑦, ⑧)
    _t = clock["tested"].filter(pl.col("jump"))
    _neg = (_t["return"] < 0).mean()
    claim(6, "Saef, Nagy, Sizov & Härdle 2024, Digital Finance (tick data, 7 exchanges)", "negative jumps dominate", "share of negative jumps above 0.55",
          f"{_neg:.1%} negative", "consistent" if _neg > 0.55 else "contradicts")  # fmt: skip
    _ratio = eth_jumps / _t.height
    claim(7, "Ben Omrane, Guesmi, Qianru & Saadi 2023, Annals of OR 330 (5m, 2016–19)", "ETH jumps about 3× as often as BTC", "ETH ÷ BTC jump count at least 2",
          f"ETH {eth_jumps:,} ÷ BTC {_t.height:,} = {_ratio:.2f}", "consistent" if _ratio >= 2 else "contradicts")  # fmt: skip
    _top = _t.group_by("ny_minute").len().sort("len", descending=True)["ny_minute"][0]
    claim(8, "Wątorek et al. 2023", "bursts at US macro release times", "the busiest adjusted-jump slot is 08:30 New York",
          f"busiest slot {_top // 60:02d}:{_top % 60:02d} New York", "consistent" if _top == 510 else "contradicts")  # fmt: skip
    _f = clock["event_table"].filter(pl.col("event") == "fomc").row(0, named=True)
    claim(9, "Yang & Wang 2026, Finance Research Letters 101", "volume in the hour after an FOMC statement 2.54× (BTC), 2.81× (ETH) a matched week", "FOMC volume ratio at the peak 5-minute bin at least 2",
          f"{_f['volume ÷ base at peak']:.1f}× at minute {_f['at minute']} (5-minute bins)", "consistent" if _f["volume ÷ base at peak"] >= 2 else "contradicts")  # fmt: skip

    # 10: bar spread estimators (⑤)
    _all = costs["calibration"]
    _c = _all.filter(pl.col("verdict") == "calibrates")
    _flat = _all.filter(pl.col("verdict") == "nothing to calibrate against").height == _all.height
    claim(10, "Brauneis, Mestel, Riordan & Theissen 2021", "bar estimators (Corwin–Schultz, Abdi–Ranaldo) track variation better than levels",
          "an estimator follows the hours (ρ ≥ 0.6) while its level is off by more than 1.5×",
          ", ".join(f"{r['estimator']} ρ {r['spearman ρ vs quoted']} at {r['estimate ÷ quoted']}×" for r in _c.iter_rows(named=True))
          or ("the measured spread barely moves by hour (max ÷ min < 1.1): " + ", ".join(
              f"{r['venue']} corwin_schultz ρ {r['spearman ρ vs quoted']}" for r in _all.filter(pl.col("estimator") == "corwin_schultz").iter_rows(named=True))),
          "can't tell" if _flat else ("consistent" if _c.filter(pl.col("estimate ÷ quoted") > 1.5).height and _c.height < 4 else ("mixed" if _c.height else "contradicts")))  # fmt: skip

    # 11: venues (⑨, ⑬)
    _o = venues["overall"].row(0, named=True)
    _y = venues["discovery_days"].filter((pl.col("pair") == "binance vs bybit") & (pl.col("every") == "100ms"))
    _ils = _y.group_by(pl.col("day").dt.year().alias("y")).agg(pl.col("ils").median()).sort("y")
    claim(11, "price-discovery literature (larger venue leads)", "the larger venue leads the smaller", "Binance ahead of Bybit in 90% of hours",
          f"ahead in {_o['Binance ahead, share']:.1%} of hours; ILS {_ils['ils'][0]:.2f} ({_ils['y'][0]}) → {_ils['ils'][-1]:.2f} ({_ils['y'][-1]})",
          "consistent" if _o["Binance ahead, share"] > 0.9 else "contradicts")  # fmt: skip

    # 12–16: forecasting, execution, stress (⑮–⑱)
    _s = stress["forecast_scores"].filter((pl.col("against") == "persistence") & (pl.col("model") == "decomposition"))
    _ok = _s.filter((pl.col("r2_oos") > 0) & (pl.col("dm_p") < 0.05)).height
    claim(12, "Bialkowski, Darolles & Le Fol 2008", "a seasonal-plus-dynamic decomposition forecasts intraday volume", "beats persistence out of sample, DM p < 0.05, on depth and volume",
          f"{_ok} of {_s.height} series; R²oos {', '.join(f'{v:.2f}' for v in _s['r2_oos'])}", "consistent" if _ok == _s.height else "mixed")  # fmt: skip
    _e = costs["dv"]
    import galata_research as _gr

    _btc = _gr.timeseries.log_elasticity(_e, "depth", "rv")
    _sig = lambda e: e["beta"] < 0 and e["t"] < -3  # noqa: E731
    claim(13, "Roşu 2009; Obizhaeva & Wang 2013 (as reviewed in Angerer, Gramlich & Hanke 2025, JRFM)", "thin books go with higher short-term volatility", "depth's elasticity to realized variance negative, t < −3",
          f"BTC {_btc['beta']:+.3f} (t {_btc['t']:.0f}); ETH {eth_elasticity['beta']:+.3f} (t {eth_elasticity['t']:.1f})",
          "consistent" if _sig(_btc) and _sig(eth_elasticity) else ("mixed" if _sig(_btc) or _sig(eth_elasticity) else "contradicts"))  # fmt: skip
    _p = stress["schedule_table"].filter(pl.col("order") == "$10M")
    _hind = _p.filter(pl.col("plan") == "hindsight")["mean saving vs TWAP"][0]
    _prof = _p.filter(pl.col("plan") == "profile")["mean saving vs TWAP"][0]
    claim(14, "Almgren & Chriss 2000, time-varying liquidity", "trading in proportion to liquidity beats TWAP", "hindsight saving above 0; material if above 5%",
          f"hindsight {_hind:.2%}, profile {_prof:.2%}, adaptive {_p.filter(pl.col('plan') == 'adaptive')['mean saving vs TWAP'][0]:.2%}",
          "consistent, immaterial" if 0 < _hind < 0.05 else ("consistent" if _hind >= 0.05 else "contradicts"))  # fmt: skip
    _half = costs["half_life"]
    claim(15, "Obizhaeva & Wang 2013", "the optimal execution depends on resilience, not on static spread or depth", "not testable here: no executions of known size",
          f"after a shock, the depth dent does not halve within 60 s ({'no half-life' if _half is None else f'{_half} s'})", "can't tell")  # fmt: skip
    _low = stress["crash_table"]["depth low ÷ base"].median()
    claim(16, "Kaiko (FTX, October 2025)", "depth collapses in liquidity crises (≈ halved)", "median depth low point across ten extreme hours at most 0.6 of baseline",
          f"median low {_low:.2f} of the week before", "consistent" if _low <= 0.6 else "contradicts")  # fmt: skip

    verdicts = pl.DataFrame(rows)
    return (verdicts,)


@app.cell
def _(mo, pl, verdicts):
    _counts = verdicts.group_by("verdict").len().sort("len", descending=True)
    mo.vstack([
        mo.md("## What survives"),
        mo.md(", ".join(f"**{r['len']} {r['verdict']}**" for r in _counts.iter_rows(named=True))),
        mo.ui.table(verdicts.select("#", "the claim", "this record", "verdict"), page_size=20, selection=None),
        mo.md("The source and the rule of each claim:"),
        mo.ui.table(verdicts.select("#", "source", "rule"), page_size=20, selection=None),
    ])  # fmt: skip
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## References

    Each was checked against its primary source on 2026-09-28; vendor and
    press pages are marked. The check corrected two claims. The FOMC volume
    multiples are the paper's abstract figures (2.54× and 2.81×), not the 2.39×
    and 2.77× first carried. And "thin books go with higher volatility" is the
    literature Angerer, Gramlich and Hanke review (Roşu; Obizhaeva and Wang),
    not their own finding.

    - Abdi, F. and Ranaldo, A. (2017). A simple estimation of bid–ask spreads from daily close, high, and low prices. *Review of Financial Studies* 30(12).
    - Almgren, R. and Chriss, N. (2000). Optimal execution of portfolio transactions. *Journal of Risk* 3(2).
    - Amberdata (vendor, 2025). The rhythm of liquidity: temporal patterns in market depth.
    - Angerer, M., Gramlich, M. and Hanke, M. (2025). Order book liquidity on crypto exchanges. *Journal of Risk and Financial Management* 18(3), 124 (cited for its review).
    - Ardia, D., Guidotti, E. and Kroencke, T. A. (2024). Efficient estimation of bid–ask spreads from open, high, low, and close prices. *Journal of Financial Economics* 161, 103916.
    - Ben Omrane, W., Guesmi, K., Qianru, Q. and Saadi, S. (2023; online 2021). The high-frequency impact of macroeconomic news on jumps and co-jumps in the cryptocurrency markets. *Annals of Operations Research* 330, 177–209.
    - Bialkowski, J., Darolles, S. and Le Fol, G. (2008). Improving VWAP strategies: a dynamic volume approach. *Journal of Banking & Finance* 32(9), 1709–1722.
    - Boudt, K., Croux, C. and Laurent, S. (2011). Robust estimation of intraweek periodicity in volatility and jump detection. *Journal of Empirical Finance* 18(2).
    - Brauneis, A., Mestel, R. and Theissen, E. (2025; online 2024). The crypto world trades at tea time. *Review of Quantitative Finance and Accounting* 64(1), 275–304.
    - Brauneis, A., Mestel, R., Riordan, R. and Theissen, E. (2021). How to measure the liquidity of cryptocurrency markets? *Journal of Banking & Finance* 124, 106041.
    - Brownlees, C., Cipollini, F. and Gallo, G. M. (2011). Intra-daily volume modeling and prediction for algorithmic trading. *Journal of Financial Econometrics* 9(3), 489–518.
    - Corwin, S. A. and Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. *Journal of Finance* 67(2).
    - CryptoSlate (press, 2026-09-11; O. Adejumo). Crypto never closes, but Bitcoin, Ethereum, XRP and Solana now move on Wall Street time (Kraken XBT/USD, 87,672 hourly observations).
    - Gonzalo, J. and Granger, C. (1995). Estimation of common long-memory components in cointegrated systems. *Journal of Business & Economic Statistics* 13(1).
    - Hasbrouck, J. (1995). One security, many markets: determining the contributions to price discovery. *Journal of Finance* 50(4).
    - Hoffmann, M., Rosenbaum, M. and Yoshida, N. (2013). Estimation of the lead-lag parameter from non-synchronous data. *Bernoulli* 19(2), 426–461.
    - Huth, N. and Abergel, F. (2014). High frequency lead/lag relationships — empirical facts. *Journal of Empirical Finance* 26, 41–58.
    - Kaiko (vendor, 2024-06-27). BTC ETFs' impact on spot market structure.
    - Kyle, A. S. (1985). Continuous auctions and insider trading. *Econometrica* 53(6).
    - Large, J. (2007). Measuring the resiliency of an electronic limit order book. *Journal of Financial Markets* 10(1).
    - Lee, S. S. and Mykland, P. A. (2008). Jumps in financial markets: a new nonparametric test and jump dynamics. *Review of Financial Studies* 21(6), 2535–2563.
    - Obizhaeva, A. A. and Wang, J. (2013). Optimal trading strategy and supply/demand dynamics. *Journal of Financial Markets* 16(1).
    - Putniņš, T. J. (2013). What do price discovery metrics really measure? *Journal of Empirical Finance* 23, 68–83.
    - Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient market. *Journal of Finance* 39(4).
    - Roşu, I. (2009). A dynamic model of the limit order book. *Review of Financial Studies* 22(11).
    - Saef, D., Nagy, O., Sizov, S. and Härdle, W. K. (2024). Understanding temporal dynamics of jumps in cryptocurrency markets: evidence from tick-by-tick data. *Digital Finance* 6(4), 605–638 (correction 2025).
    - Szűcs, B. Á. (2017). Forecasting intraday volume: comparison of two early models. *Finance Research Letters* 21, 249–258.
    - Talos (vendor, 2026-01-14; E. Hoch). Does timing matter when trading BTC?
    - Wątorek, M., Skupień, M., Kwapień, J. and Drożdż, S. (2023). Decomposing cryptocurrency high-frequency price dynamics into recurring and noisy components. *Chaos* 33, 083146.
    - Yang and Wang (2026). Scheduled FOMC statements and intraday macro event risk in cryptocurrency markets. *Finance Research Letters* 101, 110073.
    """)
    return


if __name__ == "__main__":
    app.run()
