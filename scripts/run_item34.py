"""Standalone run of har_long.py item-34 section, without marimo."""
from datetime import datetime, timezone
import galata_research as gr
import polars as pl

# ── Parameters (from the notebook) ─────────────────────────────────────────
TICKERS = ("BTC", "ETH")
BARS = ("1d", "4h")
FINE_A = {"1d": "4h", "4h": "1h"}
SAMPLE = (datetime(2020, 1, 1, tzinfo=timezone.utc), datetime(2026, 9, 27, tzinfo=timezone.utc))
SPLIT = "2024-09-01T00:00:00+00:00"
EVERY = {"1d": 5, "4h": 6}
HORIZONS = {"1d": (1, 7, 30), "4h": (1, 6, 42)}

# ── Load bars ───────────────────────────────────────────────────────────────
print("Loading bars...")
bars = {}
for t in TICKERS:
    m = gr.reference.candles(t, *SAMPLE, venues="binance-um").collect()
    for width in sorted(set(BARS) | set(FINE_A.values()) | {"5m"}):
        bars[(t, width)] = gr.timeseries.resample(m, width)
print(f"Loaded {len(bars)} bar sets")

# ── Jump walks ─────────────────────────────────────────────────────────────
from galata_research.models import vol as _vol

print("Running jump walks...")
parts, skipped = [], []
for t in TICKERS:
    for i in BARS:
        try:
            jm = _vol.realized_jumps(bars[(t, "5m")], i)
        except gr.Refused as why:
            skipped.append({"ticker": t, "interval": i, "model": "all", "why": str(why)})
            continue
        for model in ("har", "harj", "harcj", "hartcj"):
            try:
                if model == "har":
                    measures = gr.timeseries.realized_from(bars[(t, "5m")], i)
                else:
                    measures = jm
                f = _vol.har(measures, model=model, split=SPLIT, every=EVERY[i], horizons=HORIZONS[i])
                parts.append(f.with_columns(pl.lit(t).alias("ticker"), pl.lit(i).alias("interval"), pl.lit(model).alias("model")))
            except gr.Refused as why:
                skipped.append({"ticker": t, "interval": i, "model": model, "why": str(why)})

jump_walked = pl.concat(parts, how="diagonal_relaxed")
jump_skipped = pl.DataFrame(skipped, schema={"ticker": pl.String, "interval": pl.String, "model": pl.String, "why": pl.String})
print(f"Forecasts: {jump_walked.height:,}; skipped: {jump_skipped.height}")

# ── Jump scales ────────────────────────────────────────────────────────────
print("Computing jump scales...")
_rows = []
_cut = pl.lit(SPLIT).str.to_datetime(time_zone="UTC")
for _t in TICKERS:
    for _i in BARS:
        _r2 = gr.timeseries.returns(bars[(_t, _i)], kind="log").select("ts", (pl.col("return") ** 2).alias("r2"))
        _m = gr.timeseries.realized_from(bars[(_t, "5m")], _i).select("ts", "close_ts", "rv")
        _j = _m.join(_r2, on="ts").drop_nulls().filter(pl.col("close_ts") <= _cut)
        _rows.append({"ticker": _t, "interval": _i, "c": _j["r2"].sum() / _j["rv"].sum(), "bars": _j.height})
jump_scales = pl.DataFrame(_rows)
print("Scales computed")

# ── Verdicts ────────────────────────────────────────────────────────────────
print("Scoring...")
ev = gr.models.evaluate
_rows = []
for _t in TICKERS:
    for _i in BARS:
        _r2_proxy = ev.proxies(bars[(_t, _i)], "r2")
        _rv5_proxy = gr.timeseries.realized_from(bars[(_t, "5m")], _i).select("ticker", "ts", pl.col("rv").alias("proxy"))
        _c = jump_scales.filter((pl.col("ticker") == _t) & (pl.col("interval") == _i))["c"][0]
        _rv5_proxy = _rv5_proxy.with_columns((pl.col("proxy") * _c).alias("proxy"))
        for _proxy_name, _proxy in (("r2", _r2_proxy), ("rv5", _rv5_proxy)):
            _f = jump_walked.filter((pl.col("ticker") == _t) & (pl.col("interval") == _i))
            _al = ev.align(_f, _proxy)
            _card = ev.scorecard(_al, benchmark="har")
            _q = _card.select("model", "h", "qlike")
            for _model, _hyp in (("harcj", "H7"), ("hartcj", "H8")):
                _m_q = _q.filter(pl.col("model") == _model).select("h", pl.col("qlike").alias("m"))
                _h_q = _q.filter(pl.col("model") == "har").select("h", pl.col("qlike").alias("q"))
                _j = _m_q.join(_h_q, on="h").sort("h")
                _wins = int((_j["m"] < _j["q"]).sum())
                _v = "consistent" if _wins == _j.height else "contradicts" if _wins == 0 else "mixed"
                _rows.append({"#": _hyp, "ticker": _t, "bars": _i, "proxy": _proxy_name, "verdict": _v,
                              "measured": f"{_wins} of {_j.height} horizons"})
            if _proxy_name == "r2":
                try:
                    _p = ev.uspa(_al, model="hartcj", benchmark="har", reps=499)["p_value"]
                    _rows.append({"#": "H9", "ticker": _t, "bars": _i, "proxy": _proxy_name,
                                  "verdict": "yes" if _p < 0.05 else "no", "measured": f"uSPA p {_p:.3f}"})
                except Exception as _why:
                    _rows.append({"#": "H9", "ticker": _t, "bars": _i, "proxy": _proxy_name,
                                  "verdict": "can't tell", "measured": str(_why)[:80]})

jump_verdicts = pl.DataFrame(_rows).sort("#", "ticker", "bars", "proxy")
print("Done")

# ── Summary ────────────────────────────────────────────────────────────────
_h7 = jump_verdicts.filter(pl.col("#") == "H7")
_h8 = jump_verdicts.filter(pl.col("#") == "H8")
_h9 = jump_verdicts.filter(pl.col("#") == "H9")
_h7_r2 = _h7.filter(pl.col("proxy") == "r2")["verdict"].to_list()
_h7_rv5 = _h7.filter(pl.col("proxy") == "rv5")["verdict"].to_list()
_h8_r2 = _h8.filter(pl.col("proxy") == "r2")["verdict"].to_list()
_h8_rv5 = _h8.filter(pl.col("proxy") == "rv5")["verdict"].to_list()
_h7_match = sum(a == b for a, b in zip(_h7_r2, _h7_rv5, strict=True))
_h8_match = sum(a == b for a, b in zip(_h8_r2, _h8_rv5, strict=True))
_total_match = _h7_match + _h8_match
_h10_v = "robust" if _total_match >= 6 else "proxy-dependent"
_h7_n = _h7.filter(pl.col("proxy") == "r2").filter(pl.col("verdict") == "consistent").height
_h8_n = _h8.filter(pl.col("proxy") == "r2").filter(pl.col("verdict") == "consistent").height

print("\n=== ITEM 34 VERDICTS ===")
print(jump_verdicts.to_pandas().to_string(index=False))
print(f"\nH7 (HAR-CJ vs HAR): {_h7_n} of 4 cells consistent under r²")
print(f"H8 (HAR-TCJ vs HAR): {_h8_n} of 4 cells consistent under r²")
print(f"H10 (proxy robustness): {_h10_v} ({_h7_match + _h8_match} of 4 verdicts match across proxies)")
