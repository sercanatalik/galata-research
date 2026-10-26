"""Up or down over the next h: direction targets, walk-forward logistic forecasts, and the scores that judge them.

    f = gr.momentum.encoded(gr.momentum.features(bars_1m), "score")
    t = gr.models.direction.targets(f, "5h")                          # y (sign), cls (−1, 0, +1 at ±9.5 bps)
    p = gr.models.direction.walk_forward(t, horizon="5h", split="2023-01-01", encoding="score")
    gr.models.direction.brier_skill(p, model="B", benchmark="climatology")

**Targets.** r = log close_{t+h} − log close_t over h decision bars, null when
any bar between is missing. `y = 1[r > 0]` is the primary target; `cls` is
+1 above +c, −1 below −c and 0 between, with c a cost that defines the target
(9.5 bps: Hyperliquid's 0.045% taker each way and a 0.5 bps spread).

**Models.** Logistic regression, binary for `y` and multinomial (flat as the
reference class) for `cls`, fitted by Newton–IRLS on columns standardised with
the training rows' mean and standard deviation, the intercept unpenalised and
the rest under a fixed L2 penalty (1e-6 by default; it keeps a separable column
finite and is not searched). Set **A** is volatility and the clock: log σ,
the hour of day as two harmonics and the weekday as one. Set **B** is A plus
the momentum encodings and their interaction. **Climatology** is the training
rows' share of each outcome.

**Walk-forward.** Refits fall on `gr.timeseries.walk_forward_origins`' refit
origins, an expanding window. Labels overlap, so a refit at R trains only on
rows whose target was known by R (close_t + h ≤ R, the purge), and its model
applies from R + h (the embargo); the rows between keep the previous model
(López de Prado 2018, ch. 7). A and B are fitted on the same rows: those where
every B column and the target exist.

**Scores.** Log loss and Brier score per row, Brier skill against a
benchmark, Diebold–Mariano on log-loss differences through
`gr.models.evaluate.dm` (h − 1 lags, Harvey–Leybourne–Newbold), the
Pesaran–Timmermann (1992) test of the calls against the outcomes, and a
reliability table.
"""

import math

import numpy as np
import polars as pl
from scipy.stats import norm

from .. import backtest, momentum, stats, studies, timeseries, utils
from .._errors import Refused
from . import evaluate

HORIZONS = ("5m", "30m", "5h", "1d")
COST = 0.00095
RIDGE = 1e-6
SETS = ("A", "B")
CLASSES = (-1, 0, 1)
A_COLUMNS = (
    "log_sigma",
    "hour_sin1",
    "hour_cos1",
    "hour_sin2",
    "hour_cos2",
    "weekday_sin",
    "weekday_cos",
)


# ── Targets ─────────────────────────────────────────────────────────────────


def targets(frame: pl.DataFrame, horizon: str, *, decision: str = "5m", cost: float = COST) -> pl.DataFrame:
    """The frame with `horizon, h, r_ahead, y, cls, cost`: the h-bar log return ahead and its two targets, null across a hole."""
    utils.require(
        frame,
        ("ticker", "ts", "close_ts", "close"),
        "make features with gr.momentum.features",
    )
    h = momentum.bars_in(horizon, decision)
    if cost < 0:
        raise Refused(f"cost={cost} must be at least 0")
    ahead = pl.duration(microseconds=h * momentum.micros(decision))
    lead = pl.col("close").shift(-h).over("ticker")
    whole = pl.col("ts").shift(-h).over("ticker") == pl.col("ts") + ahead
    r = pl.when(whole).then(lead.log() - pl.col("close").log())
    return (
        frame.sort("ticker", "ts")
        .with_columns(
            pl.lit(horizon).alias("horizon"),
            pl.lit(h).alias("h"),
            r.alias("r_ahead"),
            pl.lit(cost).alias("cost"),
        )
        .with_columns(
            pl.when(pl.col("r_ahead").is_not_null()).then((pl.col("r_ahead") > 0).cast(pl.Int8)).alias("y"),
            pl.when(pl.col("r_ahead") > cost)
            .then(pl.lit(1, pl.Int8))
            .when(pl.col("r_ahead") < -cost)
            .then(pl.lit(-1, pl.Int8))
            .when(pl.col("r_ahead").is_not_null())
            .then(pl.lit(0, pl.Int8))
            .alias("cls"),
        )
    )


def clock(frame: pl.DataFrame) -> pl.DataFrame:
    """Set A's columns: log σ, the UTC hour as two harmonics, the weekday as one."""
    hour = pl.col("ts").dt.hour().cast(pl.Float64)
    day = pl.col("ts").dt.weekday().cast(pl.Float64)
    return frame.with_columns(
        pl.col("sigma").log().alias("log_sigma"),
        (2 * math.pi * hour / 24).sin().alias("hour_sin1"),
        (2 * math.pi * hour / 24).cos().alias("hour_cos1"),
        (4 * math.pi * hour / 24).sin().alias("hour_sin2"),
        (4 * math.pi * hour / 24).cos().alias("hour_cos2"),
        (2 * math.pi * day / 7).sin().alias("weekday_sin"),
        (2 * math.pi * day / 7).cos().alias("weekday_cos"),
    )


def columns(frame: pl.DataFrame, which: str) -> list[str]:
    """The model columns of set A or B in this frame (B: A, every `e_<L>`, and `interaction`)."""
    if which == "A":
        return list(A_COLUMNS)
    if which == "B":
        es = [c for c in frame.columns if c.startswith("e_")]
        if not es or "interaction" not in frame.columns:
            raise Refused("set B needs the encoded momentum columns: pass the frame through gr.momentum.encoded")
        return [*A_COLUMNS, *es, "interaction"]
    raise Refused(f"set {which!r} is not one of {', '.join(SETS)}")


# ── Logistic regression by Newton–IRLS ──────────────────────────────────────


def _standardise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    return mean, sd


def _design(x: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), (x - mean) / sd])


def _newton(objective, z0: np.ndarray, *, tol: float = 1e-10, max_iter: int = 100) -> np.ndarray:
    """Maximise a concave objective returning (value, gradient, Hessian), with step-halving."""
    z = z0
    value, grad, hess = objective(z)
    for _ in range(max_iter):
        step = np.linalg.solve(-hess, grad)
        t = 1.0
        while True:
            new = objective(z + t * step)
            if new[0] >= value - 1e-12 or t < 1e-8:
                break
            t /= 2
        z = z + t * step
        value, grad, hess = new
        if np.max(np.abs(t * step)) < tol:
            return z
    raise Refused(f"the logistic fit did not converge in {max_iter} Newton steps")


def penalised_loglik(beta: np.ndarray, x: np.ndarray, y: np.ndarray, ridge: float) -> float:
    """The binary objective IRLS maximises: Σ[y·η − log(1 + eη)] − (λ/2)·Σβ² over the non-intercept coefficients."""
    eta = x @ beta
    return float(np.sum(y * eta - np.logaddexp(0, eta)) - ridge / 2 * np.sum(beta[1:] ** 2))


def fit_logit(x: np.ndarray, y: np.ndarray, *, ridge: float = RIDGE, start: np.ndarray | None = None) -> dict:
    """Binary logistic regression on raw columns `x` (no intercept): `{"mean", "sd", "beta"}`, β on the standardised design.

    `start`, a previous fit's β, only shortens the Newton path: the optimum is the same.
    """
    mean, sd = _standardise(x)
    d = _design(x, mean, sd)
    penalty = np.full(d.shape[1], ridge)
    penalty[0] = 0.0

    def objective(beta):
        eta = d @ beta
        p = 1 / (1 + np.exp(-eta))
        value = float(np.sum(y * eta - np.logaddexp(0, eta)) - 0.5 * np.sum(penalty * beta**2))
        grad = d.T @ (y - p) - penalty * beta
        hess = -(d.T * (p * (1 - p))) @ d - np.diag(penalty)
        return value, grad, hess

    z0 = np.zeros(d.shape[1]) if start is None or start.shape != (d.shape[1],) else start
    beta = _newton(objective, z0)
    return {"mean": mean, "sd": sd, "beta": beta}


def fit_multinomial(x: np.ndarray, cls: np.ndarray, *, ridge: float = RIDGE, start: np.ndarray | None = None) -> dict:
    """Multinomial logistic regression over classes (−1, 0, +1) with 0 (flat) as the reference: `{"mean", "sd", "beta"}`, β of shape (2, p)."""
    mean, sd = _standardise(x)
    d = _design(x, mean, sd)
    p = d.shape[1]
    ys = np.column_stack([(cls == -1).astype(float), (cls == 1).astype(float)])
    penalty = np.tile(np.r_[0.0, np.full(p - 1, ridge)], 2)

    def objective(flat):
        b = flat.reshape(2, p)
        eta = d @ b.T
        log_norm = np.logaddexp(0, np.logaddexp(eta[:, 0], eta[:, 1]))
        prob = np.exp(eta - log_norm[:, None])
        value = float(np.sum(ys * eta) - np.sum(log_norm) - 0.5 * np.sum(penalty * flat**2))
        grad = (d.T @ (ys - prob)).T.reshape(-1) - penalty * flat
        hess = np.zeros((2 * p, 2 * p))
        for j in range(2):
            for k in range(2):
                w = prob[:, j] * ((j == k) - prob[:, k])
                hess[j * p : (j + 1) * p, k * p : (k + 1) * p] = -(d.T * w) @ d
        return value, grad, hess - np.diag(penalty)

    z0 = np.zeros(2 * p) if start is None or start.size != 2 * p else start.reshape(-1)
    beta = _newton(objective, z0).reshape(2, p)
    return {"mean": mean, "sd": sd, "beta": beta}


def predict_logit(fit: dict, x: np.ndarray) -> np.ndarray:
    """P(up) for raw columns `x`, clipped inside (0, 1)."""
    eta = _design(x, fit["mean"], fit["sd"]) @ fit["beta"]
    return np.clip(1 / (1 + np.exp(-eta)), 1e-15, 1 - 1e-15)


def predict_multinomial(fit: dict, x: np.ndarray) -> np.ndarray:
    """P(down), P(flat), P(up) per row, each row summing to 1."""
    eta = _design(x, fit["mean"], fit["sd"]) @ fit["beta"].T
    full = np.column_stack([eta[:, 0], np.zeros(len(x)), eta[:, 1]])
    full -= full.max(axis=1, keepdims=True)
    e = np.exp(full)
    return e / e.sum(axis=1, keepdims=True)


# ── Walk-forward ────────────────────────────────────────────────────────────


def _schedule(rows: pl.DataFrame, split, every: int, step_us: int, h: int) -> pl.DataFrame:
    """Per row after the split: the refit (index and instant R) whose model applies, the last with R + h ≤ close_ts."""
    origins = timeseries.walk_forward_origins(rows.select("ticker", "ts", "close_ts"), split, every=every)
    refits = (
        origins.filter(pl.col("refit"))
        .select("ticker", pl.col("close_ts").alias("refit_at"))
        .with_columns(
            (pl.col("refit_at") + pl.duration(microseconds=h * step_us)).alias("applies_from"),
            pl.int_range(pl.len()).over("ticker").alias("refit_k"),
        )
        .sort("ticker", "applies_from")
    )
    scored = origins.select("ticker", "ts", "close_ts").sort("ticker", "close_ts")
    return scored.join_asof(
        refits,
        left_on="close_ts",
        right_on="applies_from",
        by="ticker",
        strategy="backward",
        check_sortedness=False,
    ).drop_nulls("refit_k")


def walk_forward(
    frame: pl.DataFrame,
    *,
    horizon: str,
    split,
    encoding: str,
    target: str = "sign",
    every: int = 2016,
    decision: str = "5m",
    ridge: float = RIDGE,
    cache: dict | None = None,
) -> pl.DataFrame:
    """Out-of-sample forecasts of sets A, B and climatology for one horizon, one row per (set, decision row).

    `frame` comes from `targets` on `gr.momentum.encoded` features. Columns:
    `ticker, ts, close_ts, horizon, h, target, set, encoding, model, refit_at,
    fitted_through, n_train, y | cls, p_up [, p_down, p_flat]`. `cache`, a
    dict kept between calls, holds each refit's fit under (ticker, horizon,
    set, encoding, target, refit_at), so a rerun refits nothing it has.
    """
    if target not in ("sign", "three"):
        raise Refused(f"target={target!r} is not 'sign' or 'three'")
    if encoding not in momentum.ENCODINGS:
        raise Refused(f"encoding {encoding!r} is not one of {', '.join(momentum.ENCODINGS)}")
    utils.require(
        frame,
        ("ticker", "ts", "close_ts", "sigma", "r_ahead", "y", "cls"),
        "make targets with direction.targets",
    )
    h = momentum.bars_in(horizon, decision)
    step_us = momentum.micros(decision)
    framed = clock(frame) if "log_sigma" not in frame.columns else frame
    cols_b = columns(framed, "B")
    label = "y" if target == "sign" else "cls"
    usable = framed.drop_nulls([*cols_b, label]).filter(pl.all_horizontal([pl.col(c).is_finite() for c in cols_b]))
    ahead = pl.duration(microseconds=h * step_us)
    out = []
    for (ticker,), rows in usable.sort("ticker", "ts").group_by("ticker", maintain_order=True):
        plan = _schedule(rows, split, every, step_us, h)
        scored = rows.join(plan.select("ts", "refit_k", "refit_at"), on="ts", how="inner")
        known = rows.with_columns((pl.col("close_ts") + ahead).alias("_known"))
        previous: dict[str, np.ndarray] = {}
        for (_, refit_at), test in scored.group_by("refit_k", "refit_at", maintain_order=True):
            train = known.filter(pl.col("_known") <= refit_at)
            if train.height < 2 * len(cols_b):
                continue
            ys = train[label].to_numpy()
            for which in ("climatology", *SETS):
                key = (ticker, horizon, which, encoding, target, refit_at)
                fit = cache.get(key) if cache is not None else None
                cols = columns(framed, which) if which != "climatology" else []
                if fit is None:
                    if which == "climatology":
                        fit = {"share": np.array([np.mean(ys == c) for c in CLASSES]) if target == "three" else float(np.mean(ys))}
                    elif target == "sign":
                        fit = fit_logit(train.select(cols).to_numpy(), ys.astype(float), ridge=ridge, start=previous.get(which))
                    else:
                        fit = fit_multinomial(train.select(cols).to_numpy(), ys, ridge=ridge, start=previous.get(which))
                    if cache is not None:
                        cache[key] = fit
                if which != "climatology":
                    previous[which] = fit["beta"]
                base = test.select("ticker", "ts", "close_ts", label).with_columns(
                    pl.lit(horizon).alias("horizon"),
                    pl.lit(h).alias("h"),
                    pl.lit(target).alias("target"),
                    pl.lit(which).alias("set"),
                    pl.lit(encoding).alias("encoding"),
                    pl.lit("climatology" if which == "climatology" else "logit").alias("model"),
                    pl.lit(refit_at).alias("refit_at"),
                    pl.lit(train["_known"].max()).alias("fitted_through"),
                    pl.lit(train.height).alias("n_train"),
                )
                if target == "sign":
                    p = np.full(test.height, fit["share"]) if which == "climatology" else predict_logit(fit, test.select(cols).to_numpy())
                    out.append(base.with_columns(pl.Series("p_up", np.clip(p, 1e-15, 1 - 1e-15))))
                else:
                    if which == "climatology":
                        share = np.clip(fit["share"], 1e-15, None)
                        probs = np.tile(share / share.sum(), (test.height, 1))
                    else:
                        probs = predict_multinomial(fit, test.select(cols).to_numpy())
                    out.append(
                        base.with_columns(
                            pl.Series("p_down", probs[:, 0]),
                            pl.Series("p_flat", probs[:, 1]),
                            pl.Series("p_up", probs[:, 2]),
                        )
                    )
    if not out:
        raise Refused(f"no refit after {split} had enough rows to fit on")
    return pl.concat(out, how="vertical").sort("set", "ticker", "ts")


# ── Scores ──────────────────────────────────────────────────────────────────


def scored(forecasts: pl.DataFrame) -> pl.DataFrame:
    """The forecasts with `log_loss` and `brier` per row: binary for the sign target, multinomial for the three classes."""
    if "cls" in forecasts.columns:
        p_true = pl.when(pl.col("cls") == 1).then("p_up").when(pl.col("cls") == -1).then("p_down").otherwise("p_flat")
        brier = sum(((pl.col("cls") == c).cast(pl.Float64) - pl.col(p)).pow(2) for c, p in zip(CLASSES, ("p_down", "p_flat", "p_up"), strict=True))
        return forecasts.with_columns((-p_true.log()).alias("log_loss"), brier.alias("brier"))
    utils.require(forecasts, ("y", "p_up"), "make forecasts with direction.walk_forward")
    y = pl.col("y").cast(pl.Float64)
    return forecasts.with_columns(
        (-(y * pl.col("p_up").log() + (1 - y) * (1 - pl.col("p_up")).log())).alias("log_loss"),
        (pl.col("p_up") - y).pow(2).alias("brier"),
    )


def paired(forecasts: pl.DataFrame, model: str, benchmark: str) -> pl.DataFrame:
    """`ticker, ts, close_ts, h, loss_model, loss_benchmark, brier_model, brier_benchmark` on the rows both sets forecast."""
    s = scored(forecasts) if "log_loss" not in forecasts.columns else forecasts
    keys = ["ticker", "ts", "close_ts", "h"]
    a = s.filter(pl.col("set") == model).select(
        *keys,
        pl.col("log_loss").alias("loss_model"),
        pl.col("brier").alias("brier_model"),
    )
    b = s.filter(pl.col("set") == benchmark).select(
        *keys,
        pl.col("log_loss").alias("loss_benchmark"),
        pl.col("brier").alias("brier_benchmark"),
    )
    out = a.join(b, on=keys, how="inner").sort("ticker", "ts")
    if not out.height:
        raise Refused(f"sets {model!r} and {benchmark!r} share no forecast row")
    return out


def brier_skill(forecasts: pl.DataFrame, *, model: str, benchmark: str) -> float:
    """1 − Brier(model)/Brier(benchmark) on the shared rows: 0 for the benchmark itself, positive when the model's Brier is lower."""
    p = paired(forecasts, model, benchmark)
    return float(1 - p["brier_model"].mean() / p["brier_benchmark"].mean())


def compare(forecasts: pl.DataFrame, *, model: str, benchmark: str) -> dict:
    """Diebold–Mariano on loss_model − loss_benchmark (negative favours the model), over h − 1 lags, HLN-corrected, with the lag count."""
    p = paired(forecasts, model, benchmark)
    hs = p["h"].unique().to_list()
    if len(hs) != 1:
        raise Refused(f"the forecasts mix horizons {sorted(hs)}; compare one at a time")
    h = int(hs[0])
    d = evaluate.dm(p["loss_model"].to_numpy(), p["loss_benchmark"].to_numpy(), h=h)
    lags = max(h - 1, 0) if d["kernel"] == "rectangular" else max(h - 1, 1)
    return {
        **d,
        "lags": lags,
        "h": h,
        "mean_difference": float((p["loss_model"] - p["loss_benchmark"]).mean()),
    }


def pesaran_timmermann(calls, outcomes) -> dict:
    """Pesaran and Timmermann's (1992) test that the calls (1 up, 0 down) and the outcomes are independent; one-sided, large S rejects.

    P is the share called right, P* = Py·Px + (1 − Py)(1 − Px) the share an
    independent caller with the same marginals gets, and
    S = (P − P*) / √(V(P) − V(P*)) is standard normal under independence, with
    V(P) = P*(1 − P*)/n and V(P*) = (2Py − 1)²Px(1 − Px)/n + (2Px − 1)²Py(1 − Py)/n
    + 4PyPx(1 − Py)(1 − Px)/n² (their eq. 4–5). It assumes independent rows: on
    overlapping horizons it is optimistic, which is why it is one criterion of three.
    """
    x = np.asarray(calls, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = x.size
    if n < 3:
        raise Refused(f"{n} calls are too few for a Pesaran–Timmermann test")
    hit = float(np.mean(x == y))
    py, px = float(y.mean()), float(x.mean())
    star = py * px + (1 - py) * (1 - px)
    v_p = star * (1 - star) / n
    v_star = (2 * py - 1) ** 2 * px * (1 - px) / n + (2 * px - 1) ** 2 * py * (1 - py) / n + 4 * py * px * (1 - py) * (1 - px) / n**2
    v = v_p - v_star
    if v <= 0:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n": n,
            "hit_rate": hit,
            "expected": star,
        }
    s = (hit - star) / math.sqrt(v)
    return {
        "statistic": float(s),
        "p_value": float(norm.sf(s)),
        "n": n,
        "hit_rate": hit,
        "expected": star,
    }


def reliability(forecasts: pl.DataFrame, *, bins: int = 10) -> pl.DataFrame:
    """`bin, lo, hi, n, mean_forecast, share_up` over equal-width bins of P(up), for one set's sign forecasts."""
    utils.require(forecasts, ("p_up", "y"), "make sign forecasts with direction.walk_forward")
    if forecasts["set"].n_unique() != 1:
        raise Refused("pass one set's forecasts")
    return (
        forecasts.with_columns((pl.col("p_up") * bins).floor().clip(0, bins - 1).cast(pl.Int64).alias("bin"))
        .group_by("bin")
        .agg(
            pl.len().alias("n"),
            pl.col("p_up").mean().alias("mean_forecast"),
            pl.col("y").cast(pl.Float64).mean().alias("share_up"),
        )
        .with_columns((pl.col("bin") / bins).alias("lo"), ((pl.col("bin") + 1) / bins).alias("hi"))
        .select("bin", "lo", "hi", "n", "mean_forecast", "share_up")
        .sort("bin")
    )


def holm(p_values) -> list[float]:
    """Holm's (1979) step-down adjusted p-values, in the input's order."""
    ps = [float(p) for p in p_values]
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        out[i] = min(1.0, running)
    return out


def cell(forecasts: pl.DataFrame, *, halves_at, model: str = "B", benchmark: str = "A") -> dict:
    """One registered cell's S statistics: the DM comparison, Pesaran–Timmermann on the model's calls, and the mean log-loss difference in each half."""
    at = utils.instant("halves_at", halves_at)
    p = paired(forecasts, model, benchmark).with_columns((pl.col("loss_model") - pl.col("loss_benchmark")).alias("d"))
    first = p.filter(pl.col("close_ts").dt.epoch("us") < at)["d"]
    second = p.filter(pl.col("close_ts").dt.epoch("us") >= at)["d"]
    calls = forecasts.filter(pl.col("set") == model)
    pt = pesaran_timmermann(
        (calls["p_up"] > 0.5).cast(pl.Float64).to_numpy(),
        calls["y"].cast(pl.Float64).to_numpy(),
    )
    dm = compare(forecasts, model=model, benchmark=benchmark)
    return {
        "mean_difference": dm["mean_difference"],
        "dm_statistic": dm["statistic"],
        "dm_p": dm["p_value"],
        "lags": dm["lags"],
        "n": dm["n"],
        "pt_statistic": pt["statistic"],
        "pt_p": pt["p_value"],
        "hit_rate": pt["hit_rate"],
        "first_half": float(first.mean()) if first.len() else float("nan"),
        "second_half": float(second.mean()) if second.len() else float("nan"),
    }


def s_verdict(cells: pl.DataFrame, *, alpha: float = 0.05) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The registered S verdict: per cell, criteria 1–3 with Holm across every cell; per horizon, supported if one encoding meets them on both tickers.

    `cells` has one row per (horizon, encoding, ticker) with `cell`'s keys.
    Returns (cells with `holm_p, c1, c2, c3, met`, horizons with `supported`).
    """
    utils.require(
        cells,
        (
            "horizon",
            "encoding",
            "ticker",
            "mean_difference",
            "dm_p",
            "pt_p",
            "first_half",
            "second_half",
        ),
        "build the cells with direction.cell",
    )
    framed = cells.with_columns(pl.Series("holm_p", holm(cells["dm_p"].to_list()))).with_columns(
        ((pl.col("mean_difference") < 0) & (pl.col("holm_p") < alpha)).alias("c1"),
        (pl.col("pt_p") < alpha).alias("c2"),
        ((pl.col("mean_difference") < 0).all().over("horizon", "encoding") & (pl.col("first_half") < 0) & (pl.col("second_half") < 0)).alias("c3"),
    )
    framed = framed.with_columns((pl.col("c1") & pl.col("c2") & pl.col("c3")).alias("met"))
    tickers = cells["ticker"].n_unique()
    by_encoding = framed.group_by("horizon", "encoding").agg((pl.col("met").sum() == tickers).alias("both"))
    horizons = by_encoding.group_by("horizon").agg(pl.col("both").any().alias("supported")).sort("horizon")
    return framed, horizons


# ── The economic test (5h, 1d) ──────────────────────────────────────────────

FLIP_FEE = COST / 2  # per unit of position change: a flip (two units) pays the round trip
CAP = 2.0
_YEAR_US = 365 * 86_400_000_000


def _grid(frame: pl.DataFrame, anchor, h: int, step_us: int) -> pl.DataFrame:
    """h-wide bars on the decision rows whose close is a whole number of horizons from `anchor`: `ticker, ts, close_ts, close, sigma`."""
    width = h * step_us
    a = utils.instant("anchor", anchor)
    return (
        frame.filter(((pl.col("close_ts").dt.epoch("us") - a) % width) == 0)
        .select(
            "ticker",
            (pl.col("close_ts") - pl.duration(microseconds=width)).alias("ts"),
            "close_ts",
            "close",
            "sigma",
        )
        .sort("ticker", "ts")
    )


def inverse_vol(grid: pl.DataFrame, *, split, decision: str = "5m", cap: float = CAP) -> pl.DataFrame:
    """The grid with `position` = min(τ/σ̂, cap): `gr.models.vol.target`'s inverse_vol, band 0, on σ.

    σ̂ annualises the per-decision-bar σ, τ is the annualised volatility of the
    grid's log returns closed by the split. It is written here, not called,
    because `target` takes record intervals only and 5h is not one; a test
    checks the two agree at 1d.
    """
    h_us = int((grid["close_ts"][0] - grid["ts"][0]).total_seconds() * 1_000_000)
    per_year = _YEAR_US / h_us
    at = utils.instant("split", split)
    before = grid.filter(pl.col("close_ts").dt.epoch("us") <= at)
    r = timeseries.returns(before, kind="log")["return"].drop_nulls()
    if r.len() < 30:
        raise Refused(f"{r.len()} returns before the split are too few to set a target")
    tau = float(r.std(ddof=1)) * per_year**0.5
    sigma_year = pl.col("sigma") * (_YEAR_US / momentum.micros(decision)) ** 0.5
    return grid.with_columns(pl.min_horizontal(tau / sigma_year, pl.lit(cap)).alias("position"))


def _trial(bars: pl.DataFrame, position: pl.Expr, name: str, fee: float, funding) -> pl.DataFrame:
    """`studies.trial`'s columns and `funding`, which it drops: what a held position paid or received."""
    r = backtest.returns(bars, position, fee=fee, funding=funding)
    return r.select(
        pl.lit(name).alias("trial"),
        "ticker",
        "ts",
        "close_ts",
        "position",
        "bar_return",
        "gross",
        "cost",
        "funding",
        "net",
    )


def economic(
    frame: pl.DataFrame,
    forecasts: pl.DataFrame,
    *,
    split,
    decision: str = "5m",
    fee: float = FLIP_FEE,
    funding: pl.DataFrame | None = None,
    model: str = "B",
) -> pl.DataFrame:
    """The sign rule, buy-and-hold and the inverse-vol book on non-overlapping h-bars, as `studies.trial` frames.

    `frame` is one ticker's decision rows (`close`, `sigma`), `forecasts` its
    sign forecasts for one horizon. The rule holds +1 when the model's P(up)
    exceeds 0.5 and −1 otherwise, decided at every h-th forecast origin and held
    for h; every trial pays `fee` per unit of position change (4.75 bps, so a
    flip pays the 9.5 bps round trip) and, with `funding` from
    `gr.reference.funding_hours`, the settlements it holds through.
    Trials: `sign rule`, `hold`, `inverse vol`.
    """
    f = forecasts.filter(pl.col("set") == model).sort("ts")
    if f["ticker"].n_unique() != 1 or f["h"].n_unique() != 1:
        raise Refused("pass one ticker's forecasts for one horizon")
    h, step_us = int(f["h"][0]), momentum.micros(decision)
    anchor = f["close_ts"][0]
    grid = _grid(frame.filter(pl.col("ticker") == f["ticker"][0]), anchor, h, step_us)
    sized = inverse_vol(grid, split=split, decision=decision)
    oos = sized.filter(pl.col("close_ts") >= anchor)
    calls = f.select(
        "close_ts",
        pl.when(pl.col("p_up") > 0.5).then(1.0).otherwise(-1.0).alias("_call"),
    )
    bars = oos.join(calls, on="close_ts", how="left").rename({"position": "_inv"})
    trials = [
        _trial(bars, pl.col("_call"), "sign rule", fee, funding),
        _trial(bars, pl.lit(1.0), "hold", fee, funding),
        _trial(bars, pl.col("_inv"), "inverse vol", fee, funding),
    ]
    return pl.concat(trials).with_columns(pl.lit(f["horizon"][0]).alias("horizon"))


def daily(trials: pl.DataFrame) -> pl.DataFrame:
    """Each trial's net returns compounded by UTC day of `close_ts`: `trial, ticker, ts, net`, on one calendar whatever the horizon."""
    return (
        trials.drop_nulls("net")
        .with_columns(pl.col("close_ts").dt.truncate("1d").alias("_day"))
        .group_by("trial", "ticker", "_day")
        .agg(
            ((pl.col("net") + 1).product() - 1).alias("net"),
            ((pl.col("gross") + 1).product() - 1).alias("gross"),
        )
        .rename({"_day": "ts"})
        .sort("trial", "ticker", "ts")
    )


def e_verdict(
    cells: dict[str, pl.DataFrame],
    *,
    trials: int = 24,
    blocks: int = 16,
    alpha: float = 0.95,
) -> dict:
    """The registered E verdict over `cells`, {name: `economic` frame}: DSR at N = `trials`, PBO by CSCV, and Sharpe against hold and inverse vol.

    Every figure is on daily net returns. The best cell is the highest
    annualised Sharpe of its `sign rule`; its DSR deflates by `trials`, the
    registered N, with the variance across these cells' Sharpe ratios.
    """
    rows, rules = [], []
    for name, frame in cells.items():
        d = daily(frame)
        s = studies.summary(d, periods_per_year=365)
        get = {r["trial"]: r for r in s.iter_rows(named=True)}
        rows.append(
            {
                "cell": name,
                "sharpe": get["sign rule"]["sharpe"],
                "sharpe_annual": get["sign rule"]["sharpe_annual"],
                "periods": get["sign rule"]["periods"],
                "skew": get["sign rule"]["skew"],
                "kurt": get["sign rule"]["kurt"],
                "hold_annual": get["hold"]["sharpe_annual"],
                "inverse_vol_annual": get["inverse vol"]["sharpe_annual"],
            }
        )
        rules.append(d.filter(pl.col("trial") == "sign rule").with_columns(pl.lit(name).alias("trial")))
    table = pl.DataFrame(rows, infer_schema_length=None)
    scored = table.filter(pl.col("sharpe").is_not_null())
    if scored.height < 2:
        raise Refused(f"{scored.height} cells have a Sharpe; the verdict needs at least 2")
    best = scored.sort("sharpe", descending=True).row(0, named=True)
    variance = float(scored["sharpe"].var(ddof=1))
    dsr = stats.psr(
        best["sharpe"],
        best["periods"],
        best["skew"],
        best["kurt"],
        benchmark=stats.expected_max_sharpe(trials, variance),
    )
    overfit = stats.pbo(studies.matrix(pl.concat(rules)), blocks=blocks)
    beats = best["sharpe_annual"] > best["hold_annual"] and best["sharpe_annual"] > best["inverse_vol_annual"]
    return {
        "cells": table,
        "best": best["cell"],
        "dsr": dsr,
        "pbo": overfit["pbo"],
        "beats_hold_and_inverse_vol": beats,
        "supported": dsr >= alpha and overfit["pbo"] < 0.5 and beats,
    }
