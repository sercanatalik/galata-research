"""Weighting a book: inverse variance, Hierarchical Risk Parity and long-only minimum variance, on a Σ per rebalance.

    w = gr.models.portfolio.hrp(cov)                       # López de Prado (2016)
    book = gr.models.portfolio.book(panel, members, "2021-12-31")
    trials, weights = gr.models.portfolio.study(panel, book, on, end)   # the registered 9 and 1/N

Allocation, not timing (`planning/preregistered/build-the-portfolio.md`). At
each rebalance, the 7-day Σ of the book's live coins is estimated three ways
(a 90-day sample, RiskMetrics' EWMA, and DCC walked forward), each weighting
method is applied to each Σ, and the weights are held to the next rebalance
by `gr.factors.portfolio`, which charges the fee on turnover and the funding
per day held.

**The live set** at a rebalance is the book's coins with a bar that day.
When it changes (a delisting), every estimator starts again on the
survivors: DCC is walked forward afresh on their joint history.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

from .. import factors, utils
from .._errors import Refused

ESTIMATORS = ("sample", "ewma", "dcc")
METHODS = ("ivp", "hrp", "minvar")
EQUAL = "1/N"


def _square(cov) -> np.ndarray:
    c = np.asarray(cov, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise Refused(f"a covariance must be square, not {c.shape}")
    if not np.all(np.isfinite(c)) or np.any(np.diag(c) <= 0):
        raise Refused("a covariance needs finite entries and positive variances")
    return c


def ivp(cov) -> np.ndarray:
    """Inverse-variance weights: w ∝ 1/σᵢ², summing to one."""
    c = _square(cov)
    w = 1 / np.diag(c)
    return w / w.sum()


def _cluster_variance(c: np.ndarray, items: list[int]) -> float:
    sub = c[np.ix_(items, items)]
    w = ivp(sub)
    return float(w @ sub @ w)


def hrp_order(cov) -> list[int]:
    """The quasi-diagonal order: single linkage on the Euclidean distances between the columns of d = √(½(1 − ρ))."""
    c = _square(cov)
    sd = np.sqrt(np.diag(c))
    rho = np.clip(c / np.outer(sd, sd), -1.0, 1.0)
    d = np.sqrt(np.clip(0.5 * (1 - rho), 0.0, None))
    if c.shape[0] == 1:
        return [0]
    return [int(i) for i in leaves_list(linkage(pdist(d), "single"))]


def hrp(cov) -> np.ndarray:
    """López de Prado's (2016) Hierarchical Risk Parity: quasi-diagonalise, then bisect, splitting by inverse cluster variance.

    Each split gives the left half 1 − V_left/(V_left + V_right) of the
    weight above it, V being a cluster's variance under its own
    inverse-variance weights. No matrix is inverted.
    """
    c = _square(cov)
    w = np.ones(c.shape[0])
    clusters = [hrp_order(c)]
    while clusters:
        nxt = []
        for items in clusters:
            if len(items) < 2:
                continue
            half = len(items) // 2
            left, right = items[:half], items[half:]
            vl, vr = _cluster_variance(c, left), _cluster_variance(c, right)
            alpha = 1 - vl / (vl + vr)
            w[left] *= alpha
            w[right] *= 1 - alpha
            nxt += [left, right]
        clusters = nxt
    return w / w.sum()


def minvar(cov) -> np.ndarray:
    """The long-only minimum-variance weights: min wᵀΣw with w ≥ 0 and Σw = 1, by SLSQP from the inverse-variance start."""
    c = _square(cov)
    n = c.shape[0]
    if n == 1:
        return np.ones(1)
    result = minimize(
        lambda w: float(w @ c @ w),
        ivp(c),
        jac=lambda w: 2 * c @ w,
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}],
        method="SLSQP",
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    if not result.success:
        raise Refused(f"the minimum-variance solve failed: {result.message}")
    w = np.clip(result.x, 0.0, None)
    return w / w.sum()


WEIGHTS = {"ivp": ivp, "hrp": hrp, "minvar": minvar}


def ewma(returns: np.ndarray, lam: float = 0.94, seed: int = 30) -> np.ndarray:
    """RiskMetrics' covariance after the last row: S = λS + (1 − λ)rrᵀ, zero mean, started from the mean of the first `seed` outer products."""
    r = np.asarray(returns, dtype=float)
    if r.shape[0] <= seed:
        raise Refused(f"{r.shape[0]} joint returns; the EWMA needs more than its {seed}-row start")
    s = r[:seed].T @ r[:seed] / seed
    for row in r[seed:]:
        s = lam * s + (1 - lam) * np.outer(row, row)
    return s


def book(panel_: pl.DataFrame, members_: pl.DataFrame, on, *, n: int = 10, history: int = 500) -> list[str]:
    """The `n` perpetuals with the highest 30-day mean dollar volume on day `on`, among those with `history` bars by then."""
    day = _instant("on", on)
    rows = (
        members_.filter((pl.col("ts") == day) & pl.col("eligible"))
        .join(panel_.select("ticker", "ts", "bars"), on=["ticker", "ts"])
        .filter(pl.col("bars") >= history)
        .sort("dv", descending=True)
        .head(n)
    )
    if rows.height < n:
        raise Refused(f"only {rows.height} perpetuals had {history} bars on {day:%Y-%m-%d}; the book needs {n}")
    return rows["ticker"].to_list()


def _returns(panel_: pl.DataFrame, tickers: list[str]) -> pl.DataFrame:
    return panel_.filter(pl.col("ticker").is_in(tickers)).select("ticker", "ts", "close_ts", pl.col("ret").alias("return"))


def _joint(panel_: pl.DataFrame, tickers: list[str], until: datetime) -> np.ndarray:
    """The live set's joint daily returns up to and including `until`, in `tickers` order."""
    wide = (
        _returns(panel_, tickers)
        .filter(pl.col("ts") <= until)
        .pivot(on="ticker", index="ts", values="return")
        .sort("ts")
        .drop_nulls()
    )
    missing = [t for t in tickers if t not in wide.columns]
    if missing:
        raise Refused(f"no joint returns for {', '.join(missing)}")
    return wide.select(tickers).to_numpy()


def _dcc(panel_: pl.DataFrame, tickers: list[str], origins: list[datetime], horizon: int, every: int) -> dict:
    """DCC's cumulative `horizon`-day Σ at each origin, walked forward from the first: {origin: matrix}."""
    from . import corr

    rows = corr.walk_forward(
        _returns(panel_, tickers), model="gjr", dist="t", corr="dcc", split=origins[0], window="expanding", every=every, horizons=(horizon,)
    )
    days = pl.DataFrame({"ts": origins}, schema={"ts": rows.schema["ts"]})
    picked = rows.join(days, on="ts", how="inner")
    index = {t: i for i, t in enumerate(tickers)}
    out = {}
    for (ts,), part in picked.group_by("ts"):
        m = np.full((len(tickers), len(tickers)), np.nan)
        for r in part.iter_rows(named=True):
            i, j = index[r["ticker_i"]], index[r["ticker_j"]]
            m[i, j] = m[j, i] = r["cum_covariance"]
        out[ts] = m
    return out


def study(
    panel_: pl.DataFrame,
    tickers: list[str],
    on: list[datetime],
    end,
    *,
    window: int = 90,
    horizon: int = 7,
    lam: float = 0.94,
    every: int = 28,
    fee: float = factors.BINANCE_TAKER,
    estimators=ESTIMATORS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The registered trials, `"<method> <estimator>"` for each method and estimator, and `1/N`, as `gr.factors` trial frames.

    Returns `(trials, weights)`; `weights` holds `trial, rebalance, ticker, w`.
    """
    alive = {
        r: sorted(panel_.filter((pl.col("ts") == r) & pl.col("ticker").is_in(tickers))["ticker"].to_list()) for r in on
    }
    groups: list[tuple[list[str], list[datetime]]] = []
    for r in on:
        if len(alive[r]) < 2:
            raise Refused(f"on {r:%Y-%m-%d} only {alive[r]} of the book traded; a portfolio needs two")
        if groups and groups[-1][0] == alive[r]:
            groups[-1][1].append(r)
        else:
            groups.append((alive[r], [r]))
    rows = []
    for live, origins in groups:
        dcc = _dcc(panel_, live, origins, horizon, every) if "dcc" in estimators else {}
        for r in origins:
            joint = _joint(panel_, live, r)
            if joint.shape[0] < window:
                raise Refused(f"on {r:%Y-%m-%d} the live set has {joint.shape[0]} joint returns, short of {window}")
            covs = {
                "sample": np.cov(joint[-window:], rowvar=False) * horizon,
                "ewma": ewma(joint, lam) * horizon,
            }
            if "dcc" in estimators:
                if r not in dcc or not np.all(np.isfinite(dcc[r])):
                    raise Refused(f"DCC gave no Σ at {r:%Y-%m-%d}")
                covs["dcc"] = dcc[r]
            for est in estimators:
                for method in METHODS:
                    w = WEIGHTS[method](covs[est])
                    rows += [{"trial": f"{method} {est}", "rebalance": r, "ticker": t, "w": float(x)} for t, x in zip(live, w)]
            rows += [{"trial": EQUAL, "rebalance": r, "ticker": t, "w": 1 / len(live)} for t in live]
    weights = pl.DataFrame(rows, schema={"trial": pl.String, "rebalance": panel_.schema["ts"], "ticker": pl.String, "w": pl.Float64})
    trials = pl.concat(
        [
            factors.portfolio(panel_, part.drop("trial"), on, end, name, fee=fee)
            for (name,), part in weights.group_by("trial", maintain_order=True)
        ]
    )
    return trials, weights


def _instant(name: str, value) -> datetime:
    """A zone-aware time, as `gr.utils.instant` reads it, back as a UTC datetime."""
    return datetime.fromtimestamp(utils.instant(name, value) / 1e6, tz=UTC)


def schedule(start, end, *, every: int = 7) -> list[datetime]:
    """Rebalance days from `start` (inclusive) every `every` days, before `end`."""
    first, last = _instant("start", start), _instant("end", end)
    out, day = [], first
    while day < last:
        out.append(day)
        day += timedelta(days=every)
    return out
