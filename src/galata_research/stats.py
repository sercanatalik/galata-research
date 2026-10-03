"""Sharpe, and how much of it survives selection: PSR, the expected maximum, and the Deflated Sharpe Ratio.

Bailey and López de Prado, "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting and Non-Normality", *Journal of
Portfolio Management*, 2014. Pinned to the paper's own example: N=100,
V[SR]=0.5/250, T=1250, skewness −3, kurtosis 10, SR 2.5/√250 per day, which
give SR₀ 0.1132 per day and DSR 0.9004 (0.9505 at N=46).

Everything is per period. Annualise separately, with `annualize`.
"""

from math import e, sqrt
from statistics import NormalDist

import polars as pl

from . import utils
from ._errors import Refused
from .timeseries import optimal_block, stationary_bootstrap_indices

EULER_MASCHERONI = 0.5772156649015329
_N = NormalDist()


def sharpe(returns) -> float | None:
    """Mean over sample standard deviation (ddof 1), per period. Null when it is undefined."""
    r = _series(returns)
    if r.len() < 2:
        return None
    sd = r.std(ddof=1)
    return None if not sd else float(r.mean() / sd)


def moments(returns) -> tuple[float | None, float | None]:
    """Skewness and **raw** kurtosis (3 for a normal), population moments."""
    r = _series(returns)
    if r.len() < 2:
        return None, None
    skew = r.skew(bias=True)
    kurt = r.kurtosis(fisher=False, bias=True)
    return (None if skew is None else float(skew)), (None if kurt is None else float(kurt))


def annualize(sr: float | None, periods_per_year: float) -> float | None:
    """A per-period Sharpe scaled by √periods: 365 for daily crypto, 2190 for 4h."""
    return None if sr is None else sr * sqrt(periods_per_year)


def psr(sr: float, periods: int, skew: float, kurt: float, benchmark: float = 0.0) -> float:
    """The probability the true Sharpe exceeds `benchmark`, given T, skewness and raw kurtosis."""
    if periods < 2:
        raise Refused(f"periods={periods}: a Probabilistic Sharpe needs at least 2 returns")
    spread = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if spread <= 0:
        raise Refused(f"skew={skew}, kurt={kurt} at sr={sr} give a non-positive variance term; the moments are inconsistent")
    return _N.cdf((sr - benchmark) * sqrt(periods - 1) / sqrt(spread))


def expected_max_sharpe(trials: int, variance: float) -> float:
    """The Sharpe the best of `trials` independent trials reaches when every true Sharpe is 0.

    `variance` is the variance across the trials' estimated Sharpe ratios, per period.
    """
    if trials < 2:
        raise Refused(f"trials={trials}: with fewer than 2 trials there is no selection to correct")
    if variance < 0:
        raise Refused(f"variance={variance} is negative")
    g = EULER_MASCHERONI
    return sqrt(variance) * ((1 - g) * _N.inv_cdf(1 - 1 / trials) + g * _N.inv_cdf(1 - 1 / (trials * e)))


def dsr(sr: float, periods: int, skew: float, kurt: float, trials: int, variance: float) -> float:
    """The Deflated Sharpe Ratio: PSR against the expected maximum of `trials` trials."""
    return psr(sr, periods, skew, kurt, benchmark=expected_max_sharpe(trials, variance))


def percentile(observed: float, randoms) -> float:
    """`(1 + #{r ≥ observed}) / (N + 1)`: the permutation p-value of `observed` among `randoms` (nulls ignored)."""
    draws = [r for r in randoms if r is not None]
    return (1 + sum(r >= observed for r in draws)) / (len(draws) + 1)


def deflate(summary: pl.DataFrame) -> dict:
    """The best trial of `summary` (one row per trial: sharpe, periods, skew, kurt), deflated by all of them.

    Every row counts toward N, a trial with a null Sharpe included, because
    it was run. V[SR] is taken across the non-null Sharpe ratios.
    """
    needed = {"sharpe", "periods", "skew", "kurt"}
    missing = needed - set(summary.columns)
    if missing:
        raise Refused(f"deflate needs {', '.join(sorted(missing))} per trial")
    trials = summary.height
    scored = summary.filter(pl.col("sharpe").is_not_null())
    if scored.height < 2:
        raise Refused(f"{scored.height} of {trials} trials have a Sharpe; deflating needs at least 2")
    variance = float(scored["sharpe"].var(ddof=1))
    best = scored.sort("sharpe", descending=True).row(0, named=True)
    benchmark = expected_max_sharpe(trials, variance)
    return {
        **best,
        "benchmark": benchmark,
        "dsr": psr(best["sharpe"], best["periods"], best["skew"], best["kurt"], benchmark=benchmark),
        "trials": trials,
        "variance": variance,
    }


def _series(returns) -> pl.Series:
    s = returns if isinstance(returns, pl.Series) else pl.Series(list(returns), dtype=pl.Float64)
    return s.drop_nulls().cast(pl.Float64)


def pbo(matrix: pl.DataFrame, *, blocks: int = 16) -> dict:
    """The Probability of Backtest Overfitting, by combinatorially symmetric cross-validation.

    Bailey, Borwein, López de Prado and Zhu, "The Probability of Backtest
    Overfitting", *Journal of Computational Finance*, 2017, §2.2 steps a–g.
    `matrix` is T × N per-period returns on a shared calendar (a `ts` column
    is ignored), with no nulls: `studies.matrix` builds one. Rows are split
    into `blocks` equal blocks, the remainder dropped from the start so that
    the latest data is kept. Every choice of `blocks/2` blocks is a training
    set, and its complement the test set. The training winner's test rank,
    as a logit, says whether choosing on the past chose well. PBO is the
    share at or below zero: at or under the median out of sample.
    """
    from itertools import combinations
    from math import comb, log

    frame = matrix.drop("ts") if "ts" in matrix.columns else matrix
    columns = frame.columns
    if blocks < 2 or blocks % 2:
        raise Refused(f"blocks={blocks}: CSCV needs an even number of blocks, at least 2")
    if len(columns) < 2:
        raise Refused(f"{len(columns)} column(s): overfitting is a statement about choosing among at least 2")
    nulls = [c for c in columns if frame[c].null_count()]
    if nulls:
        raise Refused(f"the matrix has nulls in {', '.join(nulls[:5])}{' …' if len(nulls) > 5 else ''}; align it first (studies.matrix)")
    if frame.height < 2 * blocks:
        raise Refused(f"{frame.height} rows cannot fill {blocks} blocks of at least 2")

    dropped = frame.height % blocks
    frame = frame.slice(dropped)
    size = frame.height // blocks
    cols = [frame[c].cast(pl.Float64).to_list() for c in columns]
    # Per block and column: the sum and the sum of squares. A set's Sharpe is
    # pooled from its blocks' sums, so no combination re-reads a return.
    sums = [[sum(col[b * size : (b + 1) * size]) for col in cols] for b in range(blocks)]
    squares = [[sum(x * x for x in col[b * size : (b + 1) * size]) for col in cols] for b in range(blocks)]
    total_s = [sum(v) for v in zip(*sums)]
    total_q = [sum(v) for v in zip(*squares)]
    half = blocks // 2
    n = size * half
    n_cols = len(columns)

    def sharpe_of(s: list[float], q: list[float]) -> list[float | None]:
        out = []
        for sj, qj in zip(s, q):
            var = (qj - sj * sj / n) / (n - 1)
            out.append(sj / n / sqrt(var) if var > 1e-18 else None)
        return out

    rows = []
    for index, train in enumerate(combinations(range(blocks), half)):
        train_s = [sum(v) for v in zip(*(sums[b] for b in train))]
        train_q = [sum(v) for v in zip(*(squares[b] for b in train))]
        is_sr = sharpe_of(train_s, train_q)
        oos_sr = sharpe_of([t - x for t, x in zip(total_s, train_s)], [t - x for t, x in zip(total_q, train_q)])
        # The training winner: the highest Sharpe, the first column on a tie; no Sharpe ranks lowest.
        best = max(range(n_cols), key=lambda j: (float("-inf") if is_sr[j] is None else is_sr[j], -j))
        v = float("-inf") if oos_sr[best] is None else oos_sr[best]
        test = [float("-inf") if x is None else x for x in oos_sr]
        # Rank 1 is the worst; ties take their average rank.
        rank = sum(x < v for x in test) + (sum(x == v for x in test) + 1) / 2
        omega = rank / (n_cols + 1)
        rows.append({"combination": index, "best": columns[best], "is_sharpe": is_sr[best], "oos_sharpe": oos_sr[best], "rank": rank, "logit": log(omega / (1 - omega))})

    table = pl.DataFrame(rows)
    assert table.height == comb(blocks, half)
    paired = table.drop_nulls(["is_sharpe", "oos_sharpe"])
    slope = None
    if paired.height > 1 and paired["is_sharpe"].var() > 0:
        slope = float(paired.select(pl.cov("is_sharpe", "oos_sharpe")).item() / paired["is_sharpe"].var())
    return {
        # At or below zero: exactly the median is not outperforming it.
        "pbo": float((table["logit"] <= 0).mean()),
        "prob_loss": float((paired["oos_sharpe"] < 0).mean()) if paired.height else None,
        "slope": slope,
        "combinations": table,
        "blocks": blocks,
        "rows_per_block": size,
        "dropped": dropped,
        "trials": n_cols,
    }


def reality_check(excess: pl.DataFrame, *, reps: int = 1000, block: float | str | None = None, seed: int = 0) -> dict:
    """White's Reality Check (2000) and Hansen's SPA (2005): does any column beat its benchmark?

    `excess` is T × K per-period returns minus each column's benchmark
    (higher is better; a `ts` column is ignored). The null is that no column
    beats its benchmark on average, after searching all K.
    - **Reality Check:** `max_k √T d̄_k`, against the stationary bootstrap of
      `max_k √T (d̄*_k − d̄_k)`. Not studentized; this is arch's `upper`.
    - **SPA:** Hansen's studentized `max(max_k √T d̄_k / ω̂_k, 0)`, recentred
      `lower` (max(d̄, 0)), `consistent` (d̄ where it is above
      −√(ω̂²/T · 2 log log T), else 0) or `upper` (d̄). Then
      p_lower ≤ p_consistent ≤ p_upper.
    ω̂² is the stationary-bootstrap variance of √T d̄ (Politis and Romano),
    as in arch. `block` defaults to √T, as in arch; `"auto"` takes the median
    of the columns' Politis–White blocks (`timeseries.optimal_block`). p-values count
    replicates at or above the statistic.
    """
    import random
    from math import log

    frame = excess.drop("ts") if "ts" in excess.columns else excess
    columns = frame.columns
    t = frame.height
    if not columns:
        raise Refused("the excess matrix has no columns")
    nulls = [c for c in columns if frame[c].null_count()]
    if nulls:
        raise Refused(f"the excess matrix has nulls in {', '.join(nulls[:5])}; align it first (studies.excess)")
    if t < 10:
        raise Refused(f"{t} rows are too few to bootstrap")
    frame = frame.select(pl.all().cast(pl.Float64))
    if block == "auto":
        # One block for every column: the median of their own optimal blocks.
        blocks = sorted(optimal_block(frame[c]) for c in columns)
        block = max(1.0, blocks[len(blocks) // 2])
    block = block if block is not None else int(sqrt(t))
    means = frame.mean().row(0)

    demeaned = frame.select([(pl.col(c) - m).alias(c) for c, m in zip(columns, means)])
    p = 1.0 / block
    variance = [v / t for v in (demeaned.select(pl.all().pow(2)).sum().row(0))]
    for i in range(1, t):
        kappa = (1 - i / t) * (1 - p) ** i + (i / t) * (1 - p) ** (t - i)
        if kappa < 1e-12:
            continue
        lagged = (demeaned.head(t - i) * demeaned.tail(t - i)).sum().row(0)
        variance = [v + 2 * kappa * x / t for v, x in zip(variance, lagged)]
    flat = [c for c, v in zip(columns, variance) if v <= 0]
    if flat:
        raise Refused(f"no variation in {', '.join(flat[:5])}; a constant excess cannot be studentized")
    omega = [sqrt(v) for v in variance]
    root = sqrt(t)

    threshold = [-sqrt(v / t * 2 * log(log(t))) for v in variance]
    centres = {
        "lower": [max(m, 0.0) for m in means],
        "consistent": [m if m >= th else 0.0 for m, th in zip(means, threshold)],
        "upper": list(means),
    }
    rc_stat = max(root * m for m in means)
    spa_stat = max(max(root * m / w for m, w in zip(means, omega)), 0.0)

    rng = random.Random(seed)
    rc_hits, spa_hits = 0, {name: 0 for name in centres}
    for _ in range(reps):
        star = frame[stationary_bootstrap_indices(t, block, rng)].mean().row(0)
        if max(root * (s - m) for s, m in zip(star, means)) >= rc_stat:
            rc_hits += 1
        for name, mu in centres.items():
            if max(max(root * (s - c) / w for s, c, w in zip(star, mu, omega)), 0.0) >= spa_stat:
                spa_hits[name] += 1

    best = max(range(len(columns)), key=lambda k: means[k] / omega[k])
    return {
        "reality_check": rc_hits / reps,
        "spa": {name: hits / reps for name, hits in spa_hits.items()},
        "best": columns[best],
        "columns": pl.DataFrame({"column": columns, "mean": means, "omega": omega, "t": [root * m / w for m, w in zip(means, omega)]}),
        "block": block,
        "reps": reps,
        "seed": seed,
        "rows": t,
    }


def performance_fee(returns, benchmark, gamma: float, periods_per_year: float) -> float:
    """Fleming, Kirby and Ostdiek's (2001, eq. 8) fee, in basis points a year, to switch from `benchmark` to `returns`.

    Δ solves Σ[(Rₜ − Δ) − c(Rₜ − Δ)²] = Σ[Bₜ − cBₜ²], c = γ/(2(1+γ)): the
    per-period fee that leaves a quadratic-utility investor indifferent. It is
    quadratic in Δ and solved in closed form, taking the root nearest zero. A
    perp's return is already in excess of cash here, so no risk-free rate is
    added. FKO report γ = 1 and 10.
    """
    r, b = _series(returns).to_list(), _series(benchmark).to_list()
    if len(r) != len(b) or not r:
        raise Refused(f"returns ({len(r)}) and benchmark ({len(b)}) must be the same non-empty length")
    if gamma <= 0:
        raise Refused(f"gamma={gamma} must be positive")
    c = gamma / (2 * (1 + gamma))
    t = len(r)
    sr, sr2 = sum(r), sum(x * x for x in r)
    sb, sb2 = sum(b), sum(x * x for x in b)
    qa, qb, qc = -c * t, 2 * c * sr - t, sr - c * sr2 - sb + c * sb2
    disc = qb * qb - 4 * qa * qc
    if disc < 0:
        raise Refused("no fee equates the two utilities: the quadratic has no real root")
    # The stable form: the textbook (−b ± √disc)/2a loses the small root to
    # cancellation when a = −cT is tiny (γ near 0).
    q = -0.5 * (qb + (sqrt(disc) if qb >= 0 else -sqrt(disc)))
    roots = [x for x in ((q / qa) if qa else None, (qc / q) if q else None) if x is not None]
    fee = min(roots, key=abs)
    return fee * periods_per_year * 1e4


def max_drawdown(returns) -> float:
    """The largest fall of ∏(1 + r) from a previous peak, as a positive fraction; 0 if it never falls."""
    value, peak, worst = 1.0, 1.0, 0.0
    for x in _series(returns).to_list():
        value *= 1 + x
        peak = max(peak, value)
        worst = max(worst, 1 - value / peak)
    return worst


# What a return series looked like, beyond its Sharpe. Each is a description of
# one series, not a verdict: whether it survives selection is DSR's and PBO's
# question, asked of every trial. Nulls (a hole, a warm-up) are dropped, as
# for `sharpe` and `max_drawdown`.


def sortino(returns, target: float = 0.0) -> float | None:
    """Mean excess over `target` per unit of target downside deviation, per period. Null when undefined.

    Sortino and Price (1994). The downside deviation is √(Σ min(r − target, 0)² / N)
    over **all** N periods, not only the losing ones (Rollinger and Hoffman,
    "Sortino: a 'sharper' ratio", Red Rock Capital, whose worked example gives
    4.417). Annualise with `annualize`, as for Sharpe.
    """
    r = _series(returns).to_list()
    if len(r) < 2:
        return None
    excess = [x - target for x in r]
    downside = sqrt(sum(min(x, 0.0) ** 2 for x in excess) / len(excess))
    return None if not downside else sum(excess) / len(excess) / downside


def cagr(returns, periods_per_year: float) -> float | None:
    """∏(1 + r) compounded to a year: growth^(periods_per_year / N) − 1. −1 when the series is wiped out."""
    r = _series(returns).to_list()
    if not r:
        return None
    growth = 1.0
    for x in r:
        growth *= 1 + x
    return -1.0 if growth <= 0 else growth ** (periods_per_year / len(r)) - 1


def calmar(returns, periods_per_year: float) -> float | None:
    """CAGR over the maximum drawdown, over the whole series (Young 1991). Null when it never falls."""
    worst = max_drawdown(returns)
    growth = cagr(returns, periods_per_year)
    return None if not worst or growth is None else growth / worst


def omega(returns, threshold: float = 0.0) -> float | None:
    """Σ max(r − L, 0) over Σ max(L − r, 0): Keating and Shadwick's (2002) Omega at L, per period. Null with no loss."""
    r = _series(returns).to_list()
    up = sum(max(x - threshold, 0.0) for x in r)
    down = sum(max(threshold - x, 0.0) for x in r)
    return None if not down else up / down


def _drawdowns(returns) -> list[float]:
    """1 − value / peak after each period, the path starting at 1."""
    value, peak, out = 1.0, 1.0, []
    for x in _series(returns).to_list():
        value *= 1 + x
        peak = max(peak, value)
        out.append(1 - value / peak)
    return out


def ulcer_index(returns) -> float | None:
    """√(mean of squared drawdowns), as a fraction: Martin and McCann's (1989) Ulcer Index. Depth and length together."""
    dd = _drawdowns(returns)
    return None if not dd else sqrt(sum(d * d for d in dd) / len(dd))


def drawdown_duration(returns) -> int:
    """The most consecutive periods spent below a previous peak; 0 if it never falls. An unrecovered drawdown counts to the end."""
    longest = current = 0
    for d in _drawdowns(returns):
        current = current + 1 if d > 0 else 0
        longest = max(longest, current)
    return longest


def rolling_sharpe(frame: pl.DataFrame, window: int, *, column: str = "net") -> pl.DataFrame:
    """Per `(trial, ticker)`, the per-period Sharpe of the last `window` rows of `column`, on `ts`.

    A figure only for a full window: one holding a hole or a warm-up (a null
    return) is null, never a shorter window. A window with no variance is
    null. Annualise with `annualize`.
    """
    if window < 2:
        raise Refused(f"window={window}: a Sharpe needs at least 2 returns")
    utils.require(frame, ("trial", "ticker", "ts", column), "pass a trial frame from gr.studies")
    keys = ("trial", "ticker")
    sr = pl.col(column).rolling_mean(window, min_samples=window) / pl.col(column).rolling_std(window, min_samples=window)
    return (
        frame.sort(*keys, "ts")
        .select(*keys, "ts", sr.over(keys).alias("sharpe"))
        .with_columns(pl.when(pl.col("sharpe").is_finite()).then(pl.col("sharpe")).alias("sharpe"))
    )


def period_returns(frame: pl.DataFrame, every: str = "1mo", *, column: str = "net") -> pl.DataFrame:
    """Per `(trial, ticker)` and calendar period: `period, n, bars, full, return`, compounded from `column`.

    A bar belongs to the period its open (`ts`) falls in, so the bar that
    closes at midnight on the 1st is the previous month's. `bars` is how many
    bars the period holds; `n` is how many carry a return. `full` is false
    for a period cut by the sample's ends or by a hole, so a partial month is
    never read as a whole one.
    """
    utils.require(frame, ("trial", "ticker", "ts", "close_ts", column), "pass a trial frame from gr.studies")
    keys = ("trial", "ticker")
    period = pl.col("ts").dt.truncate(every)
    span = (period.dt.offset_by(every) - period).dt.total_microseconds()
    step = (pl.col("close_ts") - pl.col("ts")).dt.total_microseconds()
    return (
        frame.with_columns(period.alias("period"), (span // step).alias("_bars"))
        .group_by(*keys, "period", maintain_order=True)
        .agg(
            pl.col(column).count().alias("n"),
            pl.col("_bars").first().alias("bars"),
            ((pl.col(column).drop_nulls() + 1).product() - 1).alias("return"),
        )
        .with_columns((pl.col("n") == pl.col("bars")).alias("full"))
        .sort(*keys, "period")
    )


def describe(frame: pl.DataFrame, periods_per_year: float, *, column: str = "net") -> pl.DataFrame:
    """One row per `(trial, ticker)`: CAGR, Sharpe, Sortino, Calmar, Omega, max drawdown, Ulcer, drawdown duration.

    Sharpe and Sortino annualised by √periods_per_year; `drawdown_periods` in
    bars. A description of each trial, beside `studies.summary`, which feeds
    the Deflated Sharpe Ratio.
    """
    utils.require(frame, ("trial", "ticker", column), "pass a trial frame from gr.studies")
    rows = []
    for (trial, ticker), part in frame.partition_by("trial", "ticker", as_dict=True, maintain_order=True).items():
        r = part[column].drop_nulls()
        sr, so = sharpe(r), sortino(r)
        rows.append(
            {
                "trial": trial,
                "ticker": ticker,
                "periods": r.len(),
                "cagr": cagr(r, periods_per_year),
                "sharpe": annualize(sr, periods_per_year),
                "sortino": annualize(so, periods_per_year),
                "calmar": calmar(r, periods_per_year),
                "omega": omega(r),
                "max_drawdown": max_drawdown(r),
                "ulcer": ulcer_index(r),
                "drawdown_periods": drawdown_duration(r),
            }
        )
    schema = {"trial": pl.String, "ticker": pl.String, "periods": pl.Int64, "drawdown_periods": pl.Int64}
    return pl.DataFrame(rows, schema_overrides=schema, infer_schema_length=None)
