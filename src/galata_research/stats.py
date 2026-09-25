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

from ._errors import Refused

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


def stationary_bootstrap_indices(n: int, block: float, rng) -> list[int]:
    """Politis and Romano's (1994) stationary bootstrap: blocks of geometric length, mean `block`.

    Each index continues the last one (wrapping at `n`) with probability
    `1 − 1/block`, and otherwise starts a new block at a uniform position.
    """
    if n < 1 or block < 1:
        raise Refused(f"n={n}, block={block}: a stationary bootstrap needs n ≥ 1 and block ≥ 1")
    p = 1.0 / block
    out = [rng.randrange(n)]
    for _ in range(n - 1):
        out.append(rng.randrange(n) if rng.random() < p else (out[-1] + 1) % n)
    return out


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
    of the columns' Politis–White blocks (`optimal_block`). p-values count
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


def optimal_block(series) -> float:
    """Politis and White's (2004) optimal mean block for the stationary bootstrap, as corrected in 2009.

    Patton, Politis and White (2009) corrected the constants after Nordman
    (2008). This follows arch 8.0.0's `optimal_block_length` (its
    `stationary` column) step for step:
    - find m̂, the first run of `max(5, ⌊log₁₀ n⌋)` autocorrelations under
      `2√(log₁₀ n / n)`, and set `M = 2m̂`;
    - with a flat-top lag window, take `Ĝ = Σ 2λ(k/M)·k·γ̂(k)` and the
      long-run variance `ĝ = γ̂(0) + Σ 2λ(k/M)·γ̂(k)`;
    - `b = (2Ĝ² / (2ĝ²))^{1/3} n^{1/3}`, capped at `min(3√n, n/3)`.
    """
    from math import ceil, log10

    x = _series(series).to_list()
    n = len(x)
    if n < 10:
        raise Refused(f"{n} points are too few to choose a block")
    mean = sum(x) / n
    eps = [v - mean for v in x]
    b_max = ceil(min(3 * sqrt(n), n / 3))
    kn = max(5, int(log10(n)))
    m_max = int(ceil(sqrt(n))) + kn
    cv = 2 * sqrt(log10(n) / n)

    def dot(a, b):
        return sum(p * q for p, q in zip(a, b))

    acv, acorr, opt_m = [], [], None
    for i in range(m_max + 1):
        v1, v2 = dot(eps[i + 1 :], eps[i + 1 :]), dot(eps[: n - (i + 1)], eps[: n - (i + 1)])
        cross = dot(eps[i:], eps[: n - i])
        acv.append(cross / n)
        acorr.append(abs(cross) / sqrt(v1 * v2) if v1 * v2 > 0 else 0.0)
        if i >= kn and opt_m is None and all(a < cv for a in acorr[i - kn : i]):
            opt_m = i - kn
    m = min(2 * max(opt_m, 1) if opt_m is not None else m_max, m_max)

    g, long_run = 0.0, acv[0]
    for k in range(1, m + 1):
        lam = 1.0 if k / m <= 0.5 else 2 * (1 - k / m)
        g += 2 * lam * k * acv[k]
        long_run += 2 * lam * acv[k]
    if long_run <= 0:
        return 1.0
    b = ((2 * g**2) / (2 * long_run**2)) ** (1 / 3) * n ** (1 / 3)
    return min(b, b_max)
