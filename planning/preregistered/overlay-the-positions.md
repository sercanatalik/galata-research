# Pre-registered: stop-loss and take-profit overlays on three daily rules, Hyperliquid BTC and ETH

**Registered 2026-10-03, before any backtest of them was run.** This file is
committed alone, before the code that runs it, so its git history is the
evidence of that. Nothing below may change after the run without a new file
that says why.

## The claim being tested

Kaminski and Lo, "When do stop-loss rules stop losses?" (*Journal of
Financial Markets* 18, 2014), show that a stop-loss with re-entry adds to
expected return only when returns have momentum (positive serial
correlation), and costs return under a random walk. Practitioner
frameworks (investing-algorithm-framework's `StopLossRule`, `TakeProfitRule`,
`CooldownRule`) ship stops as a default, implying they help. The claim tested
here: **on this record, some stop or take-profit overlay improves a daily
rule's net Sharpe by more than the search over overlays would give by
chance.**

The expectation, stated before the run: **not supported.** The search over
bars permuted out of order beat the real best on this record
(`permuted_bars.py`, p = 0.59), which is what no exploitable momentum looks
like, and that is Kaminski and Lo's case where a stop costs.

## The specification, frozen

- **Bars:** `gr.market.candles(["BTC", "ETH"], "1d", ...)`, traded and closed
  bars, on the days both tickers have, from the first traded bar to the last
  closed bar at run time. 0.045% taker fee on every change of position,
  the overlay's exits included. **Funding is not charged.**
- **Three bases**, each a `gr.studies` trial:
  - `buy and hold` (Kaminski and Lo's own setting);
  - `ma 20/100 long_flat` (`studies.moving_average`);
  - `donchian unsized` (`studies.donchian_ensemble(sized=False)`, as registered
    in `donchian-ensemble.md`).
- **The overlay** (`gr.overlays.apply`), per ticker:
  - a trade's entry price is the close that decided it;
  - **stop:** for a long, at entry × (1 − s); **trailing**, at the highest
    high from the entry through the previous bar × (1 − s); mirrored for a
    short;
  - **take-profit:** for a long, at entry × (1 + t);
  - checked against each held bar's open, low and high, never against the
    bar that decided the position. A bar that opens beyond a level fills at
    the open, and one that touches it fills at the level. When a bar
    touches both the stop and the take-profit, the stop is taken;
  - after an exit, flat for a **cooldown of 10 bars**, then the base rule's
    position again.
- **Grid per base:** s ∈ {5%, 10%, 20%} × {fixed, trailing} × t ∈ {none, 25%}:
  **12 overlays**.

## The trials, and N

(3 bases + 36 overlaid) × {BTC, ETH} = **N = 78**. Every one is a column in
the tests below.

## What would count as support

**H1, per base and ticker (six tests).** White's Reality Check over the 12
overlays' net returns in excess of their own base (`studies.excess` against
the base, `stats.reality_check`, 1,000 replicates, Politis–White block,
seed 0). **The claim is supported for a base if p < 0.05 on both tickers.**
It is supported overall if it is supported for at least one base.

**H2, the whole set.** The best of all 78 by per-period Sharpe has
**DSR ≥ 0.95 at N = 78**, and **PBO < 0.5** by CSCV (16 blocks) over the 78
columns.

Anything short of H1 for some base is reported as **not supported on this
record**. H2 is reported either way, and cannot by itself support the claim:
it can be met by a base alone.

**Described, not tested:** each overlay's maximum drawdown and Ulcer index
as a ratio to its base's, the share of trades closed by the stop or the take-profit,
and `gr.trades.summary` per trial.
