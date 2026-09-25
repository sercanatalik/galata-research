# Pre-registered: the Donchian ensemble with volatility sizing, on Hyperliquid BTC and ETH

**Registered 2026-09-25, before any backtest of it was run.** This file is
committed before the code that runs it, so its git history is the evidence
of that. Nothing below may change after the run without a new file that
says why.

## The claim being tested

Zarattini, Pagani and Barbon, *Catching Crypto Trends: A Tactical Approach
for Bitcoin and Altcoins* (SSRN 5209907, 2025), report an ensemble of
Donchian-channel trend models with volatility sizing: Sharpe 1.58 net of
fees on a rotation of the 20 most liquid coins, January 2015 to March 2025.
galata-legacy's literature review names it as "the practitioner form" of the
strongest evidence in crypto, time-series trend (`design/gaps-vs-literature.md`
§2.1), and warns that volatility management "does not change the tail risk"
(Grobys et al., 2025).

This test is **narrower** than the paper: two perps, not a 20-coin rotation,
on one venue, over a shorter and partly later window.

## The specification, frozen

The rules are as stated in a public pre-registration of the same replication
(github.com/zebadee2kk/DeFi-TraderStack-Agent issue #137, citing the paper
through CXO Advisory's summary):

- **Bars:** `gr.market.candles(["BTC", "ETH"], "1d", ...)`, traded bars only
  (from 2023-02-26), closed, next-bar execution (`gr.backtest.returns`).
- **Lookbacks:** L ∈ {5, 10, 20, 30, 60, 90, 150, 250, 360} days.
- **Entry, per lookback:** a close above the highest close of the previous L
  days opens that lookback's position.
- **Exit, per lookback:** a trailing stop that starts at, and then ratchets up
  to, the higher of its previous value and the channel midpoint (the mean of
  the highest and lowest close of the previous L days). A close below the
  stop closes the position.
- **Ensemble:** equal weight, so the signal is the fraction of the nine
  lookbacks that are open.
- **Sizing:** the signal times `min(1, 0.25 / σ)`, where σ is the
  annualised (×√365) standard deviation of the last 90 daily close-to-close
  returns. Leverage is capped at 1.
- **Long-only.** Rebalanced at each daily close, with the 0.045% taker fee on
  every change of position. **Funding is not charged** (the record holds five
  days of it), which flatters a long-only strategy by up to about 11.6% a
  year at the interest floor.

## The trials, and N

Four trials: {BTC, ETH} × {sized as above, unsized (signal × 1)}. The
unsized pair is the ablation of the volatility sizing. **N = 4.** The 66
example trials run earlier today tested a different, un-registered family.
The report shows DSR at N = 4 and at N = 70 (all trials run in this
repository), and reads the conclusion at both.

## What would count as support

For the sized ensemble on BTC, all of:
1. **DSR ≥ 0.95 at N = 4**, per-period Sharpe with its own skewness and
   kurtosis;
2. its Sharpe is above buy-and-hold BTC's over the same days;
3. **PBO < 0.5** by CSCV (16 blocks) over the four trials plus buy-and-hold,
   as a sanity check on selection. With so few trials it is a weak test, and
   is read that way.

And, separately, **Grobys et al.'s claim:** the worst day and the 1% expected
shortfall of the sized ensemble are *no better* than the unsized one's,
relative to their volatility. Reported as a ratio, with no threshold.

Anything short of 1–3 is reported as **not supported on this record**. It is
not tuned, and not re-run with different parameters under this name.

---

## Result — run once, 2026-09-25, `notebooks/donchian_ensemble.py`

*Appended after the run. Nothing above this line was changed.*

The four registered trials and buy-and-hold were scored over the same 1,216
days (2023-05-28 → 2026-09-24): from the first day the sized ensemble has a
position, which is when 90 returns exist for its volatility.

| trial | ticker | Sharpe, annual | total net | max drawdown |
|---|---|---|---|---|
| donchian sized | BTC | 0.97 | +52.5% | −15.0% |
| donchian unsized | BTC | 0.96 | +102.6% | −20.1% |
| buy and hold | BTC | 0.97 | +214.6% | −53.0% |
| donchian sized | ETH | 0.72 | +30.4% | −14.0% |
| donchian unsized | ETH | 0.58 | +50.8% | −34.0% |
| buy and hold | ETH | 0.50 | +46.9% | −67.6% |

**The registered criteria, for the sized ensemble on BTC:**
1. DSR at N = 4: **0.921**. **Not met** (≥ 0.95). At N = 70 it is 0.824.
2. Sharpe against buy-and-hold BTC: **0.966 against 0.975**. **Not met**, by 0.01.
3. PBO by CSCV (16 blocks, five columns): **0.631**. **Not met** (< 0.5).

**Verdict: not supported on this record.**

**Grobys et al.'s claim, relative to each trial's own volatility:** sized
BTC's worst day was −7.4 σ against −5.2 σ unsized, and its 1% expected
shortfall −4.1 σ against −4.0 σ. For ETH: −4.9 σ against −5.7 σ, and −3.7 σ
against −3.7 σ. The 1% shortfall is unchanged by sizing on both, relative to
the risk taken, as they said. The worst day went the other way on each:
worse sized on BTC, better sized on ETH. It is one day each, so it is read
as noise.

**Observed, not registered** (post hoc, so it tests nothing):
- The ensemble cut the maximum drawdown to about a quarter of buy-and-hold's
  on both perps, at the same or a better Sharpe.
- On ETH, the sized ensemble's Sharpe (0.72) was above buy-and-hold's (0.50).

These are what a trend filter is expected to do, but they are observations
made after seeing the data. They could only become a test under a new
registration, run on data after 2026-09-25. Funding was not charged, which
flatters every long position here, the ensemble's included.
