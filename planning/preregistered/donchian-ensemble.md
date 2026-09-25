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
