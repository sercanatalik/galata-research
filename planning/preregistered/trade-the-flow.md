# Pre-registered: order flow, micro-price and Binance's lead, on Bybit BTC and ETH

**Registered 2026-10-03, before any statistic of them was computed.** This
file is committed alone, before the code that runs it, so its git history is
the evidence of that. Nothing below may change after the run without a new
file that says why. The archives below were being downloaded when this was
written. Nothing has been computed from them.

## The claims being tested

Each claim has a **statistical** part (S: is there predictability at all?)
and an **economic** part (E: does a taker rule that uses it pay, after
crossing the spread and paying the fee both ways?). The literature mostly
supports the S parts and is silent or negative on the E parts at a taker's
costs. That is the gap investing-algorithm-framework marks as "not possible"
(showcase 22, 24) and this record can measure.

- **H1, order-flow imbalance.** Cont, Kukanov and Stoikov (2014) show OFI
  moves the price in the same interval. The claim tested is that **it also
  predicts the next one**. Expected: S weakly supported, E not.
- **H2, the micro-price.** Stoikov (2018): the book's imbalance predicts the
  next mid, so a size-weighted mid forecasts the future mid better than the
  mid does. Expected: S supported. There is no E: the micro-price is a
  better mark, not a trade.
- **H3, Binance's lead.** `liquidity_venues.py` ⑨ found Binance leads Bybit
  in 94–96% of hours, by about 50–100 ms. **The claim tested is that
  Binance's last second predicts Bybit's next one**, and that a taker on
  Bybit can trade it. Expected: S supported, E not. The lead is far
  shorter than a second, and a 1-second move rarely clears 11 bps of fees.

## The data, frozen

- **Days:** the first Wednesday of each month, 2025-10-01 to 2026-09-02:
  **12 days** (`--sample monthly:wed`), the same days for every series.
- **Bybit linear BTCUSDT and ETHUSDT:** its book replayed to one row per
  second (`gr.reference.book`, ob200), giving the best bid and ask and their
  sizes at each second.
- **Binance USDⓈ-M BTCUSDT and ETHUSDT:** aggTrades (`gr.reference.trades`).
  Binance's price at a second is its last trade at or before it.
- A second with a crossed or locked Bybit book, or with no Binance trade in
  the last 5 s, gives no observation. Days are not joined: no window spans
  midnight, so each day is its own stretch.
- **Costs (E):** Bybit's base taker fee, **0.055% each way**. Entry and exit
  are at the touch: buy at the ask, sell at the bid. No maker rebates, no
  queue, no latency model; the decision at second t trades at second t's
  touch. **Fills at the decision second's own touch flatter E:** a real
  taker arrives later. A rule that fails E fails even with that help.

## H1: order-flow imbalance

`gr.liquidity.ofi` on the per-second book with 10-second buckets. Each
bucket's OFI is normalised by its mean depth (ofi / depth). x is that
normalised OFI in bucket b; y is the mid's log return over bucket b + 1, in bps.

- **S:** a pooled OLS of y on x per ticker, with Newey–West standard errors
  (6 lags). **Supported if the slope is positive with t > 3 on both tickers.**
- **E:** z is x's z-score over the day's previous 360 buckets (an hour). Buy
  at the ask when z > k, sell at the bid when z < −k, close at the touch
  after h. k ∈ {1, 2, 3} × h ∈ {10 s, 60 s}: **6 rules per ticker, N = 12.**

## H2: the micro-price

At each second t: the mid m = (bid + ask)/2, and the weighted mid
w = (bid × ask_sz + ask × bid_sz)/(bid_sz + ask_sz). The target is the mid
at t + h, h ∈ {1 s, 10 s}.

- **S:** squared errors e_m = (m_{t+h} − m_t)², e_w = (m_{t+h} − w_t)², in
  bps of the mid. Diebold–Mariano on d = e_m − e_w, Newey–West (h + 5 lags).
  **Supported if the weighted mid's error is smaller with p < 0.01 at both
  horizons on both tickers.**

## H3: Binance's lead

x is Binance's log return over second t (last trade to last trade); y is
Bybit's mid log return over second t + 1, in bps.

- **S:** a pooled OLS of y on x per ticker, Newey–West (6 lags). **Supported
  if the slope is positive with t > 3 on both tickers.**
- **E:** buy Bybit at the ask when x > k bps, sell at the bid when x < −k,
  close at the touch after h. k ∈ {2, 5, 10} bps × h ∈ {1 s, 5 s, 30 s}:
  **9 rules per ticker, N = 18.**

## What would count as support for E

For each of H1 and H3: the best rule (highest mean net bps per trade) must
have **a mean net return per trade above 0 with a one-sided p below 0.05/N**
(Bonferroni over its family: N = 12 and 18). The p comes from a t on the
trades' daily means (12 days, so the day is the unit and overlapping trades
within a day are not counted as independent). It must also be **positive on
both tickers.** A rule with fewer than 30 trades has no figure, but still
counts toward N.

Anything short of a part's criteria is reported as **not supported**.
Nothing is tuned, dropped or re-run under this name.
