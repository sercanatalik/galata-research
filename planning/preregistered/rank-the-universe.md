# Pre-registered: cross-sectional factors on every Binance USDT perpetual, point in time

**Registered 2026-10-04, before any statistic of them was computed.** This
file is committed alone, before the code that runs it, so its git history is
the evidence of that. Nothing below may change after the run without a new
file that says why. The archive was being downloaded when this was written.
Nothing has been computed from it except the list of symbols it holds.

**D8 is lifted** (the operator, 2026-10-04: "lift it, use all available
tickers"). This is the repository's first study across many tickers.

## The claim being tested

Liu, Tsyvinski and Wu ("Common Risk Factors in Cryptocurrency", *Journal of
Finance* 77, 2022) find cross-sectional momentum over one to four weeks and
size effects in crypto returns. Practitioner frameworks ship the same
pipeline: rank a universe, hold the top, short the bottom. The claim
tested here: **some standard cross-sectional rule, ranked point in time
across the liquid Binance perpetuals, earns a net Sharpe that survives the
search over rules, after taker fees and funding.**

The expectation, stated before the run: **not supported.** The published
factor returns are mostly from before 2021 and from spot markets without
funding. On perps the short leg pays or receives funding every eight hours,
and a weekly long–short in the top 50 turns over a large share of its book.

## The universe, frozen

- **Candidates:** every USDT perpetual Binance's archive lists, **delisted
  ones included** (`galata-fetch daily binance-um --all`; 900 symbols listed
  on 2026-10-04, among them LUNA, FTT and SRM), with daily bars
  (`gr.reference.daily`) and settled funding (`galata-fetch funding … --all`).
- **Excluded, primary span:** Binance's index perpetuals BTCDOM, DEFI,
  FOOTBALL and BLUEBIRD, and the stablecoin USDC.
- **Excluded as well, secondary span:** the non-crypto perpetuals Binance
  listed from 2025. This list was compiled by hand from the symbol list on
  2026-10-04 and is best effort. Every excluded or kept name that enters the
  traded set is reported, so a misclassification is visible.
  AAOI AAPL ACN ADBE ALAB AMAT AMC AMD AMZN ANET ANTHROPIC APLD APP ARM ASML ASTS AVGO AXTI BABA
  BITO BMNR BRKB BX BYD BZ CBRS CIEN CL COHR COIN COPPER COST CRCL CRDO CRM CRWD CRWV CSCO
  CSOPSAMSUNG2L CSOPSKHYNIX2L CVNA CVX CXMT DDOG DELL DIA DIS DJT DKNG EBAY EWJ EWT EWY EWZ GDX
  GEV GLW GME GOOGL GS GTLB HANMI HD HIMS HK0625 HK0700 HK0992 HK1810 HOOD HPE HUT HYUNDAI IBM
  INTC IONQ IREN IWM JPM KO KODEX200 KORU KUAISHOU LGELECTRONICS LLY LRCX MARA MDB MEITUAN META
  MINIMAX MP MRK MRNA MRVL MSFT MSTR MU NATGAS NAVER NBIS NFLX NKE NOK NOW NVDA NVDL NVO OKLO
  OPENAI ORCL PANW PAXG PDD PLTR POPMART PYPL QCOM QQQ RDDT RIVN RKLB SAMSUNG SAMSUNGEM SHOP
  SKHYNIX SLX SMCI SMH SNDK SNOW SOFI SONY SOXL SOXS SPCX SPY SQQQ STRC TBT TEM TENCENT TMF
  TQQQ TSLA TSLL TSM TTWO TXN TZA UBER UNH UNITREE URNM USDBRL UVXY VRT VST WDC WMT XAG XAU
  XAUT XBI XLE XOM XPD XPT ZHIPU ZM ZS.
- **Eligible on day t:** a daily bar on t, and at least 90 daily bars up to
  and including t.
- **Traded on day t:** the 50 eligible perpetuals with the highest mean
  dollar volume (close × volume) over the 30 days ending t.

## The portfolios

Decided at the close of a rebalance day from data up to that close, and held
from the next day to the next rebalance. **Rebalanced every 7 days**,
counted from the first day with a full traded set.

- **long_short:** the top fifth (10 names) of the ranking long, the bottom
  fifth short, equal weight: +0.05 each long and −0.05 each short (gross 1,
  net 0).
- **long_only:** the top fifth long, 0.1 each (gross 1). Its benchmark is
  the **equal-weight traded set**, 0.02 each, on the same schedule.

Weights are held fixed between rebalances; drift is not traded. A coin with
no bar on a day earns nothing that day: delisted, it has exited at its last
close. Turnover at a rebalance pays **Binance's base taker fee, 0.05%**.
**Funding is charged** per day held, weight × that day's settled funding (a
long pays positive funding). A coin-day with no funding archive is charged
nothing, and the share of such position-days is reported.

| Rule | Ranked on (higher is long) | Grid | Trials |
|---|---|---|---|
| `mom L` | the return over the last L days | L ∈ {7, 30, 90} | 3 × 2 sides |
| `lowvol` | −σ of daily returns over 30 days | — | 1 × 2 |
| `resmom` | the 30-day return less β × BTC's 30-day return, β over 90 days | — | 1 × 2 |
| `small` | −the 30-day mean dollar volume | — | 1 × 2 |

**6 rules × 2 sides: N = 12.**

## What would count as support

On the **primary span, 2020-01-01 to 2024-12-31**, for the best of the 12 by
per-period Sharpe, all of:

1. **DSR ≥ 0.95 at N = 12**;
2. **Reality Check p < 0.05** among its side's six rules: long_short against
   cash, long_only against the equal-weight traded set (1,000 replicates,
   Politis–White block, seed 0);
3. **PBO < 0.5** over the 12 (16 blocks).

Anything short of that is reported as **not supported**. The secondary span,
**2025-01-01 to 2026-09-30**, reports the same 12 with the same figures. It
can say whether a verdict held, and it cannot rescue a failure.

**Described, not tested:** each rule's net Sharpe per year, its mean weekly
turnover, the funding it paid or received, and how often each excluded or
borderline name would have entered the traded set.

---

*Edited 2026-10-04 at the operator's request: a reference to an external repository was removed from the motivation above. The specification and the criteria are unchanged.*
