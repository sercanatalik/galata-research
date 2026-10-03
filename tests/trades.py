from datetime import timedelta

import polars as pl
import pytest
from conftest import utc

from galata_research import backtest, trades

DAY = timedelta(days=1)
FEE = 0.001


def _bars(closes, *, highs=None, lows=None, ticker="BTC", skip=()):
    t0 = utc("2026-01-01T00:00")
    days = [d for d in range(len(closes) + len(skip)) if d not in skip][: len(closes)]
    closes = [float(c) for c in closes]
    return pl.DataFrame(
        {
            "ticker": ticker,
            "ts": [t0 + d * DAY for d in days],
            "close_ts": [t0 + (d + 1) * DAY for d in days],
            "high": [float(h) for h in highs] if highs else closes,
            "low": [float(x) for x in lows] if lows else closes,
            "close": closes,
        }
    )


def _trial(bars, decided, *, fee=FEE, name="t"):
    """`decided[i]` is the position decided at bar i's close, held through bar i + 1."""
    r = backtest.returns(bars, pl.lit(pl.Series(decided, dtype=pl.Float64)), fee=fee)
    return r.with_columns(pl.lit(name).alias("trial"))


def a_trade_is_a_stretch_held_on_one_side():
    b = _bars([100, 110, 121, 121, 110, 99, 99])
    # Held: -, 1, 1, 0, -1, -1, 0.
    t = trades.table(_trial(b, [1, 1, 0, -1, -1, 0, 0], fee=0.0))
    assert t["side"].to_list() == ["long", "short"]
    assert t["bars"].to_list() == [2, 2]
    assert t["entry_ts"].to_list() == [b["close_ts"][0], b["close_ts"][3]]
    assert t["exit_ts"].to_list() == [b["close_ts"][2], b["close_ts"][5]]
    assert t["gross"].to_list() == [pytest.approx(0.21), pytest.approx(0.2)]
    assert t["open"].to_list() == [False, False]


def a_resize_keeps_the_trade():
    t = trades.table(_trial(_bars([100] * 5), [1, 0.5, 1, 0, 0]))
    assert t.height == 1
    # Turnover 1 in, 0.5 down, 0.5 up, 1 out.
    assert t["cost"][0] == pytest.approx(3 * FEE)


def the_cost_of_leaving_is_the_trade_that_left():
    # Guard: by row, the exit's cost would sit on the flat bar and the trade would look free to leave.
    t = trades.table(_trial(_bars([100, 100, 100, 100]), [1, 1, 0, 0]))
    assert t["cost"].to_list() == [pytest.approx(2 * FEE)]
    assert t["net"].to_list() == [pytest.approx((1 - FEE) * (1 - FEE) - 1)]


def a_flip_splits_its_cost_by_size():
    t = trades.table(_trial(_bars([100] * 5), [1, 1, -0.5, -0.5, -0.5]))
    # The flip turns over 1.5: 1 closes the long, 0.5 opens the short.
    assert t["cost"].to_list() == [pytest.approx(2 * FEE), pytest.approx(0.5 * FEE)]


def the_trades_costs_add_up_to_the_frames():
    frame = _trial(_bars([100, 103, 99, 104, 101, 98, 102, 100]), [1, -1, -1, 0.5, 0, 1, -1, 0])
    assert trades.table(frame)["cost"].sum() == pytest.approx(frame["cost"].sum())


def a_trade_held_at_the_end_is_open():
    t = trades.table(_trial(_bars([100, 101, 102]), [0, 1, 1]))
    assert t["open"].to_list() == [True]
    assert t["cost"].to_list() == [pytest.approx(FEE)]


def a_trade_across_a_hole_says_so_and_earns_nothing_there():
    b = _bars([100, 110, 121, 133.1], skip=(2,))
    t = trades.table(_trial(b, [1, 1, 1, 1], fee=0.0))
    assert t["spans_gap"].to_list() == [True]
    # 100 → 110 is earned; 110 → 121 straddles the hole and is not; 121 → 133.1 is.
    assert t["gross"][0] == pytest.approx(1.1 * 1.1 - 1)


def the_funding_is_charged_to_the_trade_that_held():
    frame = _trial(_bars([100, 100, 100]), [1, 1, 0], fee=0.0).with_columns(
        pl.when(pl.col("position") == 1).then(0.0001).alias("funding")
    )
    t = trades.table(frame)
    assert t["funding"].to_list() == [pytest.approx(0.0002)]
    assert t["net"][0] == pytest.approx(0.9999**2 - 1)


def a_long_trades_excursions_come_from_lows_and_highs():
    b = _bars([100, 104, 106, 100], highs=[100, 108, 107, 101], lows=[100, 97, 103, 99])
    t = trades.table(_trial(b, [1, 1, 0, 0], fee=0.0), b)
    assert t["entry_price"].to_list() == [100.0]
    assert t["mae"].to_list() == [pytest.approx(-0.03)]
    assert t["mfe"].to_list() == [pytest.approx(0.08)]


def a_short_trades_excursions_are_mirrored():
    b = _bars([100, 104, 96, 100], highs=[100, 105, 99, 101], lows=[100, 98, 95, 99])
    t = trades.table(_trial(b, [-1, -1, 0, 0], fee=0.0), b)
    assert t["mae"].to_list() == [pytest.approx(-0.05)]
    assert t["mfe"].to_list() == [pytest.approx(0.05)]


def no_excursion_is_read_from_the_deciding_bar():
    # Guard: the deciding bar's low of 50 came before the entry, at its close.
    b = _bars([100, 101, 102], highs=[100, 101, 102], lows=[50, 100, 101])
    t = trades.table(_trial(b, [1, 1, 0], fee=0.0), b)
    assert t["mae"].to_list() == [0.0]


def the_summary_counts_closed_trades_only():
    #           win      loss     win      win      loss     open
    closes = [100, 110, 110, 99, 99, 109, 109, 120, 120, 110, 110, 120]
    decided = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
    s = trades.summary(_trial(_bars(closes), decided, fee=0.0), periods_per_year=365).row(0, named=True)
    assert (s["trades"], s["closed"], s["long"], s["short"]) == (6, 5, 6, 0)
    assert s["win_rate"] == pytest.approx(3 / 5)
    wins, losses = [0.1, 10 / 99, 11 / 109], [-0.1, -1 / 12]
    assert s["profit_factor"] == pytest.approx(sum(wins) / -sum(losses))
    assert s["win_loss"] == pytest.approx((sum(wins) / 3) / (-sum(losses) / 2))
    assert (s["max_wins"], s["max_losses"]) == (2, 1)
    assert s["short_win_rate"] is None
    # 11 rows with a return; 6 of them held.
    assert s["exposure"] == pytest.approx(6 / 11)
    assert s["trades_per_year"] == pytest.approx(6 / (11 / 365))


def a_trial_that_never_trades_still_has_a_row():
    s = trades.summary(_trial(_bars([100, 101, 102]), [0, 0, 0]))
    assert s.height == 1
    assert s["trades"][0] == 0 and s["win_rate"][0] is None and s["exposure"][0] == 0.0


def a_summary_with_no_loss_has_no_profit_factor():
    s = trades.summary(_trial(_bars([100, 110, 110]), [1, 0, 0], fee=0.0))
    assert s["profit_factor"][0] is None and s["win_rate"][0] == 1.0


def a_studies_trial_reads_as_trades():
    from galata_research import studies

    b = _bars([100, 101, 103, 102, 99, 98, 100, 104, 106, 105])
    frame = studies.moving_average(b, [2], [3], sides=["long_short"], fee=FEE)
    t = trades.table(frame, b)
    # Always in the market once the slow mean exists, so every cost the frame charged is some trade's.
    assert t.height > 0 and t["cost"].sum() == pytest.approx(frame["cost"].sum())
    assert trades.summary(frame, b)["trades"][0] == t.height
