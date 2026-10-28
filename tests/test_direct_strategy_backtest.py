"""Verification tests for the OFFLINE evaluation of the direct multi-setup setups.

Directive requirement: FIRST verify backward backtest-engine semantics
(short-trade representation, execution timing, stop/TP resolution and cost
accounting) before reporting any metric. This file does exactly that:

1. Parity with the existing long-only engine (`src.backtest.engine.run_backtest`)
   for execution timing, opening-gap stop/TP handling, ambiguous-bar
   stop-first tie-break, and turnover-based costs.
2. Short-side mirror semantics verified on hand-computed candle scenarios
   (the existing engine is long-only and cannot represent shorts).
3. Parity of the vectorized decision series with the LIVE evaluator
   (`evaluate_direct_live_signal`) candle-by-candle on the committed genuine
   5m historical CSV.
4. Walk-forward split chronology (no leakage).

All publication stays disabled here (offline research module only).
"""
import datetime

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.evaluation.direct_strategy_backtest import (
    TRANSACTION_COST,
    SLIPPAGE,
    SETUP_VARIANTS,
    expanding_anchored_folds,
    compute_decision_series,
    simulate_trades,
    _decisions_for_variant,
)
from src.evaluation.live_direct_strategy import (
    BREAKOUT_LOOKBACK,
    SLOW_MA_PERIOD,
    FAST_MA_PERIOD,
    PULLBACK_LOOKBACK,
    BUY,
    SELL,
    NO_TRADE,
)
from src.evaluation.live_execution_runtime import get_canonical_timeframe_duration
from src.evaluation.live_direct_strategy import evaluate_direct_live_signal
from src.evaluation.direct_strategy_backtest import summarize_metrics

REF_NOW = datetime.datetime(2025, 6, 1, 12, 0, tzinfo=datetime.timezone.utc)
COST_RATE = TRANSACTION_COST + SLIPPAGE


def candles(opens, highs, lows, closes, start="2025-01-01T00:00:00Z", freq="5min"):
    ts = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "open": opens, "high": highs, "low": lows, "close": closes})
    df["return"] = df["close"].pct_change().fillna(0.0)  # engine-required return column
    return df


def flat_decisions(n):
    return pd.DataFrame({
        "decision": [NO_TRADE] * n,
        "stop_loss": [np.nan] * n,
        "take_profit": [np.nan] * n,
    })


# ---------------------------------------------------------------------------
# 1. Parity with the existing long-only engine
# ---------------------------------------------------------------------------

def test_execution_timing_matches_engine_next_bar_open() -> None:
    df = candles(
        opens=[100 + i for i in range(6)],
        highs=[101 + i for i in range(6)],
        lows=[99 + i for i in range(6)],
        closes=[100 + i for i in range(6)],
    )
    dec = flat_decisions(len(df))
    dec.loc[len(df) // 2, "decision"] = BUY
    dec.loc[len(df) // 2, "stop_loss"] = 0.0          # never hit
    dec.loc[len(df) // 2, "take_profit"] = 99999.0    # never hit

    trades, net = simulate_trades(df, dec, transaction_cost=0.0001, slippage=0.0)
    enter_idx = len(df) // 2 + 1
    # Execution happens at the NEXT bar's open, exactly like the engine.
    assert trades.iloc[0]["entry_price"] == df["open"].iloc[enter_idx]
    assert trades.iloc[0]["entry_timestamp"] == df["timestamp"].iloc[enter_idx]

    # Engine equivalence: engine enters on the SAME bar (prev_sig == 1) at that
    # bar's open, and books (close - entry)/entry for the entry bar.
    sig = [0] * len(df)
    sig[len(df) // 2] = 1
    eng = run_backtest(
        df.assign(signal=sig), signal_column="signal", return_column="return",
        transaction_cost=0.0001, slippage=0.0,
        stop_loss_pct=0.9, take_profit_pct=9.9,  # never hit in this range
    )
    assert eng["gross_strategy_return"].iloc[enter_idx] == pytest.approx(
        (df["close"].iloc[enter_idx] - df["open"].iloc[enter_idx]) / df["open"].iloc[enter_idx]
    )


def test_stop_target_and_open_gap_parity_with_engine() -> None:
    # Long entry at bar 3 open; bar 4 opens BELOW the stop level: exit must use
    # the gapped open (engine-formula parity).
    df = candles(
        opens=[1000, 1001, 1002, 1003, 900, 1005],
        highs=[1004, 1005, 1006, 1007, 1008, 1009],
        lows=[998, 999, 1000, 1003, 899, 1003],
        closes=[1000.5, 1001.5, 1002.5, 1003.5, 905, 1006],
    )
    dec_entry = 2  # decision on candle 2 -> executes at bar 3's open
    dec = flat_decisions(len(df))
    dec.loc[dec_entry, "decision"] = BUY
    dec.loc[dec_entry, "stop_loss"] = 1002.5   # absolute stop (decision candle low)
    dec.loc[dec_entry, "take_profit"] = 99999.0

    trades, net = simulate_trades(df, dec, transaction_cost=0.0001, slippage=0.0)
    entry = df["open"].iloc[3]
    assert trades.iloc[0]["entry_price"] == entry
    stop_pct = (entry - 1002.5) / entry
    # Engine equivalence: hold the long signal until the stop bar's close, then exit.
    sig2 = [0, 0, 1, 1, 0, 0][: len(df)]
    eng2 = run_backtest(
        df.assign(signal=sig2), signal_column="signal", return_column="return", transaction_cost=0.0001, slippage=0.0,
        stop_loss_pct=stop_pct, take_profit_pct=99999.0 / entry,
    )
    # Entry bar return must match ((close-entry)/entry)
    assert eng2["gross_strategy_return"].iloc[3] == pytest.approx(
        (df["close"].iloc[3] - entry) / entry
    )
    # Stop bar exit: engine uses the gapped open (900) because open < stop:
    assert eng2["gross_strategy_return"].iloc[4] == pytest.approx((900 - df["close"].iloc[3]) / df["close"].iloc[3])
    # My simulator: same gapped-open exit, recorded as a trade open->exit direct return.
    assert trades.iloc[0]["exit_reason"] == "stop"
    assert trades.iloc[0]["exit_price"] == 900.0
    assert trades.iloc[0]["gross_return"] == pytest.approx((900.0 - entry) / entry)
    # Per-bar chain equality for the stop bar
    assert net.iloc[4] == pytest.approx(eng2["gross_strategy_return"].iloc[4] - eng2["turnover"].iloc[4] * 0.0001)


def test_ambiguous_stop_and_target_bar_is_stop_first_parity() -> None:
    # Long entry at bar 3 open; bar 4 hits BOTH stop (low <= sl) and target
    # (high >= tp) with open between the barriers: engine resolves stop-first.
    entry_open = 1003.0
    sl_price, tp_price = 1000.0, 1010.0
    df = candles(
        opens=[1000, 1001, 1002, entry_open, 1005],
        highs=[1004, 1005, 1006, 1007, 1011],
        lows=[998, 999, 1000, 1001, 999],
        closes=[1000.5, 1001.5, 1002.5, 1003.5, 1008],
    )
    dec_entry = 2
    dec = flat_decisions(len(df))
    dec.loc[dec_entry, "decision"] = BUY
    dec.loc[dec_entry, "stop_loss"] = sl_price
    dec.loc[dec_entry, "take_profit"] = tp_price

    trades, net = simulate_trades(df, dec, transaction_cost=0.0001, slippage=0.0)
    stop_pct = (entry_open - sl_price) / entry_open
    tp_pct = (tp_price - entry_open) / entry_open
    sig2 = [0, 0, 1, 1, 0]
    eng2 = run_backtest(
        df.assign(signal=sig2), signal_column="signal", return_column="return", transaction_cost=0.0001, slippage=0.0,
        stop_loss_pct=stop_pct, take_profit_pct=tp_pct,
    )
    stop_exit = eng2["gross_strategy_return"].iloc[4]
    assert stop_exit == pytest.approx((sl_price - df["close"].iloc[3]) / df["close"].iloc[3])
    assert trades.iloc[0]["exit_reason"] == "stop_or_target_same_bar_stop_first"
    assert trades.iloc[0]["exit_price"] == sl_price
    assert trades.iloc[0]["gross_return"] == pytest.approx((sl_price - entry_open) / entry_open)
    assert net.iloc[4] == pytest.approx(stop_exit - eng2["turnover"].iloc[4] * 0.0001)


def test_turnover_cost_accounting_matches_engine() -> None:
    # Entry+exit in the same bar books turnover 2, exactly like the engine.
    entry_open = 1003.0
    df = candles(
        opens=[1000, 1001, 1002, entry_open, 1005],
        highs=[1004, 1005, 1006, 1100.0, 1011],
        lows=[998, 999, 1000, 1001, 999],
        closes=[1000.5, 1001.5, 1002.5, 1095.0, 1008],
    )
    dec_entry = 2
    dec = flat_decisions(len(df))
    dec.loc[dec_entry, "decision"] = BUY
    dec.loc[dec_entry, "stop_loss"] = 990.0    # not hit intrabar (low 1001 > 990)
    dec.loc[dec_entry, "take_profit"] = 1090.0  # hit on entry bar

    trades, net = simulate_trades(df, dec, transaction_cost=0.0001, slippage=0.0)
    tp_pct = (1090.0 - entry_open) / entry_open
    sl_pct = (entry_open - 990.0) / entry_open
    sig3 = [0, 0, 1, 0, 0]
    eng3 = run_backtest(
        df.assign(signal=sig3), signal_column="signal", return_column="return",
        transaction_cost=0.0001, slippage=0.0,
        stop_loss_pct=sl_pct, take_profit_pct=tp_pct,
    )
    assert eng3["turnover"].iloc[3] == 2.0
    assert net.iloc[3] == pytest.approx(
        eng3["gross_strategy_return"].iloc[3] - eng3["turnover"].iloc[3] * 0.0001
    )
    assert trades.iloc[0]["exit_reason"] == "take_profit"
    assert trades.iloc[0]["exit_price"] == 1090.0
    assert trades.iloc[0]["net_return"] == pytest.approx(
        trades.iloc[0]["gross_return"] - 2 * 0.0001
    )


# ---------------------------------------------------------------------------
# 2. Short side: exact mirror semantics, hand-computed
# ---------------------------------------------------------------------------

def test_short_trade_exits_at_stop_with_gapped_open_and_costs() -> None:
    # Short entry at bar 3 open; bar 4 opens ABOVE our stop: exit at gapped open.
    df = candles(
        opens=[2000, 2001, 2002, 2003, 2100, 2005],
        highs=[2004, 2005, 2006, 2049, 2101, 2009],
        lows=[1998, 1999, 2000, 2001, 2051, 2003],
        closes=[2000.5, 2001.5, 2002.5, 2045.0, 2095.0, 2006],
    )
    dec_entry = 2
    dec = flat_decisions(len(df))
    dec.loc[dec_entry, "decision"] = SELL
    dec.loc[dec_entry, "stop_loss"] = 2050.0   # decision-candle high (invalidation)
    dec.loc[dec_entry, "take_profit"] = 1900.0

    trades, net = simulate_trades(df, dec, transaction_cost=0.0001, slippage=0.0)
    t = trades.iloc[0]
    assert t["direction"] == "short"
    assert t["entry_price"] == 2003.0
    assert t["exit_reason"] == "stop"
    assert t["exit_price"] == 2100.0  # gapped open above the stop
    assert t["gross_return"] == pytest.approx((2003.0 - 2100.0) / 2003.0)
    assert t["net_return"] == pytest.approx(t["gross_return"] - 2 * 0.0001)
    # Per-bar chain (net of the turnover costs booked on those bars):
    # bar 3 holds the short (turnover 1 = entry), bar 4 exits (turnover 1).
    assert net.iloc[3] == pytest.approx((2003.0 - df["close"].iloc[3]) / 2003.0 - 0.0001)
    assert net.iloc[4] == pytest.approx((df["close"].iloc[3] - 2100.0) / df["close"].iloc[3] - 0.0001)
    assert trades.iloc[0]["bars_held"] == 1


def test_short_trade_take_profit_then_buy_flip_and_final_liquidation() -> None:
    # Short enters at bar 3 open; hits TP intrabar at bar 4 (stop-first preserved:
    # bar 3 holds with no barriers); BUY decision at bar 5 enters long at bar 6
    # open; tail position is honestly liquidated at the final close.
    df = candles(
        opens=[2000, 2001, 2002, 2003, 2004.5, 1010, 3000],
        highs=[2004, 2005, 2006, 2040, 2030, 1012, 3005],
        lows=[1998, 1999, 2000, 2000, 1990, 1008, 2995],
        closes=[2000.5, 2001.5, 2002.5, 2002.0, 1992.0, 1011, 3002],
    )
    dec = flat_decisions(len(df))
    dec.loc[2, "decision"] = SELL
    dec.loc[2, "stop_loss"] = 2050.0
    dec.loc[2, "take_profit"] = 1995.0    # hit at bar 4 (low 1990 <= tp)
    dec.loc[5, "decision"] = BUY
    dec.loc[5, "stop_loss"] = 2000.0
    dec.loc[5, "take_profit"] = 99999.0

    trades, net = simulate_trades(df, dec, transaction_cost=0.0001, slippage=0.0)
    assert len(trades) == 2
    short, long_ = trades.iloc[0], trades.iloc[1]
    assert short["exit_reason"] == "take_profit"
    assert short["exit_price"] == 1995.0
    assert short["gross_return"] == pytest.approx((2003.0 - 1995.0) / 2003.0)
    assert short["bars_held"] == 1
    assert long_["direction"] == "long"
    assert long_["entry_price"] == 3000.0  # bar 6 open (decision at bar 5)
    assert long_["exit_reason"] == "final_close_liquidation"
    assert long_["exit_price"] == 3002.0
    # Long entry-bar gain books the open-to-close move once, net of the
    # entry+liquidation turnover (2 x 0.0001) booked on that bar.
    assert net.iloc[6] == pytest.approx((df["close"].iloc[6] - 3000.0) / 3000.0 - 0.0002)
    # Flat bar 5 contributes nothing
    assert net.iloc[5] == 0.0


def test_costs_are_nonzero_and_consistent_in_trade_records() -> None:
    df = candles(
        opens=[2000, 2001, 2002, 2003, 2004],
        highs=[2004, 2005, 2006, 2050, 2008],
        lows=[1998, 1999, 2000, 2001, 1990],
        closes=[2000.5, 2001.5, 2002.5, 2003.5, 1995.0],
    )
    dec = flat_decisions(len(df))
    dec.loc[2, "decision"] = SELL
    dec.loc[2, "stop_loss"] = 2050.0
    dec.loc[2, "take_profit"] = 1990.0
    trades, net = simulate_trades(
        df, dec, transaction_cost=TRANSACTION_COST, slippage=SLIPPAGE
    )
    t = trades.iloc[0]
    assert t["net_return"] == pytest.approx(t["gross_return"] - 2 * COST_RATE)


# ---------------------------------------------------------------------------
# 3. Decision-series parity with the LIVE evaluator (committed genuine 5m CSV)
# ---------------------------------------------------------------------------

def _load_committed_csv_slice(rows: int = 260) -> pd.DataFrame:
    raw = pd.read_csv("data/raw/xauusd_5m/ds_moving_average_5m.csv")
    prepared = pd.DataFrame({
        "timestamp": pd.to_datetime(raw["timestamp"], utc=True),
        "open": raw["open"].astype(float),
        "high": raw["high"].astype(float),
        "low": raw["low"].astype(float),
        "close": raw["close"].astype(float),
    }).sort_values("timestamp").reset_index(drop=True)
    return prepared.iloc[-rows:].reset_index(drop=True)


def test_vectorized_decisions_match_live_evaluator_on_real_history() -> None:
    prepared = _load_committed_csv_slice()
    series = compute_decision_series(prepared)
    duration = get_canonical_timeframe_duration("5m").total_seconds()
    checked = 0
    for i in range(len(prepared)):
        if i < SLOW_MA_PERIOD:
            assert series["decision"].iloc[i] == NO_TRADE
            assert series["reason"].iloc[i] == "insufficient_history"
            continue
        slice_df = prepared.iloc[: i + 1]
        live = evaluate_direct_live_signal(
            slice_df, symbol="XAUUSD", timeframe="5m",
            reference_now=slice_df["timestamp"].iloc[-1] + pd.Timedelta(seconds=duration),
            max_age_seconds=10_000_000.0,
        )
        assert series["decision"].iloc[i] == live.decision, f"row {i}"
        assert series["reason"].iloc[i] == live.reason, f"row {i}"
        if live.decision == NO_TRADE:
            assert pd.isna(series["entry_price"].iloc[i])
        else:
            assert series["entry_price"].iloc[i] == pytest.approx(live.entry_price)
            assert series["stop_loss"].iloc[i] == pytest.approx(live.stop_loss)
        checked += 1
    assert checked >= len(prepared) - SLOW_MA_PERIOD


def test_no_lookahead_decision_series_matches_truncated_inputs() -> None:
    prepared = _load_committed_csv_slice(rows=220)
    full = compute_decision_series(prepared)
    for i in range(25, len(prepared), 17):
        trunc = compute_decision_series(prepared.iloc[: i + 1].reset_index(drop=True))
        assert trunc["decision"].iloc[i] == full["decision"].iloc[i]
        assert trunc["reason"].iloc[i] == full["reason"].iloc[i]
        if full["decision"].iloc[i] != NO_TRADE:
            assert trunc["stop_loss"].iloc[i] == pytest.approx(full["stop_loss"].iloc[i])


def test_use_of_constant_parameters_is_explicit() -> None:
    assert SLOW_MA_PERIOD == 20 and FAST_MA_PERIOD == 5
    assert PULLBACK_LOOKBACK == 3 and BREAKOUT_LOOKBACK == 10


def test_summarize_metrics_handles_zero_trade_segment() -> None:
    """Regression: a segment with NO trades must not crash and reports honest zero metrics."""
    df = candles(
        opens=[100, 101, 102, 103, 104, 105],
        highs=[101, 102, 103, 104, 105, 106],
        lows=[99, 100, 101, 102, 103, 104],
        closes=[100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
    )
    summary = summarize_metrics(
        df, flat_decisions(len(df)), pd.DataFrame(), pd.Series(0.0, index=df.index), "5m"
    )
    assert summary["trades"] == 0
    assert summary["wins"] == 0
    assert summary["win_rate"] == 0.0
    assert summary["expectancy"] == 0.0
    assert summary["net_return_after_costs"] == 0.0
    assert summary["max_drawdown"] == 0.0
    assert summary["avg_trade_bars"] == 0.0
    assert summary["buy_signals"] == 0 and summary["sell_signals"] == 0
    assert summary["no_trade_signals"] == len(df)


def test_numeric_epoch_timestamp_csv_is_parsed_with_real_dates(tmp_path) -> None:
    """Regression: the committed daily CSV format (unix epoch seconds) must yield
    real 2025 date ranges in the report, never the 1970 epoch of ns misparse."""
    csv = tmp_path / "daily.csv"
    csv.write_text(
        "timestamp,open,high,low,close\n"
        "1735689600,2625.1,2626.0,2621.48,2623.56\n"
        "1735776000,2623.66,2660.38,2622.45,2658.3\n"
    )
    prepared = pd.DataFrame({
        "timestamp": pd.to_datetime(pd.read_csv(csv)["timestamp"], unit="s", utc=True),
        "open": [2625.1, 2623.66],
        "high": [2626.0, 2660.38],
        "low": [2621.48, 2622.45],
        "close": [2623.56, 2658.3],
    })
    decisions = compute_decision_series(prepared)
    summary = summarize_metrics(
        prepared, decisions, pd.DataFrame(), pd.Series(0.0, index=prepared.index), "1D"
    )
    assert summary["date_range"][0].startswith("2025-01-01")
    assert summary["date_range"][-1].startswith("2025-01-02")


# ---------------------------------------------------------------------------
# 4. Variant filtering and walk-forward splits
# ---------------------------------------------------------------------------

def test_variant_filtering_matches_v1_crossover_only_semantics() -> None:
    prepared = _load_committed_csv_slice(rows=520)
    full = compute_decision_series(prepared)
    crossover_only = _decisions_for_variant(full, "crossover_only")
    met_mask = full["reason"].astype(str).str.startswith("crossover_")
    assert (crossover_only.loc[~met_mask, "decision"] == NO_TRADE).all()
    assert (crossover_only.loc[met_mask, "decision"] == full.loc[met_mask, "decision"]).all()
    assert (crossover_only.loc[met_mask, "entry_price"] == full.loc[met_mask, "entry_price"]).all()


def test_variants_registry_contains_live_product_and_v1_baseline() -> None:
    assert SETUP_VARIANTS["multi_setup"][0] == "crossover"
    assert SETUP_VARIANTS["multi_setup"][1] == "pullback_continuation"
    assert SETUP_VARIANTS["multi_setup"][2] == "breakout_momentum"
    assert SETUP_VARIANTS["crossover_only"] == ("crossover",)


def test_walk_forward_folds_are_chronological_and_non_leaking() -> None:
    folds = expanding_anchored_folds(1000)
    assert [f.fold for f in folds] == [1, 2, 3]
    for fold in folds:
        is_s, is_e = fold.is_range
        os_s, os_e = fold.oos_range
        assert is_s == 0 and is_s <= os_s <= is_e <= os_e
    assert folds[-1].oos_range[1] == 1000
    assert folds[0].oos_range[0] >= 0.5 * 1000  # in-sample only precedes OOS
