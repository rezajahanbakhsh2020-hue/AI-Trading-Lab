"""OFFLINE research-only evaluation of the direct multi-setup live strategy.

This module is used ONLY for offline research/backtesting evidence. It is
NEVER imported by the live runtime path; live decisions always come from
`src.evaluation.live_direct_strategy.evaluate_direct_live_signal`.

Semantics are aligned with the existing long-only backtest engine
(`src.backtest.engine.run_backtest`), which was verified first (see
`tests/test_direct_strategy_backtest.py`):
- execution timing: a decision on closed candle i executes at bar i+1's OPEN
  (next-bar open execution, identical to the engine's `prev_sig` handling);
- stops/targets: evaluated intrabar against the executing bar's low/high with
  opening-gap honoring and a conservative STOP-FIRST tie-break when both the
  stop and the target are hit inside the same bar (engine-identical for the
  long side);
- costs: turnover-based, `trading_cost = turnover * (transaction_cost +
  slippage)` with turnover 1 on entry, 1 on exit (enter+exit inside one bar
  counts 2), identical to the engine;
- the engine is long-only; the short side is implemented as the exact mirror
  of the verified long formulas (stop-first tie-break preserved) and is unit
  verified against hand-computed candle scenarios.

Decision generation is a strict vectorization of the LIVE rule set; it is
parity-checked against `evaluate_direct_live_signal` candle-by-candle in
tests. Decision i uses rows 0..i only (no look-ahead).

Trade-level metrics use the direct entry-open -> exit-price return per trade;
bar-level equity/drawdown use the engine-style close-chained per-bar returns
flat gaps between trades are honestly excluded (no carried exposure).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.live_direct_strategy import (
    BREAKOUT_LOOKBACK,
    FAST_MA_PERIOD,
    PULLBACK_LOOKBACK,
    SLOW_MA_PERIOD,
    BUY,
    SELL,
    NO_TRADE,
)

TRANSACTION_COST = 0.0001  # 1 basis point per unit turnover (non-zero, explicit)
SLIPPAGE = 0.0001          # 1 basis point per unit turnover (non-zero, explicit)
RISK_REWARD_RATIO = 2.0

SETUP_VARIANTS: dict[str, tuple[str, ...]] = {
    "crossover_only": ("crossover",),                                            # v1.0-equivalent
    "pullback_only": ("pullback_continuation",),
    "breakout_only": ("breakout_momentum",),
    "multi_setup": ("crossover", "pullback_continuation", "breakout_momentum"),  # v2.0 live product
}

REASON_PREFIX_MAP = {
    "crossover": "crossover_",
    "pullback_continuation": "pullback_continuation_",
    "breakout_momentum": "breakout_momentum_",
}


def compute_decision_series(prepared: pd.DataFrame) -> pd.DataFrame:
    """Vectorized (leak-free) computation of the direct multi-setup rule per closed candle.

    Row i carries the outcome the LIVE evaluator would emit if its input slice
    ended at row i. Executors act on row i's decision at row i+1's open.
    """
    close = prepared["close"].astype(float)
    high = prepared["high"].astype(float)
    low = prepared["low"].astype(float)

    fast_now = close.rolling(FAST_MA_PERIOD).mean()
    fast_prev = fast_now.shift(1)
    slow_now = close.rolling(SLOW_MA_PERIOD).mean()
    slow_prev = slow_now.shift(1)

    pullback_low_min = low.rolling(PULLBACK_LOOKBACK).min().shift(1)
    pullback_high_max = high.rolling(PULLBACK_LOOKBACK).max().shift(1)
    breakout_high = high.rolling(BREAKOUT_LOOKBACK).max().shift(1)
    breakout_low = low.rolling(BREAKOUT_LOOKBACK).min().shift(1)
    prev_close = close.shift(1)

    crossover_buy = (fast_prev <= slow_prev) & (fast_now > slow_now)
    crossover_sell = (fast_prev >= slow_prev) & (fast_now < slow_now)

    trend_up = (fast_now > slow_now) & (slow_now > slow_prev)
    trend_down = (fast_now < slow_now) & (slow_now < slow_prev)
    pullback_buy = trend_up & (pullback_low_min <= fast_now) & (close > prev_close) & (close > fast_now)
    pullback_sell = trend_down & (pullback_high_max >= fast_now) & (close < prev_close) & (close < fast_now)

    breakout_buy = (close > breakout_high) & (close > prev_close) & (close > fast_now)
    breakout_sell = (close < breakout_low) & (close < prev_close) & (close < fast_now)

    conditions = [
        crossover_buy, crossover_sell,
        pullback_buy, pullback_sell,
        breakout_buy, breakout_sell,
    ]
    decisions_choices = [BUY, SELL, BUY, SELL, BUY, SELL]
    reason_choices = [
        "crossover_bullish", "crossover_bearish",
        "pullback_continuation_bullish", "pullback_continuation_bearish",
        "breakout_momentum_bullish", "breakout_momentum_bearish",
    ]
    n = len(prepared)
    # History gate identical to live: need >= SLOW_MA_PERIOD + 1 closed candles.
    valid = np.arange(n) >= SLOW_MA_PERIOD

    decision = np.select(conditions, decisions_choices, default=NO_TRADE)
    decision = np.where(valid, decision, NO_TRADE)
    reason_raw = np.select(conditions, reason_choices, default="no_setup_met")
    reason = np.where(valid, reason_raw, "insufficient_history")

    entry = close.values
    stop = np.where(decision == BUY, low.values, np.where(decision == SELL, high.values, np.nan))
    risk = np.abs(entry - stop)
    reward = risk * RISK_REWARD_RATIO
    take_profit = np.where(decision == BUY, entry + reward, np.where(decision == SELL, entry - reward, np.nan))

    out = pd.DataFrame(index=prepared.index)
    out["decision"] = decision
    out["reason"] = reason
    out["entry_price"] = np.where(decision == NO_TRADE, np.nan, entry)
    out["stop_loss"] = np.where(decision == NO_TRADE, np.nan, stop)
    out["take_profit"] = np.where(decision == NO_TRADE, np.nan, take_profit)
    out["risk_reward_ratio"] = np.where(decision == NO_TRADE, np.nan, RISK_REWARD_RATIO)

    # Degenerate zero-risk geometry -> NO TRADE, mirroring the live rule.
    zero_risk = (decision != NO_TRADE) & (risk <= 0)
    out.loc[zero_risk, "decision"] = NO_TRADE
    out.loc[zero_risk, "reason"] = "invalid_risk_geometry"
    out.loc[zero_risk, ["entry_price", "stop_loss", "take_profit", "risk_reward_ratio"]] = np.nan
    return out


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    is_range: tuple[int, int]
    oos_range: tuple[int, int]


def expanding_anchored_folds(n: int) -> list[WalkForwardFold]:
    """Chronological anchored walk-forward splits with NO leakage.

    Fold k uses IS = [0, start_k * n) (expanding window; the rule set is fully
    explicit and deterministic, so parameters are never refit) and
    OOS = [start_k * n, end_k * n). Boundaries: OOS windows are the
    60-70%, 70-80% and 80-100% chronology; the 0-60% chronology is the
    confirming in-sample window (parameters are fixed, never optimized).
    """
    boundaries = (0.6, 0.7, 0.8)
    folds: list[WalkForwardFold] = []
    for k in range(len(boundaries)):
        os_s = int(n * boundaries[k])
        os_e = int(n * boundaries[k + 1]) if k + 1 < len(boundaries) else n
        folds.append(WalkForwardFold(fold=k + 1, is_range=(0, os_s), oos_range=(os_s, os_e)))
    return folds


def simulate_trades(
    prepared: pd.DataFrame,
    decisions: pd.DataFrame,
    transaction_cost: float = TRANSACTION_COST,
    slippage: float = SLIPPAGE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Chronological single-position simulation with verified engine semantics.

    Returns (trade_log, per_bar_net_returns). Engine-aligned semantics:
    - Decision on candle i executes at bar i+1 open; entries only when flat.
    - Opposite-decision flip exits at that bar's open and re-enters the mirror
      position at the same open (turnover 2 -> conservative cost accounting).
    - SL/TP levels are the decision candle's absolute invalidation/TP levels;
      exits honor opening gaps and use STOP-FIRST when both barriers are hit
      in the same bar (long side engine-identical, short side mirrored).
    - An open position at the end of data is liquidated at the final close,
      flagged `final_close_liquidation` (turnover 1).
    """
    if transaction_cost < 0 or slippage < 0:
        raise ValueError("Costs must be non-negative.")
    cost_rate = float(transaction_cost) + float(slippage)

    open_ = prepared["open"].astype(float).values
    high_arr = prepared["high"].astype(float).values
    low_arr = prepared["low"].astype(float).values
    close_arr = prepared["close"].astype(float).values
    ts = prepared["timestamp"].reset_index(drop=True)
    n = len(prepared)

    dec = pd.Series(decisions["decision"]).reset_index(drop=True).fillna(NO_TRADE).values
    sl_levels = decisions["stop_loss"].reset_index(drop=True).to_numpy(dtype=float)
    tp_levels = decisions["take_profit"].reset_index(drop=True).to_numpy(dtype=float)

    def barrier_hits(direction: int, i: int, sl_price: float, tp_price: float) -> tuple[bool, str | None, float]:
        """Intrabar stop/target resolution, engine-aligned (stop-first on ambiguity).

        Returns (hit, exit_reason, exit_price)."""
        if direction > 0:
            sl_hit, tp_hit = low_arr[i] <= sl_price, high_arr[i] >= tp_price
            c_open = open_[i]
            if sl_hit and tp_hit:
                if c_open <= sl_price or c_open >= tp_price:
                    return True, "stop_or_target_same_bar_stop_first", c_open
                return True, "stop_or_target_same_bar_stop_first", sl_price
            if sl_hit:
                return True, "stop", (c_open if c_open < sl_price else sl_price)
            if tp_hit:
                return True, "take_profit", (c_open if c_open > tp_price else tp_price)
        else:
            sl_hit, tp_hit = high_arr[i] >= sl_price, low_arr[i] <= tp_price
            c_open = open_[i]
            if sl_hit and tp_hit:
                if c_open >= sl_price or c_open <= tp_price:
                    return True, "stop_or_target_same_bar_stop_first", c_open
                return True, "stop_or_target_same_bar_stop_first", sl_price
            if sl_hit:
                return True, "stop", (c_open if c_open > sl_price else sl_price)
            if tp_hit:
                return True, "take_profit", (c_open if c_open < tp_price else tp_price)
        return False, None, 0.0

    trade_log: list[dict[str, Any]] = []

    def record_trade(idx: int, i: int, direction: int, entry_price: float, sl_price: float,
                     tp_price: float, exit_price: float, exit_reason: str,
                     same_entry_bar: bool = False) -> None:
        gross = ((exit_price - entry_price) / entry_price) if direction > 0 else ((entry_price - exit_price) / entry_price)
        trade_log.append({
            "entry_timestamp": ts.iloc[idx],
            "exit_timestamp": ts.iloc[i],
            "direction": "long" if direction > 0 else "short",
            "entry_price": entry_price,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "bars_held": i - idx,
            "gross_return": float(gross),
            "net_return": float(gross - 2.0 * cost_rate),
        })

    gross = np.zeros(n)
    turnover = np.zeros(n)
    direction = 0      # +1 long, -1 short, 0 flat
    entry_price = sl_price = tp_price = 0.0
    entry_idx = -1

    for i in range(1, n):
        prev_decision = dec[i - 1]
        prev_close = close_arr[i - 1]
        c_open, c_close = open_[i], close_arr[i]

        if direction == 0:
            if prev_decision in (BUY, SELL):
                direction = 1 if prev_decision == BUY else -1
                entry_idx, entry_price = i, c_open
                sl_price, tp_price = sl_levels[i - 1], tp_levels[i - 1]
                turnover[i] += 1.0
                hit, exit_reason, exit_price = barrier_hits(direction, i, sl_price, tp_price)
                if hit:
                    gross[i] = ((exit_price - entry_price) / entry_price) if direction > 0 else ((entry_price - exit_price) / entry_price)
                    turnover[i] += 1.0
                    record_trade(entry_idx, i, direction, entry_price, sl_price, tp_price, exit_price, exit_reason, same_entry_bar=True)
                    direction = 0
                    entry_idx = -1
                else:
                    gross[i] = ((c_close - entry_price) / entry_price) if direction > 0 else ((entry_price - c_close) / entry_price)
            else:
                gross[i] = 0.0
        else:
            flipped = (direction > 0 and prev_decision == SELL) or (direction < 0 and prev_decision == BUY)
            if flipped:
                exit_price = c_open
                gross[i] = ((exit_price - prev_close) / prev_close) if direction > 0 else ((prev_close - exit_price) / prev_close)
                turnover[i] += 1.0
                record_trade(entry_idx, i, direction, entry_price, sl_price, tp_price, exit_price, "signal_flip", same_entry_bar=False)
                direction = 1 if prev_decision == BUY else -1
                entry_idx, entry_price = i, c_open
                sl_price, tp_price = sl_levels[i - 1], tp_levels[i - 1]
                turnover[i] += 1.0
                hit, exit_reason, exit_price2 = barrier_hits(direction, i, sl_price, tp_price)
                if hit:
                    gross[i] += ((exit_price2 - entry_price) / entry_price) if direction > 0 else ((entry_price - exit_price2) / entry_price)
                    turnover[i] += 1.0
                    record_trade(entry_idx, i, direction, entry_price, sl_price, tp_price, exit_price2, exit_reason, same_entry_bar=True)
                    direction = 0
                    entry_idx = -1
                else:
                    gross[i] += ((c_close - entry_price) / entry_price) if direction > 0 else ((entry_price - c_close) / entry_price)
            else:
                hit, exit_reason, exit_price = barrier_hits(direction, i, sl_price, tp_price)
                if hit:
                    gross[i] = ((exit_price - prev_close) / prev_close) if direction > 0 else ((prev_close - exit_price) / prev_close)
                    turnover[i] += 1.0
                    record_trade(entry_idx, i, direction, entry_price, sl_price, tp_price, exit_price, exit_reason, same_entry_bar=False)
                    direction, entry_idx = 0, -1
                else:
                    gross[i] = ((c_close - prev_close) / prev_close) if direction > 0 else ((prev_close - c_close) / prev_close)

    if direction != 0:
        i = n - 1
        exit_price = close_arr[i]
        turnover[i] += 1.0
        record_trade(entry_idx, i, direction, entry_price, sl_price, tp_price, exit_price, "final_close_liquidation", same_entry_bar=False)
        direction, entry_idx = 0, -1

    net = gross - turnover * cost_rate
    net[0] = 0.0
    return pd.DataFrame(trade_log), pd.Series(net, index=prepared.index)


def summarize_metrics(
    prepared: pd.DataFrame,
    decisions: pd.DataFrame,
    trades: pd.DataFrame,
    net_returns: pd.Series,
    timeframe: str,
) -> dict[str, Any]:
    """Compact, human-transcribable metrics for one evaluated segment."""
    n_evaluated = int(len(prepared))
    n_buys = int((decisions["decision"] == BUY).sum())
    n_sells = int((decisions["decision"] == SELL).sum())
    n_no_trade = int((decisions["decision"] == NO_TRADE).sum())
    no_trade_ratio = n_no_trade / n_evaluated if n_evaluated else 0.0
    reason_counts: dict[str, int] = {}
    for name, cnt in pd.Series(decisions["reason"]).value_counts().to_dict().items():
        reason_counts[str(name)] = int(cnt)
    n_trades = len(trades)
    if n_trades:
        nets = trades["net_return"].astype(float).values
        wins = int((nets > 0).sum())
        win_rate = wins / n_trades
        expectancy = float(nets.mean())
        durations = trades["bars_held"].astype(float)
        avg_bars = float(durations.mean())
        max_bars = float(durations.max())
    else:
        win_rate, expectancy, avg_bars, max_bars = 0.0, 0.0, 0.0, 0.0
    curve = np.cumprod(1.0 + net_returns.values)
    equity = float(curve[-1]) if len(curve) else 1.0
    peak = np.maximum.accumulate(curve) if len(curve) else np.array([1.0])
    dd = (curve - peak) / np.where(peak > 0, peak, 1.0)
    mdd = float(dd.min()) if len(dd) else 0.0
    exit_reasons: dict[str, int] = {}
    if n_trades:
        for name, cnt in trades["exit_reason"].value_counts().to_dict().items():
            exit_reasons[str(name)] = int(cnt)
    return {
        "timeframe": timeframe,
        "candles_evaluated": n_evaluated,
        "buy_signals": n_buys,
        "sell_signals": n_sells,
        "no_trade_signals": n_no_trade,
        "no_trade_ratio": round(no_trade_ratio, 6),
        "reason_counts": reason_counts,
        "trades": n_trades,
        "wins": int(wins if not n_trades else wins),
        "win_rate": round(win_rate, 6),
        "expectancy": round(expectancy, 8),
        "net_return_after_costs": round(equity - 1.0, 8),
        "max_drawdown": round(mdd, 8),
        "avg_trade_bars": round(avg_bars, 4),
        "max_trade_bars": max_bars,
        "exit_reasons": exit_reasons,
        "date_range": [str(prepared["timestamp"].iloc[0]), str(prepared["timestamp"].iloc[-1])],
    }


def _decisions_for_variant(full_decisions: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Restrict the multi-setup decision series to the variant's setups.

    Valid because the crossover conditions do not depend on the later setups
    and the live evaluate order (crossover first) matches reason classification.
    """
    setups = SETUP_VARIANTS[variant]
    mask = pd.Series(False, index=full_decisions.index)
    for s in setups:
        mask |= full_decisions["reason"].astype(str).str.startswith(REASON_PREFIX_MAP[s])
    d = full_decisions.copy()
    d.loc[~mask, "decision"] = NO_TRADE
    d["reason"] = np.where(mask, d["reason"], "filtered_by_variant")
    return d


def _segment_result(
    prepared_segment: pd.DataFrame,
    decisions_segment: pd.DataFrame,
    timeframe: str,
    label: str,
    fold: int,
) -> dict[str, Any]:
    trades, net_returns = simulate_trades(prepared_segment, decisions_segment, TRANSACTION_COST, SLIPPAGE)
    summary = summarize_metrics(prepared_segment, decisions_segment, trades, net_returns, timeframe)
    summary["fold"] = fold
    summary["segment"] = label
    return summary


def evaluate_variant(
    prepared: pd.DataFrame,
    full_decisions: pd.DataFrame,
    variant: str,
    timeframe: str,
) -> dict[str, Any]:
    decisions = _decisions_for_variant(full_decisions, variant)
    trades, net_returns = simulate_trades(prepared, decisions, TRANSACTION_COST, SLIPPAGE)
    summary = summarize_metrics(prepared, decisions, trades, net_returns, timeframe)
    summary["variant"] = variant
    summary["setups"] = list(SETUP_VARIANTS[variant])
    summary["costs"] = {
        "transaction_cost": TRANSACTION_COST,
        "slippage": SLIPPAGE,
        "round_trip_turnover": 2.0,
        "round_trip_cost_rate": 2.0 * (TRANSACTION_COST + SLIPPAGE),
    }
    folds = expanding_anchored_folds(len(prepared))
    summary["is_confirmation"] = [
        _segment_result(
            prepared.iloc[fold.is_range[0]:fold.is_range[1]].reset_index(drop=True),
            decisions.iloc[fold.is_range[0]:fold.is_range[1]].reset_index(drop=True),
            timeframe, f"IS_0-{fold.is_range[1]}", fold.fold,
        )
        for fold in folds
    ]
    summary["oos_folds"] = [
        _segment_result(
            prepared.iloc[fold.oos_range[0]:fold.oos_range[1]].reset_index(drop=True),
            decisions.iloc[fold.oos_range[0]:fold.oos_range[1]].reset_index(drop=True),
            timeframe, f"OOS_{fold.oos_range[0]}-{fold.oos_range[1]}", fold.fold,
        )
        for fold in folds
    ]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline anchored walk-forward evaluation of direct multi-setup variants on committed reachable XAUUSD CSVs."
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--timeframe", required=True, help="Canonical timeframe label (metric annotation only).")
    parser.add_argument("--variants", nargs="*", default=list(SETUP_VARIANTS.keys()))
    args = parser.parse_args()

    raw = pd.read_csv(args.csv)
    prepared = pd.DataFrame({
        "timestamp": pd.to_datetime(raw["timestamp"], utc=True),
        "open": raw["open"].astype(float),
        "high": raw["high"].astype(float),
        "low": raw["low"].astype(float),
        "close": raw["close"].astype(float),
    }).sort_values("timestamp").reset_index(drop=True)

    full_decisions = compute_decision_series(prepared)
    report: dict[str, Any] = {
        "timeframe": args.timeframe,
        "csv": args.csv,
        "rows": int(len(prepared)),
        "date_range": [str(prepared["timestamp"].iloc[0]), str(prepared["timestamp"].iloc[-1])],
        "variants": [
            evaluate_variant(prepared, full_decisions, variant, args.timeframe)
            for variant in args.variants
        ],
    }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
