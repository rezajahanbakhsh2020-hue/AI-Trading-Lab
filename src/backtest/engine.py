import math
import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    signal_column: str = "signal",
    return_column: str = "return",
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    tp1_multiplier: float = 1.0,
    tp2_multiplier: float = 2.0,
    tp3_multiplier: float = 3.0,
) -> pd.DataFrame:
    """Run a simple long-only backtest with optional intrabar stop-loss and take-profit risk execution.

    The signal generated at period t is executed during period t+1 to avoid look-ahead bias.
    Transaction costs and slippage are charged when position changes.
    When stop_loss_pct or take_profit_pct is provided, intrabar OHLC price levels are checked for execution.
    """
    result = df.copy()

    if signal_column not in result.columns:
        raise ValueError(f"Column '{signal_column}' not found.")

    if return_column not in result.columns:
        raise ValueError(f"Column '{return_column}' not found.")

    if transaction_cost < 0:
        raise ValueError("transaction_cost must be non-negative.")

    if slippage < 0:
        raise ValueError("slippage must be non-negative.")

    has_risk = stop_loss_pct is not None or take_profit_pct is not None

    if has_risk:
        required_ohlc = {"open", "high", "low", "close"}
        missing_ohlc = required_ohlc - set(result.columns)
        if missing_ohlc:
            raise ValueError(
                f"Backtesting with risk parameters (stop_loss_pct/take_profit_pct) requires OHLC columns. "
                f"Missing: {sorted(missing_ohlc)}"
            )

        if stop_loss_pct is not None:
            if not isinstance(stop_loss_pct, (int, float)) or isinstance(stop_loss_pct, bool) or not math.isfinite(stop_loss_pct) or stop_loss_pct <= 0:
                raise ValueError(f"stop_loss_pct must be a positive finite float, got: {stop_loss_pct}")

        if take_profit_pct is not None:
            if not isinstance(take_profit_pct, (int, float)) or isinstance(take_profit_pct, bool) or not math.isfinite(take_profit_pct) or take_profit_pct <= 0:
                raise ValueError(f"take_profit_pct must be a positive finite float, got: {take_profit_pct}")

        # Intrabar risk simulation
        n = len(result)
        positions = [0.0] * n
        gross_returns = [float("nan")] * n if n > 0 else []

        signals = result[signal_column].values
        opens = result["open"].values
        highs = result["high"].values
        lows = result["low"].values
        closes = result["close"].values

        in_position = False
        entry_price = 0.0
        sl_price = None
        tp_price = None

        eff_tp_mult = tp1_multiplier if tp1_multiplier > 0 else 1.0

        for i in range(n):
            if i == 0:
                # First period: no previous signal available
                positions[0] = 0.0
                gross_returns[0] = float("nan")
                continue

            prev_sig = signals[i - 1]
            c_open = float(opens[i])
            c_high = float(highs[i])
            c_low = float(lows[i])
            c_close = float(closes[i])
            prev_close = float(closes[i - 1])

            if in_position:
                sl_hit = (c_low <= sl_price) if sl_price is not None else False
                tp_hit = (c_high >= tp_price) if tp_price is not None else False

                if sl_hit and tp_hit:
                    # Simultaneous touch: pessimistic fill at stop loss to prevent look-ahead bias
                    exit_price = min(c_open, sl_price) if c_open < sl_price else sl_price
                    gross_returns[i] = (exit_price - prev_close) / prev_close
                    positions[i] = 0.0
                    in_position = False
                elif sl_hit:
                    exit_price = min(c_open, sl_price) if c_open < sl_price else sl_price
                    gross_returns[i] = (exit_price - prev_close) / prev_close
                    positions[i] = 0.0
                    in_position = False
                elif tp_hit:
                    exit_price = max(c_open, tp_price) if c_open > tp_price else tp_price
                    gross_returns[i] = (exit_price - prev_close) / prev_close
                    positions[i] = 0.0
                    in_position = False
                else:
                    if prev_sig == 1:
                        positions[i] = 1.0
                        gross_returns[i] = (c_close - prev_close) / prev_close
                    else:
                        positions[i] = 0.0
                        gross_returns[i] = (c_close - prev_close) / prev_close
                        in_position = False

            else:
                if prev_sig == 1:
                    in_position = True
                    entry_price = c_open if not math.isnan(c_open) else prev_close

                    sl_price = entry_price * (1.0 - stop_loss_pct) if stop_loss_pct is not None else None
                    if take_profit_pct is not None:
                        tp_price = entry_price * (1.0 + take_profit_pct)
                    elif stop_loss_pct is not None and eff_tp_mult is not None:
                        tp_price = entry_price * (1.0 + stop_loss_pct * eff_tp_mult)
                    else:
                        tp_price = None

                    sl_hit = (c_low <= sl_price) if sl_price is not None else False
                    tp_hit = (c_high >= tp_price) if tp_price is not None else False

                    if sl_hit and tp_hit:
                        exit_price = min(c_open, sl_price) if c_open < sl_price else sl_price
                        gross_returns[i] = (exit_price - entry_price) / entry_price
                        positions[i] = 0.0
                        in_position = False
                    elif sl_hit:
                        exit_price = min(c_open, sl_price) if c_open < sl_price else sl_price
                        gross_returns[i] = (exit_price - entry_price) / entry_price
                        positions[i] = 0.0
                        in_position = False
                    elif tp_hit:
                        exit_price = max(c_open, tp_price) if c_open > tp_price else tp_price
                        gross_returns[i] = (exit_price - entry_price) / entry_price
                        positions[i] = 0.0
                        in_position = False
                    else:
                        positions[i] = 1.0
                        gross_returns[i] = (c_close - entry_price) / entry_price
                else:
                    positions[i] = 0.0
                    gross_returns[i] = 0.0

        result["position"] = positions
        result["gross_strategy_return"] = gross_returns

    else:
        # Standard signal-only backtest
        result["position"] = (
            result[signal_column]
            .shift(1)
            .fillna(0.0)
        )

        result["gross_strategy_return"] = (
            result["position"] * result[return_column]
        )

        if not result.empty:
            result.loc[
                result.index[0],
                "gross_strategy_return",
            ] = float("nan")

    result["turnover"] = (
        result["position"]
        .diff()
        .abs()
        .fillna(0.0)
    )

    total_cost_rate = transaction_cost + slippage

    result["trading_cost"] = (
        result["turnover"] * total_cost_rate
    )

    result["strategy_return"] = (
        result["gross_strategy_return"]
        - result["trading_cost"]
    )

    result["equity"] = (
        1.0
        + result["strategy_return"].fillna(0.0)
    ).cumprod()

    return result
