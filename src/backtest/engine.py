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
    tp1_multiplier: float | None = None,
    tp2_multiplier: float | None = None,
    tp3_multiplier: float | None = None,
) -> pd.DataFrame:
    """Run a simple long-only backtest with next-bar signal execution and intrabar risk management.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing signal, return, and optionally OHLC columns.
    signal_column : str
        Trading signal column (1 for long, 0 for cash).
    return_column : str
        Market return column.
    transaction_cost : float
        Proportional transaction cost applied to position turnover.
    slippage : float
        Proportional slippage applied to position turnover.
    stop_loss_pct : float | None
        Stop loss percentage (e.g. 0.02 for 2%).
    take_profit_pct : float | None
        Take profit percentage (e.g. 0.03 for 3%).
    tp1_multiplier : float | None
        Multiplier on stop_loss_pct to establish TP1 target if take_profit_pct is None.
    tp2_multiplier : float | None
        Multi-target TP2 multiplier (unsupported in single-target backtesting; fails closed).
    tp3_multiplier : float | None
        Multi-target TP3 multiplier (unsupported in single-target backtesting; fails closed).
    """
    result = df.copy()

    if signal_column not in result.columns:
        raise ValueError(f"Column '{signal_column}' not found.")

    if return_column not in result.columns:
        raise ValueError(f"Column '{return_column}' not found.")

    if not isinstance(transaction_cost, (int, float)) or isinstance(transaction_cost, bool) or not math.isfinite(transaction_cost) or transaction_cost < 0:
        raise ValueError("transaction_cost must be a non-negative finite float.")

    if not isinstance(slippage, (int, float)) or isinstance(slippage, bool) or not math.isfinite(slippage) or slippage < 0:
        raise ValueError("slippage must be a non-negative finite float.")

    if tp2_multiplier is not None or tp3_multiplier is not None:
        raise ValueError("Multi-target tp2_multiplier/tp3_multiplier execution is not supported by single-target backtest engine.")

    has_risk = stop_loss_pct is not None or take_profit_pct is not None or tp1_multiplier is not None

    if has_risk:
        required_ohlc = {"open", "high", "low", "close"}
        missing_ohlc = required_ohlc - set(result.columns)
        if missing_ohlc:
            raise ValueError(
                f"Backtesting with risk parameters requires OHLC columns. "
                f"Missing: {sorted(missing_ohlc)}"
            )

        if stop_loss_pct is not None:
            if not isinstance(stop_loss_pct, (int, float)) or isinstance(stop_loss_pct, bool) or not math.isfinite(stop_loss_pct) or stop_loss_pct <= 0:
                raise ValueError(f"stop_loss_pct must be a positive finite float, got: {stop_loss_pct}")

        if take_profit_pct is not None:
            if not isinstance(take_profit_pct, (int, float)) or isinstance(take_profit_pct, bool) or not math.isfinite(take_profit_pct) or take_profit_pct <= 0:
                raise ValueError(f"take_profit_pct must be a positive finite float, got: {take_profit_pct}")

        if tp1_multiplier is not None:
            if not isinstance(tp1_multiplier, (int, float)) or isinstance(tp1_multiplier, bool) or not math.isfinite(tp1_multiplier) or tp1_multiplier <= 0:
                raise ValueError(f"tp1_multiplier must be a positive finite float, got: {tp1_multiplier}")
            if stop_loss_pct is None and take_profit_pct is None:
                raise ValueError("tp1_multiplier requires stop_loss_pct to calculate take-profit price level.")

        n = len(result)
        positions = [0.0] * n
        gross_returns = [float("nan")] * n if n > 0 else []
        turnovers = [0.0] * n

        signals = result[signal_column].values
        opens = result["open"].values
        highs = result["high"].values
        lows = result["low"].values
        closes = result["close"].values

        in_position = False
        entry_price = 0.0
        sl_price = None
        tp_price = None

        for i in range(n):
            if i == 0:
                positions[0] = 0.0
                gross_returns[0] = float("nan")
                turnovers[0] = 0.0
                continue

            prev_sig = signals[i - 1]
            c_open = float(opens[i])
            c_high = float(highs[i])
            c_low = float(lows[i])
            c_close = float(closes[i])
            prev_close = float(closes[i - 1])

            if in_position:
                if prev_sig == 0:
                    # Signal-driven exit at bar i open!
                    exit_price = c_open
                    if sl_price is not None and c_open <= sl_price:
                        exit_price = c_open
                    elif tp_price is not None and c_open >= tp_price:
                        exit_price = c_open

                    gross_returns[i] = (exit_price - prev_close) / prev_close
                    positions[i] = 0.0
                    turnovers[i] = 1.0
                    in_position = False
                    sl_price = None
                    tp_price = None
                else:
                    sl_hit = (c_low <= sl_price) if sl_price is not None else False
                    tp_hit = (c_high >= tp_price) if tp_price is not None else False

                    if sl_hit and tp_hit:
                        exit_price = min(c_open, sl_price) if c_open < sl_price else sl_price
                        gross_returns[i] = (exit_price - prev_close) / prev_close
                        positions[i] = 0.0
                        turnovers[i] = 1.0
                        in_position = False
                        sl_price = None
                        tp_price = None
                    elif sl_hit:
                        exit_price = min(c_open, sl_price) if c_open < sl_price else sl_price
                        gross_returns[i] = (exit_price - prev_close) / prev_close
                        positions[i] = 0.0
                        turnovers[i] = 1.0
                        in_position = False
                        sl_price = None
                        tp_price = None
                    elif tp_hit:
                        exit_price = max(c_open, tp_price) if c_open > tp_price else tp_price
                        gross_returns[i] = (exit_price - prev_close) / prev_close
                        positions[i] = 0.0
                        turnovers[i] = 1.0
                        in_position = False
                        sl_price = None
                        tp_price = None
                    else:
                        positions[i] = 1.0
                        gross_returns[i] = (c_close - prev_close) / prev_close
                        turnovers[i] = 0.0

            else:
                if prev_sig == 1:
                    in_position = True
                    entry_price = c_open if not math.isnan(c_open) else prev_close

                    sl_price = entry_price * (1.0 - stop_loss_pct) if stop_loss_pct is not None else None
                    if take_profit_pct is not None:
                        tp_price = entry_price * (1.0 + take_profit_pct)
                    elif stop_loss_pct is not None and tp1_multiplier is not None:
                        tp_price = entry_price * (1.0 + stop_loss_pct * tp1_multiplier)
                    else:
                        tp_price = None

                    sl_hit = (c_low <= sl_price) if sl_price is not None else False
                    tp_hit = (c_high >= tp_price) if tp_price is not None else False

                    if sl_hit and tp_hit:
                        exit_price = min(c_open, sl_price) if c_open < sl_price else sl_price
                        gross_returns[i] = (exit_price - entry_price) / entry_price
                        positions[i] = 0.0
                        turnovers[i] = 2.0
                        in_position = False
                        sl_price = None
                        tp_price = None
                    elif sl_hit:
                        exit_price = min(c_open, sl_price) if c_open < sl_price else sl_price
                        gross_returns[i] = (exit_price - entry_price) / entry_price
                        positions[i] = 0.0
                        turnovers[i] = 2.0
                        in_position = False
                        sl_price = None
                        tp_price = None
                    elif tp_hit:
                        exit_price = max(c_open, tp_price) if c_open > tp_price else tp_price
                        gross_returns[i] = (exit_price - entry_price) / entry_price
                        positions[i] = 0.0
                        turnovers[i] = 2.0
                        in_position = False
                        sl_price = None
                        tp_price = None
                    else:
                        positions[i] = 1.0
                        gross_returns[i] = (c_close - entry_price) / entry_price
                        turnovers[i] = 1.0
                else:
                    positions[i] = 0.0
                    gross_returns[i] = 0.0
                    turnovers[i] = 0.0

        result["position"] = positions
        result["turnover"] = turnovers
        result["gross_strategy_return"] = gross_returns

    else:
        # Standard signal-only backtest
        result["position"] = (
            result[signal_column]
            .shift(1)
            .fillna(0.0)
        )

        result["turnover"] = (
            result["position"]
            .diff()
            .abs()
            .fillna(0.0)
        )

        gross_ret = [float("nan")] * len(result)
        signals = result[signal_column].values
        returns = result[return_column].values
        opens = result["open"].values if "open" in result.columns else None
        closes = result["close"].values if "close" in result.columns else None

        for i in range(len(result)):
            if i == 0:
                gross_ret[0] = float("nan")
                continue

            pos_prev = result["position"].iloc[i - 1]
            pos_curr = result["position"].iloc[i]

            if pos_prev == 1.0 and pos_curr == 0.0:
                if opens is not None and closes is not None:
                    c_open = float(opens[i])
                    prev_close = float(closes[i - 1])
                    gross_ret[i] = (c_open - prev_close) / prev_close
                else:
                    gross_ret[i] = 0.0
            elif pos_curr == 1.0:
                if pos_prev == 0.0:
                    if opens is not None and closes is not None:
                        c_open = float(opens[i])
                        c_close = float(closes[i])
                        gross_ret[i] = (c_close - c_open) / c_open
                    else:
                        gross_ret[i] = returns[i]
                else:
                    gross_ret[i] = returns[i]
            else:
                gross_ret[i] = 0.0

        result["gross_strategy_return"] = gross_ret

    total_cost_rate = transaction_cost + slippage

    result["trading_cost"] = (
        result["turnover"] * total_cost_rate
    )

    result["strategy_return"] = (
        result["gross_strategy_return"].fillna(0.0)
        - result["trading_cost"]
    )

    if not result.empty:
        result.loc[result.index[0], "strategy_return"] = float("nan")

    result["equity"] = (
        1.0
        + result["strategy_return"].fillna(0.0)
    ).cumprod()

    return result
