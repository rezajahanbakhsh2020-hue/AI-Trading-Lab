import pandas as pd

from src.evaluation.metrics import (
    calmar_ratio,
    exposure,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    win_rate,
)


def combine_oos_results(
    results: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine walk-forward OOS results into one chronological DataFrame.
    """

    if not isinstance(results, list):
        raise TypeError("results must be a list.")

    if not results:
        return pd.DataFrame()

    for result in results:
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                "Every walk-forward result must be a pandas DataFrame."
            )

    combined = pd.concat(
        results,
        axis=0,
        ignore_index=True,
    )

    if "timestamp" in combined.columns:
        combined = combined.sort_values(
            "timestamp"
        ).reset_index(drop=True)

    return combined


def _calculate_window_returns(
    results: list[pd.DataFrame],
) -> list[float]:
    """
    Calculate total return for every walk-forward OOS window.
    """

    window_returns = []

    for result in results:
        if result.empty:
            window_returns.append(0.0)
            continue

        window_returns.append(
            float(total_return(result))
        )

    return window_returns


def evaluate_walk_forward(
    results: list[pd.DataFrame],
) -> dict:
    """
    Evaluate combined out-of-sample walk-forward results.

    The report contains both aggregate OOS performance metrics and
    window-level stability metrics.
    """

    combined = combine_oos_results(results)

    if combined.empty:
        return {
            "windows": 0,
            "observations": 0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0,
            "sortino_ratio": 0.0,
            "exposure": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "window_returns": [],
            "profitable_windows": 0,
            "losing_windows": 0,
            "positive_window_rate": 0.0,
        }

    required_columns = {
        "strategy_return",
        "equity",
    }

    missing_columns = [
        column
        for column in required_columns
        if column not in combined.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if "position" in combined.columns:
        exposure_column = "position"
    elif "signal" in combined.columns:
        exposure_column = "signal"
    else:
        raise ValueError(
            "Missing required exposure column: "
            "'position' or 'signal'."
        )

    window_returns = _calculate_window_returns(results)

    profitable_windows = sum(
        value > 0
        for value in window_returns
    )

    losing_windows = sum(
        value < 0
        for value in window_returns
    )

    positive_window_rate = (
        profitable_windows / len(window_returns)
        if window_returns
        else 0.0
    )

    return {
        "windows": len(results),
        "observations": len(combined),
        "total_return": total_return(combined),
        "max_drawdown": max_drawdown(combined),
        "sharpe_ratio": sharpe_ratio(combined),
        "calmar_ratio": calmar_ratio(combined),
        "sortino_ratio": sortino_ratio(combined),
        "exposure": exposure(
            combined,
            signal_column=exposure_column,
        ),
        "win_rate": win_rate(combined),
        "profit_factor": profit_factor(combined),
        "window_returns": window_returns,
        "profitable_windows": profitable_windows,
        "losing_windows": losing_windows,
        "positive_window_rate": positive_window_rate,
    }
