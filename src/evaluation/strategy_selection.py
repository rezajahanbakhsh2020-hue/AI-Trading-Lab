from __future__ import annotations

import pandas as pd


def select_eligible_strategies(
    report: pd.DataFrame,
    min_total_return: float = 0.0,
    max_drawdown: float = 0.20,
    min_sharpe_ratio: float = 0.0,
    min_positive_window_rate: float = 0.50,
) -> pd.DataFrame:
    """
    Select strategies that pass predefined performance gates.

    The function is designed for walk-forward evaluation reports.

    A strategy is eligible only when it satisfies all configured
    conditions:

    - total_return >= min_total_return
    - absolute max_drawdown <= max_drawdown
    - sharpe_ratio >= min_sharpe_ratio
    - positive_window_rate >= min_positive_window_rate

    Parameters
    ----------
    report:
        Strategy evaluation report. It must contain:
        total_return, max_drawdown, sharpe_ratio,
        and positive_window_rate.

    min_total_return:
        Minimum acceptable total return.

    max_drawdown:
        Maximum acceptable drawdown magnitude.

    min_sharpe_ratio:
        Minimum acceptable Sharpe ratio.

    min_positive_window_rate:
        Minimum percentage of profitable walk-forward windows.

    Returns
    -------
    pd.DataFrame
        Copy of the input report containing only eligible strategies.
    """

    if not isinstance(report, pd.DataFrame):
        raise TypeError("report must be a pandas DataFrame.")

    if min_total_return < 0:
        raise ValueError(
            "min_total_return must be non-negative."
        )

    if max_drawdown < 0:
        raise ValueError(
            "max_drawdown must be non-negative."
        )

    if min_sharpe_ratio < 0:
        raise ValueError(
            "min_sharpe_ratio must be non-negative."
        )

    if not 0.0 <= min_positive_window_rate <= 1.0:
        raise ValueError(
            "min_positive_window_rate must be between 0 and 1."
        )

    required_columns = {
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
        "positive_window_rate",
    }

    missing_columns = sorted(
        required_columns.difference(report.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    if report.empty:
        return report.copy()

    eligible = report[
        (report["total_return"] >= min_total_return)
        & (report["max_drawdown"].abs() <= max_drawdown)
        & (report["sharpe_ratio"] >= min_sharpe_ratio)
        & (
            report["positive_window_rate"]
            >= min_positive_window_rate
        )
    ].copy()

    return eligible
