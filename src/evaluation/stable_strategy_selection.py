from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "strategy",
    "total_return",
    "max_drawdown",
    "sharpe_ratio",
    "positive_window_rate",
    "stability_score",
    "worst_window_return",
    "max_consecutive_losses",
}


def select_stable_strategies(
    report: pd.DataFrame,
    min_total_return: float = 0.0,
    max_drawdown: float = 0.20,
    min_sharpe_ratio: float = 0.0,
    min_positive_window_rate: float = 0.50,
    min_stability_score: float = 0.50,
    min_worst_window_return: float = -1.0,
    max_consecutive_losses: int | None = None,
) -> pd.DataFrame:
    """
    Select strategies using both performance and walk-forward stability.

    A strategy must pass all configured performance and stability gates.

    Required columns
    ----------------
    strategy
    total_return
    max_drawdown
    sharpe_ratio
    positive_window_rate
    stability_score
    worst_window_return
    max_consecutive_losses
    """

    if not isinstance(report, pd.DataFrame):
        raise TypeError(
            "report must be a pandas DataFrame."
        )

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

    if not 0.0 <= min_stability_score <= 1.0:
        raise ValueError(
            "min_stability_score must be between 0 and 1."
        )

    if not -1.0 <= min_worst_window_return <= 1.0:
        raise ValueError(
            "min_worst_window_return must be between -1 and 1."
        )

    if (
        max_consecutive_losses is not None
        and max_consecutive_losses < 0
    ):
        raise ValueError(
            "max_consecutive_losses must be non-negative."
        )

    missing_columns = sorted(
        REQUIRED_COLUMNS.difference(report.columns)
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
        & (
            report["max_drawdown"].abs()
            <= max_drawdown
        )
        & (
            report["sharpe_ratio"]
            >= min_sharpe_ratio
        )
        & (
            report["positive_window_rate"]
            >= min_positive_window_rate
        )
        & (
            report["stability_score"]
            >= min_stability_score
        )
        & (
            report["worst_window_return"]
            >= min_worst_window_return
        )
    ].copy()

    if max_consecutive_losses is not None:
        eligible = eligible[
            eligible["max_consecutive_losses"]
            <= max_consecutive_losses
        ].copy()

    return eligible.reset_index(drop=True)


def rank_stable_strategies(
    report: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank eligible strategies by robustness.

    Stability is the primary ranking criterion, followed by
    positive-window consistency, worst-window performance,
    Sharpe ratio, and total return.
    """

    if not isinstance(report, pd.DataFrame):
        raise TypeError(
            "report must be a pandas DataFrame."
        )

    missing_columns = sorted(
        REQUIRED_COLUMNS.difference(report.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    if report.empty:
        return report.copy()

    return (
        report.copy()
        .sort_values(
            [
                "stability_score",
                "positive_window_rate",
                "worst_window_return",
                "sharpe_ratio",
                "total_return",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def select_best_stable_strategy(
    report: pd.DataFrame,
    **selection_kwargs,
) -> str | None:
    """
    Apply stability gates and return the strongest remaining strategy.
    """

    eligible = select_stable_strategies(
        report,
        **selection_kwargs,
    )

    if eligible.empty:
        return None

    ranked = rank_stable_strategies(
        eligible
    )

    return str(ranked.iloc[0]["strategy"])
