from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "strategy",
    "total_return",
    "max_drawdown",
    "sharpe_ratio",
    "stability_score",
    "worst_window_return",
    "max_consecutive_losses",
}


def calculate_risk_adjusted_score(
    report: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate a risk-aware score for each strategy.

    The score rewards return, Sharpe ratio and stability while penalizing
    drawdown, poor worst-window performance and long losing streaks.
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

    result = report.copy()

    if result.empty:
        result["risk_adjusted_score"] = pd.Series(
            dtype="float64"
        )
        return result

    drawdown_penalty = result["max_drawdown"].abs()

    worst_window_penalty = (
        result["worst_window_return"].clip(
            upper=0
        ).abs()
    )

    loss_streak_penalty = (
        result["max_consecutive_losses"]
        / (
            result["max_consecutive_losses"]
            + 5.0
        )
    )

    result["risk_adjusted_score"] = (
        0.30 * result["total_return"]
        + 0.25 * result["sharpe_ratio"]
        + 0.30 * result["stability_score"]
        - 0.10 * drawdown_penalty
        - 0.03 * worst_window_penalty
        - 0.02 * loss_streak_penalty
    )

    return result


def rank_risk_adjusted_strategies(
    report: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank strategies using the risk-adjusted score.
    """

    scored = calculate_risk_adjusted_score(
        report
    )

    if scored.empty:
        return scored

    return (
        scored.sort_values(
            [
                "risk_adjusted_score",
                "stability_score",
                "sharpe_ratio",
                "total_return",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def select_best_risk_adjusted_strategy(
    report: pd.DataFrame,
) -> str | None:
    """
    Return the highest-ranked risk-adjusted strategy.
    """

    ranked = rank_risk_adjusted_strategies(
        report
    )

    if ranked.empty:
        return None

    return str(
        ranked.iloc[0]["strategy"]
    )


def apply_risk_limits(
    report: pd.DataFrame,
    max_drawdown: float = 0.20,
    min_sharpe_ratio: float = 0.0,
    min_stability_score: float = 0.50,
    min_worst_window_return: float = -0.20,
    max_consecutive_losses: int = 5,
) -> pd.DataFrame:
    """
    Filter strategies using explicit risk limits.
    """

    if not isinstance(report, pd.DataFrame):
        raise TypeError(
            "report must be a pandas DataFrame."
        )

    if max_drawdown < 0:
        raise ValueError(
            "max_drawdown must be non-negative."
        )

    if min_sharpe_ratio < 0:
        raise ValueError(
            "min_sharpe_ratio must be non-negative."
        )

    if not 0.0 <= min_stability_score <= 1.0:
        raise ValueError(
            "min_stability_score must be between 0 and 1."
        )

    if not -1.0 <= min_worst_window_return <= 1.0:
        raise ValueError(
            "min_worst_window_return must be between -1 and 1."
        )

    if max_consecutive_losses < 0:
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

    result = report[
        report["max_drawdown"].abs()
        <= max_drawdown
    ]

    result = result[
        result["sharpe_ratio"]
        >= min_sharpe_ratio
    ]

    result = result[
        result["stability_score"]
        >= min_stability_score
    ]

    result = result[
        result["worst_window_return"]
        >= min_worst_window_return
    ]

    result = result[
        result["max_consecutive_losses"]
        <= max_consecutive_losses
    ]

    return result.reset_index(drop=True)


def select_with_risk_limits(
    report: pd.DataFrame,
    **risk_kwargs,
) -> str | None:
    """
    Apply risk limits and return the best remaining strategy.
    """

    eligible = apply_risk_limits(
        report,
        **risk_kwargs,
    )

    return select_best_risk_adjusted_strategy(
        eligible
    )
