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


def _validate_report(report: pd.DataFrame) -> None:
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


def calculate_portfolio_score(
    report: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate a normalized portfolio suitability score.

    The score combines return quality, risk-adjusted performance,
    stability and downside behavior.
    """

    _validate_report(report)

    result = report.copy()

    if result.empty:
        result["portfolio_score"] = pd.Series(
            dtype="float64"
        )
        return result

    drawdown = result["max_drawdown"].abs()

    downside = (
        result["worst_window_return"]
        .clip(upper=0)
        .abs()
    )

    loss_streak = (
        result["max_consecutive_losses"]
        / (
            result["max_consecutive_losses"]
            + 5.0
        )
    )

    result["portfolio_score"] = (
        0.25 * result["total_return"]
        + 0.25 * result["sharpe_ratio"]
        + 0.30 * result["stability_score"]
        + 0.10 * (1.0 - drawdown.clip(upper=1.0))
        + 0.07 * (1.0 - downside.clip(upper=1.0))
        + 0.03 * (1.0 - loss_streak)
    )

    return result


def rank_portfolio_candidates(
    report: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank strategies by portfolio suitability.
    """

    scored = calculate_portfolio_score(
        report
    )

    if scored.empty:
        return scored

    return (
        scored.sort_values(
            [
                "portfolio_score",
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


def select_portfolio_candidates(
    report: pd.DataFrame,
    max_drawdown: float = 0.20,
    min_sharpe_ratio: float = 0.50,
    min_stability_score: float = 0.60,
    min_worst_window_return: float = -0.15,
    max_consecutive_losses: int = 4,
    min_total_return: float = 0.0,
) -> pd.DataFrame:
    """
    Filter strategies suitable for portfolio construction.
    """

    _validate_report(report)

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

    if min_total_return < 0:
        raise ValueError(
            "min_total_return must be non-negative."
        )

    if report.empty:
        return report.copy()

    selected = report[
        report["total_return"]
        >= min_total_return
    ]

    selected = selected[
        selected["max_drawdown"].abs()
        <= max_drawdown
    ]

    selected = selected[
        selected["sharpe_ratio"]
        >= min_sharpe_ratio
    ]

    selected = selected[
        selected["stability_score"]
        >= min_stability_score
    ]

    selected = selected[
        selected["worst_window_return"]
        >= min_worst_window_return
    ]

    selected = selected[
        selected["max_consecutive_losses"]
        <= max_consecutive_losses
    ]

    return (
        selected.copy()
        .reset_index(drop=True)
    )


def build_strategy_portfolio(
    report: pd.DataFrame,
    max_strategies: int = 3,
    **selection_kwargs,
) -> pd.DataFrame:
    """
    Build a ranked portfolio of the strongest eligible strategies.

    Strategies are selected after applying risk and stability filters,
    then ranked by portfolio suitability.
    """

    _validate_report(report)

    if not isinstance(max_strategies, int):
        raise TypeError(
            "max_strategies must be an integer."
        )

    if max_strategies <= 0:
        raise ValueError(
            "max_strategies must be greater than zero."
        )

    eligible = select_portfolio_candidates(
        report,
        **selection_kwargs,
    )

    ranked = rank_portfolio_candidates(
        eligible
    )

    return ranked.head(
        max_strategies
    ).reset_index(drop=True)


def get_portfolio_strategy_names(
    report: pd.DataFrame,
    max_strategies: int = 3,
    **selection_kwargs,
) -> list[str]:
    """
    Return the selected strategy names in portfolio order.
    """

    portfolio = build_strategy_portfolio(
        report,
        max_strategies=max_strategies,
        **selection_kwargs,
    )

    if portfolio.empty:
        return []

    return [
        str(strategy)
        for strategy in portfolio["strategy"]
    ]
