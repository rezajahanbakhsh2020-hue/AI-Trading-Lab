from __future__ import annotations

import pandas as pd

from src.evaluation.compare import (
    comparison_dataframe,
    rank_walk_forward_strategies,
)


def build_final_strategy_report(
    comparison: dict[str, dict],
    metric: str = "total_return",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Build the final ranked strategy report for walk-forward comparison.

    Parameters
    ----------
    comparison:
        Output of compare_walk_forward_strategies().

    metric:
        Primary metric used for ranking.

    ascending:
        Ranking direction for the primary metric.

    Returns
    -------
    pd.DataFrame
        Ranked strategy report with performance and stability metrics.
    """

    ranked = rank_walk_forward_strategies(
        comparison=comparison,
        metric=metric,
        ascending=ascending,
    )

    if ranked.empty:
        return ranked

    columns = [
        "rank",
        "strategy",
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
        "calmar_ratio",
        "sortino_ratio",
        "exposure",
        "win_rate",
        "profit_factor",
        "windows",
        "observations",
        "profitable_windows",
        "losing_windows",
        "positive_window_rate",
    ]

    available_columns = [
        column
        for column in columns
        if column in ranked.columns
    ]

    return ranked[available_columns].copy()


def get_best_strategy(
    comparison: dict[str, dict],
    metric: str = "total_return",
    ascending: bool = False,
) -> str | None:
    """
    Return the name of the highest-ranked walk-forward strategy.

    Returns None when the comparison is empty.
    """

    report = build_final_strategy_report(
        comparison=comparison,
        metric=metric,
        ascending=ascending,
    )

    if report.empty:
        return None

    return str(report.iloc[0]["strategy"])


def build_final_comparison_report(
    comparison: dict[str, dict],
    metric: str = "total_return",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Build a final ranked report for a regular strategy comparison.

    This report is intended for compare_strategies() output,
    not walk-forward comparison output.

    Ranking rules
    -------------
    1. Primary metric.
    2. Absolute max drawdown.
    3. Strategy name as deterministic tie-breaker.

    For max_drawdown itself, lower absolute drawdown is better.
    """

    if not isinstance(comparison, dict):
        raise TypeError(
            "comparison must be a dictionary."
        )

    if not comparison:
        return pd.DataFrame()

    if not isinstance(metric, str):
        raise TypeError(
            "metric must be a string."
        )

    dataframe = comparison_dataframe(comparison)

    if metric not in dataframe.columns:
        raise ValueError(
            f"Unknown ranking metric: {metric}"
        )

    if "max_drawdown" not in dataframe.columns:
        raise ValueError(
            "Missing required ranking column: max_drawdown"
        )

    dataframe = dataframe.copy()
    dataframe.index.name = "strategy"
    dataframe = dataframe.reset_index()

    dataframe["_ranking_max_drawdown"] = (
        dataframe["max_drawdown"].abs()
    )

    if metric == "max_drawdown":
        primary_column = "_ranking_max_drawdown"
        primary_ascending = True
    else:
        primary_column = metric
        primary_ascending = ascending

    dataframe = dataframe.sort_values(
        by=[
            primary_column,
            "_ranking_max_drawdown",
            "strategy",
        ],
        ascending=[
            primary_ascending,
            True,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    dataframe = dataframe.drop(
        columns=["_ranking_max_drawdown"]
    )

    dataframe.insert(
        0,
        "rank",
        dataframe.index + 1,
    )

    return dataframe


def get_best_comparison_strategy(
    comparison: dict[str, dict],
    metric: str = "total_return",
    ascending: bool = False,
) -> str | None:
    """
    Return the name of the highest-ranked strategy
    from a regular strategy comparison.

    Returns None when the comparison is empty.
    """

    report = build_final_comparison_report(
        comparison=comparison,
        metric=metric,
        ascending=ascending,
    )

    if report.empty:
        return None

    return str(report.iloc[0]["strategy"])
