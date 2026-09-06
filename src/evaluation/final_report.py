from __future__ import annotations

import pandas as pd

from src.evaluation.compare import rank_walk_forward_strategies


def build_final_strategy_report(
    comparison: dict[str, dict],
    metric: str = "total_return",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Build the final ranked strategy report.

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
        Ranked strategy report with a compact, stable set of
        performance and stability metrics.
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
    Return the name of the highest-ranked strategy.

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
