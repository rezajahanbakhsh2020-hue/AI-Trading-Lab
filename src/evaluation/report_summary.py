from __future__ import annotations

from typing import Any, Mapping


def build_report_summary(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build a compact summary from an evaluation report.

    Parameters
    ----------
    report:
        Evaluation report represented as a mapping.

    Returns
    -------
    dict[str, Any]
        Summary containing the main evaluation metrics.

    Raises
    ------
    TypeError
        If report is not a mapping.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    summary: dict[str, Any] = {}

    preferred_fields = (
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "win_rate",
        "profit_factor",
    )

    for field in preferred_fields:
        if field in report:
            summary[field] = report[field]

    return summary


def calculate_report_score(
    report: Mapping[str, Any],
) -> float:
    """
    Calculate a simple composite score from available report metrics.

    Parameters
    ----------
    report:
        Evaluation report represented as a mapping.

    Returns
    -------
    float
        Composite score based on available metrics.

    Raises
    ------
    TypeError
        If report is not a mapping.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    score = 0.0

    if isinstance(report.get("total_return"), (int, float)):
        score += float(report["total_return"])

    if isinstance(report.get("sharpe_ratio"), (int, float)):
        score += float(report["sharpe_ratio"])

    if isinstance(report.get("sortino_ratio"), (int, float)):
        score += float(report["sortino_ratio"])

    if isinstance(report.get("calmar_ratio"), (int, float)):
        score += float(report["calmar_ratio"])

    if isinstance(report.get("max_drawdown"), (int, float)):
        score += float(report["max_drawdown"])

    return score


def build_report_overview(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build a complete overview containing summary and composite score.

    Parameters
    ----------
    report:
        Evaluation report represented as a mapping.

    Returns
    -------
    dict[str, Any]
        Report overview.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    return {
        "summary": build_report_summary(report),
        "score": calculate_report_score(report),
    }
