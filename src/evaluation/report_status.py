from __future__ import annotations

from typing import Any, Mapping


def determine_report_status(
    report: Mapping[str, Any],
    minimum_sharpe: float = 1.0,
    maximum_drawdown: float = -0.20,
    minimum_win_rate: float = 0.50,
) -> str:
    """
    Determine the overall status of an evaluation report.

    The report is considered:
    - "pass" when all available required metrics meet their thresholds.
    - "fail" when at least one available metric is below its threshold.
    - "incomplete" when none of the required metrics is available.

    Parameters
    ----------
    report:
        Evaluation report represented as a mapping.
    minimum_sharpe:
        Minimum acceptable Sharpe ratio.
    maximum_drawdown:
        Maximum acceptable drawdown.
    minimum_win_rate:
        Minimum acceptable win rate.

    Returns
    -------
    str
        "pass", "fail", or "incomplete".

    Raises
    ------
    TypeError
        If report is not a mapping.
    ValueError
        If thresholds are not numeric.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    thresholds = (
        minimum_sharpe,
        maximum_drawdown,
        minimum_win_rate,
    )

    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in thresholds
    ):
        raise ValueError("All thresholds must be numeric.")

    checks: list[bool] = []

    if "sharpe_ratio" in report:
        checks.append(
            isinstance(report["sharpe_ratio"], (int, float))
            and not isinstance(report["sharpe_ratio"], bool)
            and report["sharpe_ratio"] >= minimum_sharpe
        )

    if "max_drawdown" in report:
        checks.append(
            isinstance(report["max_drawdown"], (int, float))
            and not isinstance(report["max_drawdown"], bool)
            and report["max_drawdown"] >= maximum_drawdown
        )

    if "win_rate" in report:
        checks.append(
            isinstance(report["win_rate"], (int, float))
            and not isinstance(report["win_rate"], bool)
            and report["win_rate"] >= minimum_win_rate
        )

    if not checks:
        return "incomplete"

    return "pass" if all(checks) else "fail"


def is_report_acceptable(
    report: Mapping[str, Any],
    minimum_sharpe: float = 1.0,
    maximum_drawdown: float = -0.20,
    minimum_win_rate: float = 0.50,
) -> bool:
    """
    Return whether an evaluation report passes its quality thresholds.
    """
    return determine_report_status(
        report,
        minimum_sharpe=minimum_sharpe,
        maximum_drawdown=maximum_drawdown,
        minimum_win_rate=minimum_win_rate,
    ) == "pass"


def build_status_summary(
    report: Mapping[str, Any],
    minimum_sharpe: float = 1.0,
    maximum_drawdown: float = -0.20,
    minimum_win_rate: float = 0.50,
) -> dict[str, Any]:
    """
    Build a compact status summary for an evaluation report.
    """
    status = determine_report_status(
        report,
        minimum_sharpe=minimum_sharpe,
        maximum_drawdown=maximum_drawdown,
        minimum_win_rate=minimum_win_rate,
    )

    return {
        "status": status,
        "acceptable": status == "pass",
    }
