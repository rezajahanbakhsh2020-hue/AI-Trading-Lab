from __future__ import annotations

from typing import Any, Mapping


def format_report_text(
    report: Mapping[str, Any],
    precision: int = 4,
) -> str:
    """
    Format an evaluation report as readable plain text.

    Parameters
    ----------
    report:
        Evaluation report represented as a mapping.
    precision:
        Number of decimal places for numeric values.

    Returns
    -------
    str
        Formatted report text.

    Raises
    ------
    TypeError
        If report is not a mapping.
    ValueError
        If precision is negative.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    if precision < 0:
        raise ValueError("precision must be non-negative.")

    lines = []

    for field, value in report.items():
        if isinstance(value, bool):
            formatted_value = str(value)
        elif isinstance(value, (int, float)):
            formatted_value = f"{value:.{precision}f}"
        else:
            formatted_value = str(value)

        lines.append(f"{field}: {formatted_value}")

    return "\n".join(lines)


def format_report_markdown(
    report: Mapping[str, Any],
    title: str = "Evaluation Report",
) -> str:
    """
    Format an evaluation report as a Markdown table.

    Parameters
    ----------
    report:
        Evaluation report represented as a mapping.
    title:
        Markdown heading placed above the table.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    if not isinstance(title, str):
        raise TypeError("title must be a string.")

    lines = [
        f"# {title}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]

    for field, value in report.items():
        lines.append(f"| {field} | {value} |")

    return "\n".join(lines)


def format_report_summary(
    report: Mapping[str, Any],
) -> str:
    """
    Format the main evaluation metrics as a compact single-line summary.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    fields = (
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "win_rate",
        "profit_factor",
    )

    parts = [
        f"{field}={report[field]}"
        for field in fields
        if field in report
    ]

    return ", ".join(parts)
