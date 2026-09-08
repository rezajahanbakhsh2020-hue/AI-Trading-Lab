from __future__ import annotations

from typing import Any, Mapping


def rank_reports(
    reports: list[Mapping[str, Any]],
    metric: str,
    descending: bool = True,
) -> list[Mapping[str, Any]]:
    """
    Rank evaluation reports by a numeric metric.

    Parameters
    ----------
    reports:
        Collection of evaluation reports.
    metric:
        Metric used for ranking.
    descending:
        If True, highest values rank first. If False, lowest values
        rank first.

    Returns
    -------
    list[Mapping[str, Any]]
        Reports ordered by the selected metric.

    Raises
    ------
    TypeError
        If reports, metric, or descending has an invalid type.
    ValueError
        If a report does not contain a numeric value for the metric.
    """
    if not isinstance(reports, list):
        raise TypeError("reports must be a list.")

    if not isinstance(metric, str):
        raise TypeError("metric must be a string.")

    if not isinstance(descending, bool):
        raise TypeError("descending must be a boolean.")

    ranked_reports = []

    for report in reports:
        if not isinstance(report, Mapping):
            raise TypeError("Each report must be a mapping.")

        value = report.get(metric)

        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise ValueError(
                f"Report must contain a numeric value for: {metric}"
            )

        ranked_reports.append(report)

    return sorted(
        ranked_reports,
        key=lambda report: float(report[metric]),
        reverse=descending,
    )


def assign_report_ranks(
    reports: list[Mapping[str, Any]],
    metric: str,
    descending: bool = True,
) -> list[dict[str, Any]]:
    """
    Rank reports and add an explicit rank field.

    Parameters
    ----------
    reports:
        Collection of evaluation reports.
    metric:
        Metric used for ranking.
    descending:
        If True, highest values receive the best rank.

    Returns
    -------
    list[dict[str, Any]]
        Ranked copies of the reports with a "rank" field.
    """
    ranked = rank_reports(
        reports,
        metric,
        descending=descending,
    )

    return [
        {
            **dict(report),
            "rank": index,
        }
        for index, report in enumerate(ranked, start=1)
    ]


def get_top_reports(
    reports: list[Mapping[str, Any]],
    metric: str,
    count: int = 3,
    descending: bool = True,
) -> list[Mapping[str, Any]]:
    """
    Return the top reports according to a metric.

    Parameters
    ----------
    reports:
        Collection of evaluation reports.
    metric:
        Metric used for ranking.
    count:
        Maximum number of reports to return.
    descending:
        If True, highest values are considered best.

    Returns
    -------
    list[Mapping[str, Any]]
        Top ranked reports.

    Raises
    ------
    TypeError
        If an argument has an invalid type.
    ValueError
        If count is less than one.
    """
    if not isinstance(count, int) or isinstance(count, bool):
        raise TypeError("count must be an integer.")

    if count < 1:
        raise ValueError("count must be at least 1.")

    ranked = rank_reports(
        reports,
        metric,
        descending=descending,
    )

    return ranked[:count]
