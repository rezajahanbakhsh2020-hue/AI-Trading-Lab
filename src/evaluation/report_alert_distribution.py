from __future__ import annotations

from typing import Any, Mapping


def calculate_alert_distribution(
    history: list[Mapping[str, Any]],
) -> dict[int, int]:
    """
    Count how many snapshots contain each alert count.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    distribution: dict[int, int] = {}

    for item in history:
        if not isinstance(item, Mapping):
            raise TypeError(
                "Each history item must be a mapping."
            )

        value = item.get("alert_count")

        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise ValueError(
                "Each history item must contain a numeric alert_count."
            )

        if value < 0:
            raise ValueError(
                "alert_count must not be negative."
            )

        if not float(value).is_integer():
            raise ValueError(
                "alert_count must be an integer."
            )

        count = int(value)
        distribution[count] = (
            distribution.get(count, 0) + 1
        )

    return dict(sorted(distribution.items()))


def find_most_common_alert_count(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Return the alert count that occurs most frequently.

    When multiple counts have the same frequency, the smallest
    alert count is returned.
    """
    distribution = calculate_alert_distribution(history)

    if not distribution:
        raise ValueError(
            "No alert history available."
        )

    return min(
        distribution,
        key=lambda value: (
            -distribution[value],
            value,
        ),
    )


def calculate_alert_count_share(
    history: list[Mapping[str, Any]],
    alert_count: int,
) -> float:
    """
    Calculate the fraction of snapshots having a specific alert count.
    """
    if not isinstance(alert_count, int) or isinstance(
        alert_count,
        bool,
    ):
        raise TypeError(
            "alert_count must be an integer."
        )

    if alert_count < 0:
        raise ValueError(
            "alert_count must not be negative."
        )

    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    if not history:
        raise ValueError(
            "No alert history available."
        )

    distribution = calculate_alert_distribution(
        history,
    )

    occurrences = distribution.get(
        alert_count,
        0,
    )

    return round(
        occurrences / len(history),
        10,
    )


def build_alert_distribution_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete distribution summary for alert history.
    """
    distribution = calculate_alert_distribution(history)

    if not distribution:
        raise ValueError(
            "No alert history available."
        )

    most_common = find_most_common_alert_count(
        history,
    )

    return {
        "distribution": distribution,
        "most_common_alert_count": most_common,
        "most_common_share": round(
            distribution[most_common] / len(history),
            10,
        ),
    }
