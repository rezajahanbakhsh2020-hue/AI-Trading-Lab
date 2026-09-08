from __future__ import annotations

from typing import Any, Mapping


def calculate_alert_statistics(
    history: list[Mapping[str, Any]],
) -> dict[str, float]:
    """
    Calculate basic statistics for alert counts across history.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    if not history:
        raise ValueError(
            "No alert history available."
        )

    values: list[float] = []

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

        values.append(float(value))

    minimum = min(values)
    maximum = max(values)
    mean = sum(values) / len(values)

    return {
        "count": float(len(values)),
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "range": maximum - minimum,
    }


def calculate_zero_alert_snapshots(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Count snapshots that contain zero alerts.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    count = 0

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

        if value == 0:
            count += 1

    return count


def calculate_alert_free_ratio(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the fraction of snapshots with zero alerts.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    if not history:
        raise ValueError(
            "No alert history available."
        )

    zero_count = calculate_zero_alert_snapshots(
        history,
    )

    return round(
        zero_count / len(history),
        10,
    )


def build_alert_statistics_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete statistical summary of alert history.
    """
    statistics = calculate_alert_statistics(history)
    zero_count = calculate_zero_alert_snapshots(history)
    alert_free_ratio = calculate_alert_free_ratio(history)

    return {
        "statistics": statistics,
        "zero_alert_snapshots": zero_count,
        "alert_free_ratio": alert_free_ratio,
    }
