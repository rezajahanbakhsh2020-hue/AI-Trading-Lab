from __future__ import annotations

from typing import Any, Mapping


def calculate_alert_count_trend(
    history: list[Mapping[str, Any]],
) -> dict[str, float | str]:
    """
    Calculate the trend of alert counts across history.

    History is assumed to be ordered from oldest to newest.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

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

    if not values:
        raise ValueError(
            "No alert history available."
        )

    first = values[0]
    last = values[-1]
    change = round(last - first, 10)

    if change > 0:
        direction = "increasing"
    elif change < 0:
        direction = "decreasing"
    else:
        direction = "stable"

    return {
        "first": first,
        "last": last,
        "change": change,
        "direction": direction,
    }


def calculate_alert_rate(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the average number of alerts per history snapshot.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    if not history:
        raise ValueError(
            "No alert history available."
        )

    total = 0

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

        total += value

    return round(
        total / len(history),
        10,
    )


def build_alert_trend_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert trend summary.
    """
    trend = calculate_alert_count_trend(history)
    rate = calculate_alert_rate(history)

    return {
        "snapshot_count": len(history),
        "trend": trend,
        "average_alert_rate": rate,
    }
