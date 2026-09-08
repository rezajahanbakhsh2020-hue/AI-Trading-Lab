from __future__ import annotations

from typing import Any, Mapping


def _validate_history(
    history: list[Mapping[str, Any]],
) -> None:
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    for item in history:
        if not isinstance(item, Mapping):
            raise TypeError(
                "Each history item must be a mapping."
            )

        value = item.get("alert_count")

        if (
            not isinstance(value, int)
            or isinstance(value, bool)
        ):
            raise ValueError(
                "Each history item must contain an integer alert_count."
            )

        if value < 0:
            raise ValueError(
                "alert_count must not be negative."
            )


def _validate_threshold(threshold: int) -> None:
    if (
        not isinstance(threshold, int)
        or isinstance(threshold, bool)
    ):
        raise TypeError(
            "threshold must be an integer."
        )

    if threshold < 0:
        raise ValueError(
            "threshold must not be negative."
        )


def count_breaches(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> int:
    """Count observations whose alert count exceeds the threshold."""
    _validate_history(history)
    _validate_threshold(threshold)

    return sum(
        1
        for item in history
        if item["alert_count"] > threshold
    )


def calculate_breach_ratio(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> float:
    """Calculate the proportion of observations above the threshold."""
    _validate_history(history)
    _validate_threshold(threshold)

    if not history:
        return 0.0

    return count_breaches(
        history,
        threshold,
    ) / len(history)


def calculate_breach_percentage(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> float:
    """Calculate the percentage of observations above the threshold."""
    return (
        calculate_breach_ratio(
            history,
            threshold,
        )
        * 100.0
    )


def calculate_max_alert_count(
    history: list[Mapping[str, Any]],
) -> int:
    """Return the maximum alert count in the history."""
    _validate_history(history)

    if not history:
        return 0

    return max(
        item["alert_count"]
        for item in history
    )


def calculate_average_alert_count(
    history: list[Mapping[str, Any]],
) -> float:
    """Return the arithmetic mean alert count."""
    _validate_history(history)

    if not history:
        return 0.0

    return sum(
        item["alert_count"]
        for item in history
    ) / len(history)


def build_alert_threshold_summary(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> dict[str, Any]:
    """Build a summary of alert counts relative to a threshold."""
    _validate_history(history)
    _validate_threshold(threshold)

    return {
        "snapshot_count": len(history),
        "threshold": threshold,
        "breach_count": count_breaches(
            history,
            threshold,
        ),
        "breach_ratio": calculate_breach_ratio(
            history,
            threshold,
        ),
        "breach_percentage": calculate_breach_percentage(
            history,
            threshold,
        ),
        "max_alert_count": calculate_max_alert_count(
            history,
        ),
        "average_alert_count": calculate_average_alert_count(
            history,
        ),
    }
