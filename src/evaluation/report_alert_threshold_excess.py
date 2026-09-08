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


def calculate_excess(
    alert_count: int,
    threshold: int,
) -> int:
    """Calculate how far an alert count exceeds the threshold."""
    if (
        not isinstance(alert_count, int)
        or isinstance(alert_count, bool)
    ):
        raise TypeError(
            "alert_count must be an integer."
        )

    if alert_count < 0:
        raise ValueError(
            "alert_count must not be negative."
        )

    _validate_threshold(threshold)

    return max(
        0,
        alert_count - threshold,
    )


def calculate_total_excess(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> int:
    """Calculate the total amount by which observations exceed the threshold."""
    _validate_history(history)
    _validate_threshold(threshold)

    return sum(
        calculate_excess(
            item["alert_count"],
            threshold,
        )
        for item in history
    )


def calculate_average_excess(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> float:
    """Calculate the average threshold excess across observations."""
    _validate_history(history)
    _validate_threshold(threshold)

    if not history:
        return 0.0

    return calculate_total_excess(
        history,
        threshold,
    ) / len(history)


def find_excess_positions(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> list[int]:
    """Return positions whose alert count exceeds the threshold."""
    _validate_history(history)
    _validate_threshold(threshold)

    return [
        index
        for index, item in enumerate(history)
        if item["alert_count"] > threshold
    ]


def calculate_max_excess(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> int:
    """Return the largest threshold excess."""
    _validate_history(history)
    _validate_threshold(threshold)

    if not history:
        return 0

    return max(
        calculate_excess(
            item["alert_count"],
            threshold,
        )
        for item in history
    )


def build_threshold_excess_summary(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> dict[str, Any]:
    """Build a complete summary of threshold excess."""
    _validate_history(history)
    _validate_threshold(threshold)

    return {
        "snapshot_count": len(history),
        "threshold": threshold,
        "total_excess": calculate_total_excess(
            history,
            threshold,
        ),
        "average_excess": calculate_average_excess(
            history,
            threshold,
        ),
        "excess_positions": find_excess_positions(
            history,
            threshold,
        ),
        "max_excess": calculate_max_excess(
            history,
            threshold,
        ),
    }
