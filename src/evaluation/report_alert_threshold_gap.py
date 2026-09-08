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


def calculate_gap(
    alert_count: int,
    threshold: int,
) -> int:
    """Calculate the distance from an alert count to the threshold."""
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

    return alert_count - threshold


def calculate_absolute_gap(
    alert_count: int,
    threshold: int,
) -> int:
    """Calculate the absolute distance from an alert count to the threshold."""
    return abs(
        calculate_gap(
            alert_count,
            threshold,
        )
    )


def calculate_average_gap(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> float:
    """Calculate the mean signed gap from the threshold."""
    _validate_history(history)
    _validate_threshold(threshold)

    if not history:
        return 0.0

    return sum(
        calculate_gap(
            item["alert_count"],
            threshold,
        )
        for item in history
    ) / len(history)


def calculate_average_absolute_gap(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> float:
    """Calculate the mean absolute gap from the threshold."""
    _validate_history(history)
    _validate_threshold(threshold)

    if not history:
        return 0.0

    return sum(
        calculate_absolute_gap(
            item["alert_count"],
            threshold,
        )
        for item in history
    ) / len(history)


def find_above_threshold_positions(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> list[int]:
    """Return positions whose alert count is above the threshold."""
    _validate_history(history)
    _validate_threshold(threshold)

    return [
        index
        for index, item in enumerate(history)
        if item["alert_count"] > threshold
    ]


def find_below_threshold_positions(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> list[int]:
    """Return positions whose alert count is below the threshold."""
    _validate_history(history)
    _validate_threshold(threshold)

    return [
        index
        for index, item in enumerate(history)
        if item["alert_count"] < threshold
    ]


def build_threshold_gap_summary(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> dict[str, Any]:
    """Build a complete signed-gap summary."""
    _validate_history(history)
    _validate_threshold(threshold)

    return {
        "snapshot_count": len(history),
        "threshold": threshold,
        "average_gap": calculate_average_gap(
            history,
            threshold,
        ),
        "average_absolute_gap": calculate_average_absolute_gap(
            history,
            threshold,
        ),
        "above_threshold_positions": find_above_threshold_positions(
            history,
            threshold,
        ),
        "below_threshold_positions": find_below_threshold_positions(
            history,
            threshold,
        ),
    }
