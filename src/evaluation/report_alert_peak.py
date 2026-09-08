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


def calculate_peak_alert_count(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Calculate the maximum number of simultaneous alerts observed.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return max(
        item["alert_count"]
        for item in history
    )


def calculate_peak_alert_excess(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> int:
    """
    Calculate how far the peak alert count exceeds a threshold.

    Returns zero when the peak does not exceed the threshold.
    """
    _validate_history(history)

    if not isinstance(threshold, int) or isinstance(threshold, bool):
        raise ValueError(
            "threshold must be an integer."
        )

    if threshold < 0:
        raise ValueError(
            "threshold must not be negative."
        )

    peak = calculate_peak_alert_count(history)

    return max(peak - threshold, 0)


def build_alert_peak_summary(
    history: list[Mapping[str, Any]],
    threshold: int | None = None,
) -> dict[str, Any]:
    """
    Build a summary of the peak alert count.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    peak = calculate_peak_alert_count(history)

    result: dict[str, Any] = {
        "snapshot_count": len(history),
        "peak_alert_count": peak,
    }

    if threshold is not None:
        result["threshold"] = threshold
        result["peak_alert_excess"] = (
            calculate_peak_alert_excess(history, threshold)
        )

    return result
