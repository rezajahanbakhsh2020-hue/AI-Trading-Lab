from __future__ import annotations

from collections.abc import Mapping
from typing import Any


TRACKED_FIELDS = (
    "signal_label",
    "trend",
    "strategy",
    "entry_price",
    "stop_loss",
    "take_profit",
    "risk_reward_ratio",
    "stability_score",
    "market_state",
    "quote_stale",
)


def compare_live_decisions(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Compare two consecutive live decision records.

    Values are compared exactly as recorded. No trading value is
    recalculated or inferred.
    """
    if not isinstance(previous, Mapping):
        raise TypeError("previous must be a mapping")

    if not isinstance(current, Mapping):
        raise TypeError("current must be a mapping")

    changed_fields: dict[str, dict[str, Any]] = {}

    for field in TRACKED_FIELDS:
        previous_value = previous.get(field)
        current_value = current.get(field)

        if previous_value != current_value:
            changed_fields[field] = {
                "previous": previous_value,
                "current": current_value,
            }

    return {
        "changed": bool(changed_fields),
        "changed_field_count": len(changed_fields),
        "changed_fields": changed_fields,
        "previous_timestamp": previous.get("timestamp"),
        "current_timestamp": current.get("timestamp"),
    }


def summarize_live_decision_changes(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build a compact summary of changes between two decisions.
    """
    comparison = compare_live_decisions(previous, current)

    changed_fields = comparison["changed_fields"]

    signal_changed = "signal_label" in changed_fields
    trend_changed = "trend" in changed_fields
    strategy_changed = "strategy" in changed_fields
    risk_changed = any(
        field in changed_fields
        for field in (
            "entry_price",
            "stop_loss",
            "take_profit",
            "risk_reward_ratio",
        )
    )
    stability_changed = "stability_score" in changed_fields
    market_state_changed = "market_state" in changed_fields
    quote_status_changed = "quote_stale" in changed_fields

    return {
        **comparison,
        "signal_changed": signal_changed,
        "trend_changed": trend_changed,
        "strategy_changed": strategy_changed,
        "risk_changed": risk_changed,
        "stability_changed": stability_changed,
        "market_state_changed": market_state_changed,
        "quote_status_changed": quote_status_changed,
    }


def validate_live_decision_change(
    result: Mapping[str, Any],
) -> bool:
    """
    Validate the structure of a decision-change result.
    """
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")

    required_fields = (
        "changed",
        "changed_field_count",
        "changed_fields",
        "previous_timestamp",
        "current_timestamp",
    )

    missing = [
        field
        for field in required_fields
        if field not in result
    ]

    if missing:
        raise ValueError(
            "change result is missing required fields: "
            + ", ".join(missing)
        )

    if not isinstance(result["changed"], bool):
        raise ValueError("changed must be boolean")

    if not isinstance(result["changed_field_count"], int):
        raise ValueError(
            "changed_field_count must be an integer"
        )

    if not isinstance(result["changed_fields"], Mapping):
        raise ValueError("changed_fields must be a mapping")

    if result["changed_field_count"] != len(
        result["changed_fields"]
    ):
        raise ValueError(
            "changed_field_count must match changed_fields"
        )

    return True
