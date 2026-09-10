from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def build_live_runtime_preflight(
    runtime_decision: Mapping[str, Any],
    *,
    market_data_ready: bool = True,
    quote_ready: bool = True,
    timestamp_ready: bool = True,
) -> dict[str, Any]:
    """
    Build the final preflight result before entering live runtime.

    This function performs validation only.
    It does not connect to a broker and does not execute orders.
    """

    if not isinstance(runtime_decision, Mapping):
        raise TypeError(
            "runtime_decision must be a mapping."
        )

    failed_checks: list[str] = []

    if not bool(market_data_ready):
        failed_checks.append("market_data")

    if not bool(quote_ready):
        failed_checks.append("quote")

    if not bool(timestamp_ready):
        failed_checks.append("timestamp")

    if not bool(
        runtime_decision.get("runtime_allowed", False)
    ):
        failed_checks.append("runtime_permission")

    strategy = runtime_decision.get("strategy")

    if not isinstance(strategy, str) or not strategy.strip():
        failed_checks.append("strategy")

    readiness = runtime_decision.get("readiness")

    if not isinstance(readiness, Mapping):
        failed_checks.append("readiness")

    runtime_gate = runtime_decision.get(
        "runtime_gate"
    )

    if not isinstance(runtime_gate, Mapping):
        failed_checks.append("runtime_gate")

    unique_failed_checks: list[str] = []

    for check in failed_checks:
        if check not in unique_failed_checks:
            unique_failed_checks.append(check)

    ready = len(unique_failed_checks) == 0

    return {
        "preflight_ready": ready,
        "status": "READY" if ready else "BLOCKED",
        "strategy": strategy,
        "failed_checks": unique_failed_checks,
        "checked_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }


def is_live_runtime_preflight_ready(
    runtime_decision: Mapping[str, Any],
    *,
    market_data_ready: bool = True,
    quote_ready: bool = True,
    timestamp_ready: bool = True,
) -> bool:
    """
    Return only the final preflight permission.
    """

    result = build_live_runtime_preflight(
        runtime_decision,
        market_data_ready=market_data_ready,
        quote_ready=quote_ready,
        timestamp_ready=timestamp_ready,
    )

    return bool(result["preflight_ready"])


def live_runtime_preflight_message(
    runtime_decision: Mapping[str, Any],
    *,
    market_data_ready: bool = True,
    quote_ready: bool = True,
    timestamp_ready: bool = True,
) -> str:
    """
    Return a concise human-readable preflight result.
    """

    result = build_live_runtime_preflight(
        runtime_decision,
        market_data_ready=market_data_ready,
        quote_ready=quote_ready,
        timestamp_ready=timestamp_ready,
    )

    strategy = (
        result.get("strategy")
        or "no strategy selected"
    )

    if result["preflight_ready"]:
        return (
            "LIVE PREFLIGHT READY: "
            f"{strategy}"
        )

    return (
        "LIVE PREFLIGHT BLOCKED: "
        f"{strategy}; "
        f"failed checks: "
        f"{', '.join(result['failed_checks'])}."
    )
