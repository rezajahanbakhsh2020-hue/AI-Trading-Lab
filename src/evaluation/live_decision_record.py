from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any


DECISION_RECORD_FIELDS = (
    "timestamp",
    "symbol",
    "interval",
    "signal",
    "signal_label",
    "trend",
    "strategy",
    "entry_price",
    "stop_loss",
    "take_profit",
    "risk_reward_ratio",
    "stability_score",
    "market_state",
    "quote_age_seconds",
    "quote_stale",
    "candle_count",
)

AUTHORIZATION_RECORD_FIELDS = (
    "runtime_authorization_fingerprint",
    "authorization_policy_version",
    "authorized_at_utc",
    "promoted_artifact_fingerprint",
    "governance_decision_fingerprint",
    "campaign_selection_decision_fingerprint",
    "candidate_id",
    "strategy_name",
    "strategy_version",
)

LIFECYCLE_RECORD_FIELDS = (
    "decision_id",
    "signal_id",
    "canonical_live_decision_fingerprint",
    "current_lifecycle_state",
    "transition_history",
)


def build_live_decision_record(
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build a normalized, immutable-style record of one live decision.

    The function only records values already present in the supplied
    snapshot. It does not recalculate signals, risk levels, or stability.
    Missing optional values remain None.
    """
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    timestamp = snapshot.get("timestamp")
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    signal = snapshot.get("signal")
    signal_label = snapshot.get("signal_label")

    if signal_label is None and isinstance(signal, str):
        signal_label = signal

    res = {
        "timestamp": timestamp,
        "symbol": snapshot.get("symbol"),
        "interval": snapshot.get("interval"),
        "signal": signal,
        "signal_label": signal_label,
        "reason": snapshot.get("reason"),
        "trend": snapshot.get("trend"),
        "strategy": snapshot.get("strategy"),
        "entry_price": snapshot.get("entry_price"),
        "stop_loss": snapshot.get("stop_loss"),
        "take_profit": snapshot.get("take_profit"),
        "risk_reward_ratio": snapshot.get("risk_reward_ratio"),
        "stability_score": snapshot.get("stability_score"),
        "market_state": snapshot.get(
            "market_state",
            snapshot.get("marketState"),
        ),
        "quote_age_seconds": snapshot.get("quote_age_seconds"),
        "quote_stale": snapshot.get("quote_stale"),
        "candle_count": snapshot.get("candle_count"),
    }

    # Include authorization fields if present in snapshot
    for auth_field in AUTHORIZATION_RECORD_FIELDS:
        if auth_field in snapshot:
            res[auth_field] = snapshot.get(auth_field)

    # Include lifecycle fields if present in snapshot
    for life_field in LIFECYCLE_RECORD_FIELDS:
        if life_field in snapshot:
            res[life_field] = snapshot.get(life_field)

    return res


def validate_live_decision_record(
    record: Mapping[str, Any],
) -> bool:
    """
    Validate the structural integrity of a live decision record.

    Required fields must exist and core identity fields must not be empty.
    When authorization lineage is present, all mandatory authorization fields
    must be non-empty strings without missing/fabricated values.
    """
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")

    missing = [
        field
        for field in DECISION_RECORD_FIELDS
        if field not in record
    ]

    if missing:
        raise ValueError(
            f"record is missing required fields: {', '.join(missing)}"
        )

    for field in ("timestamp", "symbol", "interval"):
        value = record.get(field)
        if value is None or value == "":
            raise ValueError(
                f"record field '{field}' must not be empty"
            )

    # If runtime authorization fingerprint is present or any authorization field is set,
    # validate that all mandatory authorization fields are present and non-empty.
    has_auth = any(f in record for f in AUTHORIZATION_RECORD_FIELDS)
    if has_auth:
        mandatory_auth_fields = (
            "runtime_authorization_fingerprint",
            "authorization_policy_version",
            "authorized_at_utc",
            "promoted_artifact_fingerprint",
            "governance_decision_fingerprint",
            "candidate_id",
            "strategy_name",
            "strategy_version",
        )
        for auth_field in mandatory_auth_fields:
            if auth_field not in record:
                raise ValueError(
                    f"Production live decision record missing mandatory authorization field '{auth_field}'."
                )
            val = record.get(auth_field)
            if val is None or not str(val).strip():
                raise ValueError(
                    f"Production live decision record field '{auth_field}' must be a non-empty string."
                )

        # campaign_selection_decision_fingerprint may be None only if legitimately absent,
        # but if provided, must be a non-empty string.
        csdf = record.get("campaign_selection_decision_fingerprint")
        if csdf is not None and not str(csdf).strip():
            raise ValueError(
                "Production live decision record field 'campaign_selection_decision_fingerprint' cannot be empty if provided."
            )

    # Validate canonical lifecycle fingerprint if present
    cld_fp = record.get("canonical_live_decision_fingerprint")
    if cld_fp is not None and not str(cld_fp).strip():
        raise ValueError(
            "Production live decision record field 'canonical_live_decision_fingerprint' cannot be empty if provided."
        )

    return True


def record_live_decision(
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build and validate a live decision record in one operation.
    """
    record = build_live_decision_record(snapshot)
    validate_live_decision_record(record)
    return record
