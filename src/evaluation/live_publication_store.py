from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.evaluation.live_decision_lifecycle import (
    CanonicalLiveDecision,
    LiveDecisionLifecycleError,
    LiveDecisionLifecycleState,
    transition_live_decision,
    validate_live_decision_lifecycle,
)
from src.evaluation.live_production_decision import (
    ProductionIntelligencePublication,
    PromotedCandidateArtifact,
)
from src.evaluation.live_publication_delivery import (
    DeliveryStatus,
    PublicationDeliveryReceipt,
    append_delivery_receipt,
    find_delivery_receipt,
    map_publisher_result_to_status,
)


class PublicationIntegrityError(ValueError):
    """Raised when a conflicting publication replay or payload mismatch occurs."""


def save_publication_history(
    history: Iterable[Mapping[str, Any]],
    path: str | Path,
) -> Path:
    """Persist publication history list as JSON."""
    if isinstance(history, (str, bytes)) or not isinstance(history, Iterable):
        raise TypeError("history must be iterable")

    records: list[dict[str, Any]] = []
    for item in history:
        if not isinstance(item, Mapping):
            raise TypeError("each publication item must be a mapping")
        records.append(dict(item))

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def load_publication_history(path: str | Path) -> list[dict[str, Any]]:
    """Load publication history from JSON file."""
    target = Path(path)
    if not target.exists():
        return []

    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("stored publication history must be a list")

    return [dict(item) for item in raw if isinstance(item, Mapping)]


def append_publication_record(
    record: Mapping[str, Any],
    path: str | Path,
    enforce_idempotency: bool = True,
) -> list[dict[str, Any]]:
    """Append publication record to store with deterministic replay protection."""
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")

    new_record = dict(record)
    pub_id = new_record.get("publication_id") or new_record.get("event_id")
    if not pub_id or not str(pub_id).strip():
        raise ValueError("publication record missing publication_id")

    target = Path(path)
    history = load_publication_history(target)

    if enforce_idempotency and pub_id:
        for existing in history:
            ext_pub_id = existing.get("publication_id") or existing.get("event_id")
            if ext_pub_id == pub_id:
                # Compare authorization fingerprint in provenance explicitly for conflict detection
                new_prov = new_record.get("provenance", {})
                ext_prov = existing.get("provenance", {})
                new_raw_auth = new_prov.get("runtime_authorization_fingerprint")
                ext_raw_auth = ext_prov.get("runtime_authorization_fingerprint")

                new_auth_fp = str(new_raw_auth).strip() if new_raw_auth is not None and str(new_raw_auth).strip() != "" else None
                ext_auth_fp = str(ext_raw_auth).strip() if ext_raw_auth is not None and str(ext_raw_auth).strip() != "" else None

                existing_has_auth = ext_auth_fp is not None
                new_has_auth = new_auth_fp is not None

                if existing_has_auth != new_has_auth:
                    raise PublicationIntegrityError(
                        f"Conflicting publication replay detected for publication_id '{pub_id}': "
                        "authorization lineage presence mismatch."
                    )

                if existing_has_auth and new_has_auth and ext_auth_fp != new_auth_fp:
                    raise PublicationIntegrityError(
                        f"Conflicting publication replay detected for publication_id '{pub_id}': "
                        f"runtime_authorization_fingerprint mismatch ('{ext_auth_fp}' vs '{new_auth_fp}')."
                    )

                # Compare canonical live decision fingerprint in provenance explicitly for conflict detection
                new_cld_fp = new_prov.get("canonical_live_decision_fingerprint") or new_record.get("canonical_live_decision_fingerprint")
                ext_cld_fp = ext_prov.get("canonical_live_decision_fingerprint") or existing.get("canonical_live_decision_fingerprint")

                new_cld_str = str(new_cld_fp).strip() if new_cld_fp is not None and str(new_cld_fp).strip() != "" else None
                ext_cld_str = str(ext_cld_fp).strip() if ext_cld_fp is not None and str(ext_cld_fp).strip() != "" else None

                if (ext_cld_str is not None) != (new_cld_str is not None):
                    raise PublicationIntegrityError(
                        f"Conflicting publication replay detected for publication_id '{pub_id}': "
                        "canonical_live_decision_fingerprint presence mismatch."
                    )

                if ext_cld_str is not None and new_cld_str is not None and ext_cld_str != new_cld_str:
                    raise PublicationIntegrityError(
                        f"Conflicting publication replay detected for publication_id '{pub_id}': "
                        f"canonical_live_decision_fingerprint mismatch ('{ext_cld_str}' vs '{new_cld_str}')."
                    )

                # Compare entire publication record representation for exact match
                if existing == new_record:
                    # Identical replay: safe no-op
                    return history

                # Conflicting replay for same identity -> fail closed
                raise PublicationIntegrityError(
                    f"Conflicting publication replay detected for publication_id '{pub_id}'. "
                    "Existing publication record differs from new record."
                )

    history.append(new_record)
    save_publication_history(history, target)
    return history


def publish_canonical_live_decision(
    canonical_decision: CanonicalLiveDecision,
    publisher: Any,
    candidate: PromotedCandidateArtifact,
    path: str | Path,
    skip_if_no_trade: bool = False,
    enforce_idempotency: bool = True,
    actor: str = "live_publication_store",
    timestamp_utc: Optional[str] = None,
    delivery_path: Optional[str | Path] = None,
) -> tuple[CanonicalLiveDecision, ProductionIntelligencePublication, Any]:
    """Harden publication lifecycle boundary for a CanonicalLiveDecision.

    Enforces semantic ordering:
    PERSISTED -> construct publication artifact -> consult delivery ledger ->
    invoke publisher (if not already DELIVERED) -> map result ->
    persist immutable delivery receipt -> ONLY THEN transition CanonicalLiveDecision
    PERSISTED -> PUBLISHED iff delivery is DELIVERED.
    """
    if not isinstance(canonical_decision, CanonicalLiveDecision):
        raise TypeError(f"canonical_decision must be a CanonicalLiveDecision, got {type(canonical_decision).__name__}")

    validate_live_decision_lifecycle(canonical_decision)

    target_pub_path = Path(path)
    d_path = Path(delivery_path) if delivery_path is not None else target_pub_path.parent / "delivery_history.json"

    now_iso = timestamp_utc if timestamp_utc is not None else datetime.now(timezone.utc).isoformat()

    # If canonical decision is already marked PUBLISHED
    if canonical_decision.current_state == LiveDecisionLifecycleState.PUBLISHED:
        history = load_publication_history(target_pub_path)
        for existing in history:
            ext_cld = existing.get("provenance", {}).get("canonical_live_decision_fingerprint")
            if ext_cld == canonical_decision.canonical_live_decision_fingerprint:
                dec_obj = canonical_decision.decision
                publication = ProductionIntelligencePublication.from_artifacts(
                    decision=dec_obj,
                    signal=canonical_decision.signal,
                    risk=canonical_decision.risk_levels,
                    candidate=candidate,
                    authorization=canonical_decision.authorization_receipt,
                )
                existing_receipt = find_delivery_receipt(
                    publication.publication_id,
                    canonical_decision.canonical_live_decision_fingerprint,
                    path=d_path,
                )
                receipt_fp = existing_receipt.delivery_receipt_fingerprint if existing_receipt else None
                return canonical_decision, publication, {
                    "status": "SKIPPED_ALREADY_PUBLISHED",
                    "published": True,
                    "delivery_status": DeliveryStatus.DELIVERED.value,
                    "delivery_receipt_fingerprint": receipt_fp,
                }
        raise LiveDecisionLifecycleError("Canonical decision is marked PUBLISHED but missing from publication store.")

    if canonical_decision.current_state != LiveDecisionLifecycleState.PERSISTED:
        raise LiveDecisionLifecycleError(
            f"Cannot publish canonical live decision in state '{canonical_decision.current_state.value}': "
            f"expected state 'PERSISTED'. Publication cannot bypass persistence."
        )

    # Attach canonical fingerprint and state to decision before creating publication
    dec_obj = canonical_decision.decision
    object.__setattr__(dec_obj, "canonical_live_decision_fingerprint", canonical_decision.canonical_live_decision_fingerprint)
    object.__setattr__(dec_obj, "current_lifecycle_state", canonical_decision.current_state.value)

    publication = ProductionIntelligencePublication.from_artifacts(
        decision=dec_obj,
        signal=canonical_decision.signal,
        risk=canonical_decision.risk_levels,
        candidate=candidate,
        authorization=canonical_decision.authorization_receipt,
    )

    if publication.provenance.get("canonical_live_decision_fingerprint") != canonical_decision.canonical_live_decision_fingerprint:
        raise PublicationIntegrityError("Publication provenance fingerprint mismatch with canonical decision.")

    # Consult durable delivery ledger BEFORE performing an outbound delivery attempt
    existing_receipt = find_delivery_receipt(
        publication.publication_id,
        canonical_decision.canonical_live_decision_fingerprint,
        path=d_path,
    )

    if existing_receipt is not None and existing_receipt.delivery_status == DeliveryStatus.DELIVERED:
        # Already DELIVERED: do not invoke publisher again!
        delivery_ts = existing_receipt.last_attempt_at_utc
        published_decision = transition_live_decision(
            canonical_decision,
            LiveDecisionLifecycleState.PUBLISHED,
            actor=actor,
            timestamp_utc=delivery_ts,
            reason="canonical_publication_delivered",
        )
        validate_live_decision_lifecycle(published_decision)

        publication.provenance["canonical_live_decision_fingerprint"] = published_decision.canonical_live_decision_fingerprint
        publication.provenance["current_lifecycle_state"] = published_decision.current_state.value

        append_publication_record(publication.as_dict(), target_pub_path, enforce_idempotency=enforce_idempotency)

        pub_res = {
            "status": "PUBLISHED",
            "published": True,
            "event_id": publication.publication_id,
            "publication_id": publication.publication_id,
            "delivery_status": DeliveryStatus.DELIVERED.value,
            "delivery_receipt_fingerprint": existing_receipt.delivery_receipt_fingerprint,
            "note": "Already delivered according to delivery ledger",
        }
        return published_decision, publication, pub_res

    # Determine attempt count and timing
    attempt_count = (existing_receipt.attempt_count + 1) if (existing_receipt and existing_receipt.delivery_status.is_retryable) else 1
    first_attempt_at_utc = existing_receipt.first_attempt_at_utc if (existing_receipt and existing_receipt.delivery_status.is_retryable) else now_iso

    # Invoke publisher
    publish_result = None
    if publisher is not None and hasattr(publisher, "publish"):
        publish_result = publisher.publish(
            publication,
            skip_if_no_trade=skip_if_no_trade,
        )
    elif publisher is None:
        publish_result = {
            "status": "SKIPPED_DISABLED",
            "published": False,
            "reason": "Publisher is None",
        }

    # Map publisher result to DeliveryStatus
    del_status, http_status, remote_event_id, error_msg = map_publisher_result_to_status(publish_result)

    # Record delivery receipt in durable recovery ledger
    receipt = PublicationDeliveryReceipt(
        publication_id=publication.publication_id,
        decision_id=canonical_decision.decision.decision_id,
        signal_id=canonical_decision.signal.signal_id,
        canonical_live_decision_fingerprint=canonical_decision.canonical_live_decision_fingerprint,
        runtime_authorization_fingerprint=canonical_decision.authorization_receipt.authorization_fingerprint,
        candidate_id=candidate.candidate_id,
        strategy_name=candidate.strategy_name,
        strategy_version=candidate.strategy_version,
        symbol=canonical_decision.decision.symbol,
        timeframe=canonical_decision.decision.timeframe,
        delivery_status=del_status,
        attempt_count=attempt_count,
        first_attempt_at_utc=first_attempt_at_utc,
        last_attempt_at_utc=now_iso,
        http_status=http_status,
        remote_event_id=remote_event_id,
        error=error_msg,
        publisher_version="1.0",
        promoted_artifact_fingerprint=canonical_decision.authorization_receipt.promoted_artifact_fingerprint,
        governance_decision_fingerprint=canonical_decision.authorization_receipt.governance_decision_fingerprint,
        campaign_selection_decision_fingerprint=canonical_decision.authorization_receipt.campaign_selection_decision_fingerprint,
    )

    append_delivery_receipt(receipt, d_path, enforce_idempotency=enforce_idempotency)

    # ONLY IF delivery status is DELIVERED do we transition to PUBLISHED and append to publication history store
    if del_status == DeliveryStatus.DELIVERED:
        published_decision = transition_live_decision(
            canonical_decision,
            LiveDecisionLifecycleState.PUBLISHED,
            actor=actor,
            timestamp_utc=now_iso,
            reason="canonical_publication_delivered",
        )
        validate_live_decision_lifecycle(published_decision)
        final_decision = published_decision

        publication.provenance["canonical_live_decision_fingerprint"] = final_decision.canonical_live_decision_fingerprint
        publication.provenance["current_lifecycle_state"] = final_decision.current_state.value

        append_publication_record(publication.as_dict(), target_pub_path, enforce_idempotency=enforce_idempotency)
    else:
        final_decision = canonical_decision  # Remains PERSISTED!

    if isinstance(publish_result, Mapping):
        res_dict = dict(publish_result)
    else:
        res_dict = {}

    res_dict["delivery_status"] = del_status.value
    res_dict["delivery_receipt_fingerprint"] = receipt.delivery_receipt_fingerprint
    res_dict["current_lifecycle_state"] = final_decision.current_state.value

    return final_decision, publication, res_dict
