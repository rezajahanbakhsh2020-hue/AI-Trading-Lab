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
    load_delivery_history,
    map_publisher_result_to_status,
)
from src.evaluation.research_store import DEFAULT_RESEARCH_DIR


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
    mtf_intelligence: Optional[Any] = None,
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
                    mtf_intelligence=mtf_intelligence,
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
        mtf_intelligence=mtf_intelligence,
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
    try:
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
    except Exception as exc:
        publish_result = {
            "status": "FAILED",
            "published": False,
            "reason": f"Publisher exception: {exc}",
            "error": str(exc),
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


def _reconstitute_canonical_live_decision_from_record(
    record: Mapping[str, Any],
    candidate: PromotedCandidateArtifact,
) -> CanonicalLiveDecision:
    """Reconstitute a CanonicalLiveDecision instance from a persisted decision record and PromotedCandidateArtifact."""
    from src.evaluation.live_decision_lifecycle import LifecycleTransitionRecord
    from src.evaluation.live_production_decision import (
        Direction,
        ProductionAuthorizationReceipt,
        ProductionDecision,
        ProductionSignal,
        calculate_production_risk_levels,
    )

    dec_id = str(record["decision_id"])
    sig_id = str(record.get("signal_id", dec_id))
    sym = str(record["symbol"]).upper()
    tf = str(record["interval"])
    m_ts = str(record["timestamp"])
    sig_label = str(record.get("signal_label", "NO TRADE")).upper()
    direction = Direction(sig_label) if sig_label in ("BUY", "SELL", "NO TRADE") else Direction.NO_TRADE

    auth_receipt = ProductionAuthorizationReceipt(
        candidate_id=str(record["candidate_id"]),
        strategy_name=str(record["strategy_name"]),
        strategy_version=str(record["strategy_version"]),
        symbol=sym,
        timeframe=tf,
        promoted_artifact_fingerprint=str(record["promoted_artifact_fingerprint"]),
        governance_decision_fingerprint=str(record["governance_decision_fingerprint"]),
        campaign_selection_decision_fingerprint=record.get("campaign_selection_decision_fingerprint"),
        authorization_policy_version=str(record.get("authorization_policy_version", "runtime_auth_v1.0")),
        authorized_at_utc=str(record.get("authorized_at_utc", m_ts)),
        authorization_fingerprint=str(record["runtime_authorization_fingerprint"]),
        operational_stability_score=float(record.get("stability_score", candidate.operational_stability_score)),
    )

    reason_val = str(record["reason"]).strip() if record.get("reason") is not None and str(record["reason"]).strip() else "reconstituted_decision"
    inval_val = str(record["invalidation_condition"]).strip() if record.get("invalidation_condition") is not None and str(record["invalidation_condition"]).strip() else ("Close below stop_loss or trend turns DOWN" if direction == Direction.BUY else None)

    decision = ProductionDecision(
        candidate_id=candidate.candidate_id,
        evidence_id=candidate.evidence.evidence_id,
        experiment_fingerprint=candidate.evidence.experiment_fingerprint,
        symbol=sym,
        timeframe=tf,
        decision_timestamp=str(record.get("authorized_at_utc", m_ts)),
        market_timestamp=m_ts,
        direction=direction,
        reason=reason_val,
        entry_price=float(record["entry_price"]) if record.get("entry_price") is not None else None,
        invalidation_condition=inval_val,
        confidence=float(record.get("stability_score", candidate.operational_stability_score)),
        parameters=candidate.parameters,
    )
    object.__setattr__(decision, "decision_id", dec_id)

    signal = ProductionSignal.from_decision(decision)
    object.__setattr__(signal, "signal_id", sig_id)

    risk = calculate_production_risk_levels(decision, candidate)

    # Reconstruct transition history
    raw_history = record.get("transition_history")
    if raw_history and isinstance(raw_history, list):
        history_tuple = tuple(LifecycleTransitionRecord.from_dict(tr) for tr in raw_history)
    else:
        # Default history up to PERSISTED
        tr_auth = LifecycleTransitionRecord(
            from_state=None,
            to_state=LiveDecisionLifecycleState.AUTHORIZED,
            timestamp_utc=m_ts,
            actor="recovery",
            artifact_fingerprint=auth_receipt.authorization_fingerprint,
        )
        tr_eval = LifecycleTransitionRecord(
            from_state=LiveDecisionLifecycleState.AUTHORIZED,
            to_state=LiveDecisionLifecycleState.EVALUATED,
            timestamp_utc=m_ts,
            actor="recovery",
            artifact_fingerprint=auth_receipt.authorization_fingerprint,
        )
        tr_risk = LifecycleTransitionRecord(
            from_state=LiveDecisionLifecycleState.EVALUATED,
            to_state=LiveDecisionLifecycleState.RISK_VALIDATED,
            timestamp_utc=m_ts,
            actor="recovery",
            artifact_fingerprint=auth_receipt.authorization_fingerprint,
        )
        tr_pres = LifecycleTransitionRecord(
            from_state=LiveDecisionLifecycleState.RISK_VALIDATED,
            to_state=LiveDecisionLifecycleState.PRESENTABLE,
            timestamp_utc=m_ts,
            actor="recovery",
            artifact_fingerprint=auth_receipt.authorization_fingerprint,
        )
        tr_pers = LifecycleTransitionRecord(
            from_state=LiveDecisionLifecycleState.PRESENTABLE,
            to_state=LiveDecisionLifecycleState.PERSISTED,
            timestamp_utc=m_ts,
            actor="recovery",
            artifact_fingerprint=record.get("canonical_live_decision_fingerprint", auth_receipt.authorization_fingerprint),
        )
        history_tuple = (tr_auth, tr_eval, tr_risk, tr_pres, tr_pers)

    cld = CanonicalLiveDecision(
        live_decision_id=dec_id,
        authorization_receipt=auth_receipt,
        decision=decision,
        signal=signal,
        risk_levels=risk,
        current_state=LiveDecisionLifecycleState.PERSISTED,
        transition_history=history_tuple,
    )
    return cld


def recover_pending_publication_deliveries(
    publisher: Any,
    *,
    publication_path: str | Path,
    delivery_path: str | Path,
    decision_store_path: Optional[str | Path] = None,
    research_dir: Path | str = DEFAULT_RESEARCH_DIR,
    max_retries: int = 3,
    actor: str = "delivery_recovery_trigger",
    timestamp_utc: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Recover and re-deliver any pending FAILED_RETRYABLE publication receipts in the delivery ledger.

    For each pending receipt:
    - If attempt_count >= max_retries, transitions receipt to FAILED_PERMANENT.
    - Otherwise, resolves candidate and loads canonical decision record, reconstructing the
      exact publication artifact without creating a new or different decision/signal.
    - Calls publish_canonical_live_decision to attempt re-delivery using the exact same identity and payload.
    """
    d_path = Path(delivery_path)
    pub_path = Path(publication_path)
    dec_path = Path(decision_store_path) if decision_store_path is not None else pub_path.parent / "decision_history.json"

    if not d_path.exists():
        return []

    delivery_history = load_delivery_history(d_path)
    pending_receipts = [r for r in delivery_history if r.delivery_status == DeliveryStatus.FAILED_RETRYABLE]

    if not pending_receipts:
        return []

    from src.evaluation.live_decision_store import load_live_decision_history
    from src.evaluation.research_store import resolve_promoted_candidate

    results: list[dict[str, Any]] = []
    now_iso = timestamp_utc if timestamp_utc is not None else datetime.now(timezone.utc).isoformat()

    dec_history = load_live_decision_history(dec_path) if dec_path.exists() else []

    for receipt in pending_receipts:
        if receipt.attempt_count >= max_retries:
            exhausted_receipt = PublicationDeliveryReceipt(
                publication_id=receipt.publication_id,
                decision_id=receipt.decision_id,
                signal_id=receipt.signal_id,
                canonical_live_decision_fingerprint=receipt.canonical_live_decision_fingerprint,
                runtime_authorization_fingerprint=receipt.runtime_authorization_fingerprint,
                candidate_id=receipt.candidate_id,
                strategy_name=receipt.strategy_name,
                strategy_version=receipt.strategy_version,
                symbol=receipt.symbol,
                timeframe=receipt.timeframe,
                delivery_status=DeliveryStatus.FAILED_PERMANENT,
                attempt_count=receipt.attempt_count,
                first_attempt_at_utc=receipt.first_attempt_at_utc,
                last_attempt_at_utc=now_iso,
                http_status=receipt.http_status,
                remote_event_id=receipt.remote_event_id,
                error=f"Retry attempts exhausted ({receipt.attempt_count}/{max_retries})",
                publisher_version=receipt.publisher_version,
                promoted_artifact_fingerprint=receipt.promoted_artifact_fingerprint,
                governance_decision_fingerprint=receipt.governance_decision_fingerprint,
                campaign_selection_decision_fingerprint=receipt.campaign_selection_decision_fingerprint,
            )
            append_delivery_receipt(exhausted_receipt, d_path)
            results.append({
                "publication_id": receipt.publication_id,
                "status": "FAILED_PERMANENT",
                "reason": "Retry attempts exhausted",
                "attempt_count": receipt.attempt_count,
            })
            continue

        try:
            candidate = resolve_promoted_candidate(candidate_id=receipt.candidate_id, base_dir=Path(research_dir))
        except Exception:
            candidate = None

        if candidate is None:
            failed_receipt = PublicationDeliveryReceipt(
                publication_id=receipt.publication_id,
                decision_id=receipt.decision_id,
                signal_id=receipt.signal_id,
                canonical_live_decision_fingerprint=receipt.canonical_live_decision_fingerprint,
                runtime_authorization_fingerprint=receipt.runtime_authorization_fingerprint,
                candidate_id=receipt.candidate_id,
                strategy_name=receipt.strategy_name,
                strategy_version=receipt.strategy_version,
                symbol=receipt.symbol,
                timeframe=receipt.timeframe,
                delivery_status=DeliveryStatus.FAILED_PERMANENT,
                attempt_count=receipt.attempt_count + 1,
                first_attempt_at_utc=receipt.first_attempt_at_utc,
                last_attempt_at_utc=now_iso,
                http_status=receipt.http_status,
                remote_event_id=receipt.remote_event_id,
                error=f"Unresolvable promoted candidate '{receipt.candidate_id}'",
                publisher_version=receipt.publisher_version,
                promoted_artifact_fingerprint=receipt.promoted_artifact_fingerprint,
                governance_decision_fingerprint=receipt.governance_decision_fingerprint,
                campaign_selection_decision_fingerprint=receipt.campaign_selection_decision_fingerprint,
            )
            append_delivery_receipt(failed_receipt, d_path)
            results.append({
                "publication_id": receipt.publication_id,
                "status": "FAILED_PERMANENT",
                "reason": f"Unresolvable candidate '{receipt.candidate_id}'",
                "attempt_count": receipt.attempt_count + 1,
            })
            continue

        target_record = None
        for rec in dec_history:
            if rec.get("canonical_live_decision_fingerprint") == receipt.canonical_live_decision_fingerprint or rec.get("decision_id") == receipt.decision_id:
                target_record = rec
                break

        if target_record is None:
            failed_receipt = PublicationDeliveryReceipt(
                publication_id=receipt.publication_id,
                decision_id=receipt.decision_id,
                signal_id=receipt.signal_id,
                canonical_live_decision_fingerprint=receipt.canonical_live_decision_fingerprint,
                runtime_authorization_fingerprint=receipt.runtime_authorization_fingerprint,
                candidate_id=receipt.candidate_id,
                strategy_name=receipt.strategy_name,
                strategy_version=receipt.strategy_version,
                symbol=receipt.symbol,
                timeframe=receipt.timeframe,
                delivery_status=DeliveryStatus.FAILED_PERMANENT,
                attempt_count=receipt.attempt_count + 1,
                first_attempt_at_utc=receipt.first_attempt_at_utc,
                last_attempt_at_utc=now_iso,
                http_status=receipt.http_status,
                remote_event_id=receipt.remote_event_id,
                error=f"Missing decision record for fingerprint '{receipt.canonical_live_decision_fingerprint}'",
                publisher_version=receipt.publisher_version,
                promoted_artifact_fingerprint=receipt.promoted_artifact_fingerprint,
                governance_decision_fingerprint=receipt.governance_decision_fingerprint,
                campaign_selection_decision_fingerprint=receipt.campaign_selection_decision_fingerprint,
            )
            append_delivery_receipt(failed_receipt, d_path)
            results.append({
                "publication_id": receipt.publication_id,
                "status": "FAILED_PERMANENT",
                "reason": "Missing decision record",
                "attempt_count": receipt.attempt_count + 1,
            })
            continue

        try:
            cld = _reconstitute_canonical_live_decision_from_record(target_record, candidate)
            pub_cld, pub_obj, pub_res = publish_canonical_live_decision(
                cld,
                publisher=publisher,
                candidate=candidate,
                path=pub_path,
                delivery_path=d_path,
                actor=actor,
                timestamp_utc=now_iso,
            )
            results.append({
                "publication_id": receipt.publication_id,
                "status": pub_res.get("status"),
                "delivery_status": pub_res.get("delivery_status"),
                "attempt_count": receipt.attempt_count + 1,
            })
        except Exception as exc:
            results.append({
                "publication_id": receipt.publication_id,
                "status": "FAILED_RECOVERY_EXCEPTION",
                "error": str(exc),
                "attempt_count": receipt.attempt_count + 1,
            })

    return results
