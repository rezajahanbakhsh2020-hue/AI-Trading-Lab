"""Research Evidence Qualification Service for Project 1.

Provides explicit, deterministic, side-effect-free governance qualification
over ResearchEvidence artifacts prior to promotion binding or production entry.

Does NOT:
- generate trading signals
- select strategies or parameters
- calculate risk levels or positions
- bind production candidates
- mutate production state or stores
- publish to Project 2
- silently repair or override invalid evidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Optional, Sequence

from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    EvidencePartition,
    EvidencePartitionRole,
    ExecutionAssumptions,
    PromotionStatus,
    RejectionReason,
    ResearchEvidence,
    ResearchExperimentSpec,
    compute_experiment_fingerprint,
)


@dataclass(frozen=True)
class ResearchQualificationPolicy:
    """Configurable governance criteria for research evidence qualification."""

    policy_version: str = "qualification_v1.0"
    allowed_statuses: tuple[PromotionStatus, ...] = (
        PromotionStatus.PROMOTABLE,
        PromotionStatus.VALIDATED,
    )
    require_oos: bool = True
    require_walk_forward: bool = True
    require_robustness: bool = True
    require_statistical_evidence: bool = True
    require_friction_model: bool = True
    require_code_provenance: bool = True
    require_dataset_scope: bool = True
    require_deterministic_fingerprint: bool = True
    min_statistical_observations: int = 30
    max_evidence_age_days: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.policy_version or not self.policy_version.strip():
            raise ValueError("policy_version must be a non-empty string.")
        if not self.allowed_statuses:
            raise ValueError("allowed_statuses must not be empty.")
        if self.min_statistical_observations <= 0:
            raise ValueError("min_statistical_observations must be positive.")
        if self.max_evidence_age_days is not None and self.max_evidence_age_days <= 0:
            raise ValueError("max_evidence_age_days must be positive if specified.")


@dataclass(frozen=True)
class ResearchQualificationResult:
    """Immutable, auditable result of a research evidence qualification check."""

    qualified: bool
    status: PromotionStatus
    rejection_reasons: tuple[RejectionReason, ...]
    evidence_fingerprint: str
    qualification_notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.qualified, bool):
            raise TypeError("qualified must be a boolean.")
        if not isinstance(self.status, PromotionStatus):
            raise TypeError("status must be a PromotionStatus enum member.")
        for r in self.rejection_reasons:
            if not isinstance(r, RejectionReason):
                raise TypeError(f"Rejection reason '{r}' must be a RejectionReason enum member.")
        if not self.qualified and self.status not in (
            PromotionStatus.REJECTED,
            PromotionStatus.PROPOSED,
            PromotionStatus.EXPERIMENTAL,
        ):
            raise ValueError("Unqualified result must have a non-promoted status.")
        if self.qualified and self.rejection_reasons:
            raise ValueError("Qualified result cannot carry rejection reasons.")


def qualify_research_evidence(
    evidence: Any,
    policy: Optional[ResearchQualificationPolicy] = None,
    now: Optional[datetime] = None,
) -> ResearchQualificationResult:
    """Perform deterministic, side-effect-free qualification check on ResearchEvidence.

    Evaluates evidence lineage, spec structural validity, DatasetScope,
    ExecutionAssumptions, CodeProvenance, SHA-256 fingerprint integrity,
    partition completeness (IS, Validation, OOS, Walk-Forward), robustness,
    statistical observation sufficiency, and evidence freshness.

    Returns a ResearchQualificationResult indicating whether the evidence
    satisfies all governance requirements for candidate promotion.
    """
    if policy is None:
        policy = ResearchQualificationPolicy()

    fingerprint = getattr(evidence, "experiment_fingerprint", "") or ""
    rejection_reasons: list[RejectionReason] = []
    notes: list[str] = []

    # 1. Type validation
    if not isinstance(evidence, ResearchEvidence):
        return ResearchQualificationResult(
            qualified=False,
            status=PromotionStatus.REJECTED,
            rejection_reasons=(RejectionReason.EVIDENCE_INCOMPLETENESS,),
            evidence_fingerprint=fingerprint,
            qualification_notes="Input object is not a valid ResearchEvidence instance.",
        )

    fingerprint = evidence.experiment_fingerprint

    # 2. Spec presence and structural validity
    spec = evidence.spec
    if not isinstance(spec, ResearchExperimentSpec):
        rejection_reasons.append(RejectionReason.SPECIFICATION_INVALID)
        notes.append("Missing or invalid ResearchExperimentSpec.")
    else:
        # Check spec attributes
        for attr_name in ("hypothesis", "methodology_version", "strategy_name", "strategy_version", "benchmark_reference"):
            val = getattr(spec, attr_name, None)
            if not isinstance(val, str) or not val.strip():
                if RejectionReason.SPECIFICATION_INVALID not in rejection_reasons:
                    rejection_reasons.append(RejectionReason.SPECIFICATION_INVALID)
                notes.append(f"Spec field '{attr_name}' is missing or empty.")

    # 3. Dataset scope validation
    if policy.require_dataset_scope:
        if not isinstance(getattr(spec, "dataset_scope", None), DatasetScope):
            rejection_reasons.append(RejectionReason.INVALID_DATASET_SCOPE)
            notes.append("DatasetScope is missing or invalid.")
        else:
            ds = spec.dataset_scope
            if not ds.dataset_id or not ds.dataset_id.strip() or not ds.symbol or not ds.symbol.strip() or not ds.timeframe or not ds.timeframe.strip():
                rejection_reasons.append(RejectionReason.INVALID_DATASET_SCOPE)
                notes.append("DatasetScope fields (dataset_id, symbol, timeframe) must be non-empty.")
            if not ds.start_date or not ds.start_date.strip() or not ds.end_date or not ds.end_date.strip():
                rejection_reasons.append(RejectionReason.INVALID_DATASET_SCOPE)
                notes.append("DatasetScope date boundaries must be non-empty.")
            elif ds.start_date > ds.end_date:
                rejection_reasons.append(RejectionReason.INVALID_DATASET_SCOPE)
                notes.append(f"DatasetScope start_date '{ds.start_date}' is later than end_date '{ds.end_date}'.")

    # 4. Execution assumptions (friction & latency) validation
    if policy.require_friction_model:
        ea = getattr(spec, "execution_assumptions", None)
        if not isinstance(ea, ExecutionAssumptions):
            rejection_reasons.append(RejectionReason.EXECUTION_ASSUMPTION_VIOLATION)
            notes.append("ExecutionAssumptions missing or invalid.")
        else:
            for field_name in ("transaction_cost", "slippage", "latency_ms"):
                val = getattr(ea, field_name, None)
                if not isinstance(val, (int, float)) or math.isnan(val) or val < 0.0:
                    if RejectionReason.EXECUTION_ASSUMPTION_VIOLATION not in rejection_reasons:
                        rejection_reasons.append(RejectionReason.EXECUTION_ASSUMPTION_VIOLATION)
                    notes.append(f"Execution assumption '{field_name}' is invalid or negative: {val}")

    # 5. Code provenance validation
    if policy.require_code_provenance:
        cp = getattr(spec, "code_provenance", None)
        if not isinstance(cp, CodeProvenance):
            rejection_reasons.append(RejectionReason.EVIDENCE_INCOMPLETENESS)
            notes.append("CodeProvenance missing or invalid.")
        elif not cp.commit_sha or not cp.commit_sha.strip():
            rejection_reasons.append(RejectionReason.EVIDENCE_INCOMPLETENESS)
            notes.append("CodeProvenance commit_sha is missing or empty.")

    # 6. Fingerprint integrity validation
    if policy.require_deterministic_fingerprint and isinstance(spec, ResearchExperimentSpec):
        if not evidence.experiment_fingerprint or not evidence.experiment_fingerprint.strip():
            rejection_reasons.append(RejectionReason.FAILED_REPRODUCIBILITY)
            notes.append("Evidence experiment_fingerprint is missing or empty.")
        elif evidence.experiment_fingerprint != spec.fingerprint:
            rejection_reasons.append(RejectionReason.FAILED_REPRODUCIBILITY)
            notes.append(
                f"Evidence experiment_fingerprint '{evidence.experiment_fingerprint}' does not "
                f"match spec.fingerprint '{spec.fingerprint}'."
            )
        else:
            # Re-compute fingerprint from canonical payload
            try:
                recomputed_fp = compute_experiment_fingerprint(
                    hypothesis=spec.hypothesis,
                    methodology_version=spec.methodology_version,
                    strategy_name=spec.strategy_name,
                    strategy_version=spec.strategy_version,
                    dataset_scope=spec.dataset_scope,
                    execution_assumptions=spec.execution_assumptions,
                    code_provenance=spec.code_provenance,
                    benchmark_reference=spec.benchmark_reference,
                    parameters=spec.parameters,
                    random_seed=spec.random_seed,
                )
                if recomputed_fp != spec.fingerprint or recomputed_fp != evidence.experiment_fingerprint:
                    rejection_reasons.append(RejectionReason.FAILED_REPRODUCIBILITY)
                    notes.append("SHA-256 fingerprint re-computation mismatch (tampered evidence detected).")
            except Exception as exc:
                rejection_reasons.append(RejectionReason.FAILED_REPRODUCIBILITY)
                notes.append(f"Fingerprint re-computation failed: {exc}")

    # 7. Partition completeness validation
    partitions = getattr(evidence, "partitions", ()) or ()
    roles_present = {p.role for p in partitions if isinstance(p, EvidencePartition)}

    if EvidencePartitionRole.IN_SAMPLE not in roles_present:
        rejection_reasons.append(RejectionReason.EVIDENCE_INCOMPLETENESS)
        notes.append("Missing In-Sample partition evidence.")

    if policy.require_oos and EvidencePartitionRole.OUT_OF_SAMPLE not in roles_present:
        rejection_reasons.append(RejectionReason.FAILED_OOS)
        notes.append("Missing required Out-of-Sample partition evidence.")

    if policy.require_walk_forward and EvidencePartitionRole.WALK_FORWARD not in roles_present:
        rejection_reasons.append(RejectionReason.FAILED_WALK_FORWARD)
        notes.append("Missing required Walk-Forward partition evidence.")

    # Validate partition observation counts for statistical sufficiency
    if policy.require_statistical_evidence:
        total_obs = sum(p.observations for p in partitions if isinstance(p, EvidencePartition))
        if total_obs < policy.min_statistical_observations:
            rejection_reasons.append(RejectionReason.INSUFFICIENT_STATISTICAL_SAMPLE)
            notes.append(
                f"Total partition observations ({total_obs}) below policy threshold "
                f"({policy.min_statistical_observations})."
            )

    # 8. Robustness verdict validation
    if policy.require_robustness:
        rv = getattr(evidence, "robustness_verdict", None)
        if not rv or not isinstance(rv, dict):
            rejection_reasons.append(RejectionReason.MISSING_ROBUSTNESS_EVIDENCE)
            notes.append("Robustness verdict is missing or empty.")
        else:
            is_robust = rv.get("is_robust")
            passed = rv.get("passed")
            if is_robust is False or (is_robust is None and passed is False):
                rejection_reasons.append(RejectionReason.FAILED_ROBUSTNESS)
                notes.append(f"Robustness verdict failed (is_robust={is_robust}, passed={passed}).")

    # 9. Existing status & rejection reasons check
    if evidence.promotion_status not in policy.allowed_statuses:
        rejection_reasons.append(RejectionReason.CRITIQUE_REJECTED)
        notes.append(
            f"Evidence status '{evidence.promotion_status.value}' is not in allowed "
            f"promotion statuses {[s.value for s in policy.allowed_statuses]}."
        )

    if evidence.rejection_reasons:
        for existing_reason in evidence.rejection_reasons:
            if existing_reason not in rejection_reasons:
                rejection_reasons.append(existing_reason)
        notes.append(f"Evidence contains prior rejection reasons: {[r.value for r in evidence.rejection_reasons]}")

    # 10. Freshness validation
    if policy.max_evidence_age_days is not None:
        if not evidence.created_at_utc or not str(evidence.created_at_utc).strip():
            rejection_reasons.append(RejectionReason.EVIDENCE_INCOMPLETENESS)
            notes.append("Evidence missing created_at_utc required for freshness check.")
        else:
            try:
                created = datetime.fromisoformat(str(evidence.created_at_utc).replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                now_dt = now if now is not None else datetime.now(timezone.utc)
                if now_dt.tzinfo is None:
                    now_dt = now_dt.replace(tzinfo=timezone.utc)
                age_days = (now_dt - created).total_seconds() / 86400.0
                if age_days < 0:
                    rejection_reasons.append(RejectionReason.STALE_INVALID_DATA)
                    notes.append("Evidence created_at_utc is in the future.")
                elif age_days > policy.max_evidence_age_days:
                    rejection_reasons.append(RejectionReason.STALE_INVALID_DATA)
                    notes.append(f"Evidence is stale ({age_days:.1f} days old, max: {policy.max_evidence_age_days}).")
            except ValueError:
                rejection_reasons.append(RejectionReason.EVIDENCE_INCOMPLETENESS)
                notes.append(f"Invalid created_at_utc timestamp: '{evidence.created_at_utc}'.")

    # Deduplicate rejection reasons maintaining order
    dedup_rejections = tuple(dict.fromkeys(rejection_reasons))

    if dedup_rejections:
        return ResearchQualificationResult(
            qualified=False,
            status=PromotionStatus.REJECTED,
            rejection_reasons=dedup_rejections,
            evidence_fingerprint=fingerprint,
            qualification_notes="; ".join(notes),
        )

    return ResearchQualificationResult(
        qualified=True,
        status=evidence.promotion_status,
        rejection_reasons=(),
        evidence_fingerprint=fingerprint,
        qualification_notes="Evidence successfully qualified for promotion.",
    )
