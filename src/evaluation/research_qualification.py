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

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.evaluation.evidence_integrity import ResearchEvidenceIntegrityGate
from src.evaluation.research_constitution import (
    EvidencePartition,
    PromotionStatus,
    RejectionReason,
    ResearchEvidence,
)
from src.evaluation.research_robustness import (
    ResearchRobustnessAssessment,
    RobustnessStatus,
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
    max_evidence_age_days: float | None = None

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
    """Immutable, auditable result of a research evidence qualification check and governance decision."""

    qualified: bool
    status: PromotionStatus
    rejection_reasons: tuple[RejectionReason, ...]
    evidence_fingerprint: str
    experiment_fingerprint: str = ""
    integrity_valid: bool = True
    integrity_rejection_reasons: tuple[RejectionReason, ...] = ()
    robustness_assessment_fingerprint: str | None = None
    policy_version: str = "qualification_v1.0"
    qualification_notes: str = ""
    decision_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.qualified, bool):
            raise TypeError("qualified must be a boolean.")
        if not isinstance(self.status, PromotionStatus):
            raise TypeError("status must be a PromotionStatus enum member.")
        for r in self.rejection_reasons:
            if not isinstance(r, RejectionReason):
                raise TypeError(f"Rejection reason '{r}' must be a RejectionReason enum member.")
        for r in self.integrity_rejection_reasons:
            if not isinstance(r, RejectionReason):
                raise TypeError(f"Integrity rejection reason '{r}' must be a RejectionReason enum member.")
        if not self.qualified and self.status not in (
            PromotionStatus.REJECTED,
            PromotionStatus.PROPOSED,
            PromotionStatus.EXPERIMENTAL,
        ):
            raise ValueError("Unqualified result must have a non-promoted status.")
        if self.qualified and self.rejection_reasons:
            raise ValueError("Qualified result cannot carry rejection reasons.")

        payload = {
            "qualified": self.qualified,
            "status": self.status.value,
            "rejection_reasons": sorted([r.value for r in self.rejection_reasons]),
            "evidence_fingerprint": self.evidence_fingerprint.strip(),
            "experiment_fingerprint": self.experiment_fingerprint.strip(),
            "integrity_valid": self.integrity_valid,
            "integrity_rejection_reasons": sorted([r.value for r in self.integrity_rejection_reasons]),
            "robustness_assessment_fingerprint": (
                self.robustness_assessment_fingerprint.strip()
                if self.robustness_assessment_fingerprint
                else None
            ),
            "policy_version": self.policy_version.strip(),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        fp = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        object.__setattr__(self, "decision_fingerprint", fp)


# Aliases for canonical governance decision boundary
ResearchGovernanceDecision = ResearchQualificationResult
ResearchQualificationDecision = ResearchQualificationResult


def qualify_research_evidence(
    evidence: Any,
    policy: ResearchQualificationPolicy | None = None,
    robustness_assessment: Any | None = None,
    now: datetime | None = None,
) -> ResearchQualificationResult:
    """Perform deterministic, side-effect-free qualification check on ResearchEvidence.

    Consumes authoritative ResearchEvidenceIntegrityGate result for structural/lineage integrity
    and authoritative ResearchRobustnessAssessment for robustness verification.

    Evaluates allowed PromotionStatus, canonical robustness assessment,
    statistical observation sufficiency, prior rejection reasons, and evidence freshness.

    Returns a ResearchQualificationResult (ResearchGovernanceDecision) indicating whether the
    evidence satisfies all governance requirements for candidate promotion.
    """
    if policy is None:
        policy = ResearchQualificationPolicy()

    fingerprint = getattr(evidence, "experiment_fingerprint", "") or ""
    rejection_reasons: list[RejectionReason] = []
    notes: list[str] = []

    # 1. Authoritative Pre-Qualification Integrity Gate
    gate_res = ResearchEvidenceIntegrityGate.validate(
        evidence,
        require_exact_timestamps=False,
        require_walk_forward=policy.require_walk_forward,
        require_oos=policy.require_oos,
    )
    integrity_valid = gate_res.valid
    integrity_rejections = gate_res.rejection_reasons

    if not gate_res.valid:
        for r in gate_res.rejection_reasons:
            if r not in rejection_reasons:
                rejection_reasons.append(r)
        notes.extend(gate_res.notes)

    if not isinstance(evidence, ResearchEvidence):
        return ResearchQualificationResult(
            qualified=False,
            status=PromotionStatus.REJECTED,
            rejection_reasons=tuple(dict.fromkeys(rejection_reasons)),
            evidence_fingerprint=fingerprint,
            experiment_fingerprint=fingerprint,
            integrity_valid=False,
            integrity_rejection_reasons=integrity_rejections,
            robustness_assessment_fingerprint=None,
            policy_version=policy.policy_version,
            qualification_notes="Input object is not a valid ResearchEvidence instance.",
        )

    exp_fingerprint = evidence.experiment_fingerprint
    ev_id = evidence.evidence_id

    # 2. Authoritative Robustness Assessment Validation
    rob_fp: str | None = None
    if robustness_assessment is not None:
        if not isinstance(robustness_assessment, ResearchRobustnessAssessment):
            rejection_reasons.append(RejectionReason.FAILED_ROBUSTNESS)
            notes.append("Provided robustness_assessment is not a ResearchRobustnessAssessment instance.")
        else:
            rob_fp = robustness_assessment.robustness_fingerprint
            # Validate exact lineage binding
            if robustness_assessment.experiment_fingerprint != exp_fingerprint:
                rejection_reasons.append(RejectionReason.FAILED_REPRODUCIBILITY)
                notes.append(
                    f"Robustness assessment experiment_fingerprint '{robustness_assessment.experiment_fingerprint}' "
                    f"does not match evidence experiment_fingerprint '{exp_fingerprint}'."
                )
            if robustness_assessment.evidence_fingerprint != ev_id:
                rejection_reasons.append(RejectionReason.FAILED_REPRODUCIBILITY)
                notes.append(
                    f"Robustness assessment evidence_fingerprint '{robustness_assessment.evidence_fingerprint}' "
                    f"does not match evidence_id '{ev_id}'."
                )

            # Validate robustness status and requirements
            if policy.require_robustness:
                if not robustness_assessment.is_robust:
                    rejection_reasons.append(RejectionReason.FAILED_ROBUSTNESS)
                    notes.append(f"Canonical robustness assessment failed (is_robust=False, status={robustness_assessment.status.value}).")

                if robustness_assessment.status == RobustnessStatus.NOT_EVALUATED:
                    rejection_reasons.append(RejectionReason.MISSING_ROBUSTNESS_EVIDENCE)
                    notes.append("Canonical robustness assessment status is NOT_EVALUATED.")
                elif robustness_assessment.status == RobustnessStatus.INSUFFICIENT_DATA:
                    rejection_reasons.append(RejectionReason.INSUFFICIENT_STATISTICAL_SAMPLE)
                    notes.append("Canonical robustness assessment status is INSUFFICIENT_DATA.")

                if robustness_assessment.dimensions_unavailable:
                    rejection_reasons.append(RejectionReason.FAILED_ROBUSTNESS)
                    notes.append(
                        f"Canonical robustness assessment has unevaluated dimensions: "
                        f"{', '.join(robustness_assessment.dimensions_unavailable)}."
                    )
    else:
        if policy.require_robustness:
            rejection_reasons.append(RejectionReason.MISSING_ROBUSTNESS_EVIDENCE)
            notes.append("Policy requires robustness assessment, but no authoritative ResearchRobustnessAssessment was supplied.")
    # 3. Partition observation count validation for statistical sufficiency
    if policy.require_statistical_evidence:
        partitions = getattr(evidence, "partitions", ()) or ()
        total_obs = sum(p.observations for p in partitions if isinstance(p, EvidencePartition))
        if total_obs < policy.min_statistical_observations:
            rejection_reasons.append(RejectionReason.INSUFFICIENT_STATISTICAL_SAMPLE)
            notes.append(
                f"Total partition observations ({total_obs}) below policy threshold "
                f"({policy.min_statistical_observations})."
            )

    # 4. Status check
    if evidence.promotion_status not in policy.allowed_statuses:
        rejection_reasons.append(RejectionReason.CRITIQUE_REJECTED)
        notes.append(
            f"Evidence status '{evidence.promotion_status.value}' is not in allowed "
            f"promotion statuses {[s.value for s in policy.allowed_statuses]}."
        )

    # 5. Prior rejection reasons check
    if evidence.rejection_reasons:
        for existing_reason in evidence.rejection_reasons:
            if existing_reason not in rejection_reasons:
                rejection_reasons.append(existing_reason)
        notes.append(f"Evidence contains prior rejection reasons: {[r.value for r in evidence.rejection_reasons]}")

    # 6. Freshness validation
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
            evidence_fingerprint=exp_fingerprint,
            experiment_fingerprint=exp_fingerprint,
            integrity_valid=integrity_valid,
            integrity_rejection_reasons=integrity_rejections,
            robustness_assessment_fingerprint=rob_fp,
            policy_version=policy.policy_version,
            qualification_notes="; ".join(notes),
        )

    return ResearchQualificationResult(
        qualified=True,
        status=evidence.promotion_status,
        rejection_reasons=(),
        evidence_fingerprint=exp_fingerprint,
        experiment_fingerprint=exp_fingerprint,
        integrity_valid=integrity_valid,
        integrity_rejection_reasons=integrity_rejections,
        robustness_assessment_fingerprint=rob_fp,
        policy_version=policy.policy_version,
        qualification_notes="Evidence successfully qualified for promotion.",
    )
