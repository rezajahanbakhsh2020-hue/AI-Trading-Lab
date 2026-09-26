"""Research Selection-Aware Governance Module for Project 1.

Provides explicit, deterministic, side-effect-free governance accounting for
multiple-testing and selection-bias across research search trials and ledgers.

Consumes ResearchEvidence artifacts and ResearchTrialRecord ledgers to record
selection context, count evaluated hypotheses, and flag multiple-testing status
without executing research or connecting directly to production decisioning,
risk generation, publication, or live execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from src.evaluation.research_constitution import (
    PromotionStatus,
    RejectionReason,
    ResearchEvidence,
)

if TYPE_CHECKING:
    from src.evaluation.discovery_engine import ResearchTrialRecord


class SelectionGovernanceStatus(str, Enum):
    """Status classification for multiple-testing / selection-bias governance."""

    SELECTION_NOT_APPLICABLE = "SELECTION_NOT_APPLICABLE"
    SELECTION_CONTEXT_RECORDED = "SELECTION_CONTEXT_RECORDED"
    SELECTION_ADJUSTMENT_APPLIED = "SELECTION_ADJUSTMENT_APPLIED"
    SELECTION_REVIEW_REQUIRED = "SELECTION_REVIEW_REQUIRED"
    SELECTION_INSUFFICIENT_DATA = "SELECTION_INSUFFICIENT_DATA"


class SelectionGovernanceReason(str, Enum):
    """Machine-readable reasons for selection governance status classification."""

    SINGLE_HYPOTHESIS_NO_SELECTION = "SINGLE_HYPOTHESIS_NO_SELECTION"
    MULTI_TRIAL_SELECTION_DETECTED = "MULTI_TRIAL_SELECTION_DETECTED"
    CORRECTION_UNAVAILABLE_MISSING_DISTRIBUTIONAL_INPUTS = (
        "CORRECTION_UNAVAILABLE_MISSING_DISTRIBUTIONAL_INPUTS"
    )
    INSUFFICIENT_TRIAL_LEDGER = "INSUFFICIENT_TRIAL_LEDGER"
    SELECTION_ADJUSTMENT_NOT_REQUESTED = "SELECTION_ADJUSTMENT_NOT_REQUESTED"


def _canonical_json_value(obj: Any) -> Any:
    """Helper to convert custom types/enums for canonical JSON serialization."""
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def compute_selection_fingerprint(payload: Mapping[str, Any]) -> str:
    """Compute deterministic SHA-256 fingerprint from a canonical payload mapping.

    Uses sort_keys=True, compact separators, and UTF-8 encoding.
    NEVER uses Python's built-in hash().
    """
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_canonical_json_value,
        ensure_ascii=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResearchSelectionPolicy:
    """Policy governing multiple-testing and selection-bias governance accounting."""

    policy_version: str = "selection_v1.0"
    require_trial_ledger: bool = True
    max_uncorrected_trials: int = 1

    def __post_init__(self) -> None:
        if not self.policy_version or not self.policy_version.strip():
            raise ValueError("policy_version must be a non-empty string.")
        if self.max_uncorrected_trials < 1:
            raise ValueError("max_uncorrected_trials must be at least 1.")


@dataclass(frozen=True)
class ResearchSelectionAssessment:
    """Canonical, immutable result of a research selection-bias governance check."""

    status: SelectionGovernanceStatus
    reason: SelectionGovernanceReason
    search_fingerprint: str
    trial_count: int
    completed_trial_count: int
    failed_trial_count: int
    qualified_trial_count: int
    rejected_trial_count: int
    candidate_id: str
    candidate_fingerprint: str
    experiment_fingerprint: str
    evidence_fingerprint: str | None
    is_selected_winner: bool
    selection_notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    selection_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.status, SelectionGovernanceStatus):
            raise TypeError("status must be a SelectionGovernanceStatus enum member.")
        if not isinstance(self.reason, SelectionGovernanceReason):
            raise TypeError("reason must be a SelectionGovernanceReason enum member.")
        if self.trial_count < 0:
            raise ValueError("trial_count must be non-negative.")
        if self.completed_trial_count < 0:
            raise ValueError("completed_trial_count must be non-negative.")
        if self.failed_trial_count < 0:
            raise ValueError("failed_trial_count must be non-negative.")
        if self.qualified_trial_count < 0:
            raise ValueError("qualified_trial_count must be non-negative.")
        if self.rejected_trial_count < 0:
            raise ValueError("rejected_trial_count must be non-negative.")
        if not isinstance(self.is_selected_winner, bool):
            raise TypeError("is_selected_winner must be a boolean.")

        payload = {
            "status": self.status.value,
            "reason": self.reason.value,
            "search_fingerprint": self.search_fingerprint,
            "trial_count": self.trial_count,
            "completed_trial_count": self.completed_trial_count,
            "failed_trial_count": self.failed_trial_count,
            "qualified_trial_count": self.qualified_trial_count,
            "rejected_trial_count": self.rejected_trial_count,
            "candidate_id": self.candidate_id,
            "candidate_fingerprint": self.candidate_fingerprint,
            "experiment_fingerprint": self.experiment_fingerprint,
            "evidence_fingerprint": self.evidence_fingerprint,
            "is_selected_winner": self.is_selected_winner,
            "selection_notes": self.selection_notes.strip(),
            "metadata": self.metadata,
        }
        fp = compute_selection_fingerprint(payload)
        object.__setattr__(self, "selection_fingerprint", fp)


def assess_research_selection(
    evidence: ResearchEvidence,
    trial_records: Sequence[ResearchTrialRecord] | None = None,
    search_fingerprint: str = "",
    policy: ResearchSelectionPolicy | None = None,
) -> ResearchSelectionAssessment:
    """Perform deterministic selection-aware governance assessment on ResearchEvidence.

    Consumes existing canonical evidence and ResearchTrialRecord ledger metadata.
    Preserves exact trial counts and search fingerprint without executing research
    or manufacturing unbacked statistical corrections.
    """
    if policy is None:
        policy = ResearchSelectionPolicy()

    if not isinstance(evidence, ResearchEvidence):
        raise TypeError("evidence must be a ResearchEvidence instance.")

    records = tuple(trial_records) if trial_records is not None else ()

    for rec in records:
        if type(rec).__name__ != "ResearchTrialRecord":
            raise TypeError("All items in trial_records must be ResearchTrialRecord instances.")

    # Calculate ledger trial counts
    if records:
        total_trials = len(records)
        completed_trials = sum(1 for t in records if t.status in ("COMPLETED", "QUALIFIED", "REJECTED"))
        failed_trials = sum(1 for t in records if t.status == "FAILED")
        qualified_trials = sum(1 for t in records if t.status == "QUALIFIED")
        rejected_trials = sum(1 for t in records if t.status == "REJECTED")
        effective_search_fp = search_fingerprint or records[0].search_id
    else:
        total_trials = 1
        completed_trials = 1
        failed_trials = 0
        qualified_trials = 1 if evidence.promotion_status in (PromotionStatus.PROMOTABLE, PromotionStatus.VALIDATED) else 0
        rejected_trials = 1 if evidence.promotion_status == PromotionStatus.REJECTED else 0
        effective_search_fp = search_fingerprint or "single_experiment_search"

    # Identify matching trial record if present
    matching_record = None
    for rec in records:
        if (
            rec.experiment_fingerprint == evidence.experiment_fingerprint
            or (rec.evidence_fingerprint and rec.evidence_fingerprint == evidence.evidence_id)
        ):
            matching_record = rec
            break

    candidate_id = matching_record.candidate_id if matching_record else getattr(evidence.spec, "strategy_name", "unknown_candidate")
    candidate_fp = matching_record.candidate_fingerprint if matching_record else evidence.spec.fingerprint

    # Determine if winner inside multi-trial search
    is_winner = False
    if records:
        # Candidate is qualified if evidence is qualified or record status is QUALIFIED
        is_winner = (
            matching_record is not None
            and matching_record.status == "QUALIFIED"
            and evidence.promotion_status in (PromotionStatus.PROMOTABLE, PromotionStatus.VALIDATED)
        )
    else:
        is_winner = evidence.promotion_status in (PromotionStatus.PROMOTABLE, PromotionStatus.VALIDATED)

    # Classify selection governance status & reason
    if total_trials <= policy.max_uncorrected_trials and not records:
        status = SelectionGovernanceStatus.SELECTION_NOT_APPLICABLE
        reason = SelectionGovernanceReason.SINGLE_HYPOTHESIS_NO_SELECTION
        notes = "Single hypothesis evaluated; multiple-testing selection bias governance not applicable."
    elif total_trials <= 1:
        status = SelectionGovernanceStatus.SELECTION_NOT_APPLICABLE
        reason = SelectionGovernanceReason.SINGLE_HYPOTHESIS_NO_SELECTION
        notes = "Single trial search ledger recorded; selection bias adjustment not applicable."
    else:
        # Multi-trial search detected
        status = SelectionGovernanceStatus.SELECTION_CONTEXT_RECORDED
        reason = SelectionGovernanceReason.CORRECTION_UNAVAILABLE_MISSING_DISTRIBUTIONAL_INPUTS
        notes = (
            f"Multi-trial search recorded ({total_trials} trials, {qualified_trials} qualified). "
            "Selection context preserved; quantitative multiple-testing correction is unavailable "
            "because distribution of returns across non-selected trials is not represented in trial records."
        )

    metadata = {
        "policy_version": policy.policy_version,
        "max_uncorrected_trials": policy.max_uncorrected_trials,
        "matching_trial_found": matching_record is not None,
    }

    return ResearchSelectionAssessment(
        status=status,
        reason=reason,
        search_fingerprint=effective_search_fp,
        trial_count=total_trials,
        completed_trial_count=completed_trials,
        failed_trial_count=failed_trials,
        qualified_trial_count=qualified_trials,
        rejected_trial_count=rejected_trials,
        candidate_id=candidate_id,
        candidate_fingerprint=candidate_fp,
        experiment_fingerprint=evidence.experiment_fingerprint,
        evidence_fingerprint=evidence.evidence_id,
        is_selected_winner=is_winner,
        selection_notes=notes,
        metadata=metadata,
    )
