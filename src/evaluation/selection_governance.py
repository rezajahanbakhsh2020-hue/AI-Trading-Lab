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
import math
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

    policy_version: str = "selection_v2.0"
    require_trial_ledger: bool = True
    max_uncorrected_trials: int = 1
    alpha: float = 0.05
    correction_method: str = "holm_bonferroni"

    def __post_init__(self) -> None:
        if not self.policy_version or not self.policy_version.strip():
            raise ValueError("policy_version must be a non-empty string.")
        if self.max_uncorrected_trials < 1:
            raise ValueError("max_uncorrected_trials must be at least 1.")
        if not isinstance(self.alpha, (int, float)) or not math.isfinite(self.alpha) or not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be a finite float between 0.0 and 1.0 exclusive.")
        if self.correction_method not in ("holm_bonferroni", "bonferroni"):
            raise ValueError(f"Unsupported correction_method '{self.correction_method}'.")


def run_multiple_testing_correction(
    raw_p_values: Mapping[str, float],
    alpha: float = 0.05,
    method: str = "holm_bonferroni",
) -> dict[str, Any]:
    """Execute deterministic multiple-testing statistical correction.

    Supports Holm-Bonferroni step-down procedure and Bonferroni single-step correction.
    Fails closed if any raw p-value is missing, non-finite, or outside [0.0, 1.0].
    """
    if not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be a finite float between 0.0 and 1.0 exclusive.")
    if method not in ("holm_bonferroni", "bonferroni"):
        raise ValueError(f"Unsupported correction_method '{method}'.")
    if not raw_p_values:
        raise ValueError("raw_p_values mapping must not be empty.")

    m = len(raw_p_values)
    clean_raw: dict[str, float] = {}

    for k, p in raw_p_values.items():
        if not isinstance(p, (int, float)) or not math.isfinite(p):
            raise ValueError(f"Non-finite raw p-value for key '{k}': {p}")
        p_float = float(p)
        if not (0.0 <= p_float <= 1.0):
            raise ValueError(f"Raw p-value for key '{k}' out of bounds [0, 1]: {p_float}")
        clean_raw[k] = p_float

    adjusted_p_values: dict[str, float] = {}

    if method == "holm_bonferroni":
        # Sort keys by raw p-value ascending, breaking ties deterministically by key name
        sorted_keys = sorted(clean_raw.keys(), key=lambda k: (clean_raw[k], str(k)))
        running_max = 0.0

        for i, k in enumerate(sorted_keys):
            raw_p = clean_raw[k]
            # Holm multiplier: (m - i) where i is 0-indexed
            multiplier = m - i
            step_adj = raw_p * multiplier
            adj_p = min(1.0, max(running_max, step_adj))
            running_max = adj_p
            adjusted_p_values[k] = adj_p
    elif method == "bonferroni":
        for k, raw_p in clean_raw.items():
            adjusted_p_values[k] = min(1.0, raw_p * m)

    rejection_decisions = {
        k: (adjusted_p_values[k] <= alpha) for k in clean_raw
    }

    return {
        "method": method,
        "method_version": f"{method}_v1.0",
        "alpha": float(alpha),
        "eligible_trial_count": m,
        "raw_p_values": {k: round(v, 8) for k, v in clean_raw.items()},
        "adjusted_p_values": {k: round(v, 8) for k, v in adjusted_p_values.items()},
        "rejection_decisions": rejection_decisions,
    }


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
    evidence_collection: Sequence[ResearchEvidence] | None = None,
) -> ResearchSelectionAssessment:
    """Perform deterministic selection-aware governance assessment on ResearchEvidence.

    Consumes existing canonical evidence, ResearchTrialRecord ledger metadata, and optional
    evidence collections. Applies quantitative multiple-testing correction (e.g. Holm-Bonferroni)
    when valid statistical inputs are present, failing closed when evidence is insufficient.
    """
    if policy is None:
        policy = ResearchSelectionPolicy()

    if not isinstance(evidence, ResearchEvidence):
        raise TypeError("evidence must be a ResearchEvidence instance.")

    records = tuple(trial_records) if trial_records is not None else ()

    for rec in records:
        if type(rec).__name__ != "ResearchTrialRecord":
            raise TypeError("All items in trial_records must be ResearchTrialRecord instances.")

    ev_collection = list(evidence_collection) if evidence_collection is not None else []
    for ev_item in ev_collection:
        if not isinstance(ev_item, ResearchEvidence):
            raise TypeError("All items in evidence_collection must be ResearchEvidence instances.")

    # Assemble complete evidence map
    known_evidences: list[ResearchEvidence] = [evidence]
    for ev_item in ev_collection:
        if not any(e.experiment_fingerprint == ev_item.experiment_fingerprint for e in known_evidences):
            known_evidences.append(ev_item)

    ev_map_by_exp_fp = {e.experiment_fingerprint: e for e in known_evidences}
    ev_map_by_ev_id = {e.evidence_id: e for e in known_evidences if e.evidence_id}

    # Calculate ledger trial counts
    if records:
        total_trials = len(records)
        completed_trials = sum(1 for t in records if t.status in ("COMPLETED", "QUALIFIED", "REJECTED"))
        failed_trials = sum(1 for t in records if t.status == "FAILED")
        qualified_trials = sum(1 for t in records if t.status == "QUALIFIED")
        rejected_trials = sum(1 for t in records if t.status == "REJECTED")
        effective_search_fp = search_fingerprint or records[0].search_id
    elif len(known_evidences) > 1:
        total_trials = len(known_evidences)
        completed_trials = total_trials
        failed_trials = 0
        qualified_trials = sum(1 for e in known_evidences if e.promotion_status in (PromotionStatus.PROMOTABLE, PromotionStatus.VALIDATED))
        rejected_trials = sum(1 for e in known_evidences if e.promotion_status == PromotionStatus.REJECTED)
        effective_search_fp = search_fingerprint or "multi_experiment_search"
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
    if records:
        is_winner = (
            matching_record is not None
            and matching_record.status == "QUALIFIED"
            and evidence.promotion_status in (PromotionStatus.PROMOTABLE, PromotionStatus.VALIDATED)
        )
    else:
        is_winner = evidence.promotion_status in (PromotionStatus.PROMOTABLE, PromotionStatus.VALIDATED)

    # Classify selection governance status & reason
    if total_trials <= policy.max_uncorrected_trials and not records and len(known_evidences) <= 1:
        status = SelectionGovernanceStatus.SELECTION_NOT_APPLICABLE
        reason = SelectionGovernanceReason.SINGLE_HYPOTHESIS_NO_SELECTION
        notes = "Single hypothesis evaluated; multiple-testing selection bias governance not applicable."
        metadata = {
            "policy_version": policy.policy_version,
            "max_uncorrected_trials": policy.max_uncorrected_trials,
            "alpha": policy.alpha,
            "correction_method": policy.correction_method,
            "matching_trial_found": matching_record is not None,
        }
    elif total_trials <= 1:
        status = SelectionGovernanceStatus.SELECTION_NOT_APPLICABLE
        reason = SelectionGovernanceReason.SINGLE_HYPOTHESIS_NO_SELECTION
        notes = "Single trial search ledger recorded; selection bias adjustment not applicable."
        metadata = {
            "policy_version": policy.policy_version,
            "max_uncorrected_trials": policy.max_uncorrected_trials,
            "alpha": policy.alpha,
            "correction_method": policy.correction_method,
            "matching_trial_found": matching_record is not None,
        }
    else:
        # Multi-trial search detected: Extract trial-level statistical p-values
        raw_p_map: dict[str, float] = {}
        trial_fp_map: dict[str, str] = {}

        if records:
            for rec in records:
                if rec.status == "FAILED":
                    continue
                ev_found = ev_map_by_exp_fp.get(rec.experiment_fingerprint) or (
                    ev_map_by_ev_id.get(rec.evidence_fingerprint) if rec.evidence_fingerprint else None
                )
                if ev_found and ev_found.robustness_verdict:
                    stat_val = ev_found.robustness_verdict.get("statistical_validation")
                    if isinstance(stat_val, dict) and "p_value" in stat_val:
                        p_val = stat_val["p_value"]
                        if isinstance(p_val, (int, float)) and math.isfinite(p_val) and (0.0 <= p_val <= 1.0):
                            raw_p_map[rec.trial_id] = float(p_val)
                            trial_fp_map[rec.trial_id] = ev_found.experiment_fingerprint
        else:
            for idx, ev_item in enumerate(known_evidences):
                if ev_item.robustness_verdict:
                    stat_val = ev_item.robustness_verdict.get("statistical_validation")
                    if isinstance(stat_val, dict) and "p_value" in stat_val:
                        p_val = stat_val["p_value"]
                        if isinstance(p_val, (int, float)) and math.isfinite(p_val) and (0.0 <= p_val <= 1.0):
                            t_key = f"trial_{idx}_{ev_item.experiment_fingerprint[:8]}"
                            raw_p_map[t_key] = float(p_val)
                            trial_fp_map[t_key] = ev_item.experiment_fingerprint

        eligible_trial_count = len(raw_p_map)
        excluded_trial_count = total_trials - eligible_trial_count

        # Check if valid p-values exist for all completed trials
        if eligible_trial_count >= 2 and eligible_trial_count >= completed_trials:
            # Perform deterministic quantitative multiple-testing correction
            corr_res = run_multiple_testing_correction(
                raw_p_values=raw_p_map,
                alpha=policy.alpha,
                method=policy.correction_method,
            )

            # Find key for current target evidence
            target_key = None
            if matching_record and matching_record.trial_id in raw_p_map:
                target_key = matching_record.trial_id
            else:
                for k, fp in trial_fp_map.items():
                    if fp == evidence.experiment_fingerprint:
                        target_key = k
                        break

            target_raw_p = corr_res["raw_p_values"].get(target_key) if target_key else None
            target_adj_p = corr_res["adjusted_p_values"].get(target_key) if target_key else None
            target_passed = corr_res["rejection_decisions"].get(target_key, False) if target_key else False

            status = SelectionGovernanceStatus.SELECTION_ADJUSTMENT_APPLIED
            reason = SelectionGovernanceReason.MULTI_TRIAL_SELECTION_DETECTED
            notes = (
                f"Multi-trial selection adjustment applied using {policy.correction_method} "
                f"(m={eligible_trial_count}, alpha={policy.alpha}). "
                f"Raw p-value: {target_raw_p if target_raw_p is not None else 'N/A'}, "
                f"Adjusted p-value: {target_adj_p if target_adj_p is not None else 'N/A'}. "
                f"Correction verdict: {'PASSED' if target_passed else 'FAILED'}."
            )

            metadata = {
                "policy_version": policy.policy_version,
                "max_uncorrected_trials": policy.max_uncorrected_trials,
                "alpha": policy.alpha,
                "correction_method": policy.correction_method,
                "correction_applied": True,
                "matching_trial_found": matching_record is not None,
                "eligible_trial_count": eligible_trial_count,
                "excluded_trial_count": excluded_trial_count,
                "raw_p_values": corr_res["raw_p_values"],
                "adjusted_p_values": corr_res["adjusted_p_values"],
                "rejection_decisions": corr_res["rejection_decisions"],
                "target_raw_p_value": target_raw_p,
                "target_adjusted_p_value": target_adj_p,
                "target_passed_correction": target_passed,
                "selected_winner_identity": candidate_id,
                "relevant_experiment_fingerprints": sorted(set(trial_fp_map.values())),
            }
        else:
            # Insufficient valid statistical inputs across trials -> fail closed
            status = SelectionGovernanceStatus.SELECTION_REVIEW_REQUIRED if records else SelectionGovernanceStatus.SELECTION_INSUFFICIENT_DATA
            reason = SelectionGovernanceReason.CORRECTION_UNAVAILABLE_MISSING_DISTRIBUTIONAL_INPUTS
            notes = (
                f"Multi-trial search recorded ({total_trials} trials, {qualified_trials} qualified, {eligible_trial_count} eligible with valid p-values). "
                "Selection context preserved; quantitative multiple-testing correction is unavailable "
                "because valid trial-level statistical p-values are missing or insufficient across trial records."
            )

            metadata = {
                "policy_version": policy.policy_version,
                "max_uncorrected_trials": policy.max_uncorrected_trials,
                "alpha": policy.alpha,
                "correction_method": policy.correction_method,
                "correction_applied": False,
                "matching_trial_found": matching_record is not None,
                "eligible_trial_count": eligible_trial_count,
                "excluded_trial_count": excluded_trial_count,
                "reason_details": "Canonical trial evidence does not contain valid statistical p-values for all completed trials.",
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
