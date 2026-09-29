"""Governed Research Campaign Evidence Synthesis & Selection Lifecycle.

Establishes the authoritative research-level decision layer that transforms a
completed durable Research Campaign into a deterministic, reproducible,
evidence-backed comparative research decision.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional

from src.evaluation.research_constitution import (
    ResearchEvidence,
)
from src.evaluation.research_store import (
    ResearchCampaignStore,
    load_research_experiment,
)
from src.evaluation.research_qualification import (
    qualify_research_evidence,
    ResearchQualificationPolicy,
)
from src.evaluation.research_robustness import assess_research_robustness
from src.evaluation.selection_governance import (
    assess_research_selection,
    ResearchSelectionPolicy,
)


def _canonical_json_dumps(data: Any) -> str:
    """Helper to convert dictionary/data structures to canonical JSON for SHA-256 hashing."""
    def _normalize(obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.value
        if dataclasses.is_dataclass(obj):
            return _normalize(asdict(obj))
        if isinstance(obj, (list, tuple)):
            return [_normalize(x) for x in obj]
        if isinstance(obj, (dict, Mapping)):
            return {str(k): _normalize(v) for k, v in sorted(obj.items())}
        if isinstance(obj, float):
            if not math.isfinite(obj):
                return str(obj)
            return round(obj, 10)
        return obj

    normalized = _normalize(data)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def compute_sha256_fingerprint(data: Any) -> str:
    """Computes a deterministic SHA-256 fingerprint for arbitrary canonical data."""
    canonical = _canonical_json_dumps(data)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class CampaignSelectionStatus(str, Enum):
    """Canonical decision states for campaign selection decisions."""
    SELECTED = "SELECTED"
    NO_ELIGIBLE_CANDIDATE = "NO_ELIGIBLE_CANDIDATE"
    SELECTION_BLOCKED = "SELECTION_BLOCKED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    SELECTION_REVIEW_REQUIRED = "SELECTION_REVIEW_REQUIRED"
    TIE_UNRESOLVED = "TIE_UNRESOLVED"


@dataclass(frozen=True)
class ResearchCandidateComparison:
    """Canonical comparison representation for a single research candidate in a campaign."""
    candidate_id: str
    candidate_fingerprint: str
    experiment_fingerprint: str
    evidence_fingerprint: str
    qualification_status: str
    qualification_fingerprint: str
    robustness_fingerprint: str
    robustness_status: str
    benchmark_evidence: dict[str, Any] = field(default_factory=dict)
    regime_evidence: dict[str, Any] = field(default_factory=dict)
    statistical_evidence: dict[str, Any] = field(default_factory=dict)
    oos_evidence: dict[str, Any] = field(default_factory=dict)
    walk_forward_evidence: dict[str, Any] = field(default_factory=dict)
    execution_assumptions: dict[str, Any] = field(default_factory=dict)
    rejection_reasons: tuple[str, ...] = field(default_factory=tuple)
    selection_governance_result: dict[str, Any] = field(default_factory=dict)
    comparison_metrics: dict[str, Any] = field(default_factory=dict)
    comparison_fingerprint: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if not self.comparison_fingerprint:
            payload = {
                "candidate_id": self.candidate_id,
                "candidate_fingerprint": self.candidate_fingerprint,
                "experiment_fingerprint": self.experiment_fingerprint,
                "evidence_fingerprint": self.evidence_fingerprint,
                "qualification_status": self.qualification_status,
                "qualification_fingerprint": self.qualification_fingerprint,
                "robustness_fingerprint": self.robustness_fingerprint,
                "robustness_status": self.robustness_status,
                "benchmark_evidence": self.benchmark_evidence,
                "regime_evidence": self.regime_evidence,
                "statistical_evidence": self.statistical_evidence,
                "oos_evidence": self.oos_evidence,
                "walk_forward_evidence": self.walk_forward_evidence,
                "execution_assumptions": self.execution_assumptions,
                "rejection_reasons": list(self.rejection_reasons),
                "selection_governance_result": self.selection_governance_result,
                "comparison_metrics": self.comparison_metrics,
            }
            object.__setattr__(self, "comparison_fingerprint", compute_sha256_fingerprint(payload))

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_fingerprint": self.candidate_fingerprint,
            "experiment_fingerprint": self.experiment_fingerprint,
            "evidence_fingerprint": self.evidence_fingerprint,
            "qualification_status": self.qualification_status,
            "qualification_fingerprint": self.qualification_fingerprint,
            "robustness_fingerprint": self.robustness_fingerprint,
            "robustness_status": self.robustness_status,
            "benchmark_evidence": self.benchmark_evidence,
            "regime_evidence": self.regime_evidence,
            "statistical_evidence": self.statistical_evidence,
            "oos_evidence": self.oos_evidence,
            "walk_forward_evidence": self.walk_forward_evidence,
            "execution_assumptions": self.execution_assumptions,
            "rejection_reasons": list(self.rejection_reasons),
            "selection_governance_result": self.selection_governance_result,
            "comparison_metrics": self.comparison_metrics,
            "comparison_fingerprint": self.comparison_fingerprint,
        }


@dataclass(frozen=True)
class ResearchCampaignSelectionPolicy:
    """Immutable policy governing candidate selection at the Campaign level."""
    policy_version: str = "1.0"
    eligibility_rules: dict[str, Any] = field(default_factory=dict)
    required_governance_states: tuple[str, ...] = ("QUALIFIED", "PROMOTABLE", "VALIDATED")
    required_robustness_dimensions: tuple[str, ...] = field(default_factory=tuple)
    required_oos_evidence: bool = True
    required_walk_forward_evidence: bool = True
    selection_governance_requirements: dict[str, Any] = field(default_factory=dict)
    tie_handling_policy: dict[str, Any] = field(default_factory=dict)
    minimum_evidence_requirements: dict[str, Any] = field(default_factory=dict)
    max_selected_candidates: int = 1
    ordering_rules: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    policy_fingerprint: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if not self.policy_fingerprint:
            payload = {
                "policy_version": self.policy_version,
                "eligibility_rules": self.eligibility_rules,
                "required_governance_states": list(self.required_governance_states),
                "required_robustness_dimensions": list(self.required_robustness_dimensions),
                "required_oos_evidence": self.required_oos_evidence,
                "required_walk_forward_evidence": self.required_walk_forward_evidence,
                "selection_governance_requirements": self.selection_governance_requirements,
                "tie_handling_policy": self.tie_handling_policy,
                "minimum_evidence_requirements": self.minimum_evidence_requirements,
                "max_selected_candidates": self.max_selected_candidates,
                "ordering_rules": list(self.ordering_rules),
            }
            object.__setattr__(self, "policy_fingerprint", compute_sha256_fingerprint(payload))

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "eligibility_rules": self.eligibility_rules,
            "required_governance_states": list(self.required_governance_states),
            "required_robustness_dimensions": list(self.required_robustness_dimensions),
            "required_oos_evidence": self.required_oos_evidence,
            "required_walk_forward_evidence": self.required_walk_forward_evidence,
            "selection_governance_requirements": self.selection_governance_requirements,
            "tie_handling_policy": self.tie_handling_policy,
            "minimum_evidence_requirements": self.minimum_evidence_requirements,
            "max_selected_candidates": self.max_selected_candidates,
            "ordering_rules": list(self.ordering_rules),
            "policy_fingerprint": self.policy_fingerprint,
        }


@dataclass(frozen=True)
class ResearchCampaignEvidenceSynthesis:
    """Immutable, fingerprintable domain artifact representing complete research evidence synthesized from one Campaign."""
    campaign_id: str
    campaign_definition_fingerprint: str
    trial_plan_fingerprint: str
    search_space_fingerprint: str
    search_policy_fingerprint: str
    criteria_fingerprint: str
    dataset_identity: dict[str, Any]
    execution_assumptions: dict[str, Any]
    code_provenance: dict[str, Any]
    methodology_version: str
    ordered_trial_identities: tuple[str, ...]
    ordered_evidence_fingerprints: tuple[str, ...]
    completed_trial_count: int
    failed_trial_count: int
    blocked_trial_count: int
    qualified_candidate_count: int
    rejected_candidate_count: int
    selection_governance_status: str
    selection_policy_fingerprint: str
    candidate_comparisons: tuple[ResearchCandidateComparison, ...]
    synthesis_methodology_version: str = "1.0"
    synthesis_fingerprint: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if not self.synthesis_fingerprint:
            payload = {
                "campaign_id": self.campaign_id,
                "campaign_definition_fingerprint": self.campaign_definition_fingerprint,
                "trial_plan_fingerprint": self.trial_plan_fingerprint,
                "search_space_fingerprint": self.search_space_fingerprint,
                "search_policy_fingerprint": self.search_policy_fingerprint,
                "criteria_fingerprint": self.criteria_fingerprint,
                "dataset_identity": self.dataset_identity,
                "execution_assumptions": self.execution_assumptions,
                "code_provenance": self.code_provenance,
                "methodology_version": self.methodology_version,
                "ordered_trial_identities": list(self.ordered_trial_identities),
                "ordered_evidence_fingerprints": list(self.ordered_evidence_fingerprints),
                "completed_trial_count": self.completed_trial_count,
                "failed_trial_count": self.failed_trial_count,
                "blocked_trial_count": self.blocked_trial_count,
                "qualified_candidate_count": self.qualified_candidate_count,
                "rejected_candidate_count": self.rejected_candidate_count,
                "selection_governance_status": self.selection_governance_status,
                "selection_policy_fingerprint": self.selection_policy_fingerprint,
                "candidate_comparisons": [c.comparison_fingerprint for c in self.candidate_comparisons],
                "synthesis_methodology_version": self.synthesis_methodology_version,
            }
            object.__setattr__(self, "synthesis_fingerprint", compute_sha256_fingerprint(payload))

    def as_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "campaign_definition_fingerprint": self.campaign_definition_fingerprint,
            "trial_plan_fingerprint": self.trial_plan_fingerprint,
            "search_space_fingerprint": self.search_space_fingerprint,
            "search_policy_fingerprint": self.search_policy_fingerprint,
            "criteria_fingerprint": self.criteria_fingerprint,
            "dataset_identity": self.dataset_identity,
            "execution_assumptions": self.execution_assumptions,
            "code_provenance": self.code_provenance,
            "methodology_version": self.methodology_version,
            "ordered_trial_identities": list(self.ordered_trial_identities),
            "ordered_evidence_fingerprints": list(self.ordered_evidence_fingerprints),
            "completed_trial_count": self.completed_trial_count,
            "failed_trial_count": self.failed_trial_count,
            "blocked_trial_count": self.blocked_trial_count,
            "qualified_candidate_count": self.qualified_candidate_count,
            "rejected_candidate_count": self.rejected_candidate_count,
            "selection_governance_status": self.selection_governance_status,
            "selection_policy_fingerprint": self.selection_policy_fingerprint,
            "candidate_comparisons": [c.as_dict() for c in self.candidate_comparisons],
            "synthesis_methodology_version": self.synthesis_methodology_version,
            "synthesis_fingerprint": self.synthesis_fingerprint,
        }


@dataclass(frozen=True)
class ResearchCampaignSelectionDecision:
    """Authoritative research-level decision artifact resulting from comparing a Campaign's candidates."""
    campaign_id: str
    synthesis_fingerprint: str
    selection_policy_fingerprint: str
    selected_candidate_ids: tuple[str, ...]
    eligible_candidate_ids: tuple[str, ...]
    rejected_candidate_ids: tuple[str, ...]
    blocked_candidate_ids: tuple[str, ...]
    candidate_comparison_fingerprints: tuple[str, ...]
    selection_governance_fingerprints: tuple[str, ...]
    qualification_fingerprints: tuple[str, ...]
    robustness_fingerprints: tuple[str, ...]
    evidence_fingerprints: tuple[str, ...]
    decision_status: str
    decision_reason: str
    deterministic_ordering: tuple[str, ...]
    decision_methodology_version: str = "1.0"
    decision_fingerprint: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if not self.decision_fingerprint:
            payload = {
                "campaign_id": self.campaign_id,
                "synthesis_fingerprint": self.synthesis_fingerprint,
                "selection_policy_fingerprint": self.selection_policy_fingerprint,
                "selected_candidate_ids": list(self.selected_candidate_ids),
                "eligible_candidate_ids": list(self.eligible_candidate_ids),
                "rejected_candidate_ids": list(self.rejected_candidate_ids),
                "blocked_candidate_ids": list(self.blocked_candidate_ids),
                "candidate_comparison_fingerprints": list(self.candidate_comparison_fingerprints),
                "selection_governance_fingerprints": list(self.selection_governance_fingerprints),
                "qualification_fingerprints": list(self.qualification_fingerprints),
                "robustness_fingerprints": list(self.robustness_fingerprints),
                "evidence_fingerprints": list(self.evidence_fingerprints),
                "decision_status": self.decision_status,
                "decision_reason": self.decision_reason,
                "deterministic_ordering": list(self.deterministic_ordering),
                "decision_methodology_version": self.decision_methodology_version,
            }
            object.__setattr__(self, "decision_fingerprint", compute_sha256_fingerprint(payload))

    def as_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "synthesis_fingerprint": self.synthesis_fingerprint,
            "selection_policy_fingerprint": self.selection_policy_fingerprint,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "eligible_candidate_ids": list(self.eligible_candidate_ids),
            "rejected_candidate_ids": list(self.rejected_candidate_ids),
            "blocked_candidate_ids": list(self.blocked_candidate_ids),
            "candidate_comparison_fingerprints": list(self.candidate_comparison_fingerprints),
            "selection_governance_fingerprints": list(self.selection_governance_fingerprints),
            "qualification_fingerprints": list(self.qualification_fingerprints),
            "robustness_fingerprints": list(self.robustness_fingerprints),
            "evidence_fingerprints": list(self.evidence_fingerprints),
            "decision_status": self.decision_status,
            "decision_reason": self.decision_reason,
            "deterministic_ordering": list(self.deterministic_ordering),
            "decision_methodology_version": self.decision_methodology_version,
            "decision_fingerprint": self.decision_fingerprint,
        }


# =============================================================================
# Campaign Selection Integrity Validator
# =============================================================================

class CampaignSelectionIntegrityError(ValueError):
    """Raised when campaign synthesis or selection decision integrity validation fails."""
    pass


class CampaignSelectionIntegrityValidator:
    """Dedicated integrity validator for campaign synthesis and selection decisions."""

    @staticmethod
    def validate_synthesis_integrity(
        synthesis: ResearchCampaignEvidenceSynthesis,
        store: Optional[ResearchCampaignStore] = None,
    ) -> None:
        """Validates structural and lineage integrity of a ResearchCampaignEvidenceSynthesis artifact."""
        if store is None:
            store = ResearchCampaignStore()

        # 1. Re-compute fingerprint to verify no tampering
        expected_payload = {
            "campaign_id": synthesis.campaign_id,
            "campaign_definition_fingerprint": synthesis.campaign_definition_fingerprint,
            "trial_plan_fingerprint": synthesis.trial_plan_fingerprint,
            "search_space_fingerprint": synthesis.search_space_fingerprint,
            "search_policy_fingerprint": synthesis.search_policy_fingerprint,
            "criteria_fingerprint": synthesis.criteria_fingerprint,
            "dataset_identity": synthesis.dataset_identity,
            "execution_assumptions": synthesis.execution_assumptions,
            "code_provenance": synthesis.code_provenance,
            "methodology_version": synthesis.methodology_version,
            "ordered_trial_identities": list(synthesis.ordered_trial_identities),
            "ordered_evidence_fingerprints": list(synthesis.ordered_evidence_fingerprints),
            "completed_trial_count": synthesis.completed_trial_count,
            "failed_trial_count": synthesis.failed_trial_count,
            "blocked_trial_count": synthesis.blocked_trial_count,
            "qualified_candidate_count": synthesis.qualified_candidate_count,
            "rejected_candidate_count": synthesis.rejected_candidate_count,
            "selection_governance_status": synthesis.selection_governance_status,
            "selection_policy_fingerprint": synthesis.selection_policy_fingerprint,
            "candidate_comparisons": [c.comparison_fingerprint for c in synthesis.candidate_comparisons],
            "synthesis_methodology_version": synthesis.synthesis_methodology_version,
        }
        recomputed_fp = compute_sha256_fingerprint(expected_payload)
        if recomputed_fp != synthesis.synthesis_fingerprint:
            raise CampaignSelectionIntegrityError(
                f"Synthesis fingerprint mismatch for campaign '{synthesis.campaign_id}': "
                f"recomputed '{recomputed_fp}', got '{synthesis.synthesis_fingerprint}'."
            )

        # 2. Check each candidate comparison fingerprint
        for comp in synthesis.candidate_comparisons:
            comp_payload = {
                "candidate_id": comp.candidate_id,
                "candidate_fingerprint": comp.candidate_fingerprint,
                "experiment_fingerprint": comp.experiment_fingerprint,
                "evidence_fingerprint": comp.evidence_fingerprint,
                "qualification_status": comp.qualification_status,
                "qualification_fingerprint": comp.qualification_fingerprint,
                "robustness_fingerprint": comp.robustness_fingerprint,
                "robustness_status": comp.robustness_status,
                "benchmark_evidence": comp.benchmark_evidence,
                "regime_evidence": comp.regime_evidence,
                "statistical_evidence": comp.statistical_evidence,
                "oos_evidence": comp.oos_evidence,
                "walk_forward_evidence": comp.walk_forward_evidence,
                "execution_assumptions": comp.execution_assumptions,
                "rejection_reasons": list(comp.rejection_reasons),
                "selection_governance_result": comp.selection_governance_result,
                "comparison_metrics": comp.comparison_metrics,
            }
            recomputed_comp_fp = compute_sha256_fingerprint(comp_payload)
            if recomputed_comp_fp != comp.comparison_fingerprint:
                raise CampaignSelectionIntegrityError(
                    f"Candidate comparison fingerprint mismatch for candidate '{comp.candidate_id}' in campaign '{synthesis.campaign_id}': "
                    f"recomputed '{recomputed_comp_fp}', got '{comp.comparison_fingerprint}'."
                )

        # 3. Load campaign definition & trial plan to verify matching campaign identity
        definition = store.load_definition(synthesis.campaign_id)
        if definition.definition_fingerprint != synthesis.campaign_definition_fingerprint:
            raise CampaignSelectionIntegrityError(
                f"Campaign definition fingerprint mismatch for synthesis '{synthesis.campaign_id}': "
                f"store definition '{definition.definition_fingerprint}', synthesis got '{synthesis.campaign_definition_fingerprint}'."
            )

        trial_plan = store.load_trial_plan(synthesis.campaign_id)
        if trial_plan.plan_fingerprint != synthesis.trial_plan_fingerprint:
            raise CampaignSelectionIntegrityError(
                f"Trial plan fingerprint mismatch for synthesis '{synthesis.campaign_id}': "
                f"store plan '{trial_plan.plan_fingerprint}', synthesis got '{synthesis.trial_plan_fingerprint}'."
            )

    @staticmethod
    def validate_selection_decision_integrity(
        decision: ResearchCampaignSelectionDecision,
        synthesis: ResearchCampaignEvidenceSynthesis,
        selection_policy: Optional[ResearchCampaignSelectionPolicy] = None,
    ) -> None:
        """Validates structural and lineage integrity of a ResearchCampaignSelectionDecision artifact."""
        if selection_policy is None:
            selection_policy = ResearchCampaignSelectionPolicy()

        # 1. Re-compute decision fingerprint to verify no tampering
        expected_payload = {
            "campaign_id": decision.campaign_id,
            "synthesis_fingerprint": decision.synthesis_fingerprint,
            "selection_policy_fingerprint": decision.selection_policy_fingerprint,
            "selected_candidate_ids": list(decision.selected_candidate_ids),
            "eligible_candidate_ids": list(decision.eligible_candidate_ids),
            "rejected_candidate_ids": list(decision.rejected_candidate_ids),
            "blocked_candidate_ids": list(decision.blocked_candidate_ids),
            "candidate_comparison_fingerprints": list(decision.candidate_comparison_fingerprints),
            "selection_governance_fingerprints": list(decision.selection_governance_fingerprints),
            "qualification_fingerprints": list(decision.qualification_fingerprints),
            "robustness_fingerprints": list(decision.robustness_fingerprints),
            "evidence_fingerprints": list(decision.evidence_fingerprints),
            "decision_status": decision.decision_status,
            "decision_reason": decision.decision_reason,
            "deterministic_ordering": list(decision.deterministic_ordering),
            "decision_methodology_version": decision.decision_methodology_version,
        }
        recomputed_fp = compute_sha256_fingerprint(expected_payload)
        if recomputed_fp != decision.decision_fingerprint:
            raise CampaignSelectionIntegrityError(
                f"Decision fingerprint mismatch for campaign '{decision.campaign_id}': "
                f"recomputed '{recomputed_fp}', got '{decision.decision_fingerprint}'."
            )

        # 2. Check campaign ID and synthesis fingerprint matching
        if decision.campaign_id != synthesis.campaign_id:
            raise CampaignSelectionIntegrityError(
                f"Campaign ID mismatch in decision: decision got '{decision.campaign_id}', synthesis got '{synthesis.campaign_id}'."
            )

        if decision.synthesis_fingerprint != synthesis.synthesis_fingerprint:
            raise CampaignSelectionIntegrityError(
                f"Synthesis fingerprint mismatch in decision for campaign '{decision.campaign_id}': "
                f"decision got '{decision.synthesis_fingerprint}', synthesis got '{synthesis.synthesis_fingerprint}'."
            )


# =============================================================================
# Campaign Evidence Collection and Synthesis Logic
# =============================================================================

def collect_and_synthesize_campaign_evidence(
    campaign_id: str,
    selection_policy: Optional[ResearchCampaignSelectionPolicy] = None,
    store: Optional[ResearchCampaignStore] = None,
    experiment_store_dir: Optional[str | Path] = None,
) -> ResearchCampaignEvidenceSynthesis:
    """Authoritatively collects durable campaign state and synthesizes comparative research evidence.

    Fails closed if lineage is missing, corrupt, or inconsistent.
    """
    if store is None:
        store = ResearchCampaignStore()
    if selection_policy is None:
        selection_policy = ResearchCampaignSelectionPolicy()

    # 1. Load authoritative campaign definition, trial plan, state, and checkpoints
    definition = store.load_definition(campaign_id)
    trial_plan = store.load_trial_plan(campaign_id)
    store.load_lifecycle_state(campaign_id)
    checkpoints = store.list_trial_checkpoints(campaign_id)

    # 2. Verify basic campaign membership and counts
    planned_trials_by_id = {t.trial_id: t for t in trial_plan.trials}
    if len(planned_trials_by_id) != len(trial_plan.trials):
        raise ValueError(f"Duplicate trial IDs found in trial plan for campaign '{campaign_id}'.")

    checkpoint_by_id = {cp.trial_id: cp for cp in checkpoints}

    # Verify every trial in the plan exists as a checkpoint
    for trial_id in planned_trials_by_id:
        if trial_id not in checkpoint_by_id:
            raise ValueError(f"Missing expected checkpoint for trial '{trial_id}' in campaign '{campaign_id}'.")

    # Check for unknown trial checkpoints not in trial plan
    for cp in checkpoints:
        if cp.trial_id not in planned_trials_by_id:
            raise ValueError(f"Unknown trial checkpoint '{cp.trial_id}' found in campaign '{campaign_id}'.")

        # Verify campaign_id matches
        if cp.campaign_id != campaign_id:
            raise ValueError(
                f"Trial checkpoint '{cp.trial_id}' belongs to campaign '{cp.campaign_id}', expected '{campaign_id}'."
            )

    # Order trials according to the trial plan index
    ordered_planned_trials = sorted(trial_plan.trials, key=lambda t: t.trial_index)

    completed_count = 0
    failed_count = 0
    blocked_count = 0
    qualified_count = 0
    rejected_count = 0

    ordered_trial_identities = []
    ordered_evidence_fingerprints = []
    candidate_comparisons = []

    # Map for selection governance
    evidence_records_by_candidate: dict[str, ResearchEvidence] = {}

    for t_plan in ordered_planned_trials:
        trial_id = t_plan.trial_id
        cp = checkpoint_by_id[trial_id]
        ordered_trial_identities.append(trial_id)

        # Verify candidate identity
        if cp.candidate_id != t_plan.candidate_id:
            raise ValueError(
                f"Candidate ID mismatch in trial '{trial_id}': checkpoint got '{cp.candidate_id}', plan expected '{t_plan.candidate_id}'."
            )

        status_str = cp.status.upper() if isinstance(cp.status, str) else cp.status.value.upper()

        if status_str in ("FAILED", "ERROR", "TRUNCATED"):
            failed_count += 1
            # Record failed comparison representation
            comp = ResearchCandidateComparison(
                candidate_id=t_plan.candidate_id,
                candidate_fingerprint=t_plan.candidate_fingerprint,
                experiment_fingerprint=cp.experiment_fingerprint or "",
                evidence_fingerprint="",
                qualification_status="FAILED",
                qualification_fingerprint="",
                robustness_fingerprint="",
                robustness_status="FAILED",
                rejection_reasons=(cp.error_message or "Trial execution failed",),
            )
            candidate_comparisons.append(comp)
            continue

        if status_str in ("BLOCKED", "GOVERNANCE_BLOCKED"):
            blocked_count += 1
            comp = ResearchCandidateComparison(
                candidate_id=t_plan.candidate_id,
                candidate_fingerprint=t_plan.candidate_fingerprint,
                experiment_fingerprint=cp.experiment_fingerprint or "",
                evidence_fingerprint="",
                qualification_status="BLOCKED",
                qualification_fingerprint="",
                robustness_fingerprint="",
                robustness_status="BLOCKED",
                rejection_reasons=(cp.error_message or "Blocked by governance",),
            )
            candidate_comparisons.append(comp)
            continue

        if status_str in ("COMPLETED", "QUALIFIED", "REJECTED"):
            completed_count += 1
            if not cp.evidence_fingerprint:
                raise ValueError(f"Completed trial '{trial_id}' missing evidence_fingerprint in checkpoint.")

            # Load actual evidence artifact
            if not cp.experiment_fingerprint:
                raise ValueError(f"Completed trial '{trial_id}' missing experiment_fingerprint in checkpoint.")

            if experiment_store_dir is not None:
                evidence = load_research_experiment(cp.experiment_fingerprint, base_dir=experiment_store_dir)
            else:
                evidence = load_research_experiment(cp.experiment_fingerprint)
            if evidence is None:
                raise ValueError(f"Evidence artifact '{cp.evidence_fingerprint}' for trial '{trial_id}' not found.")

            # Lineage validations
            if evidence.evidence_id != cp.evidence_fingerprint and evidence.evidence_id != cp.experiment_fingerprint:
                # Note: evidence.evidence_id is evidence_fingerprint
                if evidence.evidence_id != cp.evidence_fingerprint:
                    raise ValueError(
                        f"Evidence ID mismatch for trial '{trial_id}': loaded '{evidence.evidence_id}', checkpoint expected '{cp.evidence_fingerprint}'."
                    )

            if evidence.spec.fingerprint != cp.experiment_fingerprint:
                raise ValueError(
                    f"Experiment fingerprint mismatch for trial '{trial_id}': loaded '{evidence.spec.fingerprint}', checkpoint expected '{cp.experiment_fingerprint}'."
                )

            # Dataset identity validation
            if (
                evidence.spec.dataset_scope.dataset_id != definition.dataset_scope.dataset_id
                or evidence.spec.dataset_scope.symbol != definition.dataset_scope.symbol
            ):
                raise ValueError(
                    f"Dataset scope mismatch in trial '{trial_id}': "
                    f"evidence dataset '{evidence.spec.dataset_scope.dataset_id}' vs campaign '{definition.dataset_scope.dataset_id}'."
                )

            ordered_evidence_fingerprints.append(evidence.evidence_id)
            evidence_records_by_candidate[t_plan.candidate_id] = evidence

            # Compute canonical robustness assessment
            robustness_assessment = assess_research_robustness(evidence)

            # Qualify evidence using canonical qualification gate
            qual_policy = ResearchQualificationPolicy()
            qual_result = qualify_research_evidence(
                evidence=evidence,
                policy=qual_policy,
                robustness_assessment=robustness_assessment,
            )

            qual_status_str = qual_result.status.value if isinstance(qual_result.status, Enum) else str(qual_result.status)

            if qual_status_str == "QUALIFIED" or qual_result.qualified:
                qualified_count += 1
            else:
                rejected_count += 1

            # Build detailed candidate comparison from canonical evidence partitions
            sharpe = 0.0
            pf = 0.0
            max_dd = 0.0
            total_trades = 0
            win_rate = 0.0
            oos_ev = {}
            wf_ev = {}

            for p in evidence.partitions:
                role_str = p.role.value.upper() if isinstance(p.role, Enum) else str(p.role).upper()
                p_dict = {
                    "sharpe_ratio": p.sharpe_ratio,
                    "profit_factor": p.profit_factor,
                    "max_drawdown": p.max_drawdown,
                    "total_return": p.total_return,
                    "win_rate": p.win_rate,
                    "observations": p.observations,
                }
                if "OOS" in role_str or role_str == "OUT_OF_SAMPLE":
                    oos_ev = p_dict
                    sharpe = p.sharpe_ratio
                    pf = p.profit_factor
                    max_dd = p.max_drawdown
                    total_trades = p.observations
                    win_rate = p.win_rate
                elif "WALK_FORWARD" in role_str:
                    wf_ev = p_dict
                if sharpe == 0.0 and p.sharpe_ratio:
                    sharpe = p.sharpe_ratio
                    pf = p.profit_factor
                    max_dd = p.max_drawdown
                    total_trades = p.observations
                    win_rate = p.win_rate

            comp_metrics = {
                "sharpe_ratio": sharpe,
                "profit_factor": pf,
                "max_drawdown": max_dd,
                "total_trades": total_trades,
                "win_rate": win_rate,
            }

            comp = ResearchCandidateComparison(
                candidate_id=t_plan.candidate_id,
                candidate_fingerprint=t_plan.candidate_fingerprint,
                experiment_fingerprint=evidence.spec.fingerprint,
                evidence_fingerprint=evidence.evidence_id,
                qualification_status=qual_status_str,
                qualification_fingerprint=qual_result.decision_fingerprint,
                robustness_fingerprint=robustness_assessment.robustness_fingerprint,
                robustness_status="PASSED" if robustness_assessment.is_robust else "FAILED",
                benchmark_evidence=asdict(robustness_assessment.benchmark_assessment) if robustness_assessment.benchmark_assessment else {},
                regime_evidence=asdict(robustness_assessment.regime_assessment) if robustness_assessment.regime_assessment else {},
                statistical_evidence={
                    "is_robust": robustness_assessment.is_robust,
                    "status": robustness_assessment.status.value if isinstance(robustness_assessment.status, Enum) else str(robustness_assessment.status),
                },
                oos_evidence=oos_ev,
                walk_forward_evidence=wf_ev,
                execution_assumptions=asdict(evidence.spec.execution_assumptions),
                rejection_reasons=tuple(str(r) for r in qual_result.rejection_reasons),
                comparison_metrics=comp_metrics,
            )
            candidate_comparisons.append(comp)

    # 3. Assess Selection Governance across candidates
    sel_governance_status = "SELECTION_NOT_APPLICABLE"
    if len(evidence_records_by_candidate) > 0:
        evidence_list = list(evidence_records_by_candidate.values())
        sel_policy = ResearchSelectionPolicy()
        sel_assessment = assess_research_selection(
            evidence=evidence_list[0],
            evidence_collection=evidence_list,
            policy=sel_policy,
        )

        sel_governance_status = sel_assessment.status.value if isinstance(sel_assessment.status, Enum) else str(sel_assessment.status)

        # Attach selection governance result to candidate comparisons
        updated_comparisons = []
        adj_p_map = sel_assessment.metadata.get("adjusted_p_values", {})
        rej_map = sel_assessment.metadata.get("rejection_decisions", {})
        for comp in candidate_comparisons:
            if comp.candidate_id in evidence_records_by_candidate:
                cand_evidence = evidence_records_by_candidate[comp.candidate_id]
                cand_gov_dict = {
                    "selection_status": sel_governance_status,
                    "assessment_fingerprint": sel_assessment.selection_fingerprint,
                    "adjusted_p_value": adj_p_map.get(cand_evidence.evidence_id),
                    "is_significant": rej_map.get(cand_evidence.evidence_id, False),
                }
                # Re-instantiate ResearchCandidateComparison with selection_governance_result attached
                comp_dict = comp.as_dict()
                comp_dict["selection_governance_result"] = cand_gov_dict
                del comp_dict["comparison_fingerprint"]
                comp = ResearchCandidateComparison(**comp_dict)
            updated_comparisons.append(comp)
        candidate_comparisons = updated_comparisons

    synthesis = ResearchCampaignEvidenceSynthesis(
        campaign_id=campaign_id,
        campaign_definition_fingerprint=definition.definition_fingerprint,
        trial_plan_fingerprint=trial_plan.plan_fingerprint,
        search_space_fingerprint=definition.search_space_fingerprint,
        search_policy_fingerprint=definition.search_policy_fingerprint,
        criteria_fingerprint=definition.criteria_fingerprint,
        dataset_identity=asdict(definition.dataset_scope),
        execution_assumptions=asdict(definition.execution_assumptions),
        code_provenance=asdict(definition.code_provenance),
        methodology_version=definition.methodology_version,
        ordered_trial_identities=tuple(ordered_trial_identities),
        ordered_evidence_fingerprints=tuple(ordered_evidence_fingerprints),
        completed_trial_count=completed_count,
        failed_trial_count=failed_count,
        blocked_trial_count=blocked_count,
        qualified_candidate_count=qualified_count,
        rejected_candidate_count=rejected_count,
        selection_governance_status=sel_governance_status,
        selection_policy_fingerprint=selection_policy.policy_fingerprint,
        candidate_comparisons=tuple(candidate_comparisons),
    )

    return synthesis


def select_campaign_candidate(
    synthesis: ResearchCampaignEvidenceSynthesis,
    selection_policy: Optional[ResearchCampaignSelectionPolicy] = None,
) -> ResearchCampaignSelectionDecision:
    """Executes governed, fail-closed selection over synthesized campaign candidate evidence.

    Fails closed if no eligible candidates remain, ties are unresolved, or selection governance is invalid.
    """
    if selection_policy is None:
        selection_policy = ResearchCampaignSelectionPolicy()

    eligible_candidate_ids = []
    rejected_candidate_ids = []
    blocked_candidate_ids = []

    comp_map = {c.candidate_id: c for c in synthesis.candidate_comparisons}

    for comp in synthesis.candidate_comparisons:
        cid = comp.candidate_id

        if comp.qualification_status == "BLOCKED":
            blocked_candidate_ids.append(cid)
            continue

        if comp.qualification_status not in selection_policy.required_governance_states:
            rejected_candidate_ids.append(cid)
            continue

        if comp.robustness_status != "PASSED":
            rejected_candidate_ids.append(cid)
            continue

        if selection_policy.required_oos_evidence and not comp.oos_evidence:
            rejected_candidate_ids.append(cid)
            continue

        if selection_policy.required_walk_forward_evidence and not comp.walk_forward_evidence:
            rejected_candidate_ids.append(cid)
            continue

        # Check selection governance if applicable
        sel_gov = comp.selection_governance_result
        if sel_gov:
            sel_status = sel_gov.get("selection_status", "")
            if sel_status in ("SELECTION_REVIEW_REQUIRED", "SELECTION_INSUFFICIENT_DATA"):
                rejected_candidate_ids.append(cid)
                continue

        eligible_candidate_ids.append(cid)

    # Gather required fingerprints
    qualification_fps = tuple(c.qualification_fingerprint for c in synthesis.candidate_comparisons if c.qualification_fingerprint)
    robustness_fps = tuple(c.robustness_fingerprint for c in synthesis.candidate_comparisons if c.robustness_fingerprint)
    evidence_fps = tuple(c.evidence_fingerprint for c in synthesis.candidate_comparisons if c.evidence_fingerprint)
    comp_fps = tuple(c.comparison_fingerprint for c in synthesis.candidate_comparisons)
    sel_gov_fps = tuple(
        c.selection_governance_result.get("assessment_fingerprint")
        for c in synthesis.candidate_comparisons
        if c.selection_governance_result.get("assessment_fingerprint")
    )

    if not eligible_candidate_ids:
        return ResearchCampaignSelectionDecision(
            campaign_id=synthesis.campaign_id,
            synthesis_fingerprint=synthesis.synthesis_fingerprint,
            selection_policy_fingerprint=selection_policy.policy_fingerprint,
            selected_candidate_ids=(),
            eligible_candidate_ids=(),
            rejected_candidate_ids=tuple(rejected_candidate_ids),
            blocked_candidate_ids=tuple(blocked_candidate_ids),
            candidate_comparison_fingerprints=comp_fps,
            selection_governance_fingerprints=sel_gov_fps,
            qualification_fingerprints=qualification_fps,
            robustness_fingerprints=robustness_fps,
            evidence_fingerprints=evidence_fps,
            decision_status=CampaignSelectionStatus.NO_ELIGIBLE_CANDIDATE.value,
            decision_reason="No candidates satisfied all qualification, robustness, OOS, and selection governance requirements.",
            deterministic_ordering=(),
        )

    # Sort eligible candidates using deterministic ordering rules
    eligible_comps = [comp_map[cid] for cid in eligible_candidate_ids]

    def _get_sort_key(comp: ResearchCandidateComparison) -> tuple:
        m = comp.comparison_metrics
        sharpe = round(m.get("sharpe_ratio", 0.0), 6)
        pf = round(m.get("profit_factor", 0.0), 6)
        win_rate = round(m.get("win_rate", 0.0), 6)
        return (sharpe, pf, win_rate)

    # Group candidates by sort key to detect ties
    sorted_comps = sorted(eligible_comps, key=_get_sort_key, reverse=True)
    top_key = _get_sort_key(sorted_comps[0])
    top_tied = [c for c in sorted_comps if _get_sort_key(c) == top_key]

    deterministic_ordering = tuple(c.candidate_id for c in sorted_comps)

    if len(top_tied) > 1:
        # Attempt deterministic secondary tie-breaking if tie-handling policy defines one
        tie_policy = selection_policy.tie_handling_policy
        secondary_metric = tie_policy.get("secondary_metric")

        if secondary_metric:
            def _get_secondary_key(comp: ResearchCandidateComparison) -> float:
                return round(comp.comparison_metrics.get(secondary_metric, 0.0), 6)

            top_tied_sorted = sorted(top_tied, key=_get_secondary_key, reverse=True)
            top_sec_key = _get_secondary_key(top_tied_sorted[0])
            top_sec_tied = [c for c in top_tied_sorted if _get_secondary_key(c) == top_sec_key]

            if len(top_sec_tied) == 1:
                selected_ids = (top_sec_tied[0].candidate_id,)
                return ResearchCampaignSelectionDecision(
                    campaign_id=synthesis.campaign_id,
                    synthesis_fingerprint=synthesis.synthesis_fingerprint,
                    selection_policy_fingerprint=selection_policy.policy_fingerprint,
                    selected_candidate_ids=selected_ids,
                    eligible_candidate_ids=tuple(eligible_candidate_ids),
                    rejected_candidate_ids=tuple(rejected_candidate_ids),
                    blocked_candidate_ids=tuple(blocked_candidate_ids),
                    candidate_comparison_fingerprints=comp_fps,
                    selection_governance_fingerprints=sel_gov_fps,
                    qualification_fingerprints=qualification_fps,
                    robustness_fingerprints=robustness_fps,
                    evidence_fingerprints=evidence_fps,
                    decision_status=CampaignSelectionStatus.SELECTED.value,
                    decision_reason=f"Candidate '{selected_ids[0]}' uniquely selected via primary metrics and secondary tie-break rule '{secondary_metric}'.",
                    deterministic_ordering=deterministic_ordering,
                )

        # Unresolved tie - fail closed
        return ResearchCampaignSelectionDecision(
            campaign_id=synthesis.campaign_id,
            synthesis_fingerprint=synthesis.synthesis_fingerprint,
            selection_policy_fingerprint=selection_policy.policy_fingerprint,
            selected_candidate_ids=(),
            eligible_candidate_ids=tuple(eligible_candidate_ids),
            rejected_candidate_ids=tuple(rejected_candidate_ids),
            blocked_candidate_ids=tuple(blocked_candidate_ids),
            candidate_comparison_fingerprints=comp_fps,
            selection_governance_fingerprints=sel_gov_fps,
            qualification_fingerprints=qualification_fps,
            robustness_fingerprints=robustness_fps,
            evidence_fingerprints=evidence_fps,
            decision_status=CampaignSelectionStatus.TIE_UNRESOLVED.value,
            decision_reason=f"Unresolved tie between {len(top_tied)} candidates ({[c.candidate_id for c in top_tied]}).",
            deterministic_ordering=deterministic_ordering,
        )

    selected_ids = (top_tied[0].candidate_id,)
    return ResearchCampaignSelectionDecision(
        campaign_id=synthesis.campaign_id,
        synthesis_fingerprint=synthesis.synthesis_fingerprint,
        selection_policy_fingerprint=selection_policy.policy_fingerprint,
        selected_candidate_ids=selected_ids,
        eligible_candidate_ids=tuple(eligible_candidate_ids),
        rejected_candidate_ids=tuple(rejected_candidate_ids),
        blocked_candidate_ids=tuple(blocked_candidate_ids),
        candidate_comparison_fingerprints=comp_fps,
        selection_governance_fingerprints=sel_gov_fps,
        qualification_fingerprints=qualification_fps,
        robustness_fingerprints=robustness_fps,
        evidence_fingerprints=evidence_fps,
        decision_status=CampaignSelectionStatus.SELECTED.value,
        decision_reason=f"Candidate '{selected_ids[0]}' selected deterministically as top eligible candidate.",
        deterministic_ordering=deterministic_ordering,
    )


# =============================================================================
# Authoritative Public API Functions
# =============================================================================

def synthesize_campaign_evidence(
    campaign_id: str,
    selection_policy: Optional[ResearchCampaignSelectionPolicy] = None,
    store: Optional[ResearchCampaignStore] = None,
    experiment_store_dir: Optional[str | Path] = None,
) -> ResearchCampaignEvidenceSynthesis:
    """Authoritative API to synthesize evidence for a completed research campaign."""
    synthesis = collect_and_synthesize_campaign_evidence(
        campaign_id=campaign_id,
        selection_policy=selection_policy,
        store=store,
        experiment_store_dir=experiment_store_dir,
    )
    CampaignSelectionIntegrityValidator.validate_synthesis_integrity(synthesis, store=store)
    return synthesis


def load_campaign_selection_decision(
    campaign_id: str,
    store: Optional[ResearchCampaignStore] = None,
) -> ResearchCampaignSelectionDecision:
    """Authoritative API to load a persisted ResearchCampaignSelectionDecision artifact."""
    if store is None:
        store = ResearchCampaignStore()
    return store.load_selection_decision(campaign_id)


def validate_campaign_selection_integrity(
    synthesis: ResearchCampaignEvidenceSynthesis,
    decision: Optional[ResearchCampaignSelectionDecision] = None,
    store: Optional[ResearchCampaignStore] = None,
) -> None:
    """Authoritative API to validate synthesis and selection decision integrity."""
    CampaignSelectionIntegrityValidator.validate_synthesis_integrity(synthesis, store=store)
    if decision is not None:
        CampaignSelectionIntegrityValidator.validate_selection_decision_integrity(decision, synthesis=synthesis)
