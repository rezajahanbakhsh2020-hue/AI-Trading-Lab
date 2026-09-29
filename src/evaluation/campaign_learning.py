"""Governed Campaign Research-Learning Lifecycle & Feedback Loop Service.

Establishes the missing, governed research feedback loop connecting completed
Research Campaigns -> Campaign Evidence Synthesis -> Campaign Selection Decision
-> Governed Research Learning Artifact -> Research Knowledge Patterns
-> Governed Research Hypotheses (GENERATED) -> Explicit ACCEPTED_FOR_RESEARCH.

Explicit Disclaimers & Governance Rules:
- This service operates exclusively within RESEARCH / DISCOVERY memory.
- Research memory can inform future hypothesis generation; it can NEVER establish production authority.
- This feedback loop does NOT create ProductionRuntimeAuthorization, create live trading signals,
  mutate strategy source code or production parameters, execute live trades, or publish Project 2 payloads.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
import enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.evaluation.campaign_synthesis import (
    CampaignSelectionIntegrityError,
    CampaignSelectionIntegrityValidator,
    CampaignSelectionStatus,
    ResearchCampaignEvidenceSynthesis,
    ResearchCampaignSelectionDecision,
)
from src.evaluation.hypothesis_generator import (
    HypothesisGenerationContext,
    HypothesisGenerationError,
    KnowledgeHypothesisGenerator,
    KnowledgeHypothesisGeneratorPolicy,
)
from src.evaluation.research_constitution import (
    HypothesisStatus,
    ResearchCampaignStatus,
    ResearchHypothesis,
)
from src.evaluation.research_knowledge import (
    ResearchKnowledgeDerivationPolicy,
    ResearchKnowledgePattern,
    derive_research_knowledge_patterns,
)
from src.evaluation.research_registry import (
    RegistryConflictError,
    RegistryValidationError,
    ResearchLearningRecord,
    ResearchRegistryStore,
    construct_learning_record_from_registry_record,
)
from src.evaluation.research_store import ResearchCampaignStore


def _canonical_json_dumps(data: Any) -> str:
    """Helper to convert data structures to canonical JSON for SHA-256 hashing."""
    def _normalize(obj: Any) -> Any:
        if isinstance(obj, enum.Enum):
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


def _canonical_tuple(items: Sequence[str] | None, field_name: str = "field") -> tuple[str, ...]:
    """Returns a deterministic, sorted, deduplicated tuple of strings.

    Raises CampaignLearningIntegrityError if raw input contains duplicate identifiers.
    """
    if items is None:
        return ()
    cleaned = [str(x).strip() for x in items if x is not None and str(x).strip()]
    if len(cleaned) != len(set(cleaned)):
        raise CampaignLearningIntegrityError(f"{field_name} contains invalid duplicate identifiers.")
    return tuple(sorted(set(cleaned)))


class CampaignLearningIntegrityError(ValueError):
    """Raised when campaign learning artifact or feedback-loop lineage fails integrity validation."""
    pass


@dataclass(frozen=True)
class GovernedCampaignLearningArtifact:
    """Canonical, immutable domain artifact representing learning derived from a validated Campaign Selection Decision.

    Binds campaign selection decision, evidence synthesis, selected candidates, supporting evidence,
    robustness, qualification, registry learning records, and derived knowledge patterns into one
    fingerprintable, reproducible research memory artifact.
    """

    campaign_id: str
    campaign_selection_decision_fingerprint: str
    synthesis_fingerprint: str
    selected_candidate_ids: tuple[str, ...]
    supporting_evidence_fingerprints: tuple[str, ...]
    robustness_fingerprints: tuple[str, ...]
    qualification_fingerprints: tuple[str, ...]
    learning_record_ids: tuple[str, ...]
    knowledge_pattern_ids: tuple[str, ...]
    methodology_version: str = "1.0"
    learning_artifact_id: str = field(default="", init=False)
    artifact_fingerprint: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if not self.campaign_id or not self.campaign_id.strip():
            raise CampaignLearningIntegrityError("campaign_id must be a non-empty string.")
        if not self.campaign_selection_decision_fingerprint:
            raise CampaignLearningIntegrityError("campaign_selection_decision_fingerprint is required.")
        if not self.synthesis_fingerprint:
            raise CampaignLearningIntegrityError("synthesis_fingerprint is required.")

        object.__setattr__(self, "selected_candidate_ids", _canonical_tuple(self.selected_candidate_ids, "selected_candidate_ids"))
        object.__setattr__(self, "supporting_evidence_fingerprints", _canonical_tuple(self.supporting_evidence_fingerprints, "supporting_evidence_fingerprints"))
        object.__setattr__(self, "robustness_fingerprints", _canonical_tuple(self.robustness_fingerprints, "robustness_fingerprints"))
        object.__setattr__(self, "qualification_fingerprints", _canonical_tuple(self.qualification_fingerprints, "qualification_fingerprints"))
        object.__setattr__(self, "learning_record_ids", _canonical_tuple(self.learning_record_ids, "learning_record_ids"))
        object.__setattr__(self, "knowledge_pattern_ids", _canonical_tuple(self.knowledge_pattern_ids, "knowledge_pattern_ids"))

        payload = {
            "campaign_id": self.campaign_id,
            "campaign_selection_decision_fingerprint": self.campaign_selection_decision_fingerprint,
            "synthesis_fingerprint": self.synthesis_fingerprint,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "supporting_evidence_fingerprints": list(self.supporting_evidence_fingerprints),
            "robustness_fingerprints": list(self.robustness_fingerprints),
            "qualification_fingerprints": list(self.qualification_fingerprints),
            "learning_record_ids": list(self.learning_record_ids),
            "knowledge_pattern_ids": list(self.knowledge_pattern_ids),
            "methodology_version": self.methodology_version,
        }
        fp = compute_sha256_fingerprint(payload)
        object.__setattr__(self, "artifact_fingerprint", fp)

        art_id_key = f"campaign_learning:{self.campaign_id}:{fp}"
        art_id = hashlib.sha256(art_id_key.encode("utf-8")).hexdigest()[:24]
        object.__setattr__(self, "learning_artifact_id", art_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "campaign_selection_decision_fingerprint": self.campaign_selection_decision_fingerprint,
            "synthesis_fingerprint": self.synthesis_fingerprint,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "supporting_evidence_fingerprints": list(self.supporting_evidence_fingerprints),
            "robustness_fingerprints": list(self.robustness_fingerprints),
            "qualification_fingerprints": list(self.qualification_fingerprints),
            "learning_record_ids": list(self.learning_record_ids),
            "knowledge_pattern_ids": list(self.knowledge_pattern_ids),
            "methodology_version": self.methodology_version,
            "learning_artifact_id": self.learning_artifact_id,
            "artifact_fingerprint": self.artifact_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GovernedCampaignLearningArtifact:
        if not isinstance(data, dict):
            raise CampaignLearningIntegrityError("Artifact data must be a dictionary.")

        art = cls(
            campaign_id=data.get("campaign_id", ""),
            campaign_selection_decision_fingerprint=data.get("campaign_selection_decision_fingerprint", ""),
            synthesis_fingerprint=data.get("synthesis_fingerprint", ""),
            selected_candidate_ids=tuple(data.get("selected_candidate_ids", [])),
            supporting_evidence_fingerprints=tuple(data.get("supporting_evidence_fingerprints", [])),
            robustness_fingerprints=tuple(data.get("robustness_fingerprints", [])),
            qualification_fingerprints=tuple(data.get("qualification_fingerprints", [])),
            learning_record_ids=tuple(data.get("learning_record_ids", [])),
            knowledge_pattern_ids=tuple(data.get("knowledge_pattern_ids", [])),
            methodology_version=data.get("methodology_version", "1.0"),
        )

        expected_fp = data.get("artifact_fingerprint")
        if expected_fp and art.artifact_fingerprint != expected_fp:
            raise CampaignLearningIntegrityError(
                f"Artifact fingerprint mismatch: recomputed '{art.artifact_fingerprint}', got '{expected_fp}'."
            )
        return art


class CampaignLearningIntegrityValidator:
    """Validator enforcing strict fail-closed integrity rules for campaign learning artifacts."""

    @staticmethod
    def validate_campaign_learning_integrity(
        artifact: GovernedCampaignLearningArtifact,
        store: Optional[ResearchCampaignStore] = None,
        registry_store: Optional[ResearchRegistryStore] = None,
    ) -> None:
        """Validates structural, lifecycle, and lineage integrity for a GovernedCampaignLearningArtifact.

        Fails closed on any broken link, missing file, decision mismatch, or cross-campaign contamination.
        """
        if store is None:
            store = ResearchCampaignStore()

        # 1. Re-compute fingerprint check
        payload = {
            "campaign_id": artifact.campaign_id,
            "campaign_selection_decision_fingerprint": artifact.campaign_selection_decision_fingerprint,
            "synthesis_fingerprint": artifact.synthesis_fingerprint,
            "selected_candidate_ids": list(artifact.selected_candidate_ids),
            "supporting_evidence_fingerprints": list(artifact.supporting_evidence_fingerprints),
            "robustness_fingerprints": list(artifact.robustness_fingerprints),
            "qualification_fingerprints": list(artifact.qualification_fingerprints),
            "learning_record_ids": list(artifact.learning_record_ids),
            "knowledge_pattern_ids": list(artifact.knowledge_pattern_ids),
            "methodology_version": artifact.methodology_version,
        }
        recomputed_fp = compute_sha256_fingerprint(payload)
        if recomputed_fp != artifact.artifact_fingerprint:
            raise CampaignLearningIntegrityError(
                f"Learning artifact fingerprint mismatch for campaign '{artifact.campaign_id}': "
                f"recomputed '{recomputed_fp}', got '{artifact.artifact_fingerprint}'."
            )

        # 2. Check campaign status (must be COMPLETED or TRUNCATED according to durable state)
        try:
            state_data = store.load_lifecycle_state(artifact.campaign_id)
            status_str = state_data.get("status")
            if status_str not in (
                ResearchCampaignStatus.COMPLETED.value,
                ResearchCampaignStatus.TRUNCATED.value,
            ):
                raise CampaignLearningIntegrityError(
                    f"Campaign '{artifact.campaign_id}' is not in a completed lifecycle state "
                    f"(current status: '{status_str}')."
                )
        except Exception as exc:
            if isinstance(exc, CampaignLearningIntegrityError):
                raise
            raise CampaignLearningIntegrityError(
                f"Failed to load campaign state for '{artifact.campaign_id}': {exc}"
            ) from exc

        # 3. Load and validate evidence synthesis
        try:
            synthesis = store.load_evidence_synthesis(artifact.campaign_id)
            CampaignSelectionIntegrityValidator.validate_synthesis_integrity(synthesis, store=store)
        except Exception as exc:
            raise CampaignLearningIntegrityError(
                f"Evidence synthesis validation failed for campaign '{artifact.campaign_id}': {exc}"
            ) from exc

        if synthesis.synthesis_fingerprint != artifact.synthesis_fingerprint:
            raise CampaignLearningIntegrityError(
                f"Synthesis fingerprint mismatch in learning artifact: "
                f"artifact has '{artifact.synthesis_fingerprint}', store synthesis has '{synthesis.synthesis_fingerprint}'."
            )

        # 4. Load and validate selection decision
        try:
            decision = store.load_selection_decision(artifact.campaign_id)
            CampaignSelectionIntegrityValidator.validate_selection_decision_integrity(
                decision, synthesis=synthesis
            )
        except Exception as exc:
            raise CampaignLearningIntegrityError(
                f"Selection decision validation failed for campaign '{artifact.campaign_id}': {exc}"
            ) from exc

        if decision.decision_fingerprint != artifact.campaign_selection_decision_fingerprint:
            raise CampaignLearningIntegrityError(
                f"Selection decision fingerprint mismatch in learning artifact: "
                f"artifact has '{artifact.campaign_selection_decision_fingerprint}', "
                f"store decision has '{decision.decision_fingerprint}'."
            )

        if decision.campaign_id != artifact.campaign_id:
            raise CampaignLearningIntegrityError(
                f"Selection decision campaign_id '{decision.campaign_id}' does not match "
                f"artifact campaign_id '{artifact.campaign_id}'."
            )

        # 5. Validate duplicate identities in artifact and selection decision
        for field_name, items in [
            ("selected_candidate_ids", artifact.selected_candidate_ids),
            ("supporting_evidence_fingerprints", artifact.supporting_evidence_fingerprints),
            ("robustness_fingerprints", artifact.robustness_fingerprints),
            ("qualification_fingerprints", artifact.qualification_fingerprints),
        ]:
            if len(items) != len(set(items)):
                raise CampaignLearningIntegrityError(f"Artifact {field_name} contains invalid duplicate identifiers.")

        for field_name, items in [
            ("selected_candidate_ids", decision.selected_candidate_ids),
            ("evidence_fingerprints", decision.evidence_fingerprints),
            ("robustness_fingerprints", decision.robustness_fingerprints),
            ("qualification_fingerprints", decision.qualification_fingerprints),
        ]:
            if len(items) != len(set(items)):
                raise CampaignLearningIntegrityError(f"Selection decision {field_name} contains invalid duplicate identifiers.")

        # 6. Assert exact lineage equality between learning artifact and authoritative selection decision
        if _canonical_tuple(artifact.selected_candidate_ids) != _canonical_tuple(decision.selected_candidate_ids):
            raise CampaignLearningIntegrityError(
                f"Artifact selected_candidate_ids {_canonical_tuple(artifact.selected_candidate_ids)} "
                f"does not match decision selected_candidate_ids {_canonical_tuple(decision.selected_candidate_ids)}."
            )

        if _canonical_tuple(artifact.supporting_evidence_fingerprints) != _canonical_tuple(decision.evidence_fingerprints):
            raise CampaignLearningIntegrityError(
                f"Artifact supporting_evidence_fingerprints {_canonical_tuple(artifact.supporting_evidence_fingerprints)} "
                f"does not match decision evidence_fingerprints {_canonical_tuple(decision.evidence_fingerprints)}."
            )

        if _canonical_tuple(artifact.robustness_fingerprints) != _canonical_tuple(decision.robustness_fingerprints):
            raise CampaignLearningIntegrityError(
                f"Artifact robustness_fingerprints {_canonical_tuple(artifact.robustness_fingerprints)} "
                f"does not match decision robustness_fingerprints {_canonical_tuple(decision.robustness_fingerprints)}."
            )

        if _canonical_tuple(artifact.qualification_fingerprints) != _canonical_tuple(decision.qualification_fingerprints):
            raise CampaignLearningIntegrityError(
                f"Artifact qualification_fingerprints {_canonical_tuple(artifact.qualification_fingerprints)} "
                f"does not match decision qualification_fingerprints {_canonical_tuple(decision.qualification_fingerprints)}."
            )

        # 6. Fail closed if selection decision status is blocked, insufficient, or unresolved tie
        if decision.decision_status != CampaignSelectionStatus.SELECTED.value:
            raise CampaignLearningIntegrityError(
                f"Cannot derive learning artifact from non-SELECTED campaign decision status: '{decision.decision_status}'."
            )

        if not decision.selected_candidate_ids:
            raise CampaignLearningIntegrityError(
                f"Campaign selection decision for '{artifact.campaign_id}' has no selected candidate IDs."
            )

        # Check selected candidates exist in campaign synthesis candidate comparisons
        comp_map = {c.candidate_id: c for c in synthesis.candidate_comparisons}
        for sel_id in artifact.selected_candidate_ids:
            if sel_id not in comp_map:
                raise CampaignLearningIntegrityError(
                    f"Selected candidate ID '{sel_id}' not found in campaign '{artifact.campaign_id}' candidate comparisons."
                )

        if registry_store is None:
            registry_store = ResearchRegistryStore()

        # Check learning records existence and campaign lineage
        for lid in artifact.learning_record_ids:
            found = False
            for lr in registry_store.list_learning_records():
                if lr.learning_id == lid:
                    found = True
                    break
            if not found:
                raise CampaignLearningIntegrityError(
                    f"Referenced learning record ID '{lid}' not found in registry store."
                )

        # Check knowledge patterns existence
        for pid in artifact.knowledge_pattern_ids:
            if registry_store.get_pattern_by_id(pid) is None:
                raise CampaignLearningIntegrityError(
                    f"Referenced knowledge pattern ID '{pid}' not found in registry store."
                )


# =============================================================================
# Authoritative Feedback-Loop API Functions
# =============================================================================

def _resolve_materialization_learning_records(
    learning_records: Sequence[ResearchLearningRecord],
    selected_candidate_ids: Sequence[str],
) -> tuple[ResearchLearningRecord, ...]:
    """Validates and filters explicitly presented learning records for materialization.

    Fail-closed contract:
    - Every resolved learning record MUST have a non-empty candidate_id.
    - Every resolved learning record MUST belong to a candidate in selected_candidate_ids.
    - If any presented record has a missing/ambiguous candidate_id or belongs to a non-selected candidate,
      raises CampaignLearningIntegrityError.
    """
    selected_set = set(selected_candidate_ids)
    validated: list[ResearchLearningRecord] = []

    for lr in learning_records:
        cid = getattr(lr, "candidate_id", None)
        if not cid or not str(cid).strip():
            raise CampaignLearningIntegrityError(
                f"Learning record '{lr.learning_id}' has missing or ambiguous candidate_id."
            )

        cid_str = str(cid).strip()
        if cid_str not in selected_set:
            raise CampaignLearningIntegrityError(
                f"Learning record '{lr.learning_id}' belongs to candidate '{cid_str}', "
                f"which is not present in authoritative selected candidate IDs {selected_set}."
            )

        validated.append(lr)

    return tuple(validated)


def derive_governed_campaign_learning(
    campaign_id: str,
    store: Optional[ResearchCampaignStore] = None,
    registry_store: Optional[ResearchRegistryStore] = None,
    methodology_version: str = "1.0",
) -> GovernedCampaignLearningArtifact:
    """Authoritatively derives a GovernedCampaignLearningArtifact from a validated, completed campaign decision.

    Fails closed if the campaign is incomplete, decision is not SELECTED, or lineage validation fails.
    Idempotent and deterministic.
    """
    if store is None:
        store = ResearchCampaignStore()
    if registry_store is None:
        registry_store = ResearchRegistryStore()

    # 1. Load durable campaign lifecycle state, synthesis, and decision
    state_data = store.load_lifecycle_state(campaign_id)
    status_str = state_data.get("status")
    if status_str not in (
        ResearchCampaignStatus.COMPLETED.value,
        ResearchCampaignStatus.TRUNCATED.value,
    ):
        raise CampaignLearningIntegrityError(
            f"Cannot derive learning artifact: Campaign '{campaign_id}' is in status '{status_str}', expected COMPLETED/TRUNCATED."
        )

    synthesis = store.load_evidence_synthesis(campaign_id)
    CampaignSelectionIntegrityValidator.validate_synthesis_integrity(synthesis, store=store)

    decision = store.load_selection_decision(campaign_id)
    CampaignSelectionIntegrityValidator.validate_selection_decision_integrity(decision, synthesis=synthesis)

    if decision.decision_status != CampaignSelectionStatus.SELECTED.value:
        raise CampaignLearningIntegrityError(
            f"Cannot derive learning artifact: Campaign selection decision for '{campaign_id}' "
            f"has status '{decision.decision_status}', expected '{CampaignSelectionStatus.SELECTED.value}'."
        )

    if not decision.selected_candidate_ids:
        raise CampaignLearningIntegrityError(
            f"Cannot derive learning artifact: Campaign selection decision for '{campaign_id}' has no selected candidates."
        )

    # 2. Gather registry records strictly for selected candidates in this campaign
    try:
        checkpoints = store.list_trial_checkpoints(campaign_id)
    except Exception as exc:
        raise CampaignLearningIntegrityError(
            f"Cannot derive learning artifact: Failed to list trial checkpoints for campaign '{campaign_id}': {exc}"
        ) from exc

    learning_records: list[ResearchLearningRecord] = []
    selected_cands_set = set(decision.selected_candidate_ids)

    for cp in checkpoints:
        if not cp.candidate_id or not cp.candidate_id.strip():
            raise CampaignLearningIntegrityError(
                f"Trial checkpoint '{cp.trial_id}' in campaign '{campaign_id}' has missing or ambiguous candidate_id."
            )

        # Ordinary non-selected checkpoints may exist in the campaign and are cleanly skipped
        if cp.candidate_id not in selected_cands_set:
            continue

        if cp.status in ("COMPLETED", "QUALIFIED", "REJECTED") and cp.evidence_fingerprint:
            reg_rec = registry_store.get_by_evidence_fingerprint(cp.evidence_fingerprint)
            if reg_rec is None and cp.experiment_fingerprint:
                # Fallback load experiment evidence if not registered in registry store
                ev_path = Path(store.base_dir) / campaign_id / f"{cp.trial_id}_evidence.json"
                if not ev_path.exists():
                    ev_path = Path(store.base_dir).parent / "research_experiments" / cp.experiment_fingerprint / "evidence.json"
                if ev_path.exists():
                    from src.evaluation.research_store import load_research_experiment
                    from src.evaluation.research_registry import construct_registry_record_from_evidence
                    try:
                        ev = load_research_experiment(ev_path)
                        reg_rec = construct_registry_record_from_evidence(
                            evidence=ev, candidate_id=cp.candidate_id, trial_id=cp.trial_id, qualification_status=cp.qualification_status or "QUALIFIED"
                        )
                        registry_store.register(reg_rec)
                    except RegistryConflictError:
                        reg_rec = registry_store.get_by_evidence_fingerprint(cp.evidence_fingerprint)
                    except Exception as exc:
                        raise CampaignLearningIntegrityError(
                            "Campaign learning materialization failed while resolving "
                            f"evidence/registry lineage for campaign '{campaign_id}': {exc}"
                        ) from exc

            if reg_rec is not None:
                try:
                    learnings = registry_store.query_learning(
                        experiment_fingerprint=reg_rec.experiment_fingerprint,
                        evidence_fingerprint=reg_rec.evidence_fingerprint,
                    )
                    if learnings:
                        lr = learnings[0]
                    else:
                        lr = construct_learning_record_from_registry_record(reg_rec)
                        try:
                            registry_store.register_learning_record(lr)
                        except RegistryConflictError:
                            pass

                    learning_records.append(lr)
                except Exception as exc:
                    if isinstance(exc, CampaignLearningIntegrityError):
                        raise
                    raise CampaignLearningIntegrityError(
                        "Campaign learning materialization failed while resolving "
                        f"learning records for campaign '{campaign_id}': {exc}"
                    ) from exc

    # Validate resolved learning records against authoritative selected candidate IDs
    validated_learning_records = _resolve_materialization_learning_records(
        learning_records=learning_records,
        selected_candidate_ids=decision.selected_candidate_ids,
    )

    # 3. Derive research knowledge patterns from validated learning records
    knowledge_patterns = derive_research_knowledge_patterns(
        validated_learning_records,
        registry_store=registry_store,
        policy=ResearchKnowledgeDerivationPolicy(allow_single_observation_patterns=True),
    )

    registered_patterns = []
    for pat in knowledge_patterns:
        try:
            reg_pat = registry_store.register_pattern(pat)
            registered_patterns.append(reg_pat)
        except RegistryConflictError:
            p_existing = registry_store.get_pattern_by_id(pat.pattern_id)
            if p_existing:
                registered_patterns.append(p_existing)
            else:
                raise CampaignLearningIntegrityError(
                    f"Conflicting pattern '{pat.pattern_id}' could not be resolved from registry store."
                )
        except Exception as exc:
            raise CampaignLearningIntegrityError(
                f"Failed to register derived knowledge pattern '{pat.pattern_id}' for campaign '{campaign_id}': {exc}"
            ) from exc

    sup_ev_fps = _canonical_tuple(decision.evidence_fingerprints)
    rob_fps = _canonical_tuple(decision.robustness_fingerprints)
    qual_fps = _canonical_tuple(decision.qualification_fingerprints)
    lr_ids = _canonical_tuple(lr.learning_id for lr in validated_learning_records)
    kp_ids = _canonical_tuple(p.pattern_id for p in registered_patterns)

    artifact = GovernedCampaignLearningArtifact(
        campaign_id=campaign_id,
        campaign_selection_decision_fingerprint=decision.decision_fingerprint,
        synthesis_fingerprint=synthesis.synthesis_fingerprint,
        selected_candidate_ids=_canonical_tuple(decision.selected_candidate_ids),
        supporting_evidence_fingerprints=sup_ev_fps,
        robustness_fingerprints=rob_fps,
        qualification_fingerprints=qual_fps,
        learning_record_ids=lr_ids,
        knowledge_pattern_ids=kp_ids,
        methodology_version=methodology_version,
    )

    CampaignLearningIntegrityValidator.validate_campaign_learning_integrity(
        artifact, store=store, registry_store=registry_store
    )

    return artifact


def register_governed_campaign_learning(
    artifact: GovernedCampaignLearningArtifact,
    store: Optional[ResearchCampaignStore] = None,
    registry_store: Optional[ResearchRegistryStore] = None,
) -> GovernedCampaignLearningArtifact:
    """Registers a GovernedCampaignLearningArtifact idempotently and atomically in store."""
    if store is None:
        store = ResearchCampaignStore()

    CampaignLearningIntegrityValidator.validate_campaign_learning_integrity(
        artifact, store=store, registry_store=registry_store
    )

    try:
        store.save_campaign_learning(artifact)
    except FileExistsError as exc:
        raise CampaignLearningIntegrityError(
            f"Conflicting campaign learning artifact already exists for campaign '{artifact.campaign_id}': {exc}"
        ) from exc
    return artifact


def materialize_governed_campaign_feedback(
    campaign_id: str,
    *,
    store: Optional[ResearchCampaignStore] = None,
    registry_store: Optional[ResearchRegistryStore] = None,
    methodology_version: str = "1.0",
) -> GovernedCampaignLearningArtifact:
    """Orchestrates authoritative, idempotent materialization of campaign feedback into research memory.

    Responsibilities:
    load campaign -> validate synthesis -> validate selection decision -> resolve authoritative learning records
    -> derive/register knowledge -> construct canonical learning artifact -> validate complete artifact -> persist idempotently -> return artifact.

    Does NOT execute experiments, accept hypotheses, promote candidates, or modify production state.
    """
    if store is None:
        store = ResearchCampaignStore()
    if registry_store is None:
        registry_store = ResearchRegistryStore()

    artifact = derive_governed_campaign_learning(
        campaign_id=campaign_id,
        store=store,
        registry_store=registry_store,
        methodology_version=methodology_version,
    )

    register_governed_campaign_learning(
        artifact,
        store=store,
        registry_store=registry_store,
    )

    return artifact


def generate_hypotheses_from_campaign_learning(
    campaign_id: str,
    context: HypothesisGenerationContext,
    store: Optional[ResearchCampaignStore] = None,
    registry_store: Optional[ResearchRegistryStore] = None,
    generator_policy: Optional[KnowledgeHypothesisGeneratorPolicy] = None,
) -> tuple[ResearchHypothesis, ...]:
    """Generates future ResearchHypothesis objects from validated campaign learning memory.

    DOES NOT automatically accept hypotheses or execute research.
    All generated hypotheses start strictly in status GENERATED.
    Integrates complete campaign lineage into hypothesis constraints.
    """
    if store is None:
        store = ResearchCampaignStore()
    if registry_store is None:
        registry_store = ResearchRegistryStore()

    # Load or derive campaign learning artifact
    try:
        learning_artifact = store.load_campaign_learning(campaign_id)
    except FileNotFoundError:
        learning_artifact = derive_governed_campaign_learning(
            campaign_id=campaign_id,
            store=store,
            registry_store=registry_store,
        )
        store.save_campaign_learning(learning_artifact)

    CampaignLearningIntegrityValidator.validate_campaign_learning_integrity(
        learning_artifact, store=store, registry_store=registry_store
    )

    # Load patterns referenced by learning artifact
    patterns: list[ResearchKnowledgePattern] = []
    for pid in learning_artifact.knowledge_pattern_ids:
        pat = registry_store.get_pattern_by_id(pid)
        if pat is None:
            raise CampaignLearningIntegrityError(
                f"Referenced knowledge pattern '{pid}' missing from registry store."
            )
        patterns.append(pat)

    if not patterns:
        raise HypothesisGenerationError(
            f"No active knowledge patterns found for campaign learning artifact '{learning_artifact.learning_artifact_id}'."
        )

    # Attach campaign lineage to context overrides
    enhanced_params = dict(context.parameters_override)
    enhanced_params["campaign_id"] = campaign_id
    enhanced_params["campaign_selection_decision_fingerprint"] = learning_artifact.campaign_selection_decision_fingerprint
    enhanced_params["synthesis_fingerprint"] = learning_artifact.synthesis_fingerprint
    enhanced_params["campaign_learning_artifact_fingerprint"] = learning_artifact.artifact_fingerprint

    enhanced_context = HypothesisGenerationContext(
        dataset_scope=context.dataset_scope,
        execution_assumptions=context.execution_assumptions,
        code_provenance=context.code_provenance,
        benchmark_reference=context.benchmark_reference,
        parameters_override=enhanced_params,
        walk_forward_protocol=context.walk_forward_protocol,
        random_seed=context.random_seed,
    )

    generator = KnowledgeHypothesisGenerator(policy=generator_policy)
    generated = generator.generate(patterns, context=enhanced_context)

    # Verify all generated hypotheses are strictly in GENERATED status and contain campaign lineage
    verified_hypotheses = []
    for h in generated:
        if h.status != HypothesisStatus.GENERATED:
            raise CampaignLearningIntegrityError(
                f"Generated hypothesis '{h.hypothesis_id}' is not in GENERATED status (got '{h.status.value}')."
            )

        constraints = dict(h.constraints)
        constraints["campaign_id"] = campaign_id
        constraints["campaign_selection_decision_fingerprint"] = learning_artifact.campaign_selection_decision_fingerprint
        constraints["synthesis_fingerprint"] = learning_artifact.synthesis_fingerprint
        constraints["campaign_learning_artifact_fingerprint"] = learning_artifact.artifact_fingerprint

        updated_h = ResearchHypothesis(
            statement=h.statement,
            methodology_version=h.methodology_version,
            strategy_name=h.strategy_name,
            strategy_version=h.strategy_version,
            dataset_scope=h.dataset_scope,
            execution_assumptions=h.execution_assumptions,
            code_provenance=h.code_provenance,
            benchmark_reference=h.benchmark_reference,
            parameters=dict(h.parameters),
            random_seed=h.random_seed,
            walk_forward_protocol=h.walk_forward_protocol,
            source_knowledge_ids=h.source_knowledge_ids,
            source_evidence_ids=h.source_evidence_ids,
            hypothesis_version=h.hypothesis_version,
            generation_method=h.generation_method,
            generator_version=h.generator_version,
            constraints=constraints,
            status=HypothesisStatus.GENERATED,
            created_at_utc=h.created_at_utc,
        )
        registry_store.register_hypothesis(updated_h)
        verified_hypotheses.append(updated_h)

    return tuple(verified_hypotheses)


def validate_feedback_loop_lineage(
    campaign_id: str,
    hypothesis: ResearchHypothesis,
    store: Optional[ResearchCampaignStore] = None,
    registry_store: Optional[ResearchRegistryStore] = None,
) -> None:
    """Validates complete, unbroken feedback-loop lineage from ResearchHypothesis back to Campaign.

    Fails closed if any lineage link (campaign, decision, synthesis, learning, knowledge, evidence) is missing,
    mismatched, or corrupted.
    """
    if store is None:
        store = ResearchCampaignStore()
    if registry_store is None:
        registry_store = ResearchRegistryStore()

    if not isinstance(hypothesis, ResearchHypothesis):
        raise TypeError("hypothesis must be a ResearchHypothesis instance.")

    # 1. Check campaign and artifact lineage in hypothesis constraints
    h_cid = hypothesis.constraints.get("campaign_id")
    if h_cid != campaign_id:
        raise CampaignLearningIntegrityError(
            f"Hypothesis campaign_id mismatch: hypothesis has '{h_cid}', expected '{campaign_id}'."
        )

    h_dec_fp = hypothesis.constraints.get("campaign_selection_decision_fingerprint")
    if not h_dec_fp:
        raise CampaignLearningIntegrityError("Hypothesis missing required campaign_selection_decision_fingerprint constraint.")

    h_syn_fp = hypothesis.constraints.get("synthesis_fingerprint")
    if not h_syn_fp:
        raise CampaignLearningIntegrityError("Hypothesis missing required synthesis_fingerprint constraint.")

    h_art_fp = hypothesis.constraints.get("campaign_learning_artifact_fingerprint")
    if not h_art_fp:
        raise CampaignLearningIntegrityError("Hypothesis missing required campaign_learning_artifact_fingerprint constraint.")

    # 2. Validate durable campaign learning artifact, decision, and synthesis
    learning_artifact = store.load_campaign_learning(campaign_id)
    if learning_artifact.artifact_fingerprint != h_art_fp:
        raise CampaignLearningIntegrityError(
            f"Campaign learning artifact fingerprint mismatch: hypothesis has '{h_art_fp}', store artifact has '{learning_artifact.artifact_fingerprint}'."
        )

    CampaignLearningIntegrityValidator.validate_campaign_learning_integrity(
        learning_artifact, store=store, registry_store=registry_store
    )

    if learning_artifact.campaign_selection_decision_fingerprint != h_dec_fp:
        raise CampaignLearningIntegrityError(
            f"Selection decision fingerprint mismatch between hypothesis '{h_dec_fp}' and learning artifact '{learning_artifact.campaign_selection_decision_fingerprint}'."
        )

    if learning_artifact.synthesis_fingerprint != h_syn_fp:
        raise CampaignLearningIntegrityError(
            f"Synthesis fingerprint mismatch between hypothesis '{h_syn_fp}' and learning artifact '{learning_artifact.synthesis_fingerprint}'."
        )

    synthesis = store.load_evidence_synthesis(campaign_id)
    CampaignSelectionIntegrityValidator.validate_synthesis_integrity(synthesis, store=store)

    decision = store.load_selection_decision(campaign_id)
    CampaignSelectionIntegrityValidator.validate_selection_decision_integrity(decision, synthesis=synthesis)

    # 3. Check source knowledge lineage and cross-campaign contamination
    if not hypothesis.source_knowledge_ids:
        raise CampaignLearningIntegrityError("Hypothesis missing source_knowledge_ids.")

    art_pattern_ids = set(learning_artifact.knowledge_pattern_ids)
    for k_id in hypothesis.source_knowledge_ids:
        if k_id not in art_pattern_ids:
            raise CampaignLearningIntegrityError(
                f"Referenced knowledge pattern '{k_id}' in hypothesis lineage does not belong to campaign '{campaign_id}' learning artifact."
            )
        pat = registry_store.get_pattern_by_id(k_id)
        if pat is None:
            raise CampaignLearningIntegrityError(
                f"Referenced knowledge pattern '{k_id}' in hypothesis lineage does not exist in registry store."
            )

    # 4. Check source evidence lineage
    if not hypothesis.source_evidence_ids:
        raise CampaignLearningIntegrityError("Hypothesis missing source_evidence_ids.")

    synthesis_evs = set(synthesis.ordered_evidence_fingerprints)
    for ev_id in hypothesis.source_evidence_ids:
        if ev_id not in synthesis_evs and ev_id not in decision.evidence_fingerprints:
            raise CampaignLearningIntegrityError(
                f"Referenced source evidence '{ev_id}' in hypothesis lineage does not belong to campaign '{campaign_id}'."
            )
