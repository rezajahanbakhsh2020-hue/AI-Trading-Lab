"""Canonical Research Discovery Feedback abstraction and evaluation engine for Project 1.

Provides controlled, auditable, deterministic knowledge feedback to DiscoveryEngine research
candidate evaluation without modifying strategy code, live execution, risk management, or promotion gates.

Design Principles:
- Inputs knowledge context to research candidate evaluation ONLY.
- Does NOT modify strategy code, live execution, risk rules, promotion gates, or Project2 publication.
- DoNotRepeatConstraint governance remains authoritative for hard blocking candidates.
- Informational knowledge patterns must NOT silently become hard constraints.
- Contradictory or inconclusive patterns remain informational uncertainty context.
- Deterministic relevance evaluation and SHA-256 fingerprinting.
- Fail-closed behavior on missing, malformed, lineage-inconsistent, or superseded knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import enum
import hashlib
import json
from typing import Any, Sequence

from src.evaluation.candidate_generator import CandidateSpec
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
    ResearchExperimentSpec,
)
from src.evaluation.research_knowledge import (
    PatternObservationSummary,
    ResearchKnowledgePattern,
    ResearchPatternCategory,
)
from src.evaluation.research_registry import (
    RegistryError,
    RegistryValidationError,
    SchemaVersionError,
    SCHEMA_VERSION_1_0,
    _compute_cp_id,
    _compute_ea_id,
    _compute_scope_id,
)

KNOWLEDGE_FEEDBACK_METHODOLOGY_VERSION_1_0 = "feedback_v1.0"


class DiscoveryFeedbackType(str, enum.Enum):
    """Authoritative feedback types derived from knowledge pattern evaluation for discovery."""

    RELEVANT_SUCCESS_PATTERN = "RELEVANT_SUCCESS_PATTERN"
    RELEVANT_FAILURE_PATTERN = "RELEVANT_FAILURE_PATTERN"
    INCONCLUSIVE_PATTERN = "INCONCLUSIVE_PATTERN"
    GENERAL_OBSERVATION = "GENERAL_OBSERVATION"
    NO_FEEDBACK = "NO_FEEDBACK"
    FAIL_CLOSED = "FAIL_CLOSED"


class DiscoveryFeedbackError(RegistryError, ValueError):
    """Raised when feedback evaluation encounters unrecoverable structural or lineage errors."""


@dataclass(frozen=True)
class ResearchDiscoveryFeedback:
    """Canonical, immutable domain artifact representing discovery feedback for a research candidate."""

    feedback_id: str
    feedback_type: DiscoveryFeedbackType
    candidate_id: str
    experiment_fingerprint: str
    pattern_id: str | None
    pattern_fingerprint: str | None
    search_id: str | None
    search_fingerprint: str | None
    supporting_learning_ids: tuple[str, ...]
    supporting_experiment_fingerprints: tuple[str, ...]
    supporting_evidence_fingerprints: tuple[str, ...]
    reason: str
    normalized_conditions: PatternObservationSummary | None
    methodology_version: str = KNOWLEDGE_FEEDBACK_METHODOLOGY_VERSION_1_0
    schema_version: str = SCHEMA_VERSION_1_0
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if not self.feedback_id or not self.feedback_id.strip():
            raise RegistryValidationError("feedback_id must be a non-empty string.")
        if not self.candidate_id or not self.candidate_id.strip():
            raise RegistryValidationError("candidate_id must be a non-empty string.")
        if self.schema_version != SCHEMA_VERSION_1_0:
            raise SchemaVersionError(
                f"Unsupported feedback schema version: '{self.schema_version}'. Expected '{SCHEMA_VERSION_1_0}'."
            )
        if not isinstance(self.feedback_type, DiscoveryFeedbackType):
            if isinstance(self.feedback_type, str) and self.feedback_type in DiscoveryFeedbackType.__members__:
                object.__setattr__(self, "feedback_type", DiscoveryFeedbackType(self.feedback_type))
            else:
                raise RegistryValidationError(f"Invalid DiscoveryFeedbackType: '{self.feedback_type}'.")

    @property
    def semantic_content(self) -> dict[str, Any]:
        """Return canonical dictionary representation of semantic content for fingerprinting."""
        return {
            "schema_version": self.schema_version,
            "feedback_type": self.feedback_type.value,
            "candidate_id": self.candidate_id,
            "experiment_fingerprint": self.experiment_fingerprint,
            "pattern_id": self.pattern_id,
            "pattern_fingerprint": self.pattern_fingerprint,
            "search_id": self.search_id,
            "search_fingerprint": self.search_fingerprint,
            "supporting_learning_ids": sorted(self.supporting_learning_ids),
            "supporting_experiment_fingerprints": sorted(self.supporting_experiment_fingerprints),
            "supporting_evidence_fingerprints": sorted(self.supporting_evidence_fingerprints),
            "reason": self.reason,
            "normalized_conditions": self.normalized_conditions.as_dict() if self.normalized_conditions else None,
            "methodology_version": self.methodology_version,
        }

    @property
    def canonical_fingerprint(self) -> str:
        """Compute deterministic SHA-256 fingerprint from canonical semantic content."""
        serialized = json.dumps(self.semantic_content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @property
    def feedback_fingerprint(self) -> str:
        """Alias for canonical_fingerprint."""
        return self.canonical_fingerprint

    def as_dict(self) -> dict[str, Any]:
        """Convert feedback to dictionary for JSON persistence."""
        res = self.semantic_content
        res["feedback_id"] = self.feedback_id
        res["created_at_utc"] = self.created_at_utc
        res["canonical_fingerprint"] = self.canonical_fingerprint
        res["feedback_fingerprint"] = self.feedback_fingerprint
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchDiscoveryFeedback:
        """Reconstruct ResearchDiscoveryFeedback from dictionary. Fail closed on invalid data."""
        if not isinstance(data, dict):
            raise RegistryValidationError("Feedback data must be a dictionary.")

        schema_ver = data.get("schema_version")
        if schema_ver != SCHEMA_VERSION_1_0:
            raise SchemaVersionError(
                f"Unsupported feedback schema version: '{schema_ver}'. Expected '{SCHEMA_VERSION_1_0}'."
            )

        type_str = data.get("feedback_type")
        if not type_str or type_str not in DiscoveryFeedbackType.__members__:
            raise RegistryValidationError(f"Invalid or missing feedback_type in data: '{type_str}'.")

        norm_dict = data.get("normalized_conditions")
        norm_summary = PatternObservationSummary.from_dict(norm_dict) if isinstance(norm_dict, dict) else None

        return cls(
            feedback_id=data.get("feedback_id", ""),
            feedback_type=DiscoveryFeedbackType(type_str),
            candidate_id=data.get("candidate_id", ""),
            experiment_fingerprint=data.get("experiment_fingerprint", ""),
            pattern_id=data.get("pattern_id"),
            pattern_fingerprint=data.get("pattern_fingerprint"),
            search_id=data.get("search_id"),
            search_fingerprint=data.get("search_fingerprint"),
            supporting_learning_ids=tuple(data.get("supporting_learning_ids", [])),
            supporting_experiment_fingerprints=tuple(data.get("supporting_experiment_fingerprints", [])),
            supporting_evidence_fingerprints=tuple(data.get("supporting_evidence_fingerprints", [])),
            reason=data.get("reason", ""),
            normalized_conditions=norm_summary,
            methodology_version=data.get("methodology_version", KNOWLEDGE_FEEDBACK_METHODOLOGY_VERSION_1_0),
            schema_version=schema_ver,
            created_at_utc=data.get("created_at_utc", ""),
        )


def _map_category_to_feedback_type(category: ResearchPatternCategory) -> DiscoveryFeedbackType:
    if category == ResearchPatternCategory.SUCCESS_PATTERN:
        return DiscoveryFeedbackType.RELEVANT_SUCCESS_PATTERN
    elif category == ResearchPatternCategory.FAILURE_PATTERN:
        return DiscoveryFeedbackType.RELEVANT_FAILURE_PATTERN
    elif category == ResearchPatternCategory.INCONCLUSIVE_PATTERN:
        return DiscoveryFeedbackType.INCONCLUSIVE_PATTERN
    elif category == ResearchPatternCategory.GENERAL_OBSERVATION:
        return DiscoveryFeedbackType.GENERAL_OBSERVATION
    return DiscoveryFeedbackType.GENERAL_OBSERVATION


def evaluate_candidate_discovery_feedback(
    *,
    candidate: CandidateSpec,
    dataset_scope: DatasetScope,
    execution_assumptions: ExecutionAssumptions,
    code_provenance: CodeProvenance,
    knowledge_patterns: Sequence[ResearchKnowledgePattern],
    search_id: str | None = None,
    search_fingerprint: str | None = None,
    registry_store: Any | None = None,
    methodology_version: str = KNOWLEDGE_FEEDBACK_METHODOLOGY_VERSION_1_0,
) -> tuple[ResearchDiscoveryFeedback, ...]:
    """Perform deterministic, fail-closed discovery feedback evaluation for a research candidate.

    Evaluates candidate parameters and context against canonical ResearchKnowledgePattern records.
    Returns a tuple of ResearchDiscoveryFeedback objects sorted deterministically by pattern_id.
    Fail closed if any knowledge pattern or supporting lineage is malformed, missing, or inconsistent.
    """
    if not isinstance(candidate, CandidateSpec):
        raise TypeError("candidate must be a CandidateSpec instance.")
    if not isinstance(dataset_scope, DatasetScope):
        raise TypeError("dataset_scope must be a DatasetScope instance.")
    if not isinstance(execution_assumptions, ExecutionAssumptions):
        raise TypeError("execution_assumptions must be an ExecutionAssumptions instance.")
    if not isinstance(code_provenance, CodeProvenance):
        raise TypeError("code_provenance must be a CodeProvenance instance.")

    hypothesis_stmt = (
        candidate.hypothesis_template.replace("{candidate_id}", candidate.candidate_id)
        if candidate.hypothesis_template
        else f"Hypothesis for candidate {candidate.candidate_id}"
    )

    spec = ResearchExperimentSpec(
        hypothesis=hypothesis_stmt,
        methodology_version="discovery_v1.0",
        strategy_name=candidate.strategy_name,
        strategy_version="1.0.0",
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        benchmark_reference="buy_and_hold",
        parameters=dict(candidate.parameters),
        random_seed=candidate.random_seed,
    )
    cand_exp_fp = spec.fingerprint

    cand_id = candidate.candidate_id
    cand_strat = candidate.strategy_name
    cand_symbol = dataset_scope.symbol
    cand_tf = dataset_scope.timeframe
    cand_ds_id = _compute_scope_id(dataset_scope)
    cand_ea_id = _compute_ea_id(execution_assumptions)
    cand_cp_id = _compute_cp_id(code_provenance)

    if not knowledge_patterns:
        fb_key = f"no_feedback:{cand_id}:{cand_exp_fp}:{methodology_version}"
        fb_id = hashlib.sha256(fb_key.encode("utf-8")).hexdigest()[:24]
        no_fb = ResearchDiscoveryFeedback(
            feedback_id=fb_id,
            feedback_type=DiscoveryFeedbackType.NO_FEEDBACK,
            candidate_id=cand_id,
            experiment_fingerprint=cand_exp_fp,
            pattern_id=None,
            pattern_fingerprint=None,
            search_id=search_id,
            search_fingerprint=search_fingerprint,
            supporting_learning_ids=(),
            supporting_experiment_fingerprints=(),
            supporting_evidence_fingerprints=(),
            reason="No knowledge patterns available for discovery feedback.",
            normalized_conditions=None,
            methodology_version=methodology_version,
        )
        return (no_fb,)

    # Fail closed checks on patterns and registry store
    # 1. Structural check on patterns sequence
    if not isinstance(knowledge_patterns, (list, tuple)):
        fb_key = f"fail_closed:{cand_id}:{cand_exp_fp}:invalid_sequence"
        fb_id = hashlib.sha256(fb_key.encode("utf-8")).hexdigest()[:24]
        return (
            ResearchDiscoveryFeedback(
                feedback_id=fb_id,
                feedback_type=DiscoveryFeedbackType.FAIL_CLOSED,
                candidate_id=cand_id,
                experiment_fingerprint=cand_exp_fp,
                pattern_id=None,
                pattern_fingerprint=None,
                search_id=search_id,
                search_fingerprint=search_fingerprint,
                supporting_learning_ids=(),
                supporting_experiment_fingerprints=(),
                supporting_evidence_fingerprints=(),
                reason="Invalid knowledge_patterns sequence type.",
                normalized_conditions=None,
                methodology_version=methodology_version,
            ),
        )

    # Sort patterns deterministically by pattern_id or canonical_fingerprint
    try:
        sorted_patterns = sorted(
            knowledge_patterns,
            key=lambda p: (
                getattr(p, "pattern_id", ""),
                getattr(p, "canonical_fingerprint", ""),
            ),
        )
    except Exception as exc:
        fb_key = f"fail_closed:{cand_id}:{cand_exp_fp}:sorting_error"
        fb_id = hashlib.sha256(fb_key.encode("utf-8")).hexdigest()[:24]
        return (
            ResearchDiscoveryFeedback(
                feedback_id=fb_id,
                feedback_type=DiscoveryFeedbackType.FAIL_CLOSED,
                candidate_id=cand_id,
                experiment_fingerprint=cand_exp_fp,
                pattern_id=None,
                pattern_fingerprint=None,
                search_id=search_id,
                search_fingerprint=search_fingerprint,
                supporting_learning_ids=(),
                supporting_experiment_fingerprints=(),
                supporting_evidence_fingerprints=(),
                reason=f"Failed to process knowledge patterns: {exc}",
                normalized_conditions=None,
                methodology_version=methodology_version,
            ),
        )

    feedbacks: list[ResearchDiscoveryFeedback] = []

    for pattern in sorted_patterns:
        # Validate pattern instance integrity
        if not isinstance(pattern, ResearchKnowledgePattern):
            fb_key = f"fail_closed:{cand_id}:{cand_exp_fp}:not_pattern_instance"
            fb_id = hashlib.sha256(fb_key.encode("utf-8")).hexdigest()[:24]
            return (
                ResearchDiscoveryFeedback(
                    feedback_id=fb_id,
                    feedback_type=DiscoveryFeedbackType.FAIL_CLOSED,
                    candidate_id=cand_id,
                    experiment_fingerprint=cand_exp_fp,
                    pattern_id=None,
                    pattern_fingerprint=None,
                    search_id=search_id,
                    search_fingerprint=search_fingerprint,
                    supporting_learning_ids=(),
                    supporting_experiment_fingerprints=(),
                    supporting_evidence_fingerprints=(),
                    reason=f"Invalid pattern item: expected ResearchKnowledgePattern, got {type(pattern).__name__}.",
                    normalized_conditions=None,
                    methodology_version=methodology_version,
                ),
            )

        # Validate pattern field sanity
        if not pattern.pattern_id or not pattern.pattern_id.strip():
            fb_key = f"fail_closed:{cand_id}:{cand_exp_fp}:empty_pattern_id"
            fb_id = hashlib.sha256(fb_key.encode("utf-8")).hexdigest()[:24]
            return (
                ResearchDiscoveryFeedback(
                    feedback_id=fb_id,
                    feedback_type=DiscoveryFeedbackType.FAIL_CLOSED,
                    candidate_id=cand_id,
                    experiment_fingerprint=cand_exp_fp,
                    pattern_id=None,
                    pattern_fingerprint=None,
                    search_id=search_id,
                    search_fingerprint=search_fingerprint,
                    supporting_learning_ids=(),
                    supporting_experiment_fingerprints=(),
                    supporting_evidence_fingerprints=(),
                    reason="Malformed knowledge pattern: empty pattern_id.",
                    normalized_conditions=None,
                    methodology_version=methodology_version,
                ),
            )

        # Skip inactive or superseded patterns without failing closed
        if not pattern.is_active or pattern.superseded_by_pattern_id is not None:
            continue

        # If registry store is provided, validate pattern & lineage against store
        if registry_store is not None:
            store_pattern = registry_store.get_pattern_by_id(pattern.pattern_id)
            if store_pattern is None:
                fb_key = f"fail_closed:{cand_id}:{cand_exp_fp}:{pattern.pattern_id}:missing"
                fb_id = hashlib.sha256(fb_key.encode("utf-8")).hexdigest()[:24]
                return (
                    ResearchDiscoveryFeedback(
                        feedback_id=fb_id,
                        feedback_type=DiscoveryFeedbackType.FAIL_CLOSED,
                        candidate_id=cand_id,
                        experiment_fingerprint=cand_exp_fp,
                        pattern_id=pattern.pattern_id,
                        pattern_fingerprint=pattern.canonical_fingerprint,
                        search_id=search_id,
                        search_fingerprint=search_fingerprint,
                        supporting_learning_ids=pattern.supporting_learning_ids,
                        supporting_experiment_fingerprints=pattern.supporting_experiment_fingerprints,
                        supporting_evidence_fingerprints=pattern.supporting_evidence_fingerprints,
                        reason=f"Knowledge pattern '{pattern.pattern_id}' is missing from registry store.",
                        normalized_conditions=pattern.normalized_conditions,
                        methodology_version=methodology_version,
                    ),
                )

            if store_pattern.canonical_fingerprint != pattern.canonical_fingerprint:
                fb_key = f"fail_closed:{cand_id}:{cand_exp_fp}:{pattern.pattern_id}:mismatch"
                fb_id = hashlib.sha256(fb_key.encode("utf-8")).hexdigest()[:24]
                return (
                    ResearchDiscoveryFeedback(
                        feedback_id=fb_id,
                        feedback_type=DiscoveryFeedbackType.FAIL_CLOSED,
                        candidate_id=cand_id,
                        experiment_fingerprint=cand_exp_fp,
                        pattern_id=pattern.pattern_id,
                        pattern_fingerprint=pattern.canonical_fingerprint,
                        search_id=search_id,
                        search_fingerprint=search_fingerprint,
                        supporting_learning_ids=pattern.supporting_learning_ids,
                        supporting_experiment_fingerprints=pattern.supporting_experiment_fingerprints,
                        supporting_evidence_fingerprints=pattern.supporting_evidence_fingerprints,
                        reason=f"Fingerprint mismatch for pattern '{pattern.pattern_id}'. Store: {store_pattern.canonical_fingerprint}, Input: {pattern.canonical_fingerprint}.",
                        normalized_conditions=pattern.normalized_conditions,
                        methodology_version=methodology_version,
                    ),
                )

            # Validate supporting learning record lineage
            for lid in pattern.supporting_learning_ids:
                found_lr = False
                for exp_fp in pattern.supporting_experiment_fingerprints:
                    lr = registry_store.get_learning_by_id(lid, exp_fp)
                    if lr is not None:
                        found_lr = True
                        break
                if not found_lr:
                    # Search entire learning store
                    all_lrs = registry_store.list_learning_records()
                    for lr in all_lrs:
                        if lr.learning_id == lid:
                            found_lr = True
                            break
                if not found_lr:
                    fb_key = f"fail_closed:{cand_id}:{cand_exp_fp}:{pattern.pattern_id}:missing_learning"
                    fb_id = hashlib.sha256(fb_key.encode("utf-8")).hexdigest()[:24]
                    return (
                        ResearchDiscoveryFeedback(
                            feedback_id=fb_id,
                            feedback_type=DiscoveryFeedbackType.FAIL_CLOSED,
                            candidate_id=cand_id,
                            experiment_fingerprint=cand_exp_fp,
                            pattern_id=pattern.pattern_id,
                            pattern_fingerprint=pattern.canonical_fingerprint,
                            search_id=search_id,
                            search_fingerprint=search_fingerprint,
                            supporting_learning_ids=pattern.supporting_learning_ids,
                            supporting_experiment_fingerprints=pattern.supporting_experiment_fingerprints,
                            supporting_evidence_fingerprints=pattern.supporting_evidence_fingerprints,
                            reason=f"Supporting learning record '{lid}' for pattern '{pattern.pattern_id}' is missing from registry store.",
                            normalized_conditions=pattern.normalized_conditions,
                            methodology_version=methodology_version,
                        ),
                    )

        # Check Deterministic Relevance against pattern's normalized conditions
        cond = pattern.normalized_conditions

        # Field 1: strategy_name
        if cond.strategy_name and cond.strategy_name != cand_strat:
            continue

        # Field 2: symbol
        if cond.symbol and cond.symbol != cand_symbol:
            continue

        # Field 3: timeframe
        if cond.timeframe and cond.timeframe != cand_tf:
            continue

        # Field 4: dataset_scope_id
        if cond.dataset_scope_id and cond.dataset_scope_id != cand_ds_id:
            # Allow prefix match or exact match
            if not (cand_ds_id.startswith(cond.dataset_scope_id[:8]) or cond.dataset_scope_id.startswith(cand_ds_id[:8])):
                continue

        # Field 5: execution_assumptions_id
        if cond.execution_assumptions_id and cond.execution_assumptions_id != cand_ea_id:
            if not (cand_ea_id.startswith(cond.execution_assumptions_id[:8]) or cond.execution_assumptions_id.startswith(cand_ea_id[:8])):
                continue

        # Field 6: code_provenance_id
        if cond.code_provenance_id and cond.code_provenance_id != cand_cp_id:
            if not (cand_cp_id.startswith(cond.code_provenance_id[:8]) or cond.code_provenance_id.startswith(cand_cp_id[:8])):
                continue

        # Relevant pattern matched!
        fb_type = _map_category_to_feedback_type(pattern.category)
        fb_reason = (
            f"Candidate '{cand_id}' matched active knowledge pattern '{pattern.pattern_id}' "
            f"({pattern.category.value}): {pattern.statement}"
        )

        fb_key = f"feedback:{cand_id}:{cand_exp_fp}:{pattern.pattern_id}:{methodology_version}"
        fb_id = hashlib.sha256(fb_key.encode("utf-8")).hexdigest()[:24]

        feedback_item = ResearchDiscoveryFeedback(
            feedback_id=fb_id,
            feedback_type=fb_type,
            candidate_id=cand_id,
            experiment_fingerprint=cand_exp_fp,
            pattern_id=pattern.pattern_id,
            pattern_fingerprint=pattern.canonical_fingerprint,
            search_id=search_id,
            search_fingerprint=search_fingerprint,
            supporting_learning_ids=pattern.supporting_learning_ids,
            supporting_experiment_fingerprints=pattern.supporting_experiment_fingerprints,
            supporting_evidence_fingerprints=pattern.supporting_evidence_fingerprints,
            reason=fb_reason,
            normalized_conditions=pattern.normalized_conditions,
            methodology_version=methodology_version,
        )
        feedbacks.append(feedback_item)

    if not feedbacks:
        fb_key = f"no_feedback:{cand_id}:{cand_exp_fp}:{methodology_version}"
        fb_id = hashlib.sha256(fb_key.encode("utf-8")).hexdigest()[:24]
        no_fb = ResearchDiscoveryFeedback(
            feedback_id=fb_id,
            feedback_type=DiscoveryFeedbackType.NO_FEEDBACK,
            candidate_id=cand_id,
            experiment_fingerprint=cand_exp_fp,
            pattern_id=None,
            pattern_fingerprint=None,
            search_id=search_id,
            search_fingerprint=search_fingerprint,
            supporting_learning_ids=(),
            supporting_experiment_fingerprints=(),
            supporting_evidence_fingerprints=(),
            reason=f"Candidate '{cand_id}' did not match any active knowledge pattern.",
            normalized_conditions=None,
            methodology_version=methodology_version,
        )
        return (no_fb,)

    # Return feedbacks sorted deterministically by feedback_id
    feedbacks.sort(key=lambda fb: fb.feedback_id)
    return tuple(feedbacks)
