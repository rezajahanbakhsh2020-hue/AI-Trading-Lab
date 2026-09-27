"""Canonical Research Pattern Knowledge abstraction and derivation engine for Project 1.

Transforms canonical research learning memory (ResearchLearningRecord) into deterministic,
evidence-backed knowledge about recurring success and failure patterns.

Design Principles:
- Observational knowledge layer only. Does NOT mutate strategy code, live execution, risk rules,
  promotion gates, or Project2 configuration.
- Retains complete provenance to supporting ResearchLearningRecord and ResearchRegistryRecord identities.
- Deterministic, reproducible, versioned, and auditable SHA-256 fingerprinting.
- Explicit contradiction tracking when historical evidence yields conflicting outcomes for identical conditions.
- Strict condition normalization without arbitrary fuzzy matching or probabilistic guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import enum
import hashlib
import json
from typing import Any, Mapping, Sequence

from src.evaluation.research_registry import (
    RegistryConflictError,
    RegistryError,
    RegistryValidationError,
    ResearchLearningRecord,
    ResearchOutcomeClassification,
    SchemaVersionError,
    StructuredObservedConditions,
    SCHEMA_VERSION_1_0,
)

KNOWLEDGE_METHODOLOGY_VERSION_1_0 = "knowledge_v1.0"


class ResearchPatternCategory(str, enum.Enum):
    """Authoritative categories for derived research knowledge patterns."""

    SUCCESS_PATTERN = "SUCCESS_PATTERN"
    FAILURE_PATTERN = "FAILURE_PATTERN"
    INCONCLUSIVE_PATTERN = "INCONCLUSIVE_PATTERN"
    GENERAL_OBSERVATION = "GENERAL_OBSERVATION"


class PatternDerivationError(RegistryError, ValueError):
    """Raised when pattern derivation encounters invalid, missing, or inconsistent learning memory."""


@dataclass(frozen=True)
class PatternObservationSummary:
    """Normalized, machine-readable summary of observed conditions shared across a pattern."""

    symbol: str | None
    timeframe: str | None
    strategy_name: str | None
    strategy_version: str | None
    dataset_scope_id: str
    execution_assumptions_id: str
    code_provenance_id: str
    methodology_version: str
    schema_version: str = SCHEMA_VERSION_1_0

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "dataset_scope_id": self.dataset_scope_id,
            "execution_assumptions_id": self.execution_assumptions_id,
            "code_provenance_id": self.code_provenance_id,
            "methodology_version": self.methodology_version,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PatternObservationSummary:
        if not isinstance(data, dict):
            raise RegistryValidationError("PatternObservationSummary data must be a dictionary.")
        ver = data.get("schema_version")
        if ver != SCHEMA_VERSION_1_0:
            raise SchemaVersionError(f"Unsupported summary schema version: '{ver}'. Expected '{SCHEMA_VERSION_1_0}'.")
        return cls(
            symbol=data.get("symbol"),
            timeframe=data.get("timeframe"),
            strategy_name=data.get("strategy_name"),
            strategy_version=data.get("strategy_version"),
            dataset_scope_id=data.get("dataset_scope_id", ""),
            execution_assumptions_id=data.get("execution_assumptions_id", ""),
            code_provenance_id=data.get("code_provenance_id", ""),
            methodology_version=data.get("methodology_version", ""),
            schema_version=ver,
        )


@dataclass(frozen=True)
class ResearchKnowledgePattern:
    """Canonical, immutable domain artifact representing a derived research knowledge pattern."""

    pattern_id: str
    category: ResearchPatternCategory
    statement: str
    normalized_conditions: PatternObservationSummary
    supporting_learning_ids: tuple[str, ...]
    supporting_experiment_fingerprints: tuple[str, ...]
    supporting_evidence_fingerprints: tuple[str, ...]
    observation_count: int
    success_count: int
    failure_count: int
    inconclusive_count: int
    is_contradictory: bool
    methodology_version: str = KNOWLEDGE_METHODOLOGY_VERSION_1_0
    schema_version: str = SCHEMA_VERSION_1_0
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    superseded_by_pattern_id: str | None = None
    supersedes_pattern_id: str | None = None
    is_active: bool = True

    def __post_init__(self) -> None:
        if not self.pattern_id or not self.pattern_id.strip():
            raise RegistryValidationError("pattern_id must be a non-empty string.")
        if not self.statement or not self.statement.strip():
            raise RegistryValidationError("statement must be a non-empty string.")
        if self.schema_version != SCHEMA_VERSION_1_0:
            raise SchemaVersionError(
                f"Unsupported pattern schema version: '{self.schema_version}'. Expected '{SCHEMA_VERSION_1_0}'."
            )
        if not isinstance(self.category, ResearchPatternCategory):
            if isinstance(self.category, str) and self.category in ResearchPatternCategory.__members__:
                object.__setattr__(self, "category", ResearchPatternCategory(self.category))
            else:
                raise RegistryValidationError(f"Invalid ResearchPatternCategory: '{self.category}'.")
        if self.observation_count <= 0:
            raise RegistryValidationError("observation_count must be greater than zero.")
        if not self.supporting_learning_ids:
            raise RegistryValidationError("Pattern must contain at least one supporting_learning_id.")
        if (self.success_count + self.failure_count + self.inconclusive_count) != self.observation_count:
            raise RegistryValidationError(
                f"Count mismatch in pattern '{self.pattern_id}': "
                f"success({self.success_count}) + failure({self.failure_count}) + inconclusive({self.inconclusive_count}) "
                f"!= total({self.observation_count})."
            )

    @property
    def semantic_content(self) -> dict[str, Any]:
        """Return canonical dictionary representation of semantic content for fingerprinting."""
        return {
            "schema_version": self.schema_version,
            "category": self.category.value,
            "statement": self.statement,
            "normalized_conditions": self.normalized_conditions.as_dict(),
            "supporting_learning_ids": sorted(self.supporting_learning_ids),
            "supporting_experiment_fingerprints": sorted(self.supporting_experiment_fingerprints),
            "supporting_evidence_fingerprints": sorted(self.supporting_evidence_fingerprints),
            "observation_count": self.observation_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "inconclusive_count": self.inconclusive_count,
            "is_contradictory": self.is_contradictory,
            "methodology_version": self.methodology_version,
            "superseded_by_pattern_id": self.superseded_by_pattern_id,
            "supersedes_pattern_id": self.supersedes_pattern_id,
            "is_active": self.is_active,
        }

    @property
    def canonical_fingerprint(self) -> str:
        """Compute deterministic SHA-256 fingerprint from canonical semantic content."""
        serialized = json.dumps(self.semantic_content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        """Convert pattern to dictionary for JSON persistence."""
        res = self.semantic_content
        res["pattern_id"] = self.pattern_id
        res["created_at_utc"] = self.created_at_utc
        res["canonical_fingerprint"] = self.canonical_fingerprint
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchKnowledgePattern:
        """Reconstruct ResearchKnowledgePattern from dictionary. Fail closed on invalid data."""
        if not isinstance(data, dict):
            raise RegistryValidationError("Pattern data must be a dictionary.")

        schema_ver = data.get("schema_version")
        if schema_ver != SCHEMA_VERSION_1_0:
            raise SchemaVersionError(f"Unsupported pattern schema version: '{schema_ver}'. Expected '{SCHEMA_VERSION_1_0}'.")

        cat_str = data.get("category")
        if not cat_str or cat_str not in ResearchPatternCategory.__members__:
            raise RegistryValidationError(f"Invalid or missing category in pattern data: '{cat_str}'.")

        norm_dict = data.get("normalized_conditions")
        if not isinstance(norm_dict, dict):
            raise RegistryValidationError("Missing or invalid 'normalized_conditions' in pattern data.")

        return cls(
            pattern_id=data.get("pattern_id", ""),
            category=ResearchPatternCategory(cat_str),
            statement=data.get("statement", ""),
            normalized_conditions=PatternObservationSummary.from_dict(norm_dict),
            supporting_learning_ids=tuple(data.get("supporting_learning_ids", [])),
            supporting_experiment_fingerprints=tuple(data.get("supporting_experiment_fingerprints", [])),
            supporting_evidence_fingerprints=tuple(data.get("supporting_evidence_fingerprints", [])),
            observation_count=int(data.get("observation_count", 0)),
            success_count=int(data.get("success_count", 0)),
            failure_count=int(data.get("failure_count", 0)),
            inconclusive_count=int(data.get("inconclusive_count", 0)),
            is_contradictory=bool(data.get("is_contradictory", False)),
            methodology_version=data.get("methodology_version", KNOWLEDGE_METHODOLOGY_VERSION_1_0),
            schema_version=schema_ver,
            created_at_utc=data.get("created_at_utc", ""),
            superseded_by_pattern_id=data.get("superseded_by_pattern_id"),
            supersedes_pattern_id=data.get("supersedes_pattern_id"),
            is_active=bool(data.get("is_active", True)),
        )


@dataclass(frozen=True)
class ResearchKnowledgeDerivationPolicy:
    """Policy governing deterministic research knowledge pattern derivation."""

    min_observations: int = 2
    methodology_version: str = KNOWLEDGE_METHODOLOGY_VERSION_1_0
    fail_on_missing_record: bool = True
    allow_single_observation_patterns: bool = False


def derive_research_knowledge_patterns(
    learning_records: Sequence[ResearchLearningRecord],
    *,
    registry_store: Any | None = None,
    policy: ResearchKnowledgeDerivationPolicy | None = None,
) -> tuple[ResearchKnowledgePattern, ...]:
    """Deterministically derive evidence-backed ResearchKnowledgePattern records from learning records.

    Fail closed if any learning record is missing, malformed, or lineage-inconsistent.
    Normalizes observed conditions deterministically and represents outcome contradictions explicitly.
    """
    if not isinstance(learning_records, (list, tuple)):
        raise TypeError("learning_records must be a sequence of ResearchLearningRecord instances.")

    if not learning_records:
        return ()

    effective_policy = policy or ResearchKnowledgeDerivationPolicy()
    min_obs = (
        1 if effective_policy.allow_single_observation_patterns else effective_policy.min_observations
    )

    # 1. Sort inputs deterministically by learning_id
    sorted_learnings = sorted(
        learning_records,
        key=lambda lr: (
            getattr(lr, "learning_id", ""),
            getattr(lr, "canonical_fingerprint", ""),
        ),
    )

    # 2. Validate records and optional store provenance/lineage
    for lr in sorted_learnings:
        if not isinstance(lr, ResearchLearningRecord):
            raise PatternDerivationError(
                f"Invalid item in learning_records: expected ResearchLearningRecord, got {type(lr).__name__}."
            )

        if not lr.learning_id or not lr.learning_id.strip():
            raise PatternDerivationError("Malformed learning record: empty learning_id.")

        if registry_store is not None:
            # Check learning record existence in store
            store_lr = registry_store.get_learning_by_id(lr.learning_id, lr.experiment_fingerprint)
            if store_lr is None and effective_policy.fail_on_missing_record:
                raise PatternDerivationError(
                    f"Derivation failure: Supporting learning record '{lr.learning_id}' "
                    f"for experiment '{lr.experiment_fingerprint}' is missing from registry store."
                )

            if store_lr is not None and store_lr.canonical_fingerprint != lr.canonical_fingerprint:
                raise PatternDerivationError(
                    f"Derivation failure: Fingerprint mismatch for learning record '{lr.learning_id}'. "
                    f"Store: {store_lr.canonical_fingerprint}, Input: {lr.canonical_fingerprint}."
                )

            # Check source registry record existence and lineage in store
            source_rec = registry_store.get_by_record_id(lr.source_record_id, lr.experiment_fingerprint)
            if source_rec is None and lr.evidence_fingerprint:
                source_rec = registry_store.get_by_evidence_fingerprint(lr.evidence_fingerprint)

            if source_rec is None and effective_policy.fail_on_missing_record:
                raise PatternDerivationError(
                    f"Derivation failure: Source registry record '{lr.source_record_id}' "
                    f"for learning record '{lr.learning_id}' is missing from registry store."
                )

            if source_rec is not None:
                if source_rec.experiment_fingerprint != lr.experiment_fingerprint:
                    raise PatternDerivationError(
                        f"Lineage mismatch for learning record '{lr.learning_id}': "
                        f"Source record experiment_fingerprint '{source_rec.experiment_fingerprint}' "
                        f"does not match learning record experiment_fingerprint '{lr.experiment_fingerprint}'."
                    )

    # 3. Group by deterministic condition normalization
    groups: dict[tuple, list[ResearchLearningRecord]] = {}
    for lr in sorted_learnings:
        cond = lr.observed_conditions
        norm_key = (
            cond.strategy_name or "unknown_strategy",
            cond.symbol or "unknown_symbol",
            cond.timeframe or "unknown_tf",
            lr.dataset_scope_id,
            lr.execution_assumptions_id,
            lr.code_provenance_id,
            lr.methodology_version,
        )
        groups.setdefault(norm_key, []).append(lr)

    # 4. Process groups deterministically
    derived_patterns: list[ResearchKnowledgePattern] = []
    for norm_key in sorted(groups.keys()):
        group = groups[norm_key]
        obs_count = len(group)

        if obs_count < min_obs:
            continue

        success_count = sum(1 for r in group if r.classification == ResearchOutcomeClassification.SUCCESS)
        failure_count = sum(1 for r in group if r.classification == ResearchOutcomeClassification.FAILURE)
        inconclusive_count = sum(1 for r in group if r.classification == ResearchOutcomeClassification.INCONCLUSIVE)

        is_contradictory = (success_count > 0 and failure_count > 0)

        if is_contradictory:
            category = ResearchPatternCategory.INCONCLUSIVE_PATTERN
        elif success_count == obs_count:
            category = ResearchPatternCategory.SUCCESS_PATTERN
        elif failure_count == obs_count:
            category = ResearchPatternCategory.FAILURE_PATTERN
        elif inconclusive_count == obs_count:
            category = ResearchPatternCategory.INCONCLUSIVE_PATTERN
        else:
            category = ResearchPatternCategory.GENERAL_OBSERVATION

        sup_learning_ids = tuple(sorted(set(r.learning_id for r in group)))
        sup_exp_fps = tuple(sorted(set(r.experiment_fingerprint for r in group)))
        sup_ev_fps = tuple(sorted(set(r.evidence_fingerprint for r in group if r.evidence_fingerprint)))

        strat_name, symbol, tf, ds_id, ea_id, cp_id, meth_ver = norm_key

        if category == ResearchPatternCategory.SUCCESS_PATTERN:
            stmt = f"Repeated success pattern ({obs_count} observations) for strategy '{strat_name}' on {symbol} {tf}."
        elif category == ResearchPatternCategory.FAILURE_PATTERN:
            stmt = f"Repeated failure pattern ({obs_count} observations) for strategy '{strat_name}' on {symbol} {tf}."
        elif is_contradictory:
            stmt = (
                f"Contradictory research outcome pattern ({success_count} success, {failure_count} failure) "
                f"for strategy '{strat_name}' on {symbol} {tf}."
            )
        elif category == ResearchPatternCategory.INCONCLUSIVE_PATTERN:
            stmt = f"Repeated inconclusive research pattern ({obs_count} observations) for strategy '{strat_name}' on {symbol} {tf}."
        else:
            stmt = f"Repeated general observation pattern ({obs_count} observations) for strategy '{strat_name}' on {symbol} {tf}."

        key_str = (
            f"pattern:{category.value}:{strat_name}:{symbol}:{tf}:{ds_id[:8]}:"
            f"{':'.join(sup_learning_ids)}:{effective_policy.methodology_version}"
        )
        pattern_id = hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:24]

        first_lr = group[0]
        first_cond = first_lr.observed_conditions

        norm_summary = PatternObservationSummary(
            symbol=first_cond.symbol,
            timeframe=first_cond.timeframe,
            strategy_name=first_cond.strategy_name,
            strategy_version=first_cond.strategy_version,
            dataset_scope_id=first_lr.dataset_scope_id,
            execution_assumptions_id=first_lr.execution_assumptions_id,
            code_provenance_id=first_lr.code_provenance_id,
            methodology_version=first_lr.methodology_version,
        )

        pattern = ResearchKnowledgePattern(
            pattern_id=pattern_id,
            category=category,
            statement=stmt,
            normalized_conditions=norm_summary,
            supporting_learning_ids=sup_learning_ids,
            supporting_experiment_fingerprints=sup_exp_fps,
            supporting_evidence_fingerprints=sup_ev_fps,
            observation_count=obs_count,
            success_count=success_count,
            failure_count=failure_count,
            inconclusive_count=inconclusive_count,
            is_contradictory=is_contradictory,
            methodology_version=effective_policy.methodology_version,
        )

        derived_patterns.append(pattern)

    # Return patterns sorted deterministically by pattern_id
    derived_patterns.sort(key=lambda p: p.pattern_id)
    return tuple(derived_patterns)
