"""Canonical Research Evidence Registry, Experiment Memory, and Reproducible Lineage layer for Project 1.

Provides observational, persistent research memory for completed research artifacts.
Allows the system to answer, from authoritative persisted records:
- What experiments have actually been run?
- Which experiment produced this evidence?
- Which candidate/trial/search produced it?
- Which dataset scope, execution assumptions, and code provenance were used?
- Which evidence fingerprint identifies the result?
- Was the evidence qualified, selection-assessed, and robustness-assessed?
- What benchmark/regime evidence was available?
- Was the evidence promoted?
- What happened to rejected/failed/insufficient research?
- Has the exact experiment/evidence already been evaluated?
- Can a previous result be located deterministically without rerunning research?

This layer is OBSERVATIONAL/PERSISTENT RESEARCH MEMORY.
It does NOT make production decisions, bind candidates, calculate production risk,
publish live signals, execute trades, or rerun backtests.

Explicit Disclaimers:
- The registry does NOT prove that a strategy is profitable.
- The registry does NOT prove that a strategy is production-safe.
- The registry does NOT eliminate overfitting.
- The registry does NOT replace qualification.
- The registry does NOT replace selection-bias governance.
- The registry does NOT replace robustness evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import enum
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
    PromotionStatus,
    RejectionReason,
    ResearchEvidence,
    ResearchExperimentSpec,
)
from src.evaluation.research_robustness import (
    ResearchRobustnessAssessment,
    RobustnessStatus,
)
from src.evaluation.selection_governance import (
    ResearchSelectionAssessment,
)

DEFAULT_REGISTRY_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "research_registry"
)

SCHEMA_VERSION_1_0 = "1.0"
RECORD_FILENAME = "record.json"
INDEX_FILENAME = "index.json"


class RegistryStatus(str, enum.Enum):
    """Execution/evaluation status of a research trial represented in the registry."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    QUALIFIED = "QUALIFIED"
    REJECTED = "REJECTED"
    INSUFFICIENT = "INSUFFICIENT"
    UNAVAILABLE = "UNAVAILABLE"


class ResearchOutcomeClassification(str, enum.Enum):
    """Authoritative research outcome classification derived deterministically from evidence."""

    SUCCESS = "SUCCESS"
    POSITIVE_EXPERIENCE = "SUCCESS"  # Alias
    FAILURE = "FAILURE"
    NEGATIVE_EXPERIENCE = "FAILURE"  # Alias
    INCONCLUSIVE = "INCONCLUSIVE"


class LessonCategory(str, enum.Enum):
    """Structured categories for research learning lessons."""

    STRATEGY_PERFORMANCE = "STRATEGY_PERFORMANCE"
    PARAMETER_SENSITIVITY = "PARAMETER_SENSITIVITY"
    EXECUTION_FRICTION = "EXECUTION_FRICTION"
    GOVERNANCE_REJECTION = "GOVERNANCE_REJECTION"
    DATASET_INSUFFICIENCY = "DATASET_INSUFFICIENCY"
    ROBUSTNESS_FAILURE = "ROBUSTNESS_FAILURE"
    GENERAL_OBSERVATION = "GENERAL_OBSERVATION"


class RegistryError(Exception):
    """Base exception for research registry errors."""


class RegistryConflictError(RegistryError, FileExistsError):
    """Raised when registering a record with duplicate identity but conflicting content."""


class SchemaVersionError(RegistryError, ValueError):
    """Raised when encountering an unknown or unsupported schema version."""


class RegistryValidationError(RegistryError, ValueError):
    """Raised when registry record structure or content is malformed or invalid."""


@dataclass(frozen=True)
class ResearchReproducibilityDescriptor:
    """Descriptor capturing all parameters required to reconstruct what was evaluated."""

    experiment_fingerprint: str
    evidence_fingerprint: str | None
    dataset_scope_id: str
    execution_assumptions_id: str
    code_provenance_id: str
    methodology_version: str
    search_space_fingerprint: str | None
    trial_id: str | None
    candidate_id: str | None
    schema_version: str = SCHEMA_VERSION_1_0

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_fingerprint": self.experiment_fingerprint,
            "evidence_fingerprint": self.evidence_fingerprint,
            "dataset_scope_id": self.dataset_scope_id,
            "execution_assumptions_id": self.execution_assumptions_id,
            "code_provenance_id": self.code_provenance_id,
            "methodology_version": self.methodology_version,
            "search_space_fingerprint": self.search_space_fingerprint,
            "trial_id": self.trial_id,
            "candidate_id": self.candidate_id,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchReproducibilityDescriptor:
        if not isinstance(data, dict):
            raise RegistryValidationError("Descriptor data must be a dictionary.")
        ver = data.get("schema_version")
        if ver != SCHEMA_VERSION_1_0:
            raise SchemaVersionError(f"Unsupported descriptor schema version: '{ver}'. Expected '{SCHEMA_VERSION_1_0}'.")
        return cls(
            experiment_fingerprint=data.get("experiment_fingerprint", ""),
            evidence_fingerprint=data.get("evidence_fingerprint"),
            dataset_scope_id=data.get("dataset_scope_id", ""),
            execution_assumptions_id=data.get("execution_assumptions_id", ""),
            code_provenance_id=data.get("code_provenance_id", ""),
            methodology_version=data.get("methodology_version", ""),
            search_space_fingerprint=data.get("search_space_fingerprint"),
            trial_id=data.get("trial_id"),
            candidate_id=data.get("candidate_id"),
            schema_version=ver,
        )


@dataclass(frozen=True)
class ResearchEvidenceLineage:
    """Lineage provenance chain tracing an evidence artifact back to its evaluation context."""

    search_id: str | None
    search_fingerprint: str | None
    trial_id: str | None
    trial_index: int | None
    candidate_id: str | None
    experiment_fingerprint: str
    evidence_fingerprint: str | None
    qualification_status: str
    selection_assessment_id: str | None
    robustness_assessment_id: str | None
    promotion_status: str
    schema_version: str = SCHEMA_VERSION_1_0

    def as_dict(self) -> dict[str, Any]:
        return {
            "search_id": self.search_id,
            "search_fingerprint": self.search_fingerprint,
            "trial_id": self.trial_id,
            "trial_index": self.trial_index,
            "candidate_id": self.candidate_id,
            "experiment_fingerprint": self.experiment_fingerprint,
            "evidence_fingerprint": self.evidence_fingerprint,
            "qualification_status": self.qualification_status,
            "selection_assessment_id": self.selection_assessment_id,
            "robustness_assessment_id": self.robustness_assessment_id,
            "promotion_status": self.promotion_status,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchEvidenceLineage:
        if not isinstance(data, dict):
            raise RegistryValidationError("Lineage data must be a dictionary.")
        ver = data.get("schema_version")
        if ver != SCHEMA_VERSION_1_0:
            raise SchemaVersionError(f"Unsupported lineage schema version: '{ver}'. Expected '{SCHEMA_VERSION_1_0}'.")
        return cls(
            search_id=data.get("search_id"),
            search_fingerprint=data.get("search_fingerprint"),
            trial_id=data.get("trial_id"),
            trial_index=data.get("trial_index"),
            candidate_id=data.get("candidate_id"),
            experiment_fingerprint=data.get("experiment_fingerprint", ""),
            evidence_fingerprint=data.get("evidence_fingerprint"),
            qualification_status=data.get("qualification_status", ""),
            selection_assessment_id=data.get("selection_assessment_id"),
            robustness_assessment_id=data.get("robustness_assessment_id"),
            promotion_status=data.get("promotion_status", ""),
            schema_version=ver,
        )


@dataclass(frozen=True)
class ResearchRegistryRecord:
    """Canonical, immutable registry record of a completed or attempted research trial."""

    record_id: str
    experiment_fingerprint: str
    evidence_fingerprint: str | None
    candidate_id: str | None
    search_fingerprint: str | None
    search_id: str | None
    trial_id: str | None
    trial_index: int | None
    status: RegistryStatus
    qualification_status: str
    promotion_status: str
    rejection_reasons: tuple[str, ...]
    dataset_scope_id: str
    execution_assumptions_id: str
    code_provenance_id: str
    methodology_version: str
    selection_assessment_id: str | None
    robustness_assessment_id: str | None
    benchmark_status: str | None
    regime_status: str | None
    error_message: str
    reproducibility: ResearchReproducibilityDescriptor
    lineage: ResearchEvidenceLineage
    evidence_payload: dict[str, Any] | None = None
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: str = SCHEMA_VERSION_1_0

    def __post_init__(self) -> None:
        if not self.record_id or not self.record_id.strip():
            raise RegistryValidationError("record_id must be a non-empty string.")
        if not self.experiment_fingerprint or not self.experiment_fingerprint.strip():
            raise RegistryValidationError("experiment_fingerprint must be a non-empty string.")
        if self.schema_version != SCHEMA_VERSION_1_0:
            raise SchemaVersionError(
                f"Unsupported registry record schema version: '{self.schema_version}'. "
                f"Expected '{SCHEMA_VERSION_1_0}'."
            )
        if not isinstance(self.status, RegistryStatus):
            if isinstance(self.status, str) and self.status in RegistryStatus.__members__:
                object.__setattr__(self, "status", RegistryStatus(self.status))
            else:
                raise RegistryValidationError(f"Invalid RegistryStatus: '{self.status}'.")

    @property
    def semantic_content(self) -> dict[str, Any]:
        """Return canonical dictionary representation of semantic content for fingerprinting."""
        clean_evidence_payload = None
        if self.evidence_payload is not None:
            clean_evidence_payload = {
                k: v for k, v in self.evidence_payload.items() if k != "created_at_utc"
            }

        return {
            "schema_version": self.schema_version,
            "experiment_fingerprint": self.experiment_fingerprint,
            "evidence_fingerprint": self.evidence_fingerprint,
            "candidate_id": self.candidate_id,
            "search_fingerprint": self.search_fingerprint,
            "search_id": self.search_id,
            "trial_id": self.trial_id,
            "trial_index": self.trial_index,
            "status": self.status.value,
            "qualification_status": self.qualification_status,
            "promotion_status": self.promotion_status,
            "rejection_reasons": sorted(self.rejection_reasons),
            "dataset_scope_id": self.dataset_scope_id,
            "execution_assumptions_id": self.execution_assumptions_id,
            "code_provenance_id": self.code_provenance_id,
            "methodology_version": self.methodology_version,
            "selection_assessment_id": self.selection_assessment_id,
            "robustness_assessment_id": self.robustness_assessment_id,
            "benchmark_status": self.benchmark_status,
            "regime_status": self.regime_status,
            "error_message": self.error_message,
            "reproducibility": self.reproducibility.as_dict(),
            "lineage": self.lineage.as_dict(),
            "evidence_payload": clean_evidence_payload,
        }

    @property
    def canonical_fingerprint(self) -> str:
        """Compute deterministic SHA-256 fingerprint from canonical semantic content."""
        serialized = json.dumps(self.semantic_content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        """Convert record to dictionary for JSON persistence."""
        res = self.semantic_content
        res["record_id"] = self.record_id
        res["created_at_utc"] = self.created_at_utc
        res["canonical_fingerprint"] = self.canonical_fingerprint
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchRegistryRecord:
        """Reconstruct a ResearchRegistryRecord from a dictionary. Fail closed on invalid data."""
        if not isinstance(data, dict):
            raise RegistryValidationError("Record data must be a dictionary.")

        schema_ver = data.get("schema_version")
        if schema_ver != SCHEMA_VERSION_1_0:
            raise SchemaVersionError(
                f"Unsupported record schema version: '{schema_ver}'. Expected '{SCHEMA_VERSION_1_0}'."
            )

        status_str = data.get("status")
        if not status_str or status_str not in RegistryStatus.__members__:
            raise RegistryValidationError(f"Invalid or missing status in record data: '{status_str}'.")

        repro_dict = data.get("reproducibility")
        if not isinstance(repro_dict, dict):
            raise RegistryValidationError("Missing or invalid 'reproducibility' dictionary in record data.")

        lineage_dict = data.get("lineage")
        if not isinstance(lineage_dict, dict):
            raise RegistryValidationError("Missing or invalid 'lineage' dictionary in record data.")

        rejection_reasons = tuple(data.get("rejection_reasons", []))

        return cls(
            record_id=data.get("record_id", ""),
            experiment_fingerprint=data.get("experiment_fingerprint", ""),
            evidence_fingerprint=data.get("evidence_fingerprint"),
            candidate_id=data.get("candidate_id"),
            search_fingerprint=data.get("search_fingerprint"),
            search_id=data.get("search_id"),
            trial_id=data.get("trial_id"),
            trial_index=data.get("trial_index"),
            status=RegistryStatus(status_str),
            qualification_status=data.get("qualification_status", ""),
            promotion_status=data.get("promotion_status", ""),
            rejection_reasons=rejection_reasons,
            dataset_scope_id=data.get("dataset_scope_id", ""),
            execution_assumptions_id=data.get("execution_assumptions_id", ""),
            code_provenance_id=data.get("code_provenance_id", ""),
            methodology_version=data.get("methodology_version", ""),
            selection_assessment_id=data.get("selection_assessment_id"),
            robustness_assessment_id=data.get("robustness_assessment_id"),
            benchmark_status=data.get("benchmark_status"),
            regime_status=data.get("regime_status"),
            error_message=data.get("error_message", ""),
            reproducibility=ResearchReproducibilityDescriptor.from_dict(repro_dict),
            lineage=ResearchEvidenceLineage.from_dict(lineage_dict),
            evidence_payload=data.get("evidence_payload"),
            created_at_utc=data.get("created_at_utc", ""),
            schema_version=schema_ver,
        )


def _compute_scope_id(scope: DatasetScope) -> str:
    serialized = f"{scope.dataset_id}|{scope.symbol}|{scope.timeframe}|{scope.start_date}|{scope.end_date}"
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _compute_ea_id(ea: ExecutionAssumptions) -> str:
    serialized = f"{ea.transaction_cost:.6f}|{ea.slippage:.6f}|{ea.latency_ms:.2f}"
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _compute_cp_id(cp: CodeProvenance) -> str:
    serialized = f"{cp.commit_sha}|{cp.repository_status}|{cp.author}"
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def construct_registry_record_from_evidence(
    *,
    evidence: ResearchEvidence,
    candidate_id: str | None = None,
    search_id: str | None = None,
    search_fingerprint: str | None = None,
    trial_id: str | None = None,
    trial_index: int | None = None,
    selection_assessment: ResearchSelectionAssessment | None = None,
    robustness_assessment: ResearchRobustnessAssessment | None = None,
    qualification_status: str | None = None,
    error_message: str = "",
) -> ResearchRegistryRecord:
    """Construct a canonical ResearchRegistryRecord from authoritative research artifacts."""
    if not isinstance(evidence, ResearchEvidence):
        raise TypeError("evidence must be a ResearchEvidence instance.")

    spec = evidence.spec
    ds_id = _compute_scope_id(spec.dataset_scope)
    ea_id = _compute_ea_id(spec.execution_assumptions)
    cp_id = _compute_cp_id(spec.code_provenance)

    # Determine status
    if error_message:
        status = RegistryStatus.FAILED
    elif RejectionReason.INSUFFICIENT_DATA in evidence.rejection_reasons or RejectionReason.INSUFFICIENT_SAMPLE in evidence.rejection_reasons:
        status = RegistryStatus.INSUFFICIENT
    elif not evidence.partitions:
        status = RegistryStatus.UNAVAILABLE
    elif evidence.promotion_status in (PromotionStatus.PROMOTABLE, PromotionStatus.VALIDATED) or qualification_status == "QUALIFIED":
        status = RegistryStatus.QUALIFIED
    elif evidence.promotion_status == PromotionStatus.REJECTED or qualification_status == "REJECTED":
        status = RegistryStatus.REJECTED
    else:
        status = RegistryStatus.COMPLETED

    qual_status = qualification_status or (
        "QUALIFIED" if evidence.promotion_status in (PromotionStatus.PROMOTABLE, PromotionStatus.VALIDATED) else "REJECTED"
    )

    rejection_reasons = tuple(r.value for r in evidence.rejection_reasons)

    sel_id = selection_assessment.selection_fingerprint if selection_assessment else None
    rob_id = robustness_assessment.robustness_fingerprint if robustness_assessment else None
    bm_status = robustness_assessment.benchmark_assessment.status.value if robustness_assessment else None
    rg_status = robustness_assessment.regime_assessment.status.value if robustness_assessment else None

    reproducibility = ResearchReproducibilityDescriptor(
        experiment_fingerprint=evidence.experiment_fingerprint,
        evidence_fingerprint=evidence.evidence_id,
        dataset_scope_id=ds_id,
        execution_assumptions_id=ea_id,
        code_provenance_id=cp_id,
        methodology_version=spec.methodology_version,
        search_space_fingerprint=search_fingerprint,
        trial_id=trial_id,
        candidate_id=candidate_id,
    )

    lineage = ResearchEvidenceLineage(
        search_id=search_id,
        search_fingerprint=search_fingerprint,
        trial_id=trial_id,
        trial_index=trial_index,
        candidate_id=candidate_id,
        experiment_fingerprint=evidence.experiment_fingerprint,
        evidence_fingerprint=evidence.evidence_id,
        qualification_status=qual_status,
        selection_assessment_id=sel_id,
        robustness_assessment_id=rob_id,
        promotion_status=evidence.promotion_status.value,
    )

    record_key = (
        f"{evidence.experiment_fingerprint}:{evidence.evidence_id}:{trial_id or 'none'}"
    )
    record_id = hashlib.sha256(record_key.encode("utf-8")).hexdigest()[:24]

    return ResearchRegistryRecord(
        record_id=record_id,
        experiment_fingerprint=evidence.experiment_fingerprint,
        evidence_fingerprint=evidence.evidence_id,
        candidate_id=candidate_id,
        search_fingerprint=search_fingerprint,
        search_id=search_id,
        trial_id=trial_id,
        trial_index=trial_index,
        status=status,
        qualification_status=qual_status,
        promotion_status=evidence.promotion_status.value,
        rejection_reasons=rejection_reasons,
        dataset_scope_id=ds_id,
        execution_assumptions_id=ea_id,
        code_provenance_id=cp_id,
        methodology_version=spec.methodology_version,
        selection_assessment_id=sel_id,
        robustness_assessment_id=rob_id,
        benchmark_status=bm_status,
        regime_status=rg_status,
        error_message=error_message,
        reproducibility=reproducibility,
        lineage=lineage,
        evidence_payload=evidence.as_dict(),
    )


class ResearchRegistryStore:
    """Persistent, atomic, idempotent research registry store."""

    def __init__(self, base_dir: str | Path = DEFAULT_REGISTRY_DIR) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _record_dir(self, experiment_fingerprint: str) -> Path:
        return self.base_dir / experiment_fingerprint

    def register(self, record: ResearchRegistryRecord) -> ResearchRegistryRecord:
        """Register a research record idempotently and atomically.

        - If identical record (same canonical_fingerprint) exists: return existing record.
        - If conflicting record (same experiment_fingerprint and trial_id/record_id but different fingerprint) exists: fail closed.
        - Otherwise, save atomically using tempfile + atomic rename.
        """
        if not isinstance(record, ResearchRegistryRecord):
            raise TypeError("record must be a ResearchRegistryRecord instance.")

        target_dir = self._record_dir(record.experiment_fingerprint)
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{record.record_id}.json"

        if file_path.exists():
            existing = self._load_record_file(file_path)
            if existing.canonical_fingerprint == record.canonical_fingerprint:
                return existing
            raise RegistryConflictError(
                f"Conflicting registry record exists for experiment '{record.experiment_fingerprint}' "
                f"and record_id '{record.record_id}'. "
                f"Existing fingerprint: {existing.canonical_fingerprint}, "
                f"New fingerprint: {record.canonical_fingerprint}."
            )

        # Validate JSON serialization before writing
        record_dict = record.as_dict()
        serialized_content = json.dumps(record_dict, indent=2, sort_keys=True)

        # Atomic write pattern: write to temp file in same filesystem, then atomic replace
        fd, temp_path = tempfile.mkstemp(dir=target_dir, prefix="rec_tmp_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(serialized_content)
            os.replace(temp_path, file_path)
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise

        return record

    def _load_record_file(self, file_path: Path) -> ResearchRegistryRecord:
        if not file_path.exists():
            raise FileNotFoundError(f"Registry record file not found: {file_path}")
        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RegistryValidationError(f"Failed to parse JSON from {file_path}: {exc}") from exc
        return ResearchRegistryRecord.from_dict(raw)

    def get_by_record_id(
        self, record_id: str, experiment_fingerprint: str
    ) -> ResearchRegistryRecord | None:
        """Retrieve a record by record_id and experiment_fingerprint."""
        file_path = self._record_dir(experiment_fingerprint) / f"{record_id}.json"
        if not file_path.exists():
            return None
        return self._load_record_file(file_path)

    def get_by_evidence_fingerprint(
        self, evidence_fingerprint: str
    ) -> ResearchRegistryRecord | None:
        """Retrieve record matching evidence_fingerprint."""
        for file_path in self.base_dir.glob("*/*.json"):
            if file_path.name.startswith("rec_tmp_"):
                continue
            rec = self._load_record_file(file_path)
            if rec.evidence_fingerprint == evidence_fingerprint:
                return rec
        return None

    def get_by_experiment_fingerprint(
        self, experiment_fingerprint: str
    ) -> tuple[ResearchRegistryRecord, ...]:
        """Retrieve all records matching experiment_fingerprint."""
        exp_dir = self._record_dir(experiment_fingerprint)
        if not exp_dir.exists():
            return ()
        records: list[ResearchRegistryRecord] = []
        for file_path in sorted(exp_dir.glob("*.json")):
            if file_path.name.startswith("rec_tmp_"):
                continue
            records.append(self._load_record_file(file_path))
        return tuple(records)

    def get_by_trial_id(self, trial_id: str) -> ResearchRegistryRecord | None:
        """Retrieve record matching trial_id."""
        for file_path in self.base_dir.glob("*/*.json"):
            if file_path.name.startswith("rec_tmp_"):
                continue
            rec = self._load_record_file(file_path)
            if rec.trial_id == trial_id:
                return rec
        return None

    def get_by_search_fingerprint(
        self, search_fingerprint: str
    ) -> tuple[ResearchRegistryRecord, ...]:
        """Retrieve all records matching search_fingerprint."""
        records: list[ResearchRegistryRecord] = []
        for file_path in sorted(self.base_dir.glob("*/*.json")):
            if file_path.name.startswith("rec_tmp_"):
                continue
            rec = self._load_record_file(file_path)
            if rec.search_fingerprint == search_fingerprint:
                records.append(rec)
        return tuple(records)

    def get_by_candidate_id(
        self, candidate_id: str
    ) -> tuple[ResearchRegistryRecord, ...]:
        """Retrieve all records matching candidate_id."""
        records: list[ResearchRegistryRecord] = []
        for file_path in sorted(self.base_dir.glob("*/*.json")):
            if file_path.name.startswith("rec_tmp_"):
                continue
            rec = self._load_record_file(file_path)
            if rec.candidate_id == candidate_id:
                records.append(rec)
        return tuple(records)

    def list_records(self) -> tuple[ResearchRegistryRecord, ...]:
        """List all persisted research registry records."""
        records: list[ResearchRegistryRecord] = []
        for file_path in sorted(self.base_dir.glob("*/*.json")):
            if file_path.name.startswith("rec_tmp_"):
                continue
            records.append(self._load_record_file(file_path))
        return tuple(records)

    def get_evidence_lineage(
        self, evidence_fingerprint: str
    ) -> ResearchEvidenceLineage | None:
        """Trace complete evidence lineage by evidence_fingerprint without re-executing research."""
        rec = self.get_by_evidence_fingerprint(evidence_fingerprint)
        if rec is None:
            return None
        return rec.lineage

    def has_experiment_been_evaluated(self, experiment_fingerprint: str) -> bool:
        """Return True if an experiment with experiment_fingerprint has already been evaluated."""
        records = self.get_by_experiment_fingerprint(experiment_fingerprint)
        return len(records) > 0

    def get_prior_experiment_record(
        self, experiment_fingerprint: str
    ) -> ResearchRegistryRecord | None:
        """Retrieve prior registry record for experiment_fingerprint if available."""
        records = self.get_by_experiment_fingerprint(experiment_fingerprint)
        if records:
            return records[0]
        return None

    def _learning_dir(self, experiment_fingerprint: str) -> Path:
        return self.base_dir / "learning" / experiment_fingerprint

    def record_learning_supersession(
        self,
        newer_learning: ResearchLearningRecord,
        older_learning: ResearchLearningRecord,
    ) -> tuple[ResearchLearningRecord, ResearchLearningRecord]:
        """Atomically record supersession/contradiction between two learning observations.

        Updates older_learning to record superseded_by_learning_id and deactivate constraints,
        and registers newer_learning with supersedes_learning_id set.
        """
        updated_newer, updated_older = mark_learning_supersession(
            newer_learning=newer_learning,
            older_learning=older_learning,
        )

        # Register newer learning
        registered_newer = self.register_learning_record(updated_newer)

        # Update older learning file atomically
        target_dir = self._learning_dir(updated_older.experiment_fingerprint)
        file_path = target_dir / f"{updated_older.learning_id}.json"
        serialized_content = json.dumps(updated_older.as_dict(), indent=2, sort_keys=True)
        fd, temp_path = tempfile.mkstemp(dir=target_dir, prefix="learn_tmp_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(serialized_content)
            os.replace(temp_path, file_path)
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise

        return registered_newer, updated_older

    def register_learning_record(self, record: ResearchLearningRecord) -> ResearchLearningRecord:
        """Register a research learning record idempotently and atomically."""
        if not isinstance(record, ResearchLearningRecord):
            raise TypeError("record must be a ResearchLearningRecord instance.")

        source_rec = self.get_by_record_id(record.source_record_id, record.experiment_fingerprint)
        if source_rec is None and record.evidence_fingerprint:
            source_rec = self.get_by_evidence_fingerprint(record.evidence_fingerprint)
        if source_rec is None:
            raise RegistryValidationError(
                f"Cannot register learning record '{record.learning_id}': "
                f"Source registry record '{record.source_record_id}' for experiment '{record.experiment_fingerprint}' does not exist."
            )

        if record.evidence_fingerprint and source_rec.evidence_fingerprint != record.evidence_fingerprint:
            raise RegistryValidationError(
                f"Cannot register learning record '{record.learning_id}': "
                f"Evidence fingerprint mismatch. Source: '{source_rec.evidence_fingerprint}', Record: '{record.evidence_fingerprint}'."
            )

        target_dir = self._learning_dir(record.experiment_fingerprint)
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{record.learning_id}.json"

        if file_path.exists():
            existing = self._load_learning_file(file_path)
            if existing.canonical_fingerprint == record.canonical_fingerprint:
                return existing
            raise RegistryConflictError(
                f"Conflicting learning record exists for experiment '{record.experiment_fingerprint}' "
                f"and learning_id '{record.learning_id}'. "
                f"Existing fingerprint: {existing.canonical_fingerprint}, "
                f"New fingerprint: {record.canonical_fingerprint}."
            )

        serialized_content = json.dumps(record.as_dict(), indent=2, sort_keys=True)
        fd, temp_path = tempfile.mkstemp(dir=target_dir, prefix="learn_tmp_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(serialized_content)
            os.replace(temp_path, file_path)
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise

        return record

    def _load_learning_file(self, file_path: Path) -> ResearchLearningRecord:
        if not file_path.exists():
            raise FileNotFoundError(f"Learning record file not found: {file_path}")
        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RegistryValidationError(f"Failed to parse JSON from {file_path}: {exc}") from exc
        return ResearchLearningRecord.from_dict(raw)

    def get_learning_by_id(
        self, learning_id: str, experiment_fingerprint: str
    ) -> ResearchLearningRecord | None:
        file_path = self._learning_dir(experiment_fingerprint) / f"{learning_id}.json"
        if not file_path.exists():
            return None
        return self._load_learning_file(file_path)

    def list_learning_records(self) -> tuple[ResearchLearningRecord, ...]:
        learning_dir = self.base_dir / "learning"
        if not learning_dir.exists():
            return ()
        records: list[ResearchLearningRecord] = []
        for file_path in sorted(learning_dir.glob("*/*.json")):
            if file_path.name.startswith("learn_tmp_"):
                continue
            records.append(self._load_learning_file(file_path))
        return tuple(records)

    def query_learning(
        self,
        *,
        experiment_fingerprint: str | None = None,
        evidence_fingerprint: str | None = None,
        candidate_id: str | None = None,
        classification: ResearchOutcomeClassification | str | None = None,
        lesson_category: LessonCategory | str | None = None,
        constraint_pattern_key: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        active_constraints_only: bool = False,
    ) -> tuple[ResearchLearningRecord, ...]:
        """Read-only query capability over persisted canonical learning memory."""
        all_learnings = self.list_learning_records()
        results: list[ResearchLearningRecord] = []

        class_val = classification.value if isinstance(classification, ResearchOutcomeClassification) else classification
        cat_val = lesson_category.value if isinstance(lesson_category, LessonCategory) else lesson_category

        for lr in all_learnings:
            if experiment_fingerprint and lr.experiment_fingerprint != experiment_fingerprint:
                continue
            if evidence_fingerprint and lr.evidence_fingerprint != evidence_fingerprint:
                continue
            if candidate_id and lr.candidate_id != candidate_id:
                continue
            if class_val and lr.classification.value != class_val:
                continue
            if cat_val and not any(l.category.value == cat_val for l in lr.lessons):
                continue
            if constraint_pattern_key and not any(c.pattern_key == constraint_pattern_key for c in lr.constraints):
                continue
            if symbol and lr.observed_conditions.symbol != symbol:
                continue
            if timeframe and lr.observed_conditions.timeframe != timeframe:
                continue
            if active_constraints_only and not any(c.is_active for c in lr.constraints):
                continue

            results.append(lr)

        return tuple(results)

    def get_active_do_not_repeat_constraints(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        strategy_name: str | None = None,
    ) -> tuple[DoNotRepeatConstraint, ...]:
        """Expose active read-only do-not-repeat constraints for future discovery feedback."""
        learnings = self.list_learning_records()
        active: list[DoNotRepeatConstraint] = []
        for lr in learnings:
            if symbol and lr.observed_conditions.symbol != symbol:
                continue
            if timeframe and lr.observed_conditions.timeframe != timeframe:
                continue
            if strategy_name and lr.observed_conditions.strategy_name != strategy_name:
                continue

            for c in lr.constraints:
                if c.is_active:
                    active.append(c)

        return tuple(active)

    def _knowledge_dir(self) -> Path:
        return self.base_dir / "knowledge"

    def register_pattern(self, pattern: Any) -> Any:
        """Register a research knowledge pattern idempotently and atomically.

        Fail closed if any supporting learning record identity is missing from registry store.
        """
        from src.evaluation.research_knowledge import ResearchKnowledgePattern

        if not isinstance(pattern, ResearchKnowledgePattern):
            raise TypeError("pattern must be a ResearchKnowledgePattern instance.")

        # Fail closed: validate that all supporting learning record IDs exist in store
        for lid in pattern.supporting_learning_ids:
            found = False
            for exp_fp in pattern.supporting_experiment_fingerprints:
                if self.get_learning_by_id(lid, exp_fp) is not None:
                    found = True
                    break
            if not found:
                for lr in self.list_learning_records():
                    if lr.learning_id == lid:
                        found = True
                        break
            if not found:
                raise RegistryValidationError(
                    f"Cannot register pattern '{pattern.pattern_id}': "
                    f"Supporting learning record '{lid}' does not exist in registry store."
                )

        target_dir = self._knowledge_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{pattern.pattern_id}.json"

        if file_path.exists():
            existing = self._load_pattern_file(file_path)
            if existing.canonical_fingerprint == pattern.canonical_fingerprint:
                return existing
            raise RegistryConflictError(
                f"Conflicting knowledge pattern exists for pattern_id '{pattern.pattern_id}'. "
                f"Existing fingerprint: {existing.canonical_fingerprint}, "
                f"New fingerprint: {pattern.canonical_fingerprint}."
            )

        serialized_content = json.dumps(pattern.as_dict(), indent=2, sort_keys=True)
        fd, temp_path = tempfile.mkstemp(dir=target_dir, prefix="pat_tmp_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(serialized_content)
            os.replace(temp_path, file_path)
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise

        return pattern

    def _load_pattern_file(self, file_path: Path) -> Any:
        from src.evaluation.research_knowledge import ResearchKnowledgePattern

        if not file_path.exists():
            raise FileNotFoundError(f"Pattern file not found: {file_path}")
        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RegistryValidationError(f"Failed to parse JSON from {file_path}: {exc}") from exc
        return ResearchKnowledgePattern.from_dict(raw)

    def get_pattern_by_id(self, pattern_id: str) -> Any | None:
        """Retrieve pattern by pattern_id."""
        file_path = self._knowledge_dir() / f"{pattern_id}.json"
        if not file_path.exists():
            return None
        return self._load_pattern_file(file_path)

    def list_patterns(self) -> tuple[Any, ...]:
        """List all persisted research knowledge patterns."""
        k_dir = self._knowledge_dir()
        if not k_dir.exists():
            return ()
        patterns: list[Any] = []
        for file_path in sorted(k_dir.glob("*.json")):
            if file_path.name.startswith("pat_tmp_"):
                continue
            patterns.append(self._load_pattern_file(file_path))
        patterns.sort(key=lambda p: p.pattern_id)
        return tuple(patterns)

    def query_patterns(
        self,
        *,
        category: Any | str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        strategy_name: str | None = None,
        is_contradictory: bool | None = None,
        active_only: bool = True,
    ) -> tuple[Any, ...]:
        """Read-only query capability over persisted research knowledge patterns."""
        from src.evaluation.research_knowledge import ResearchPatternCategory

        all_patterns = self.list_patterns()
        results: list[Any] = []

        cat_val = category.value if isinstance(category, ResearchPatternCategory) else category

        for pat in all_patterns:
            if cat_val and pat.category.value != cat_val:
                continue
            if symbol and pat.normalized_conditions.symbol != symbol:
                continue
            if timeframe and pat.normalized_conditions.timeframe != timeframe:
                continue
            if strategy_name and pat.normalized_conditions.strategy_name != strategy_name:
                continue
            if is_contradictory is not None and pat.is_contradictory != is_contradictory:
                continue
            if active_only and not pat.is_active:
                continue

            results.append(pat)

        results.sort(key=lambda p: p.pattern_id)
        return tuple(results)

    def record_pattern_supersession(
        self,
        newer_pattern: Any,
        older_pattern: Any,
    ) -> tuple[Any, Any]:
        """Atomically record supersession between two knowledge patterns."""
        from src.evaluation.research_knowledge import ResearchKnowledgePattern

        if not isinstance(newer_pattern, ResearchKnowledgePattern) or not isinstance(older_pattern, ResearchKnowledgePattern):
            raise TypeError("Both newer_pattern and older_pattern must be ResearchKnowledgePattern instances.")

        older_dict = older_pattern.as_dict()
        older_dict["is_active"] = False
        older_dict["superseded_by_pattern_id"] = newer_pattern.pattern_id
        updated_older = ResearchKnowledgePattern.from_dict(older_dict)

        newer_dict = newer_pattern.as_dict()
        newer_dict["supersedes_pattern_id"] = older_pattern.pattern_id
        updated_newer = ResearchKnowledgePattern.from_dict(newer_dict)

        registered_newer = self.register_pattern(updated_newer)

        # Write updated older pattern file atomically
        target_dir = self._knowledge_dir()
        file_path = target_dir / f"{updated_older.pattern_id}.json"
        serialized_content = json.dumps(updated_older.as_dict(), indent=2, sort_keys=True)
        fd, temp_path = tempfile.mkstemp(dir=target_dir, prefix="pat_tmp_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(serialized_content)
            os.replace(temp_path, file_path)
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise

        return registered_newer, updated_older


@dataclass(frozen=True)
class StructuredObservedConditions:
    """Machine-readable observed conditions extracted from research evidence/spec."""

    symbol: str | None
    timeframe: str | None
    strategy_name: str | None
    strategy_version: str | None
    dataset_scope_id: str
    execution_assumptions_id: str
    code_provenance_id: str
    methodology_version: str
    out_of_sample_evaluated: bool
    walk_forward_evaluated: bool
    benchmark_status: str | None
    regime_status: str | None
    selection_status: str | None
    robustness_status: str | None
    rejection_reasons: tuple[str, ...]
    metrics: dict[str, Any]
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
            "out_of_sample_evaluated": self.out_of_sample_evaluated,
            "walk_forward_evaluated": self.walk_forward_evaluated,
            "benchmark_status": self.benchmark_status,
            "regime_status": self.regime_status,
            "selection_status": self.selection_status,
            "robustness_status": self.robustness_status,
            "rejection_reasons": list(self.rejection_reasons),
            "metrics": dict(self.metrics),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuredObservedConditions:
        if not isinstance(data, dict):
            raise RegistryValidationError("Conditions data must be a dictionary.")
        ver = data.get("schema_version")
        if ver != SCHEMA_VERSION_1_0:
            raise SchemaVersionError(f"Unsupported conditions schema version: '{ver}'. Expected '{SCHEMA_VERSION_1_0}'.")
        return cls(
            symbol=data.get("symbol"),
            timeframe=data.get("timeframe"),
            strategy_name=data.get("strategy_name"),
            strategy_version=data.get("strategy_version"),
            dataset_scope_id=data.get("dataset_scope_id", ""),
            execution_assumptions_id=data.get("execution_assumptions_id", ""),
            code_provenance_id=data.get("code_provenance_id", ""),
            methodology_version=data.get("methodology_version", ""),
            out_of_sample_evaluated=bool(data.get("out_of_sample_evaluated", False)),
            walk_forward_evaluated=bool(data.get("walk_forward_evaluated", False)),
            benchmark_status=data.get("benchmark_status"),
            regime_status=data.get("regime_status"),
            selection_status=data.get("selection_status"),
            robustness_status=data.get("robustness_status"),
            rejection_reasons=tuple(data.get("rejection_reasons", [])),
            metrics=dict(data.get("metrics", {})),
            schema_version=ver,
        )


@dataclass(frozen=True)
class ResearchLesson:
    """Structured knowledge artifact derived from source evidence."""

    lesson_id: str
    category: LessonCategory
    source_record_id: str
    experiment_fingerprint: str
    evidence_fingerprint: str | None
    observed_conditions: StructuredObservedConditions
    conclusion: str
    confidence_score: float
    methodology_version: str
    schema_version: str = SCHEMA_VERSION_1_0

    def __post_init__(self) -> None:
        if not self.lesson_id or not self.lesson_id.strip():
            raise RegistryValidationError("lesson_id must be a non-empty string.")
        if not self.conclusion or not self.conclusion.strip():
            raise RegistryValidationError("conclusion must be a non-empty string.")
        if not (0.0 <= self.confidence_score <= 1.0):
            raise RegistryValidationError("confidence_score must be between 0.0 and 1.0.")
        if self.schema_version != SCHEMA_VERSION_1_0:
            raise SchemaVersionError(f"Unsupported lesson schema version: '{self.schema_version}'. Expected '{SCHEMA_VERSION_1_0}'.")

    @property
    def semantic_content(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "source_record_id": self.source_record_id,
            "experiment_fingerprint": self.experiment_fingerprint,
            "evidence_fingerprint": self.evidence_fingerprint,
            "observed_conditions": self.observed_conditions.as_dict(),
            "conclusion": self.conclusion,
            "confidence_score": round(self.confidence_score, 6),
            "methodology_version": self.methodology_version,
            "schema_version": self.schema_version,
        }

    @property
    def canonical_fingerprint(self) -> str:
        serialized = json.dumps(self.semantic_content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        res = self.semantic_content
        res["lesson_id"] = self.lesson_id
        res["canonical_fingerprint"] = self.canonical_fingerprint
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchLesson:
        if not isinstance(data, dict):
            raise RegistryValidationError("Lesson data must be a dictionary.")
        ver = data.get("schema_version")
        if ver != SCHEMA_VERSION_1_0:
            raise SchemaVersionError(f"Unsupported lesson schema version: '{ver}'. Expected '{SCHEMA_VERSION_1_0}'.")

        cat_str = data.get("category")
        if not cat_str or cat_str not in LessonCategory.__members__:
            raise RegistryValidationError(f"Invalid or missing LessonCategory: '{cat_str}'.")

        cond_dict = data.get("observed_conditions")
        if not isinstance(cond_dict, dict):
            raise RegistryValidationError("Missing or invalid 'observed_conditions' dictionary in lesson data.")

        return cls(
            lesson_id=data.get("lesson_id", ""),
            category=LessonCategory(cat_str),
            source_record_id=data.get("source_record_id", ""),
            experiment_fingerprint=data.get("experiment_fingerprint", ""),
            evidence_fingerprint=data.get("evidence_fingerprint"),
            observed_conditions=StructuredObservedConditions.from_dict(cond_dict),
            conclusion=data.get("conclusion", ""),
            confidence_score=float(data.get("confidence_score", 0.0)),
            methodology_version=data.get("methodology_version", ""),
            schema_version=ver,
        )


@dataclass(frozen=True)
class DoNotRepeatConstraint:
    """Durable prohibition constraint generated from verified negative research evidence."""

    constraint_id: str
    source_record_id: str
    experiment_fingerprint: str
    evidence_fingerprint: str | None
    pattern_key: str
    condition_description: str
    reason: str
    confidence_score: float
    is_active: bool = True
    superseded_by_learning_id: str | None = None
    schema_version: str = SCHEMA_VERSION_1_0

    def __post_init__(self) -> None:
        if not self.constraint_id or not self.constraint_id.strip():
            raise RegistryValidationError("constraint_id must be a non-empty string.")
        if not self.pattern_key or not self.pattern_key.strip():
            raise RegistryValidationError("pattern_key must be a non-empty string.")
        if not self.reason or not self.reason.strip():
            raise RegistryValidationError("reason must be a non-empty string.")
        if not (0.0 <= self.confidence_score <= 1.0):
            raise RegistryValidationError("confidence_score must be between 0.0 and 1.0.")
        if self.schema_version != SCHEMA_VERSION_1_0:
            raise SchemaVersionError(f"Unsupported constraint schema version: '{self.schema_version}'. Expected '{SCHEMA_VERSION_1_0}'.")

    @property
    def semantic_content(self) -> dict[str, Any]:
        return {
            "source_record_id": self.source_record_id,
            "experiment_fingerprint": self.experiment_fingerprint,
            "evidence_fingerprint": self.evidence_fingerprint,
            "pattern_key": self.pattern_key,
            "condition_description": self.condition_description,
            "reason": self.reason,
            "confidence_score": round(self.confidence_score, 6),
            "is_active": self.is_active,
            "superseded_by_learning_id": self.superseded_by_learning_id,
            "schema_version": self.schema_version,
        }

    @property
    def canonical_fingerprint(self) -> str:
        serialized = json.dumps(self.semantic_content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        res = self.semantic_content
        res["constraint_id"] = self.constraint_id
        res["canonical_fingerprint"] = self.canonical_fingerprint
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DoNotRepeatConstraint:
        if not isinstance(data, dict):
            raise RegistryValidationError("Constraint data must be a dictionary.")
        ver = data.get("schema_version")
        if ver != SCHEMA_VERSION_1_0:
            raise SchemaVersionError(f"Unsupported constraint schema version: '{ver}'. Expected '{SCHEMA_VERSION_1_0}'.")

        return cls(
            constraint_id=data.get("constraint_id", ""),
            source_record_id=data.get("source_record_id", ""),
            experiment_fingerprint=data.get("experiment_fingerprint", ""),
            evidence_fingerprint=data.get("evidence_fingerprint"),
            pattern_key=data.get("pattern_key", ""),
            condition_description=data.get("condition_description", ""),
            reason=data.get("reason", ""),
            confidence_score=float(data.get("confidence_score", 0.0)),
            is_active=bool(data.get("is_active", True)),
            superseded_by_learning_id=data.get("superseded_by_learning_id"),
            schema_version=ver,
        )


@dataclass(frozen=True)
class ResearchLearningRecord:
    """Canonical, immutable domain record of an evidence-backed learning observation."""

    learning_id: str
    source_record_id: str
    experiment_fingerprint: str
    evidence_fingerprint: str | None
    candidate_id: str | None
    search_fingerprint: str | None
    trial_id: str | None
    dataset_scope_id: str
    execution_assumptions_id: str
    code_provenance_id: str
    methodology_version: str
    classification: ResearchOutcomeClassification
    observed_conditions: StructuredObservedConditions
    lessons: tuple[ResearchLesson, ...]
    constraints: tuple[DoNotRepeatConstraint, ...]
    confidence_score: float
    rejection_reasons: tuple[str, ...]
    superseded_by_learning_id: str | None = None
    supersedes_learning_id: str | None = None
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: str = SCHEMA_VERSION_1_0

    def __post_init__(self) -> None:
        if not self.learning_id or not self.learning_id.strip():
            raise RegistryValidationError("learning_id must be a non-empty string.")
        if not self.source_record_id or not self.source_record_id.strip():
            raise RegistryValidationError("source_record_id must be a non-empty string.")
        if not self.experiment_fingerprint or not self.experiment_fingerprint.strip():
            raise RegistryValidationError("experiment_fingerprint must be a non-empty string.")
        if self.schema_version != SCHEMA_VERSION_1_0:
            raise SchemaVersionError(
                f"Unsupported learning record schema version: '{self.schema_version}'. "
                f"Expected '{SCHEMA_VERSION_1_0}'."
            )
        if not isinstance(self.classification, ResearchOutcomeClassification):
            if isinstance(self.classification, str) and self.classification in ResearchOutcomeClassification.__members__:
                object.__setattr__(self, "classification", ResearchOutcomeClassification(self.classification))
            else:
                raise RegistryValidationError(f"Invalid ResearchOutcomeClassification: '{self.classification}'.")

    @property
    def semantic_content(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_record_id": self.source_record_id,
            "experiment_fingerprint": self.experiment_fingerprint,
            "evidence_fingerprint": self.evidence_fingerprint,
            "candidate_id": self.candidate_id,
            "search_fingerprint": self.search_fingerprint,
            "trial_id": self.trial_id,
            "dataset_scope_id": self.dataset_scope_id,
            "execution_assumptions_id": self.execution_assumptions_id,
            "code_provenance_id": self.code_provenance_id,
            "methodology_version": self.methodology_version,
            "classification": self.classification.value,
            "observed_conditions": self.observed_conditions.as_dict(),
            "lessons": [l.as_dict() for l in self.lessons],
            "constraints": [c.as_dict() for c in self.constraints],
            "confidence_score": round(self.confidence_score, 6),
            "rejection_reasons": sorted(self.rejection_reasons),
            "superseded_by_learning_id": self.superseded_by_learning_id,
            "supersedes_learning_id": self.supersedes_learning_id,
        }

    @property
    def canonical_fingerprint(self) -> str:
        serialized = json.dumps(self.semantic_content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        res = self.semantic_content
        res["learning_id"] = self.learning_id
        res["created_at_utc"] = self.created_at_utc
        res["canonical_fingerprint"] = self.canonical_fingerprint
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchLearningRecord:
        if not isinstance(data, dict):
            raise RegistryValidationError("Learning record data must be a dictionary.")

        schema_ver = data.get("schema_version")
        if schema_ver != SCHEMA_VERSION_1_0:
            raise SchemaVersionError(f"Unsupported learning schema version: '{schema_ver}'. Expected '{SCHEMA_VERSION_1_0}'.")

        class_str = data.get("classification")
        if not class_str or class_str not in ResearchOutcomeClassification.__members__:
            raise RegistryValidationError(f"Invalid or missing classification in learning data: '{class_str}'.")

        cond_dict = data.get("observed_conditions")
        if not isinstance(cond_dict, dict):
            raise RegistryValidationError("Missing or invalid 'observed_conditions' dictionary in learning data.")

        lessons_raw = data.get("lessons", [])
        if not isinstance(lessons_raw, (list, tuple)):
            raise RegistryValidationError("lessons must be a list/tuple.")
        lessons = tuple(ResearchLesson.from_dict(l) for l in lessons_raw)

        constraints_raw = data.get("constraints", [])
        if not isinstance(constraints_raw, (list, tuple)):
            raise RegistryValidationError("constraints must be a list/tuple.")
        constraints = tuple(DoNotRepeatConstraint.from_dict(c) for c in constraints_raw)

        return cls(
            learning_id=data.get("learning_id", ""),
            source_record_id=data.get("source_record_id", ""),
            experiment_fingerprint=data.get("experiment_fingerprint", ""),
            evidence_fingerprint=data.get("evidence_fingerprint"),
            candidate_id=data.get("candidate_id"),
            search_fingerprint=data.get("search_fingerprint"),
            trial_id=data.get("trial_id"),
            dataset_scope_id=data.get("dataset_scope_id", ""),
            execution_assumptions_id=data.get("execution_assumptions_id", ""),
            code_provenance_id=data.get("code_provenance_id", ""),
            methodology_version=data.get("methodology_version", ""),
            classification=ResearchOutcomeClassification(class_str),
            observed_conditions=StructuredObservedConditions.from_dict(cond_dict),
            lessons=lessons,
            constraints=constraints,
            confidence_score=float(data.get("confidence_score", 0.0)),
            rejection_reasons=tuple(data.get("rejection_reasons", [])),
            superseded_by_learning_id=data.get("superseded_by_learning_id"),
            supersedes_learning_id=data.get("supersedes_learning_id"),
            created_at_utc=data.get("created_at_utc", ""),
            schema_version=schema_ver,
        )


def construct_learning_record_from_registry_record(
    registry_record: ResearchRegistryRecord,
    *,
    custom_lessons: Sequence[ResearchLesson] = (),
    custom_constraints: Sequence[DoNotRepeatConstraint] = (),
) -> ResearchLearningRecord:
    """Construct an evidence-backed ResearchLearningRecord from a valid ResearchRegistryRecord."""
    if not isinstance(registry_record, ResearchRegistryRecord):
        raise TypeError("registry_record must be a ResearchRegistryRecord instance.")

    if not registry_record.record_id or not registry_record.experiment_fingerprint:
        raise RegistryValidationError("source registry record must have valid record_id and experiment_fingerprint.")

    payload = registry_record.evidence_payload or {}
    spec_dict = payload.get("spec", {})
    ds_dict = spec_dict.get("dataset_scope", {})

    symbol = ds_dict.get("symbol")
    timeframe = ds_dict.get("timeframe")
    strategy_name = spec_dict.get("strategy_name")
    strategy_version = spec_dict.get("strategy_version")

    partitions_raw = payload.get("partitions", [])
    has_oos = any(p.get("role") == "OUT_OF_SAMPLE" for p in partitions_raw if isinstance(p, dict))
    has_wf = any(p.get("role") == "WALK_FORWARD" for p in partitions_raw if isinstance(p, dict))

    metrics: dict[str, Any] = {}
    for p in partitions_raw:
        if isinstance(p, dict) and "role" in p:
            r = p["role"]
            metrics[f"{r}_sharpe"] = p.get("sharpe_ratio")
            metrics[f"{r}_total_return"] = p.get("total_return")
            metrics[f"{r}_max_drawdown"] = p.get("max_drawdown")
            metrics[f"{r}_win_rate"] = p.get("win_rate")

    observed_conditions = StructuredObservedConditions(
        symbol=symbol,
        timeframe=timeframe,
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        dataset_scope_id=registry_record.dataset_scope_id,
        execution_assumptions_id=registry_record.execution_assumptions_id,
        code_provenance_id=registry_record.code_provenance_id,
        methodology_version=registry_record.methodology_version,
        out_of_sample_evaluated=has_oos,
        walk_forward_evaluated=has_wf,
        benchmark_status=registry_record.benchmark_status,
        regime_status=registry_record.regime_status,
        selection_status=registry_record.lineage.selection_assessment_id,
        robustness_status=registry_record.lineage.robustness_assessment_id,
        rejection_reasons=registry_record.rejection_reasons,
        metrics=metrics,
    )

    rej_set = set(registry_record.rejection_reasons)
    is_insufficient = (
        registry_record.status == RegistryStatus.INSUFFICIENT
        or "INSUFFICIENT_DATA" in rej_set
        or "INSUFFICIENT_SAMPLE" in rej_set
        or registry_record.status == RegistryStatus.UNAVAILABLE
    )

    if is_insufficient:
        classification = ResearchOutcomeClassification.INCONCLUSIVE
        confidence = 0.5
    elif (
        registry_record.status == RegistryStatus.QUALIFIED
        and registry_record.promotion_status in (PromotionStatus.PROMOTABLE.value, PromotionStatus.VALIDATED.value)
        and registry_record.qualification_status == "QUALIFIED"
        and not registry_record.rejection_reasons
        and not registry_record.error_message
    ):
        classification = ResearchOutcomeClassification.SUCCESS
        confidence = 0.95
    elif (
        registry_record.status in (RegistryStatus.REJECTED, RegistryStatus.FAILED)
        or registry_record.rejection_reasons
        or registry_record.qualification_status == "REJECTED"
        or registry_record.error_message
    ):
        classification = ResearchOutcomeClassification.FAILURE
        confidence = 0.90
    else:
        classification = ResearchOutcomeClassification.INCONCLUSIVE
        confidence = 0.50

    auto_lessons: list[ResearchLesson] = list(custom_lessons)
    auto_constraints: list[DoNotRepeatConstraint] = list(custom_constraints)

    learning_key = f"learning:{registry_record.record_id}:{registry_record.experiment_fingerprint}"
    learning_id = hashlib.sha256(learning_key.encode("utf-8")).hexdigest()[:24]

    if not auto_lessons:
        if classification == ResearchOutcomeClassification.SUCCESS:
            lesson_cat = LessonCategory.STRATEGY_PERFORMANCE
            conclusion = (
                f"Strategy '{strategy_name}' (v{strategy_version}) passed evaluation on {symbol} {timeframe} "
                f"with robust metrics and zero rejection reasons."
            )
        elif classification == ResearchOutcomeClassification.FAILURE:
            lesson_cat = LessonCategory.GOVERNANCE_REJECTION
            reasons_str = ", ".join(registry_record.rejection_reasons) or registry_record.error_message or "failed evaluation criteria"
            conclusion = (
                f"Experiment '{registry_record.experiment_fingerprint[:8]}' failed or was rejected. "
                f"Rejection reasons / error: {reasons_str}."
            )
        else:
            lesson_cat = LessonCategory.DATASET_INSUFFICIENCY
            conclusion = (
                f"Experiment '{registry_record.experiment_fingerprint[:8]}' produced inconclusive results "
                f"due to insufficient data or unavailable partitions."
            )

        lesson_key = f"lesson:{learning_id}:auto_0"
        lesson_id = hashlib.sha256(lesson_key.encode("utf-8")).hexdigest()[:24]
        auto_lessons.append(
            ResearchLesson(
                lesson_id=lesson_id,
                category=lesson_cat,
                source_record_id=registry_record.record_id,
                experiment_fingerprint=registry_record.experiment_fingerprint,
                evidence_fingerprint=registry_record.evidence_fingerprint,
                observed_conditions=observed_conditions,
                conclusion=conclusion,
                confidence_score=confidence,
                methodology_version=registry_record.methodology_version,
            )
        )

    if not auto_constraints and classification == ResearchOutcomeClassification.FAILURE:
        pattern_key = f"fail_pattern:{strategy_name}:{symbol}:{timeframe}:{registry_record.dataset_scope_id[:8]}"
        constraint_key = f"constraint:{learning_id}:auto_0"
        constraint_id = hashlib.sha256(constraint_key.encode("utf-8")).hexdigest()[:24]
        reason_str = ", ".join(registry_record.rejection_reasons) or registry_record.error_message or "Evaluation failure"
        auto_constraints.append(
            DoNotRepeatConstraint(
                constraint_id=constraint_id,
                source_record_id=registry_record.record_id,
                experiment_fingerprint=registry_record.experiment_fingerprint,
                evidence_fingerprint=registry_record.evidence_fingerprint,
                pattern_key=pattern_key,
                condition_description=f"Strategy {strategy_name} on {symbol} {timeframe}",
                reason=f"Rejected during evaluation: {reason_str}",
                confidence_score=confidence,
                is_active=True,
            )
        )

    return ResearchLearningRecord(
        learning_id=learning_id,
        source_record_id=registry_record.record_id,
        experiment_fingerprint=registry_record.experiment_fingerprint,
        evidence_fingerprint=registry_record.evidence_fingerprint,
        candidate_id=registry_record.candidate_id,
        search_fingerprint=registry_record.search_fingerprint,
        trial_id=registry_record.trial_id,
        dataset_scope_id=registry_record.dataset_scope_id,
        execution_assumptions_id=registry_record.execution_assumptions_id,
        code_provenance_id=registry_record.code_provenance_id,
        methodology_version=registry_record.methodology_version,
        classification=classification,
        observed_conditions=observed_conditions,
        lessons=tuple(auto_lessons),
        constraints=tuple(auto_constraints),
        confidence_score=confidence,
        rejection_reasons=registry_record.rejection_reasons,
    )


def mark_learning_supersession(
    *,
    newer_learning: ResearchLearningRecord,
    older_learning: ResearchLearningRecord,
) -> tuple[ResearchLearningRecord, ResearchLearningRecord]:
    """Explicitly link supersession/contradiction between two learning observations."""
    if not isinstance(newer_learning, ResearchLearningRecord) or not isinstance(older_learning, ResearchLearningRecord):
        raise TypeError("Both newer_learning and older_learning must be ResearchLearningRecord instances.")

    updated_constraints = tuple(
        DoNotRepeatConstraint(
            constraint_id=c.constraint_id,
            source_record_id=c.source_record_id,
            experiment_fingerprint=c.experiment_fingerprint,
            evidence_fingerprint=c.evidence_fingerprint,
            pattern_key=c.pattern_key,
            condition_description=c.condition_description,
            reason=c.reason,
            confidence_score=c.confidence_score,
            is_active=False,
            superseded_by_learning_id=newer_learning.learning_id,
            schema_version=c.schema_version,
        )
        for c in older_learning.constraints
    )

    updated_older = ResearchLearningRecord(
        learning_id=older_learning.learning_id,
        source_record_id=older_learning.source_record_id,
        experiment_fingerprint=older_learning.experiment_fingerprint,
        evidence_fingerprint=older_learning.evidence_fingerprint,
        candidate_id=older_learning.candidate_id,
        search_fingerprint=older_learning.search_fingerprint,
        trial_id=older_learning.trial_id,
        dataset_scope_id=older_learning.dataset_scope_id,
        execution_assumptions_id=older_learning.execution_assumptions_id,
        code_provenance_id=older_learning.code_provenance_id,
        methodology_version=older_learning.methodology_version,
        classification=older_learning.classification,
        observed_conditions=older_learning.observed_conditions,
        lessons=older_learning.lessons,
        constraints=updated_constraints,
        confidence_score=older_learning.confidence_score,
        rejection_reasons=older_learning.rejection_reasons,
        superseded_by_learning_id=newer_learning.learning_id,
        supersedes_learning_id=older_learning.supersedes_learning_id,
        created_at_utc=older_learning.created_at_utc,
        schema_version=older_learning.schema_version,
    )

    updated_newer = ResearchLearningRecord(
        learning_id=newer_learning.learning_id,
        source_record_id=newer_learning.source_record_id,
        experiment_fingerprint=newer_learning.experiment_fingerprint,
        evidence_fingerprint=newer_learning.evidence_fingerprint,
        candidate_id=newer_learning.candidate_id,
        search_fingerprint=newer_learning.search_fingerprint,
        trial_id=newer_learning.trial_id,
        dataset_scope_id=newer_learning.dataset_scope_id,
        execution_assumptions_id=newer_learning.execution_assumptions_id,
        code_provenance_id=newer_learning.code_provenance_id,
        methodology_version=newer_learning.methodology_version,
        classification=newer_learning.classification,
        observed_conditions=newer_learning.observed_conditions,
        lessons=newer_learning.lessons,
        constraints=newer_learning.constraints,
        confidence_score=newer_learning.confidence_score,
        rejection_reasons=newer_learning.rejection_reasons,
        superseded_by_learning_id=newer_learning.superseded_by_learning_id,
        supersedes_learning_id=older_learning.learning_id,
        created_at_utc=newer_learning.created_at_utc,
        schema_version=newer_learning.schema_version,
    )

    return updated_newer, updated_older
