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
