"""Research Constitution & Experiment Evidence Contract for Project 1.

This module establishes an authoritative, reproducible representation of a research experiment
and its evidence:

    Hypothesis -> Experiment -> Evaluation -> Validation -> Evidence -> Critique -> Promotion/Reject

It enforces strict fail-closed validation, deterministic experiment fingerprinting,
and structured evidence tracking without altering Protected Core trading or evaluation logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


class PromotionStatus(str, Enum):
    """Minimum status model for research experiment candidates."""

    PROPOSED = "PROPOSED"
    EXPERIMENTAL = "EXPERIMENTAL"
    VALIDATED = "VALIDATED"
    PROMOTABLE = "PROMOTABLE"
    REJECTED = "REJECTED"


class ResearchCampaignStatus(str, Enum):
    """Explicit lifecycle status for a Research Campaign / Discovery Run."""

    COMPLETED = "COMPLETED"
    TRUNCATED = "TRUNCATED"
    FAILED = "FAILED"


class HypothesisStatus(str, Enum):
    """Explicit lifecycle status for a governed Research Hypothesis."""

    GENERATED = "generated"
    ACCEPTED_FOR_RESEARCH = "accepted_for_research"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class RejectionReason(str, Enum):
    """Explicit failure and critique reasons for research experiments."""

    DATA_LEAKAGE_CONCERN = "DATA_LEAKAGE_CONCERN"
    LOOK_AHEAD_CONCERN = "LOOK_AHEAD_CONCERN"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INVALID_DATASET_SCOPE = "INVALID_DATASET_SCOPE"
    STALE_INVALID_DATA = "STALE_INVALID_DATA"
    EXECUTION_ASSUMPTION_VIOLATION = "EXECUTION_ASSUMPTION_VIOLATION"
    EVIDENCE_INCOMPLETENESS = "EVIDENCE_INCOMPLETENESS"
    IS_ONLY_SUCCESS = "IS_ONLY_SUCCESS"
    FAILED_VALIDATION = "FAILED_VALIDATION"
    FAILED_ROBUSTNESS = "FAILED_ROBUSTNESS"
    FAILED_BENCHMARK = "FAILED_BENCHMARK"
    FAILED_OOS = "FAILED_OOS"
    FAILED_WALK_FORWARD = "FAILED_WALK_FORWARD"
    FAILED_REPRODUCIBILITY = "FAILED_REPRODUCIBILITY"
    SPECIFICATION_INVALID = "SPECIFICATION_INVALID"
    CRITIQUE_REJECTED = "CRITIQUE_REJECTED"
    DUPLICATE_CANDIDATE = "DUPLICATE_CANDIDATE"
    RESEARCH_LEAKAGE = "RESEARCH_LEAKAGE"
    FAILED_PARAMETER_SENSITIVITY = "FAILED_PARAMETER_SENSITIVITY"
    FAILED_COST_STRESS = "FAILED_COST_STRESS"
    INSUFFICIENT_STATISTICAL_SAMPLE = "INSUFFICIENT_STATISTICAL_SAMPLE"
    FAILED_STATISTICAL_VALIDATION = "FAILED_STATISTICAL_VALIDATION"
    ANTI_OVERFITTING_VIOLATION = "ANTI_OVERFITTING_VIOLATION"
    MISSING_ROBUSTNESS_EVIDENCE = "MISSING_ROBUSTNESS_EVIDENCE"
    PERSISTENCE_FAILURE = "PERSISTENCE_FAILURE"
    GOVERNANCE_BLOCKED = "GOVERNANCE_BLOCKED"


@dataclass(frozen=True)
class RobustnessCriteria:
    """Configurable quantitative thresholds for candidate robustness & statistical validation."""

    perturbation_pcts: tuple[float, ...] = (-0.10, 0.10)
    min_perturbation_pass_rate: float = 0.80
    cost_stress_multipliers: tuple[float, ...] = (1.5, 2.0)
    min_cost_stress_pass_rate: float = 1.0
    subsample_slices_count: int = 3
    min_subsample_pass_rate: float = 0.66
    min_statistical_observations: int = 30
    min_t_stat: float = 1.65
    max_p_value: float = 0.05
    version: str = "robustness_v1.0"

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_perturbation_pass_rate <= 1.0):
            raise ValueError("min_perturbation_pass_rate must be between 0.0 and 1.0.")
        if not (0.0 <= self.min_cost_stress_pass_rate <= 1.0):
            raise ValueError("min_cost_stress_pass_rate must be between 0.0 and 1.0.")
        if not (0.0 <= self.min_subsample_pass_rate <= 1.0):
            raise ValueError("min_subsample_pass_rate must be between 0.0 and 1.0.")
        if self.subsample_slices_count <= 0:
            raise ValueError("subsample_slices_count must be positive.")
        if self.min_statistical_observations <= 0:
            raise ValueError("min_statistical_observations must be positive.")
        if not math.isfinite(self.min_t_stat):
            raise ValueError("min_t_stat must be a finite float.")
        if not (0.0 <= self.max_p_value <= 1.0):
            raise ValueError("max_p_value must be between 0.0 and 1.0.")
        for mult in self.cost_stress_multipliers:
            if not math.isfinite(mult) or mult < 0.0:
                raise ValueError("cost_stress_multipliers must be finite non-negative numbers.")


class EvidencePartitionRole(str, Enum):
    """Partition roles for data/evaluation windows."""

    IN_SAMPLE = "IN_SAMPLE"
    VALIDATION = "VALIDATION"
    OUT_OF_SAMPLE = "OUT_OF_SAMPLE"
    WALK_FORWARD = "WALK_FORWARD"


@dataclass(frozen=True)
class DatasetScope:
    """Identity and time boundaries of dataset used in research."""

    dataset_id: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.dataset_id.strip():
            raise ValueError("dataset_id must be a non-empty string.")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string.")
        if not self.timeframe or not self.timeframe.strip():
            raise ValueError("timeframe must be a non-empty string.")
        if not self.start_date or not self.start_date.strip():
            raise ValueError("start_date must be a non-empty string.")
        if not self.end_date or not self.end_date.strip():
            raise ValueError("end_date must be a non-empty string.")
        if self.start_date > self.end_date:
            raise ValueError(
                f"start_date '{self.start_date}' cannot be later than end_date '{self.end_date}'."
            )


@dataclass(frozen=True)
class ExecutionAssumptions:
    """Execution and market friction assumptions."""

    transaction_cost: float
    slippage: float
    latency_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.transaction_cost, (int, float)) or math.isnan(self.transaction_cost):
            raise TypeError("transaction_cost must be a valid float.")
        if not isinstance(self.slippage, (int, float)) or math.isnan(self.slippage):
            raise TypeError("slippage must be a valid float.")
        if not isinstance(self.latency_ms, (int, float)) or math.isnan(self.latency_ms):
            raise TypeError("latency_ms must be a valid float.")

        if self.transaction_cost < 0.0:
            raise ValueError("transaction_cost cannot be negative.")
        if self.slippage < 0.0:
            raise ValueError("slippage cannot be negative.")
        if self.latency_ms < 0.0:
            raise ValueError("latency_ms cannot be negative.")


@dataclass(frozen=True)
class CodeProvenance:
    """Code and methodology version provenance."""

    commit_sha: str
    repository_status: str = "clean"
    author: str = ""

    def __post_init__(self) -> None:
        if not self.commit_sha or not self.commit_sha.strip():
            raise ValueError("commit_sha must be a non-empty string.")


@dataclass(frozen=True)
class WalkForwardProtocol:
    """Authoritative representation of the Walk-Forward execution protocol.

    Holds the effective train_size and test_size used for walk-forward evaluation.
    """

    train_size: int
    test_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.train_size, int) or isinstance(self.train_size, bool):
            raise TypeError("train_size must be an integer.")
        if not isinstance(self.test_size, int) or isinstance(self.test_size, bool):
            raise TypeError("test_size must be an integer.")
        if self.train_size <= 0:
            raise ValueError(f"train_size must be a positive integer, got {self.train_size}.")
        if self.test_size <= 0:
            raise ValueError(f"test_size must be a positive integer, got {self.test_size}.")


def resolve_walk_forward_protocol(
    n_observations: int,
    train_size: int | None = None,
    test_size: int | None = None,
) -> WalkForwardProtocol:
    """Resolve effective WalkForwardProtocol for a given dataset length and optional overrides.

    Default rules:
        train_size = int(n_observations * 0.40)
        test_size  = int(n_observations * 0.15)
    """
    if not isinstance(n_observations, int) or isinstance(n_observations, bool) or n_observations <= 0:
        raise ValueError(f"n_observations must be a positive integer, got {n_observations}.")

    eff_train = train_size if train_size is not None else int(n_observations * 0.40)
    eff_test = test_size if test_size is not None else int(n_observations * 0.15)

    return WalkForwardProtocol(train_size=eff_train, test_size=eff_test)


@dataclass(frozen=True)
class ResearchExperimentSpec:
    """Authoritative spec defining a research experiment hypothesis and setup.

    Fingerprint is calculated strictly from logical configuration parameters
    and excludes volatile runtime timestamps or execution IDs.
    """

    hypothesis: str
    methodology_version: str
    strategy_name: str
    strategy_version: str
    dataset_scope: DatasetScope
    execution_assumptions: ExecutionAssumptions
    code_provenance: CodeProvenance
    benchmark_reference: str
    parameters: dict[str, Any] = field(default_factory=dict)
    random_seed: int | None = None
    walk_forward_protocol: WalkForwardProtocol | None = None
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.hypothesis or not self.hypothesis.strip():
            raise ValueError("hypothesis must be a non-empty string.")
        if not self.methodology_version or not self.methodology_version.strip():
            raise ValueError("methodology_version must be a non-empty string.")
        if not self.strategy_name or not self.strategy_name.strip():
            raise ValueError("strategy_name must be a non-empty string.")
        if not self.strategy_version or not self.strategy_version.strip():
            raise ValueError("strategy_version must be a non-empty string.")
        if not isinstance(self.dataset_scope, DatasetScope):
            raise TypeError("dataset_scope must be a DatasetScope instance.")
        if not isinstance(self.execution_assumptions, ExecutionAssumptions):
            raise TypeError("execution_assumptions must be an ExecutionAssumptions instance.")
        if not isinstance(self.code_provenance, CodeProvenance):
            raise TypeError("code_provenance must be a CodeProvenance instance.")
        if not self.benchmark_reference or not self.benchmark_reference.strip():
            raise ValueError("benchmark_reference must be a non-empty string.")
        if self.walk_forward_protocol is not None and not isinstance(
            self.walk_forward_protocol, WalkForwardProtocol
        ):
            raise TypeError("walk_forward_protocol must be a WalkForwardProtocol instance or None.")

        computed_fingerprint = compute_experiment_fingerprint(
            hypothesis=self.hypothesis,
            methodology_version=self.methodology_version,
            strategy_name=self.strategy_name,
            strategy_version=self.strategy_version,
            dataset_scope=self.dataset_scope,
            execution_assumptions=self.execution_assumptions,
            code_provenance=self.code_provenance,
            benchmark_reference=self.benchmark_reference,
            parameters=self.parameters,
            random_seed=self.random_seed,
            walk_forward_protocol=self.walk_forward_protocol,
        )
        object.__setattr__(self, "fingerprint", computed_fingerprint)


def compute_experiment_fingerprint(
    *,
    hypothesis: str,
    methodology_version: str,
    strategy_name: str,
    strategy_version: str,
    dataset_scope: DatasetScope,
    execution_assumptions: ExecutionAssumptions,
    code_provenance: CodeProvenance,
    benchmark_reference: str,
    parameters: Mapping[str, Any] | None = None,
    random_seed: int | None = None,
    walk_forward_protocol: WalkForwardProtocol | None = None,
) -> str:
    """Compute deterministic SHA-256 fingerprint for a research experiment.

    Excludes volatile timestamps or execution run IDs to ensure identical specs
    produce identical fingerprints.
    """
    if not benchmark_reference or not benchmark_reference.strip():
        raise ValueError("benchmark_reference must be a non-empty string.")

    canonical_payload = {
        "hypothesis": hypothesis.strip(),
        "methodology_version": methodology_version.strip(),
        "strategy_name": strategy_name.strip(),
        "strategy_version": strategy_version.strip(),
        "dataset_scope": {
            "dataset_id": dataset_scope.dataset_id.strip(),
            "symbol": dataset_scope.symbol.strip(),
            "timeframe": dataset_scope.timeframe.strip(),
            "start_date": dataset_scope.start_date.strip(),
            "end_date": dataset_scope.end_date.strip(),
        },
        "execution_assumptions": {
            "transaction_cost": float(execution_assumptions.transaction_cost),
            "slippage": float(execution_assumptions.slippage),
            "latency_ms": float(execution_assumptions.latency_ms),
        },
        "code_provenance": {
            "commit_sha": code_provenance.commit_sha.strip(),
        },
        "parameters": parameters or {},
        "benchmark_reference": benchmark_reference.strip(),
        "random_seed": random_seed,
    }
    if walk_forward_protocol is not None:
        canonical_payload["walk_forward_protocol"] = {
            "train_size": int(walk_forward_protocol.train_size),
            "test_size": int(walk_forward_protocol.test_size),
        }

    serialized = json.dumps(
        canonical_payload,
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvidencePartition:
    """Metrics and evidence bound to a specific partition role (IS, Validation, OOS, Walk-Forward)."""

    role: EvidencePartitionRole
    start_date: str
    end_date: str
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float = 0.0
    profit_factor: float = 0.0
    observations: int = 0
    additional_metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.role, EvidencePartitionRole):
            raise TypeError("role must be an EvidencePartitionRole enum member.")
        if not self.start_date or not self.start_date.strip():
            raise ValueError("start_date must be a non-empty string.")
        if not self.end_date or not self.end_date.strip():
            raise ValueError("end_date must be a non-empty string.")
        if self.start_date > self.end_date:
            raise ValueError(
                f"start_date '{self.start_date}' cannot be later than end_date '{self.end_date}'."
            )

        for metric_name, val in (
            ("total_return", self.total_return),
            ("max_drawdown", self.max_drawdown),
            ("sharpe_ratio", self.sharpe_ratio),
            ("win_rate", self.win_rate),
            ("profit_factor", self.profit_factor),
        ):
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(f"Partition metric '{metric_name}' must be a finite float, got: {val}")


@dataclass(frozen=True)
class ResearchEvidence:
    """Authoritative research evidence linked to a spec and its originating experiment."""

    experiment_fingerprint: str
    spec: ResearchExperimentSpec
    partitions: tuple[EvidencePartition, ...]
    robustness_verdict: dict[str, Any] = field(default_factory=dict)
    benchmark_comparison: dict[str, Any] = field(default_factory=dict)
    promotion_status: PromotionStatus = PromotionStatus.PROPOSED
    rejection_reasons: tuple[RejectionReason, ...] = field(default_factory=tuple)
    critique_notes: str = ""
    created_at_utc: str = ""
    evidence_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ResearchExperimentSpec):
            raise TypeError("spec must be a ResearchExperimentSpec instance.")
        if self.experiment_fingerprint != self.spec.fingerprint:
            raise ValueError(
                f"experiment_fingerprint '{self.experiment_fingerprint}' does not match "
                f"spec fingerprint '{self.spec.fingerprint}'."
            )
        if not isinstance(self.promotion_status, PromotionStatus):
            raise TypeError("promotion_status must be a PromotionStatus enum member.")

        for r in self.rejection_reasons:
            if not isinstance(r, RejectionReason):
                raise TypeError(f"Rejection reason '{r}' must be a RejectionReason enum member.")

        if self.promotion_status == PromotionStatus.REJECTED and not self.rejection_reasons:
            raise ValueError("REJECTED evidence must specify at least one RejectionReason.")

        evidence_payload = {
            "experiment_fingerprint": self.experiment_fingerprint,
            "partitions": [asdict(p) for p in self.partitions],
            "robustness_verdict": self.robustness_verdict,
            "benchmark_comparison": self.benchmark_comparison,
            "promotion_status": self.promotion_status.value,
            "rejection_reasons": [r.value for r in self.rejection_reasons],
            "critique_notes": self.critique_notes.strip(),
        }
        serialized = json.dumps(evidence_payload, sort_keys=True, ensure_ascii=True)
        evidence_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        object.__setattr__(self, "evidence_id", f"ev_{evidence_hash}")

    def as_dict(self) -> dict[str, Any]:
        """Convert ResearchEvidence to a dictionary representation."""
        return {
            "evidence_id": self.evidence_id,
            "experiment_fingerprint": self.experiment_fingerprint,
            "spec": {
                "hypothesis": self.spec.hypothesis,
                "methodology_version": self.spec.methodology_version,
                "strategy_name": self.spec.strategy_name,
                "strategy_version": self.spec.strategy_version,
                "dataset_scope": asdict(self.spec.dataset_scope),
                "execution_assumptions": asdict(self.spec.execution_assumptions),
                "code_provenance": asdict(self.spec.code_provenance),
                "benchmark_reference": self.spec.benchmark_reference,
                "parameters": self.spec.parameters,
                "random_seed": self.spec.random_seed,
                "walk_forward_protocol": asdict(self.spec.walk_forward_protocol)
                if self.spec.walk_forward_protocol
                else None,
                "fingerprint": self.spec.fingerprint,
            },
            "partitions": [
                {
                    "role": p.role.value,
                    "start_date": p.start_date,
                    "end_date": p.end_date,
                    "total_return": p.total_return,
                    "max_drawdown": p.max_drawdown,
                    "sharpe_ratio": p.sharpe_ratio,
                    "win_rate": p.win_rate,
                    "profit_factor": p.profit_factor,
                    "observations": p.observations,
                    "additional_metrics": p.additional_metrics,
                }
                for p in self.partitions
            ],
            "robustness_verdict": self.robustness_verdict,
            "benchmark_comparison": self.benchmark_comparison,
            "promotion_status": self.promotion_status.value,
            "rejection_reasons": [r.value for r in self.rejection_reasons],
            "critique_notes": self.critique_notes,
            "created_at_utc": self.created_at_utc,
        }


@dataclass(frozen=True)
class ResearchHypothesis:
    """Authoritative domain representation of a trading research hypothesis.

    Captures hypothesis statement, source knowledge lineage, source evidence lineage,
    dataset scope, execution assumptions, code provenance, benchmark reference,
    generator version, constraints, parameters, and lifecycle status.
    Computes a deterministic `hypothesis_id` and fingerprint using SHA-256 without
    relying on volatile timestamps.
    """

    statement: str
    methodology_version: str
    strategy_name: str
    strategy_version: str
    dataset_scope: DatasetScope
    execution_assumptions: ExecutionAssumptions
    code_provenance: CodeProvenance
    benchmark_reference: str
    parameters: dict[str, Any] = field(default_factory=dict)
    random_seed: int | None = None
    walk_forward_protocol: WalkForwardProtocol | None = None
    source_knowledge_ids: tuple[str, ...] = field(default_factory=tuple)
    source_evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    hypothesis_version: str = "1.0"
    generation_method: str = "DIRECT_SPEC"
    generator_version: str = "1.0"
    constraints: dict[str, Any] = field(default_factory=dict)
    status: HypothesisStatus = HypothesisStatus.GENERATED
    created_at_utc: str = ""
    hypothesis_id: str = field(init=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.statement or not self.statement.strip():
            raise ValueError("statement must be a non-empty string.")
        if not self.methodology_version or not self.methodology_version.strip():
            raise ValueError("methodology_version must be a non-empty string.")
        if not self.strategy_name or not self.strategy_name.strip():
            raise ValueError("strategy_name must be a non-empty string.")
        if not self.strategy_version or not self.strategy_version.strip():
            raise ValueError("strategy_version must be a non-empty string.")
        if not isinstance(self.dataset_scope, DatasetScope):
            raise TypeError("dataset_scope must be a DatasetScope instance.")
        if not isinstance(self.execution_assumptions, ExecutionAssumptions):
            raise TypeError("execution_assumptions must be an ExecutionAssumptions instance.")
        if not isinstance(self.code_provenance, CodeProvenance):
            raise TypeError("code_provenance must be a CodeProvenance instance.")
        if not self.benchmark_reference or not self.benchmark_reference.strip():
            raise ValueError("benchmark_reference must be a non-empty string.")
        if self.walk_forward_protocol is not None and not isinstance(
            self.walk_forward_protocol, WalkForwardProtocol
        ):
            raise TypeError("walk_forward_protocol must be a WalkForwardProtocol instance or None.")
        if not isinstance(self.status, HypothesisStatus):
            if isinstance(self.status, str) and self.status in HypothesisStatus.__members__:
                object.__setattr__(self, "status", HypothesisStatus(self.status))
            else:
                raise TypeError(f"status must be a HypothesisStatus enum member, got {type(self.status).__name__}.")

        fp = compute_experiment_fingerprint(
            hypothesis=self.statement,
            methodology_version=self.methodology_version,
            strategy_name=self.strategy_name,
            strategy_version=self.strategy_version,
            dataset_scope=self.dataset_scope,
            execution_assumptions=self.execution_assumptions,
            code_provenance=self.code_provenance,
            benchmark_reference=self.benchmark_reference,
            parameters=self.parameters,
            random_seed=self.random_seed,
            walk_forward_protocol=self.walk_forward_protocol,
        )
        object.__setattr__(self, "fingerprint", fp)
        object.__setattr__(self, "hypothesis_id", f"hyp_{fp[:16]}")

    @property
    def canonical_hypothesis_fingerprint(self) -> str:
        """Compute extended deterministic SHA-256 fingerprint for hypothesis governance.

        Includes statement, source lineage, scope, assumptions, code provenance,
        parameters, constraints, and generator version.
        """
        return compute_hypothesis_fingerprint(
            statement=self.statement,
            methodology_version=self.methodology_version,
            strategy_name=self.strategy_name,
            strategy_version=self.strategy_version,
            dataset_scope=self.dataset_scope,
            execution_assumptions=self.execution_assumptions,
            code_provenance=self.code_provenance,
            benchmark_reference=self.benchmark_reference,
            parameters=self.parameters,
            random_seed=self.random_seed,
            walk_forward_protocol=self.walk_forward_protocol,
            source_knowledge_ids=self.source_knowledge_ids,
            source_evidence_ids=self.source_evidence_ids,
            generation_method=self.generation_method,
            generator_version=self.generator_version,
            constraints=self.constraints,
        )

    def to_experiment_spec(self) -> ResearchExperimentSpec:
        """Convert hypothesis into a ResearchExperimentSpec."""
        return ResearchExperimentSpec(
            hypothesis=self.statement,
            methodology_version=self.methodology_version,
            strategy_name=self.strategy_name,
            strategy_version=self.strategy_version,
            dataset_scope=self.dataset_scope,
            execution_assumptions=self.execution_assumptions,
            code_provenance=self.code_provenance,
            benchmark_reference=self.benchmark_reference,
            parameters=dict(self.parameters),
            random_seed=self.random_seed,
            walk_forward_protocol=self.walk_forward_protocol,
        )

    @classmethod
    def from_experiment_spec(cls, spec: ResearchExperimentSpec) -> ResearchHypothesis:
        if not isinstance(spec, ResearchExperimentSpec):
            raise TypeError("spec must be a ResearchExperimentSpec instance.")
        return cls(
            statement=spec.hypothesis,
            methodology_version=spec.methodology_version,
            strategy_name=spec.strategy_name,
            strategy_version=spec.strategy_version,
            dataset_scope=spec.dataset_scope,
            execution_assumptions=spec.execution_assumptions,
            code_provenance=spec.code_provenance,
            benchmark_reference=spec.benchmark_reference,
            parameters=dict(spec.parameters),
            random_seed=spec.random_seed,
            walk_forward_protocol=spec.walk_forward_protocol,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "methodology_version": self.methodology_version,
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "dataset_scope": asdict(self.dataset_scope),
            "execution_assumptions": asdict(self.execution_assumptions),
            "code_provenance": asdict(self.code_provenance),
            "benchmark_reference": self.benchmark_reference,
            "parameters": dict(self.parameters),
            "random_seed": self.random_seed,
            "walk_forward_protocol": asdict(self.walk_forward_protocol)
            if self.walk_forward_protocol
            else None,
            "source_knowledge_ids": list(self.source_knowledge_ids),
            "source_evidence_ids": list(self.source_evidence_ids),
            "hypothesis_version": self.hypothesis_version,
            "generation_method": self.generation_method,
            "generator_version": self.generator_version,
            "constraints": dict(self.constraints),
            "status": self.status.value,
            "created_at_utc": self.created_at_utc,
            "fingerprint": self.fingerprint,
        }


def compute_hypothesis_fingerprint(
    *,
    statement: str,
    methodology_version: str,
    strategy_name: str,
    strategy_version: str,
    dataset_scope: DatasetScope,
    execution_assumptions: ExecutionAssumptions,
    code_provenance: CodeProvenance,
    benchmark_reference: str,
    parameters: Mapping[str, Any] | None = None,
    random_seed: int | None = None,
    walk_forward_protocol: WalkForwardProtocol | None = None,
    source_knowledge_ids: Sequence[str] = (),
    source_evidence_ids: Sequence[str] = (),
    generation_method: str = "DIRECT_SPEC",
    generator_version: str = "1.0",
    constraints: Mapping[str, Any] | None = None,
) -> str:
    """Compute deterministic SHA-256 fingerprint for a ResearchHypothesis.

    Includes statement, source lineage, scope, assumptions, code provenance,
    parameters, constraints, and generator version. Excludes volatile creation timestamps.
    """
    canonical_payload = {
        "statement": statement.strip(),
        "methodology_version": methodology_version.strip(),
        "strategy_name": strategy_name.strip(),
        "strategy_version": strategy_version.strip(),
        "dataset_scope": {
            "dataset_id": dataset_scope.dataset_id.strip(),
            "symbol": dataset_scope.symbol.strip(),
            "timeframe": dataset_scope.timeframe.strip(),
            "start_date": dataset_scope.start_date.strip(),
            "end_date": dataset_scope.end_date.strip(),
        },
        "execution_assumptions": {
            "transaction_cost": float(execution_assumptions.transaction_cost),
            "slippage": float(execution_assumptions.slippage),
            "latency_ms": float(execution_assumptions.latency_ms),
        },
        "code_provenance": {
            "commit_sha": code_provenance.commit_sha.strip(),
        },
        "parameters": parameters or {},
        "benchmark_reference": benchmark_reference.strip(),
        "random_seed": random_seed,
        "source_knowledge_ids": sorted(str(k) for k in source_knowledge_ids),
        "source_evidence_ids": sorted(str(e) for e in source_evidence_ids),
        "generation_method": generation_method.strip(),
        "generator_version": generator_version.strip(),
        "constraints": constraints or {},
    }
    if walk_forward_protocol is not None:
        canonical_payload["walk_forward_protocol"] = {
            "train_size": int(walk_forward_protocol.train_size),
            "test_size": int(walk_forward_protocol.test_size),
        }

    serialized = json.dumps(
        canonical_payload,
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResearchCandidate:
    """Authoritative domain representation of a generated/evaluated research candidate.

    Binds hypothesis, experiment spec, optional evidence, validation state,
    promotion status, and rejection reasons.

    Enforces strict separation from live production signals and production decisions.
    """

    candidate_id: str
    hypothesis: ResearchHypothesis
    evidence: ResearchEvidence | None = None
    validation_status: PromotionStatus = PromotionStatus.PROPOSED
    promotion_status: PromotionStatus = PromotionStatus.PROPOSED
    rejection_reasons: tuple[RejectionReason, ...] = field(default_factory=tuple)
    created_at_utc: str = ""

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string.")
        if not isinstance(self.hypothesis, ResearchHypothesis):
            raise TypeError("hypothesis must be a ResearchHypothesis instance.")
        if self.evidence is not None:
            if not isinstance(self.evidence, ResearchEvidence):
                raise TypeError("evidence must be a ResearchEvidence instance.")
            if self.evidence.experiment_fingerprint != self.hypothesis.fingerprint:
                raise ValueError(
                    f"Candidate evidence fingerprint '{self.evidence.experiment_fingerprint}' "
                    f"does not match hypothesis fingerprint '{self.hypothesis.fingerprint}'."
                )
        if not isinstance(self.validation_status, PromotionStatus):
            raise TypeError("validation_status must be a PromotionStatus enum member.")
        if not isinstance(self.promotion_status, PromotionStatus):
            raise TypeError("promotion_status must be a PromotionStatus enum member.")

        for r in self.rejection_reasons:
            if not isinstance(r, RejectionReason):
                raise TypeError(f"Rejection reason '{r}' must be a RejectionReason enum member.")

        if self.promotion_status in (PromotionStatus.VALIDATED, PromotionStatus.PROMOTABLE):
            if self.evidence is None:
                raise ValueError("Validated or Promotable ResearchCandidate requires non-None evidence.")

    def has_provenance(self) -> bool:
        """Verify that candidate has non-empty code provenance."""
        cp = self.hypothesis.code_provenance
        return bool(cp and cp.commit_sha and cp.commit_sha.strip())

    def has_evidence(self) -> bool:
        """Verify that research evidence is attached and matching fingerprint."""
        if self.evidence is None:
            return False
        return self.evidence.experiment_fingerprint == self.hypothesis.fingerprint

    def has_reproducible_scope(self) -> bool:
        """Verify dataset scope and execution assumptions are validly attached."""
        ds = self.hypothesis.dataset_scope
        ea = self.hypothesis.execution_assumptions
        return bool(
            isinstance(ds, DatasetScope)
            and ds.dataset_id
            and ds.symbol
            and ds.timeframe
            and isinstance(ea, ExecutionAssumptions)
        )

    def has_validation_state(self) -> bool:
        """Verify candidate carries an explicit validation status."""
        return isinstance(self.validation_status, PromotionStatus)

    def is_distinct_from_live_decision(self) -> bool:
        """Assert that ResearchCandidate is strictly a research object, distinct from live trading signals or decisions."""
        class_name = self.__class__.__name__
        return class_name == "ResearchCandidate" and class_name not in ("ProductionDecision", "ProductionSignal", "ProductionIntelligencePublication")

    def lineage_reaches_original_experiment(self) -> bool:
        """Trace candidate lineage back to hypothesis and experiment spec."""
        if not self.hypothesis.fingerprint:
            return False
        if self.evidence is not None:
            return (
                self.evidence.experiment_fingerprint == self.hypothesis.fingerprint
                and self.evidence.spec.fingerprint == self.hypothesis.fingerprint
            )
        return True

    def can_promote(self, policy: Any | None = None) -> bool:
        """Check whether candidate can be promoted to production."""
        if self.evidence is None:
            return False
        if self.promotion_status not in (PromotionStatus.VALIDATED, PromotionStatus.PROMOTABLE):
            return False
        from src.evaluation.research_qualification import (
            ResearchQualificationPolicy,
            qualify_research_evidence,
        )
        qual_res = qualify_research_evidence(
            self.evidence,
            policy=policy if isinstance(policy, ResearchQualificationPolicy) else None,
        )
        return qual_res.qualified

    def promote_to_production_artifact(
        self,
        symbol: str,
        timeframe: str,
        policy: Any | None = None,
    ) -> Any:
        """Convert a qualified ResearchCandidate into a PromotedCandidateArtifact for production.

        Fails closed if the candidate is unvalidated, rejected, or unqualified.
        """
        if self.evidence is None:
            raise ValueError(
                f"Cannot promote ResearchCandidate '{self.candidate_id}': evidence is missing."
            )
        from src.evaluation.live_production_decision import PromotedCandidateArtifact
        return PromotedCandidateArtifact.from_persisted_research(
            candidate_id=self.candidate_id,
            evidence=self.evidence,
            symbol=symbol,
            timeframe=timeframe,
            parameters=dict(self.hypothesis.parameters),
            policy=policy,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "hypothesis": self.hypothesis.as_dict(),
            "evidence": self.evidence.as_dict() if self.evidence else None,
            "validation_status": self.validation_status.value,
            "promotion_status": self.promotion_status.value,
            "rejection_reasons": [r.value for r in self.rejection_reasons],
            "created_at_utc": self.created_at_utc,
        }


def compute_search_policy_fingerprint(*, max_trials: int, fail_fast: bool) -> str:
    """Compute deterministic SHA-256 fingerprint for a ResearchSearchPolicy."""
    payload = {
        "max_trials": int(max_trials),
        "fail_fast": bool(fail_fast),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def compute_criteria_fingerprint(criteria: Any) -> str:
    """Compute deterministic SHA-256 fingerprint for DiscoveryCriteria."""
    payload = {
        "min_observations_is": int(criteria.min_observations_is),
        "min_observations_oos": int(criteria.min_observations_oos),
        "min_is_sharpe": float(criteria.min_is_sharpe),
        "min_is_total_return": float(criteria.min_is_total_return),
        "max_is_drawdown": float(criteria.max_is_drawdown),
        "min_validation_sharpe": float(criteria.min_validation_sharpe),
        "min_oos_sharpe": float(criteria.min_oos_sharpe),
        "max_oos_sharpe_degradation": float(criteria.max_oos_sharpe_degradation),
        "min_walk_forward_positive_ratio": float(criteria.min_walk_forward_positive_ratio),
        "benchmark_reference": str(criteria.benchmark_reference),
        "methodology_version": str(criteria.methodology_version),
        "strategy_version": str(criteria.strategy_version),
        "robustness_criteria": {
            "version": str(criteria.robustness_criteria.version),
            "perturbation_pcts": [float(p) for p in criteria.robustness_criteria.perturbation_pcts],
            "min_perturbation_pass_rate": float(criteria.robustness_criteria.min_perturbation_pass_rate),
            "cost_stress_multipliers": [float(m) for m in criteria.robustness_criteria.cost_stress_multipliers],
            "min_cost_stress_pass_rate": float(criteria.robustness_criteria.min_cost_stress_pass_rate),
            "subsample_slices_count": int(criteria.robustness_criteria.subsample_slices_count),
            "min_subsample_pass_rate": float(criteria.robustness_criteria.min_subsample_pass_rate),
            "min_statistical_observations": int(criteria.robustness_criteria.min_statistical_observations),
            "min_t_stat": float(criteria.robustness_criteria.min_t_stat),
            "max_p_value": float(criteria.robustness_criteria.max_p_value),
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def compute_campaign_fingerprint(
    *,
    search_space_fingerprint: str,
    search_policy_fingerprint: str,
    criteria_fingerprint: str,
    dataset_scope: DatasetScope,
    execution_assumptions: ExecutionAssumptions,
    code_provenance: CodeProvenance,
    candidate_ids: Sequence[str],
) -> str:
    """Compute deterministic SHA-256 identity fingerprint for a Research Campaign."""
    payload = {
        "search_space_fingerprint": str(search_space_fingerprint).strip(),
        "search_policy_fingerprint": str(search_policy_fingerprint).strip(),
        "criteria_fingerprint": str(criteria_fingerprint).strip(),
        "dataset_scope": {
            "dataset_id": dataset_scope.dataset_id.strip(),
            "symbol": dataset_scope.symbol.strip(),
            "timeframe": dataset_scope.timeframe.strip(),
            "start_date": dataset_scope.start_date.strip(),
            "end_date": dataset_scope.end_date.strip(),
        },
        "execution_assumptions": {
            "transaction_cost": float(execution_assumptions.transaction_cost),
            "slippage": float(execution_assumptions.slippage),
            "latency_ms": float(execution_assumptions.latency_ms),
        },
        "code_provenance": {
            "commit_sha": code_provenance.commit_sha.strip(),
            "repository_status": code_provenance.repository_status.strip(),
            "author": code_provenance.author.strip(),
        },
        "candidate_ids": sorted(str(cid) for cid in candidate_ids),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class ResearchCampaign:
    """Authoritative domain representation of a multi-experiment Research Campaign / Discovery Run.

    Preserves a multi-experiment discovery run as a single scientific unit,
    binding search space, search policy, criteria, dataset scope, execution assumptions,
    trial ledger, evaluated evidence fingerprints, and selection results with deterministic
    reproducibility fingerprinting.
    """

    campaign_id: str
    search_space_fingerprint: str
    search_policy_fingerprint: str
    criteria_fingerprint: str
    dataset_scope: DatasetScope
    execution_assumptions: ExecutionAssumptions
    code_provenance: CodeProvenance
    candidate_ids: tuple[str, ...]
    evidence_fingerprints: tuple[str, ...]
    selected_candidate_ids: tuple[str, ...]
    status: ResearchCampaignStatus
    created_at_utc: str = ""
    reproducibility_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.campaign_id or not self.campaign_id.strip():
            raise ValueError("campaign_id must be a non-empty string.")
        if not self.search_space_fingerprint or not self.search_space_fingerprint.strip():
            raise ValueError("search_space_fingerprint must be a non-empty string.")
        if not self.search_policy_fingerprint or not self.search_policy_fingerprint.strip():
            raise ValueError("search_policy_fingerprint must be a non-empty string.")
        if not self.criteria_fingerprint or not self.criteria_fingerprint.strip():
            raise ValueError("criteria_fingerprint must be a non-empty string.")
        if not isinstance(self.dataset_scope, DatasetScope):
            raise TypeError("dataset_scope must be a DatasetScope instance.")
        if not isinstance(self.execution_assumptions, ExecutionAssumptions):
            raise TypeError("execution_assumptions must be an ExecutionAssumptions instance.")
        if not isinstance(self.code_provenance, CodeProvenance):
            raise TypeError("code_provenance must be a CodeProvenance instance.")
        if not isinstance(self.status, ResearchCampaignStatus):
            raise TypeError("status must be a ResearchCampaignStatus enum member.")

        expected_campaign_id = compute_campaign_fingerprint(
            search_space_fingerprint=self.search_space_fingerprint,
            search_policy_fingerprint=self.search_policy_fingerprint,
            criteria_fingerprint=self.criteria_fingerprint,
            dataset_scope=self.dataset_scope,
            execution_assumptions=self.execution_assumptions,
            code_provenance=self.code_provenance,
            candidate_ids=self.candidate_ids,
        )

        if self.campaign_id != expected_campaign_id:
            raise ValueError(
                f"Invalid campaign_id '{self.campaign_id}'. Expected calculated fingerprint '{expected_campaign_id}'."
            )

        # Calculate reproducibility fingerprint over specs and evaluation outputs
        repro_payload = {
            "campaign_id": self.campaign_id,
            "search_space_fingerprint": self.search_space_fingerprint,
            "search_policy_fingerprint": self.search_policy_fingerprint,
            "criteria_fingerprint": self.criteria_fingerprint,
            "dataset_scope_id": f"{self.dataset_scope.dataset_id}:{self.dataset_scope.symbol}:{self.dataset_scope.timeframe}:{self.dataset_scope.start_date}:{self.dataset_scope.end_date}",
            "execution_assumptions": {
                "transaction_cost": float(self.execution_assumptions.transaction_cost),
                "slippage": float(self.execution_assumptions.slippage),
                "latency_ms": float(self.execution_assumptions.latency_ms),
            },
            "code_provenance": self.code_provenance.commit_sha,
            "candidate_ids": sorted(self.candidate_ids),
            "evidence_fingerprints": sorted(self.evidence_fingerprints),
            "selected_candidate_ids": sorted(self.selected_candidate_ids),
            "status": self.status.value,
        }
        computed_repro = hashlib.sha256(json.dumps(repro_payload, sort_keys=True).encode("utf-8")).hexdigest()
        object.__setattr__(self, "reproducibility_fingerprint", computed_repro)

    def as_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "search_space_fingerprint": self.search_space_fingerprint,
            "search_policy_fingerprint": self.search_policy_fingerprint,
            "criteria_fingerprint": self.criteria_fingerprint,
            "dataset_scope": asdict(self.dataset_scope),
            "execution_assumptions": asdict(self.execution_assumptions),
            "code_provenance": asdict(self.code_provenance),
            "candidate_ids": list(self.candidate_ids),
            "evidence_fingerprints": list(self.evidence_fingerprints),
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "status": self.status.value,
            "created_at_utc": self.created_at_utc,
            "reproducibility_fingerprint": self.reproducibility_fingerprint,
        }
