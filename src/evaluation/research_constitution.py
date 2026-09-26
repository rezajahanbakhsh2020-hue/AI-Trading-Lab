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
