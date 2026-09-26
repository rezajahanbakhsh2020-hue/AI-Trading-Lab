"""Canonical Research Robustness, Benchmark, and Regime-Aware Evidence Layer for Project 1.

Provides immutable, machine-readable representations of candidate robustness, partition provenance,
benchmark comparisons, and market regime evidence derived strictly from canonical ResearchEvidence artifacts.

Enforces:
- Deterministic SHA-256 fingerprinting for assessment reproducibility without Python hash()
- Partition-level temporal boundary and assumption provenance (In-Sample, Validation, OOS, Walk-Forward)
- Explicit machine-readable states for un-evaluated or unavailable robustness, benchmark, and regime analysis
- Anti-overfitting temporal separation and statistical observation tracking
- Zero execution side-effects (does not fetch market data, run strategy backtests, or mutate production state)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from src.evaluation.research_constitution import (
    EvidencePartition,
    EvidencePartitionRole,
    PromotionStatus,
    RejectionReason,
    ResearchEvidence,
    RobustnessCriteria,
)


class RobustnessStatus(str, Enum):
    """Machine-readable status for candidate robustness assessment."""

    NOT_EVALUATED = "NOT_EVALUATED"
    EVALUATED = "EVALUATED"
    PARTIALLY_EVALUATED = "PARTIALLY_EVALUATED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class BenchmarkStatus(str, Enum):
    """Machine-readable status for benchmark/baseline evaluation."""

    BENCHMARK_NOT_EVALUATED = "BENCHMARK_NOT_EVALUATED"
    BENCHMARK_EVALUATED = "BENCHMARK_EVALUATED"
    BENCHMARK_UNAVAILABLE = "BENCHMARK_UNAVAILABLE"
    BENCHMARK_INSUFFICIENT_DATA = "BENCHMARK_INSUFFICIENT_DATA"


class RegimeStatus(str, Enum):
    """Machine-readable status for market regime performance analysis."""

    REGIME_NOT_EVALUATED = "REGIME_NOT_EVALUATED"
    REGIME_EVALUATED = "REGIME_EVALUATED"
    REGIME_UNAVAILABLE = "REGIME_UNAVAILABLE"
    REGIME_INSUFFICIENT_DATA = "REGIME_INSUFFICIENT_DATA"


def _canonical_json_value(obj: Any) -> Any:
    """Helper to convert custom types/enums for canonical JSON serialization."""
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def compute_robustness_fingerprint(payload: Mapping[str, Any]) -> str:
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
class EvaluatedPartitionRecord:
    """Canonical, immutable provenance record for an evaluated dataset/evidence partition."""

    role: str
    dataset_id: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    execution_assumptions: dict[str, float]
    evidence_fingerprint: str
    observations: int
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    evaluation_status: str
    partition_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.role or not self.role.strip():
            raise ValueError("role must be a non-empty string.")
        if not self.start_date or not self.start_date.strip():
            raise ValueError("start_date must be a non-empty string.")
        if not self.end_date or not self.end_date.strip():
            raise ValueError("end_date must be a non-empty string.")
        if self.start_date > self.end_date:
            raise ValueError(
                f"start_date '{self.start_date}' cannot be later than end_date '{self.end_date}'."
            )
        if self.observations < 0:
            raise ValueError("observations must be non-negative.")

        payload = {
            "role": self.role.strip(),
            "dataset_id": self.dataset_id.strip(),
            "symbol": self.symbol.strip(),
            "timeframe": self.timeframe.strip(),
            "start_date": self.start_date.strip(),
            "end_date": self.end_date.strip(),
            "execution_assumptions": {
                k: float(v) for k, v in sorted(self.execution_assumptions.items())
            },
            "evidence_fingerprint": self.evidence_fingerprint.strip(),
            "observations": self.observations,
            "total_return": float(self.total_return),
            "max_drawdown": float(self.max_drawdown),
            "sharpe_ratio": float(self.sharpe_ratio),
            "evaluation_status": self.evaluation_status.strip(),
        }
        fp = compute_robustness_fingerprint(payload)
        object.__setattr__(self, "partition_fingerprint", fp)


@dataclass(frozen=True)
class BenchmarkAssessment:
    """Canonical assessment of benchmark/baseline evidence."""

    status: BenchmarkStatus
    benchmark_reference: str
    benchmark_total_return: float | None = None
    candidate_total_return: float | None = None
    excess_return: float | None = None
    outperformed_benchmark: bool | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, BenchmarkStatus):
            raise TypeError("status must be a BenchmarkStatus enum member.")
        if not self.benchmark_reference or not self.benchmark_reference.strip():
            raise ValueError("benchmark_reference must be a non-empty string.")


@dataclass(frozen=True)
class RegimeAssessment:
    """Canonical assessment of regime-aware performance and stability evidence."""

    status: RegimeStatus
    regime_methodology: str
    evaluated_regimes: tuple[str, ...] = ()
    regime_stability_score: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, RegimeStatus):
            raise TypeError("status must be a RegimeStatus enum member.")
        if not self.regime_methodology or not self.regime_methodology.strip():
            raise ValueError("regime_methodology must be a non-empty string.")


@dataclass(frozen=True)
class ResearchRobustnessAssessment:
    """Canonical, immutable research robustness, benchmark, and regime evidence artifact."""

    status: RobustnessStatus
    is_robust: bool
    experiment_fingerprint: str
    evidence_fingerprint: str
    dimensions_evaluated: tuple[str, ...]
    dimensions_unavailable: tuple[str, ...]
    partition_records: tuple[EvaluatedPartitionRecord, ...]
    benchmark_assessment: BenchmarkAssessment
    regime_assessment: RegimeAssessment
    stability_summary: str
    limitations: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    robustness_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.status, RobustnessStatus):
            raise TypeError("status must be a RobustnessStatus enum member.")
        if not isinstance(self.is_robust, bool):
            raise TypeError("is_robust must be a boolean.")
        if not self.experiment_fingerprint or not self.experiment_fingerprint.strip():
            raise ValueError("experiment_fingerprint must be a non-empty string.")
        if not self.evidence_fingerprint or not self.evidence_fingerprint.strip():
            raise ValueError("evidence_fingerprint must be a non-empty string.")
        if not isinstance(self.benchmark_assessment, BenchmarkAssessment):
            raise TypeError("benchmark_assessment must be a BenchmarkAssessment instance.")
        if not isinstance(self.regime_assessment, RegimeAssessment):
            raise TypeError("regime_assessment must be a RegimeAssessment instance.")

        for p in self.partition_records:
            if not isinstance(p, EvaluatedPartitionRecord):
                raise TypeError("All items in partition_records must be EvaluatedPartitionRecord instances.")

        payload = {
            "status": self.status.value,
            "is_robust": self.is_robust,
            "experiment_fingerprint": self.experiment_fingerprint.strip(),
            "evidence_fingerprint": self.evidence_fingerprint.strip(),
            "dimensions_evaluated": list(self.dimensions_evaluated),
            "dimensions_unavailable": list(self.dimensions_unavailable),
            "partition_fingerprints": [p.partition_fingerprint for p in self.partition_records],
            "benchmark_assessment": {
                "status": self.benchmark_assessment.status.value,
                "benchmark_reference": self.benchmark_assessment.benchmark_reference.strip(),
                "benchmark_total_return": self.benchmark_assessment.benchmark_total_return,
                "candidate_total_return": self.benchmark_assessment.candidate_total_return,
                "excess_return": self.benchmark_assessment.excess_return,
                "outperformed_benchmark": self.benchmark_assessment.outperformed_benchmark,
                "notes": self.benchmark_assessment.notes.strip(),
            },
            "regime_assessment": {
                "status": self.regime_assessment.status.value,
                "regime_methodology": self.regime_assessment.regime_methodology.strip(),
                "evaluated_regimes": list(self.regime_assessment.evaluated_regimes),
                "regime_stability_score": self.regime_assessment.regime_stability_score,
                "notes": self.regime_assessment.notes.strip(),
            },
            "stability_summary": self.stability_summary.strip(),
            "limitations": list(self.limitations),
            "metadata": self.metadata,
        }
        fp = compute_robustness_fingerprint(payload)
        object.__setattr__(self, "robustness_fingerprint", fp)


def assess_research_robustness(
    evidence: ResearchEvidence,
    robustness_criteria: RobustnessCriteria | None = None,
) -> ResearchRobustnessAssessment:
    """Perform deterministic research robustness, benchmark, and regime evidence assessment.

    Consumes an existing ResearchEvidence artifact strictly without re-executing backtests,
    re-fetching market data, or inventing arbitrary performance thresholds.
    """
    if not isinstance(evidence, ResearchEvidence):
        raise TypeError("evidence must be a ResearchEvidence instance.")

    crit = robustness_criteria or RobustnessCriteria()
    spec = evidence.spec
    verdict = evidence.robustness_verdict or {}
    benchmark_data = evidence.benchmark_comparison or {}

    # 1. Build partition records
    partition_records: list[EvaluatedPartitionRecord] = []
    for part in evidence.partitions:
        eval_status = "EVALUATED" if part.observations > 0 else "INSUFFICIENT_DATA"
        exec_assump = {
            "transaction_cost": float(spec.execution_assumptions.transaction_cost),
            "slippage": float(spec.execution_assumptions.slippage),
            "latency_ms": float(spec.execution_assumptions.latency_ms),
        }
        rec = EvaluatedPartitionRecord(
            role=part.role.value,
            dataset_id=spec.dataset_scope.dataset_id,
            symbol=spec.dataset_scope.symbol,
            timeframe=spec.dataset_scope.timeframe,
            start_date=part.start_date,
            end_date=part.end_date,
            execution_assumptions=exec_assump,
            evidence_fingerprint=evidence.evidence_id,
            observations=part.observations,
            total_return=part.total_return,
            max_drawdown=part.max_drawdown,
            sharpe_ratio=part.sharpe_ratio,
            evaluation_status=eval_status,
        )
        partition_records.append(rec)

    # 2. Evaluate Robustness Status & Dimensions
    all_known_dimensions = (
        "parameter_sensitivity",
        "subsample_stability",
        "execution_cost_stress",
        "statistical_validation",
        "anti_overfitting",
    )

    if not verdict:
        robustness_status = RobustnessStatus.NOT_EVALUATED
        is_robust = False
        dims_eval: tuple[str, ...] = ()
        dims_unavail: tuple[str, ...] = all_known_dimensions
        limitations: list[str] = ["Robustness checks were not executed for this research evidence."]
        summary = "Robustness evaluation not executed."
    else:
        dims_eval_list = []
        dims_unavail_list = []

        for dim in all_known_dimensions:
            if dim in verdict:
                dims_eval_list.append(dim)
            else:
                dims_unavail_list.append(dim)

        dims_eval = tuple(dims_eval_list)
        dims_unavail = tuple(dims_unavail_list)
        is_robust = bool(verdict.get("is_robust", False))

        reasons = verdict.get("rejection_reasons", [])
        limitations = []
        if "INSUFFICIENT_STATISTICAL_SAMPLE" in reasons or "INSUFFICIENT_DATA" in reasons:
            robustness_status = RobustnessStatus.INSUFFICIENT_DATA
            limitations.append("Insufficient statistical observations for high-confidence inference.")
        elif dims_unavail:
            robustness_status = RobustnessStatus.PARTIALLY_EVALUATED
            limitations.append(f"Unevaluated robustness dimensions: {', '.join(dims_unavail)}.")
        else:
            robustness_status = RobustnessStatus.EVALUATED

        if not is_robust:
            limitations.append(f"Candidate failed robustness checks: {verdict.get('verdict_summary', 'failed')}")

        summary = verdict.get("verdict_summary", "Robustness evaluation completed.")

    # 3. Evaluate Benchmark
    bm_ref = benchmark_data.get("benchmark_reference") or getattr(spec, "benchmark_reference", "none")
    if not benchmark_data or bm_ref.lower() in ("none", "unavailable", ""):
        bm_assessment = BenchmarkAssessment(
            status=BenchmarkStatus.BENCHMARK_UNAVAILABLE,
            benchmark_reference=bm_ref or "none",
            notes="No authoritative benchmark dataset or reference comparison was configured.",
        )
    else:
        bm_ret = benchmark_data.get("benchmark_total_return")
        is_part = next((p for p in evidence.partitions if p.role == EvidencePartitionRole.IN_SAMPLE), None)
        cand_ret = is_part.total_return if is_part else (evidence.partitions[0].total_return if evidence.partitions else None)
        excess = (cand_ret - bm_ret) if (cand_ret is not None and bm_ret is not None) else None
        outperf = benchmark_data.get("outperformed_benchmark")

        bm_status = BenchmarkStatus.BENCHMARK_EVALUATED if bm_ret is not None else BenchmarkStatus.BENCHMARK_INSUFFICIENT_DATA
        bm_assessment = BenchmarkAssessment(
            status=bm_status,
            benchmark_reference=bm_ref,
            benchmark_total_return=bm_ret,
            candidate_total_return=cand_ret,
            excess_return=excess,
            outperformed_benchmark=outperf,
            notes="Benchmark comparison evaluated against In-Sample partition." if bm_status == BenchmarkStatus.BENCHMARK_EVALUATED else "Benchmark data insufficient.",
        )

    # 4. Evaluate Market Regime Analysis
    regime_data = verdict.get("regime_analysis") or spec.parameters.get("regime_analysis")
    if not regime_data:
        regime_assessment = RegimeAssessment(
            status=RegimeStatus.REGIME_UNAVAILABLE,
            regime_methodology="none",
            notes="Authoritative market regime classification was not configured or evaluated for this experiment.",
        )
    else:
        regime_method = regime_data.get("methodology", "volatility_regime_v1")
        regimes_tuple = tuple(regime_data.get("evaluated_regimes", ()))
        stability_score = regime_data.get("regime_stability_score")
        regime_assessment = RegimeAssessment(
            status=RegimeStatus.REGIME_EVALUATED,
            regime_methodology=regime_method,
            evaluated_regimes=regimes_tuple,
            regime_stability_score=stability_score,
            notes=regime_data.get("notes", "Authoritative regime analysis evaluated."),
        )

    meta = {
        "robustness_criteria_version": crit.version,
        "robustness_criteria": asdict(crit),
        "partition_count": len(partition_records),
        "rejection_reasons": [r.value for r in evidence.rejection_reasons],
    }

    return ResearchRobustnessAssessment(
        status=robustness_status,
        is_robust=is_robust,
        experiment_fingerprint=evidence.experiment_fingerprint,
        evidence_fingerprint=evidence.evidence_id,
        dimensions_evaluated=dims_eval,
        dimensions_unavailable=dims_unavail,
        partition_records=tuple(partition_records),
        benchmark_assessment=bm_assessment,
        regime_assessment=regime_assessment,
        stability_summary=summary,
        limitations=tuple(limitations),
        metadata=meta,
    )
