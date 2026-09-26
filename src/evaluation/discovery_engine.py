"""Research Discovery Engine for Project 1.

Orchestrates candidate generation, chronological dataset partitioning (In-Sample, Validation,
Out-Of-Sample, Walk-Forward), strategy evaluation via canonical research experiment runner,
Research Constitution evidence creation, fail-closed rejection/promotion decisioning, and
optional evidence persistence.

No market data generation, silent substitution, or OOS leakage permitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Mapping, Sequence

import pandas as pd

from src.evaluation.candidate_generator import CandidateSpec
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    EvidencePartition,
    EvidencePartitionRole,
    ExecutionAssumptions,
    PromotionStatus,
    RejectionReason,
    ResearchEvidence,
    ResearchExperimentSpec,
    RobustnessCriteria,
)
from src.evaluation.research_runner import run_research_experiment
from src.evaluation.research_store import save_research_experiment
from src.features.indicators import add_returns
from src.strategies.registry import DEFAULT_REGISTRY, StrategyRegistry


@dataclass(frozen=True)
class DiscoveryCriteria:
    """Configurable quantitative thresholds and criteria for Research Discovery Engine evaluation."""

    min_observations_is: int = 30
    min_observations_oos: int = 10
    min_is_sharpe: float = 0.0
    min_is_total_return: float = -1.0
    max_is_drawdown: float = 1.0  # drawdown is negative, e.g., -0.5
    min_validation_sharpe: float = -0.5
    min_oos_sharpe: float = -0.5
    max_oos_sharpe_degradation: float = 0.8  # Max OOS Sharpe drop relative to IS
    min_walk_forward_positive_ratio: float = 0.5
    benchmark_reference: str = "buy_and_hold"
    methodology_version: str = "discovery_v1.0"
    strategy_version: str = "1.0.0"
    robustness_criteria: RobustnessCriteria = field(default_factory=RobustnessCriteria)

    def __post_init__(self) -> None:
        if self.min_observations_is <= 0:
            raise ValueError("min_observations_is must be positive.")
        if self.min_observations_oos <= 0:
            raise ValueError("min_observations_oos must be positive.")
        if not isinstance(self.robustness_criteria, RobustnessCriteria):
            raise TypeError("robustness_criteria must be a RobustnessCriteria instance.")


@dataclass(frozen=True)
class DiscoveryRunResult:
    """Aggregated result of a Research Discovery Engine execution run."""

    dataset_scope: DatasetScope
    execution_assumptions: ExecutionAssumptions
    code_provenance: CodeProvenance
    candidates_evaluated: int
    promoted_evidence: tuple[ResearchEvidence, ...]
    rejected_evidence: tuple[ResearchEvidence, ...]

    @property
    def total_candidates(self) -> int:
        return len(self.promoted_evidence) + len(self.rejected_evidence)


class DiscoveryEngine:
    """Orchestrates candidate evaluation and evidence synthesis across partitioned data."""

    def __init__(
        self,
        criteria: DiscoveryCriteria | None = None,
        registry: StrategyRegistry = DEFAULT_REGISTRY,
    ) -> None:
        self.criteria = criteria or DiscoveryCriteria()
        self.registry = registry

    def run_discovery(
        self,
        df: pd.DataFrame,
        candidates: Sequence[CandidateSpec],
        dataset_scope: DatasetScope,
        execution_assumptions: ExecutionAssumptions,
        code_provenance: CodeProvenance,
        *,
        val_ratio: float = 0.2,
        oos_ratio: float = 0.3,
        wf_train_size: int | None = None,
        wf_test_size: int | None = None,
        persist_evidence: bool = False,
    ) -> DiscoveryRunResult:
        """Execute discovery workflow over candidates using dataset partitioning.

        Delegates candidate evaluation strictly through run_research_experiment.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("df must not be empty.")
        if "timestamp" not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("df must have a 'timestamp' column or DatetimeIndex.")

        data = df.copy()
        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data = data.sort_values("timestamp").reset_index(drop=True)
        else:
            data = data.sort_index()

        if "return" not in data.columns:
            data = add_returns(data)

        # Validate dataset scope vs DataFrame boundaries
        self._validate_dataset_scope(data, dataset_scope)

        # Chronological partitioning
        n = len(data)
        is_ratio = 1.0 - val_ratio - oos_ratio
        if is_ratio <= 0:
            raise ValueError(
                f"Invalid partition ratios: val_ratio ({val_ratio}) + oos_ratio ({oos_ratio}) >= 1.0"
            )

        n_is = int(n * is_ratio)
        n_val = int(n * val_ratio)
        n_oos = n - n_is - n_val

        df_is = data.iloc[:n_is]
        df_val = data.iloc[n_is : n_is + n_val]
        df_oos = data.iloc[n_is + n_val :]

        promoted: list[ResearchEvidence] = []
        rejected: list[ResearchEvidence] = []
        seen_candidate_fingerprints: set[str] = set()

        for cand in candidates:
            evidence = self._evaluate_candidate(
                cand=cand,
                df_full=data,
                df_is=df_is,
                df_val=df_val,
                df_oos=df_oos,
                dataset_scope=dataset_scope,
                execution_assumptions=execution_assumptions,
                code_provenance=code_provenance,
                wf_train_size=wf_train_size,
                wf_test_size=wf_test_size,
                seen_fingerprints=seen_candidate_fingerprints,
            )

            seen_candidate_fingerprints.add(evidence.experiment_fingerprint)

            if persist_evidence:
                save_research_experiment(evidence)

            from src.evaluation.research_qualification import qualify_research_evidence
            qual_res = qualify_research_evidence(evidence)

            if qual_res.qualified:
                promoted.append(evidence)
            else:
                rejected.append(evidence)

        return DiscoveryRunResult(
            dataset_scope=dataset_scope,
            execution_assumptions=execution_assumptions,
            code_provenance=code_provenance,
            candidates_evaluated=len(candidates),
            promoted_evidence=tuple(promoted),
            rejected_evidence=tuple(rejected),
        )

    def _validate_dataset_scope(
        self, df: pd.DataFrame, dataset_scope: DatasetScope
    ) -> None:
        """Fail closed if dataset scope start/end dates violate data limits."""
        if "timestamp" in df.columns:
            ts = df["timestamp"]
        else:
            ts = df.index

        df_start = pd.to_datetime(ts.min()).strftime("%Y-%m-%d")
        df_end = pd.to_datetime(ts.max()).strftime("%Y-%m-%d")

        scope_start = dataset_scope.start_date[:10]
        scope_end = dataset_scope.end_date[:10]

        if scope_start < df_start or scope_end > df_end:
            raise ValueError(
                f"DatasetScope dates [{scope_start}, {scope_end}] extend beyond actual "
                f"data boundaries [{df_start}, {df_end}]."
            )

    def _evaluate_candidate(
        self,
        *,
        cand: CandidateSpec,
        df_full: pd.DataFrame,
        df_is: pd.DataFrame,
        df_val: pd.DataFrame,
        df_oos: pd.DataFrame,
        dataset_scope: DatasetScope,
        execution_assumptions: ExecutionAssumptions,
        code_provenance: CodeProvenance,
        wf_train_size: int | None,
        wf_test_size: int | None,
        seen_fingerprints: set[str],
    ) -> ResearchEvidence:
        """Evaluate a single candidate by constructing a ResearchExperimentSpec and delegating to run_research_experiment."""
        hypothesis = (
            cand.hypothesis_template.replace("{candidate_id}", cand.candidate_id)
            if cand.hypothesis_template
            else f"Hypothesis for candidate {cand.candidate_id}"
        )

        spec = ResearchExperimentSpec(
            hypothesis=hypothesis,
            methodology_version=self.criteria.methodology_version,
            strategy_name=cand.strategy_name,
            strategy_version=self.criteria.strategy_version,
            dataset_scope=dataset_scope,
            execution_assumptions=execution_assumptions,
            code_provenance=code_provenance,
            benchmark_reference=self.criteria.benchmark_reference,
            parameters=cand.parameters,
            random_seed=cand.random_seed,
        )

        evidence = run_research_experiment(
            spec=spec,
            df=df_full,
            criteria=self.criteria,
            registry=self.registry,
            wf_train_size=wf_train_size,
            wf_test_size=wf_test_size,
            persist_evidence=False,
        )

        if spec.fingerprint in seen_fingerprints and RejectionReason.DUPLICATE_CANDIDATE not in evidence.rejection_reasons:
            rejection_reasons = list(evidence.rejection_reasons) + [RejectionReason.DUPLICATE_CANDIDATE]
            evidence = ResearchEvidence(
                experiment_fingerprint=evidence.experiment_fingerprint,
                spec=evidence.spec,
                partitions=evidence.partitions,
                robustness_verdict=evidence.robustness_verdict,
                benchmark_comparison=evidence.benchmark_comparison,
                promotion_status=PromotionStatus.REJECTED,
                rejection_reasons=tuple(dict.fromkeys(rejection_reasons)),
                critique_notes=evidence.critique_notes,
                created_at_utc=evidence.created_at_utc,
            )

        return evidence
