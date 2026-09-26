"""Research Discovery Engine for Project 1.

Orchestrates candidate generation, chronological dataset partitioning (In-Sample, Validation,
Out-Of-Sample, Walk-Forward), strategy evaluation, Research Constitution evidence creation,
fail-closed rejection/promotion decisioning, and optional evidence persistence.

No market data generation, silent substitution, or OOS leakage permitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Mapping, Sequence

import pandas as pd

from src.backtest.engine import run_backtest
from src.evaluation.candidate_generator import CandidateSpec
from src.evaluation.metrics import (
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    total_return,
    win_rate,
)
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
from src.evaluation.robustness_evaluator import RobustnessEvaluator
from src.evaluation.research_store import save_research_experiment
from src.evaluation.strategy_evaluator import evaluate_strategy
from src.evaluation.walk_forward import generate_walk_forward_windows
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

        Dataset is split chronologically into:
          - In-Sample (IS)
          - Validation
          - Out-of-Sample (OOS)
          - Walk-Forward (if specified or default windows constructed)
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

            if evidence.promotion_status in (
                PromotionStatus.PROMOTABLE,
                PromotionStatus.VALIDATED,
            ):
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
        """Evaluate a single candidate and produce a ResearchEvidence artifact."""
        rejection_reasons: list[RejectionReason] = []

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

        # Check duplicate candidate
        if spec.fingerprint in seen_fingerprints:
            rejection_reasons.append(RejectionReason.DUPLICATE_CANDIDATE)

        # Check strategy existence in registry
        if not self.registry.contains(cand.strategy_name):
            rejection_reasons.append(RejectionReason.SPECIFICATION_INVALID)

        # Build partitions
        partitions: list[EvidencePartition] = []

        # 1. In-Sample Partition
        is_part, is_rejections = self._evaluate_partition(
            role=EvidencePartitionRole.IN_SAMPLE,
            df_part=df_is,
            cand=cand,
            execution_assumptions=execution_assumptions,
            min_obs=self.criteria.min_observations_is,
        )
        partitions.append(is_part)
        rejection_reasons.extend(is_rejections)

        # Check IS threshold criteria
        if is_part.sharpe_ratio < self.criteria.min_is_sharpe:
            rejection_reasons.append(RejectionReason.FAILED_VALIDATION)
        if is_part.total_return < self.criteria.min_is_total_return:
            rejection_reasons.append(RejectionReason.FAILED_VALIDATION)

        # 2. Validation Partition
        val_part, val_rejections = self._evaluate_partition(
            role=EvidencePartitionRole.VALIDATION,
            df_part=df_val,
            cand=cand,
            execution_assumptions=execution_assumptions,
            min_obs=self.criteria.min_observations_oos,
        )
        partitions.append(val_part)
        rejection_reasons.extend(val_rejections)

        if val_part.sharpe_ratio < self.criteria.min_validation_sharpe:
            rejection_reasons.append(RejectionReason.FAILED_VALIDATION)

        # 3. Out-Of-Sample Partition
        oos_part, oos_rejections = self._evaluate_partition(
            role=EvidencePartitionRole.OUT_OF_SAMPLE,
            df_part=df_oos,
            cand=cand,
            execution_assumptions=execution_assumptions,
            min_obs=self.criteria.min_observations_oos,
        )
        partitions.append(oos_part)
        rejection_reasons.extend(oos_rejections)

        if oos_part.sharpe_ratio < self.criteria.min_oos_sharpe:
            rejection_reasons.append(RejectionReason.FAILED_OOS)

        # OOS Degradation check relative to IS
        if is_part.sharpe_ratio > 0:
            deg = (is_part.sharpe_ratio - oos_part.sharpe_ratio) / is_part.sharpe_ratio
            if deg > self.criteria.max_oos_sharpe_degradation:
                rejection_reasons.append(RejectionReason.FAILED_OOS)
                rejection_reasons.append(RejectionReason.IS_ONLY_SUCCESS)

        # 4. Walk-Forward Partition
        wf_part, wf_rejections = self._evaluate_walk_forward(
            df_full=df_full,
            cand=cand,
            execution_assumptions=execution_assumptions,
            wf_train_size=wf_train_size,
            wf_test_size=wf_test_size,
        )
        rejection_reasons.extend(wf_rejections)
        if wf_part is not None:
            partitions.append(wf_part)

        # Benchmark comparison on full dataset
        benchmark_comp = self._evaluate_benchmark(df_is, is_part.total_return)

        # 5. Robustness, Stress & Statistical Validation Step
        df_ref = pd.concat([df_is, df_val]).sort_values("timestamp").reset_index(drop=True) if "timestamp" in df_is.columns else pd.concat([df_is, df_val])

        robustness_eval = RobustnessEvaluator(
            criteria=self.criteria.robustness_criteria,
            registry=self.registry,
        )
        robustness_res = robustness_eval.evaluate_candidate_robustness(
            candidate=cand,
            df_reference=df_ref,
            df_oos=df_oos,
            dataset_scope=dataset_scope,
            execution_assumptions=execution_assumptions,
        )

        rejection_reasons.extend(robustness_res.rejection_reasons)

        # Determine final promotion status
        dedup_rejections = tuple(dict.fromkeys(rejection_reasons))

        if dedup_rejections:
            status = PromotionStatus.REJECTED
        else:
            status = PromotionStatus.PROMOTABLE

        now_utc = datetime.now(timezone.utc).isoformat()

        return ResearchEvidence(
            experiment_fingerprint=spec.fingerprint,
            spec=spec,
            partitions=tuple(partitions),
            robustness_verdict=robustness_res.as_dict(),
            benchmark_comparison=benchmark_comp,
            promotion_status=status,
            rejection_reasons=dedup_rejections,
            critique_notes=f"Discovery engine evaluation completed at {now_utc}.",
            created_at_utc=now_utc,
        )

    def _evaluate_partition(
        self,
        role: EvidencePartitionRole,
        df_part: pd.DataFrame,
        cand: CandidateSpec,
        execution_assumptions: ExecutionAssumptions,
        min_obs: int,
    ) -> tuple[EvidencePartition, list[RejectionReason]]:
        """Evaluate candidate on a single partition DataFrame."""
        rejections: list[RejectionReason] = []

        if "timestamp" in df_part.columns:
            ts = df_part["timestamp"]
        else:
            ts = df_part.index

        start_str = pd.to_datetime(ts.min()).strftime("%Y-%m-%d") if len(df_part) > 0 else "1970-01-01"
        end_str = pd.to_datetime(ts.max()).strftime("%Y-%m-%d") if len(df_part) > 0 else "1970-01-01"

        if len(df_part) < min_obs:
            rejections.append(RejectionReason.INSUFFICIENT_DATA)
            return (
                EvidencePartition(
                    role=role,
                    start_date=start_str,
                    end_date=end_str,
                    total_return=0.0,
                    max_drawdown=0.0,
                    sharpe_ratio=0.0,
                    observations=len(df_part),
                ),
                rejections,
            )

        try:
            eval_res = evaluate_strategy(
                df=df_part,
                name=cand.strategy_name,
                registry=self.registry,
                strategy_kwargs=cand.parameters,
                transaction_cost=execution_assumptions.transaction_cost,
                slippage=execution_assumptions.slippage,
            )

            part = EvidencePartition(
                role=role,
                start_date=start_str,
                end_date=end_str,
                total_return=eval_res.total_return,
                max_drawdown=eval_res.max_drawdown,
                sharpe_ratio=eval_res.sharpe_ratio,
                win_rate=eval_res.win_rate,
                profit_factor=eval_res.profit_factor,
                observations=eval_res.observations,
            )
            return part, rejections
        except Exception:
            rejections.append(RejectionReason.SPECIFICATION_INVALID)
            return (
                EvidencePartition(
                    role=role,
                    start_date=start_str,
                    end_date=end_str,
                    total_return=0.0,
                    max_drawdown=0.0,
                    sharpe_ratio=0.0,
                    observations=len(df_part),
                ),
                rejections,
            )

    def _evaluate_walk_forward(
        self,
        df_full: pd.DataFrame,
        cand: CandidateSpec,
        execution_assumptions: ExecutionAssumptions,
        wf_train_size: int | None,
        wf_test_size: int | None,
    ) -> tuple[EvidencePartition | None, list[RejectionReason]]:
        """Perform walk-forward validation across windows and return aggregated partition."""
        rejections: list[RejectionReason] = []
        n = len(df_full)

        train_sz = wf_train_size or int(n * 0.4)
        test_sz = wf_test_size or int(n * 0.15)

        windows = generate_walk_forward_windows(
            df_full, train_size=train_sz, test_size=test_sz
        )

        if not windows:
            rejections.append(RejectionReason.INSUFFICIENT_DATA)
            return None, rejections

        wf_returns: list[float] = []
        positive_windows = 0

        for w in windows:
            df_test = df_full.iloc[w.test_start : w.test_end]
            if df_test.empty:
                continue

            try:
                eval_res = evaluate_strategy(
                    df=df_test,
                    name=cand.strategy_name,
                    registry=self.registry,
                    strategy_kwargs=cand.parameters,
                    transaction_cost=execution_assumptions.transaction_cost,
                    slippage=execution_assumptions.slippage,
                )
                ret = eval_res.total_return
                wf_returns.append(ret)
                if ret > 0:
                    positive_windows += 1
            except Exception:
                pass

        if not wf_returns:
            rejections.append(RejectionReason.FAILED_WALK_FORWARD)
            return None, rejections

        pos_ratio = positive_windows / len(wf_returns)
        if pos_ratio < self.criteria.min_walk_forward_positive_ratio:
            rejections.append(RejectionReason.FAILED_WALK_FORWARD)

        if "timestamp" in df_full.columns:
            ts = df_full["timestamp"]
        else:
            ts = df_full.index

        start_str = pd.to_datetime(ts.min()).strftime("%Y-%m-%d")
        end_str = pd.to_datetime(ts.max()).strftime("%Y-%m-%d")

        mean_ret = sum(wf_returns) / len(wf_returns)

        wf_partition = EvidencePartition(
            role=EvidencePartitionRole.WALK_FORWARD,
            start_date=start_str,
            end_date=end_str,
            total_return=round(mean_ret, 6),
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            observations=len(wf_returns),
            additional_metrics={
                "walk_forward_windows": float(len(wf_returns)),
                "positive_window_ratio": float(pos_ratio),
            },
        )

        return wf_partition, rejections

    def _evaluate_benchmark(
        self, df_is: pd.DataFrame, candidate_is_return: float
    ) -> dict[str, Any]:
        """Calculate benchmark comparison (buy and hold return) over In-Sample period."""
        if "return" in df_is.columns:
            benchmark_return = float((1.0 + df_is["return"]).prod() - 1.0)
        elif "close" in df_is.columns and len(df_is) > 1:
            benchmark_return = float(
                (df_is["close"].iloc[-1] - df_is["close"].iloc[0])
                / df_is["close"].iloc[0]
            )
        else:
            benchmark_return = 0.0

        return {
            "benchmark_reference": self.criteria.benchmark_reference,
            "benchmark_total_return": round(benchmark_return, 6),
            "outperformed_benchmark": candidate_is_return > benchmark_return,
        }
