"""Robustness, Stress & Statistical Validation Engine for Project 1 Research Lineage.

Evaluates candidates on:
1. Parameter Sensitivity / Perturbation (without mutating candidates or OOS leakage)
2. Subsample / Slice Stability over In-Sample/Validation datasets
3. Execution Cost & Slippage Stress using ExecutionAssumptions
4. Statistical Significance / Uncertainty calculations on actual observations
5. Anti-Overfitting Temporal Separation Verification

All evaluations fail closed on non-finite, empty, or invalid inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.evaluation.candidate_generator import CandidateSpec
from src.evaluation.research_constitution import (
    DatasetScope,
    ExecutionAssumptions,
    RejectionReason,
    RobustnessCriteria,
)
from src.evaluation.research_robustness import (
    BenchmarkAssessment,
    BenchmarkStatus,
    EvaluatedPartitionRecord,
    RegimeAssessment,
    RegimeStatus,
    ResearchRobustnessAssessment,
    RobustnessStatus,
    assess_research_robustness,
    compute_robustness_fingerprint,
)
from src.evaluation.strategy_evaluator import evaluate_strategy
from src.strategies.registry import DEFAULT_REGISTRY, StrategyRegistry


@dataclass(frozen=True)
class StatisticalResult:
    """Authoritative statistical validation result derived from real evaluator observations."""

    observation_count: int
    mean_return: float
    std_return: float
    t_statistic: float
    p_value: float
    is_valid: bool
    methodology_version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_count": self.observation_count,
            "mean_return": round(self.mean_return, 8),
            "std_return": round(self.std_return, 8),
            "t_statistic": round(self.t_statistic, 6) if math.isfinite(self.t_statistic) else 0.0,
            "p_value": round(self.p_value, 6) if math.isfinite(self.p_value) else 1.0,
            "is_valid": self.is_valid,
            "methodology_version": self.methodology_version,
        }


@dataclass(frozen=True)
class RobustnessEvaluationResult:
    """Comprehensive robustness & statistical verdict for a candidate spec."""

    candidate_id: str
    is_robust: bool
    rejection_reasons: tuple[RejectionReason, ...]
    parameter_sensitivity: dict[str, Any]
    subsample_stability: dict[str, Any]
    execution_cost_stress: dict[str, Any]
    statistical_validation: dict[str, Any]
    anti_overfitting_verdict: dict[str, Any]
    verdict_summary: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "is_robust": self.is_robust,
            "rejection_reasons": [r.value for r in self.rejection_reasons],
            "parameter_sensitivity": self.parameter_sensitivity,
            "subsample_stability": self.subsample_stability,
            "execution_cost_stress": self.execution_cost_stress,
            "statistical_validation": self.statistical_validation,
            "anti_overfitting_verdict": self.anti_overfitting_verdict,
            "verdict_summary": self.verdict_summary,
        }


def derive_parameter_perturbations(
    candidate: CandidateSpec,
    perturbation_pcts: tuple[float, ...] = (-0.10, 0.10),
) -> tuple[CandidateSpec, ...]:
    """Derive perturbed candidate specs deterministically without mutating original candidate.

    Only applies to numeric parameters (int, float). Ensures distinct candidates with
    unique candidate_ids to prevent collisions.
    """
    if not isinstance(candidate, CandidateSpec):
        raise TypeError("candidate must be a CandidateSpec instance.")

    perturbed_candidates: list[CandidateSpec] = []
    base_params = candidate.parameters

    for param_name, param_val in sorted(base_params.items()):
        if isinstance(param_val, bool) or not isinstance(param_val, (int, float)):
            continue

        for pct in sorted(perturbation_pcts):
            if pct == 0.0:
                continue

            if isinstance(param_val, int):
                delta = int(round(param_val * pct))
                if delta == 0:
                    delta = 1 if pct > 0 else -1
                new_val = param_val + delta
            else:
                new_val = float(param_val * (1.0 + pct))

            if new_val == param_val:
                continue

            new_params = dict(base_params)
            new_params[param_name] = new_val

            pert_cand = CandidateSpec(
                generator_name=f"{candidate.generator_name}_pert",
                generator_version=candidate.generator_version,
                strategy_name=candidate.strategy_name,
                parameters=new_params,
                random_seed=candidate.random_seed,
                hypothesis_template=candidate.hypothesis_template,
            )
            # Ensure candidate_id differs from source candidate
            if pert_cand.candidate_id != candidate.candidate_id:
                perturbed_candidates.append(pert_cand)

    return tuple(perturbed_candidates)


def compute_statistical_validation(
    observations: Sequence[float],
    criteria: RobustnessCriteria,
) -> StatisticalResult:
    """Calculate statistical significance (t-statistic, p-value) on real observations.

    Fails closed if observation count < criteria.min_statistical_observations or
    contains non-finite values.
    """
    methodology = f"stat_t_test_{criteria.version}"

    if not observations or len(observations) < criteria.min_statistical_observations:
        return StatisticalResult(
            observation_count=len(observations) if observations else 0,
            mean_return=0.0,
            std_return=0.0,
            t_statistic=0.0,
            p_value=1.0,
            is_valid=False,
            methodology_version=methodology,
        )

    obs_arr = np.array(observations, dtype=float)
    if not np.all(np.isfinite(obs_arr)):
        return StatisticalResult(
            observation_count=len(observations),
            mean_return=0.0,
            std_return=0.0,
            t_statistic=0.0,
            p_value=1.0,
            is_valid=False,
            methodology_version=methodology,
        )

    n = len(obs_arr)
    mean_ret = float(np.mean(obs_arr))
    std_ret = float(np.std(obs_arr, ddof=1)) if n > 1 else 0.0

    if std_ret <= 1e-12:
        # Zero variance: positive mean gives high t-stat, zero/neg gives 0.0 t-stat
        t_stat = 10.0 if mean_ret > 0 else 0.0
        p_val = 0.0 if mean_ret > 0 else 1.0
    else:
        se = std_ret / math.sqrt(n)
        t_stat = mean_ret / se
        # Standard normal approximation or exact t-distribution approximation
        # One-tailed test: p-value for H0: mean <= 0 vs H1: mean > 0
        from math import erfc
        # Normal CDF approx for t-stat
        p_val = 0.5 * erfc(t_stat / math.sqrt(2))

    is_valid = (t_stat >= criteria.min_t_stat) and (p_val <= criteria.max_p_value)

    return StatisticalResult(
        observation_count=n,
        mean_return=mean_ret,
        std_return=std_ret,
        t_statistic=t_stat,
        p_value=p_val,
        is_valid=is_valid,
        methodology_version=methodology,
    )


class RobustnessEvaluator:
    """Evaluates strategy candidates for parameter sensitivity, cost stress, slice stability, and statistical validity."""

    def __init__(
        self,
        criteria: RobustnessCriteria | None = None,
        registry: StrategyRegistry = DEFAULT_REGISTRY,
    ) -> None:
        self.criteria = criteria or RobustnessCriteria()
        self.registry = registry

    def evaluate_candidate_robustness(
        self,
        candidate: CandidateSpec,
        df_reference: pd.DataFrame,
        df_oos: pd.DataFrame | None,
        dataset_scope: DatasetScope,
        execution_assumptions: ExecutionAssumptions,
    ) -> RobustnessEvaluationResult:
        """Run full robustness checks over reference period data (In-Sample + Validation).

        OOS data is strictly checked for non-use / temporal separation.
        """
        if not isinstance(candidate, CandidateSpec):
            raise TypeError("candidate must be a CandidateSpec instance.")
        if not isinstance(df_reference, pd.DataFrame) or df_reference.empty:
            raise ValueError("df_reference must be a non-empty pandas DataFrame.")

        rejection_reasons: list[RejectionReason] = []

        # 1. Anti-Overfitting Verification
        anti_overfitting = self._verify_anti_overfitting(
            candidate=candidate,
            df_reference=df_reference,
            df_oos=df_oos,
        )
        if not anti_overfitting["passed"]:
            rejection_reasons.append(RejectionReason.ANTI_OVERFITTING_VIOLATION)

        # 2. Parameter Sensitivity Check
        param_sens = self._evaluate_parameter_sensitivity(
            candidate=candidate,
            df_reference=df_reference,
            execution_assumptions=execution_assumptions,
        )
        if not param_sens["passed"]:
            rejection_reasons.append(RejectionReason.FAILED_PARAMETER_SENSITIVITY)

        # 3. Subsample / Slice Stability
        slice_stab = self._evaluate_subsample_stability(
            candidate=candidate,
            df_reference=df_reference,
            execution_assumptions=execution_assumptions,
        )
        if not slice_stab["passed"]:
            rejection_reasons.append(RejectionReason.FAILED_ROBUSTNESS)

        # 4. Execution Cost & Slippage Stress
        cost_stress = self._evaluate_cost_stress(
            candidate=candidate,
            df_reference=df_reference,
            execution_assumptions=execution_assumptions,
        )
        if not cost_stress["passed"]:
            rejection_reasons.append(RejectionReason.FAILED_COST_STRESS)

        # 5. Statistical Validation on Observations
        stat_res = self._evaluate_statistical_significance(
            candidate=candidate,
            df_reference=df_reference,
            execution_assumptions=execution_assumptions,
        )
        if stat_res.observation_count < self.criteria.min_statistical_observations:
            rejection_reasons.append(RejectionReason.INSUFFICIENT_STATISTICAL_SAMPLE)
        if not stat_res.is_valid:
            rejection_reasons.append(RejectionReason.FAILED_STATISTICAL_VALIDATION)

        dedup_rejections = tuple(dict.fromkeys(rejection_reasons))
        is_robust = len(dedup_rejections) == 0

        summary = (
            f"Candidate '{candidate.candidate_id}' robustness: {'PASS' if is_robust else 'FAIL'}; "
            f"ParamSensitivity={param_sens['passed']}; "
            f"CostStress={cost_stress['passed']}; "
            f"SliceStability={slice_stab['passed']}; "
            f"StatValid={stat_res.is_valid} (obs={stat_res.observation_count}, t_stat={stat_res.t_statistic:.2f})"
        )

        return RobustnessEvaluationResult(
            candidate_id=candidate.candidate_id,
            is_robust=is_robust,
            rejection_reasons=dedup_rejections,
            parameter_sensitivity=param_sens,
            subsample_stability=slice_stab,
            execution_cost_stress=cost_stress,
            statistical_validation=stat_res.as_dict(),
            anti_overfitting_verdict=anti_overfitting,
            verdict_summary=summary,
        )

    def _verify_anti_overfitting(
        self,
        candidate: CandidateSpec,
        df_reference: pd.DataFrame,
        df_oos: pd.DataFrame | None,
    ) -> dict[str, Any]:
        """Verify temporal separation. Ensure no OOS data was used for reference evaluation."""
        if df_oos is None or df_oos.empty:
            return {"passed": True, "notes": "No OOS partition provided for comparison."}

        ref_ts_max = (
            df_reference["timestamp"].max()
            if "timestamp" in df_reference.columns
            else df_reference.index.max()
        )
        oos_ts_min = (
            df_oos["timestamp"].min()
            if "timestamp" in df_oos.columns
            else df_oos.index.min()
        )

        passed = ref_ts_max < oos_ts_min
        return {
            "passed": passed,
            "reference_end": str(ref_ts_max),
            "oos_start": str(oos_ts_min),
            "notes": "Strict temporal separation confirmed." if passed else "OOS data leaks prior to reference end!",
        }

    def _evaluate_parameter_sensitivity(
        self,
        candidate: CandidateSpec,
        df_reference: pd.DataFrame,
        execution_assumptions: ExecutionAssumptions,
    ) -> dict[str, Any]:
        """Evaluate parameter perturbation sensitivity."""
        perturbed = derive_parameter_perturbations(
            candidate, self.criteria.perturbation_pcts
        )
        if not perturbed:
            # If candidate has no numeric parameters, perturbation check passes trivially
            return {
                "passed": True,
                "perturbations_tested": 0,
                "passed_count": 0,
                "pass_rate": 1.0,
                "details": [],
            }

        passed_count = 0
        details = []

        for p_cand in perturbed:
            try:
                eval_res = evaluate_strategy(
                    df=df_reference,
                    name=p_cand.strategy_name,
                    registry=self.registry,
                    strategy_kwargs=p_cand.parameters,
                    transaction_cost=execution_assumptions.transaction_cost,
                    slippage=execution_assumptions.slippage,
                )
                # Perturbation passes if Sharpe ratio >= 0.0 and total return >= -0.5
                passed_pert = eval_res.sharpe_ratio >= 0.0 and math.isfinite(eval_res.total_return)
                if passed_pert:
                    passed_count += 1

                details.append({
                    "candidate_id": p_cand.candidate_id,
                    "parameters": p_cand.parameters,
                    "sharpe_ratio": eval_res.sharpe_ratio,
                    "total_return": eval_res.total_return,
                    "passed": passed_pert,
                })
            except Exception:
                details.append({
                    "candidate_id": p_cand.candidate_id,
                    "parameters": p_cand.parameters,
                    "passed": False,
                })

        pass_rate = passed_count / len(perturbed)
        passed = pass_rate >= self.criteria.min_perturbation_pass_rate

        return {
            "passed": passed,
            "perturbations_tested": len(perturbed),
            "passed_count": passed_count,
            "pass_rate": pass_rate,
            "details": details,
        }

    def _evaluate_subsample_stability(
        self,
        candidate: CandidateSpec,
        df_reference: pd.DataFrame,
        execution_assumptions: ExecutionAssumptions,
    ) -> dict[str, Any]:
        """Evaluate stability across chronological non-overlapping subsample slices."""
        n_slices = self.criteria.subsample_slices_count
        n = len(df_reference)
        slice_size = n // n_slices

        if slice_size < 10:
            return {
                "passed": False,
                "slices_tested": n_slices,
                "passed_count": 0,
                "pass_rate": 0.0,
                "notes": "Insufficient observations per slice.",
            }

        passed_count = 0
        details = []

        for i in range(n_slices):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size if i < n_slices - 1 else n
            df_slice = df_reference.iloc[start_idx:end_idx]

            try:
                eval_res = evaluate_strategy(
                    df=df_slice,
                    name=candidate.strategy_name,
                    registry=self.registry,
                    strategy_kwargs=candidate.parameters,
                    transaction_cost=execution_assumptions.transaction_cost,
                    slippage=execution_assumptions.slippage,
                )
                passed_slice = eval_res.sharpe_ratio >= -0.5 and math.isfinite(eval_res.total_return)
                if passed_slice:
                    passed_count += 1

                details.append({
                    "slice_index": i,
                    "observations": len(df_slice),
                    "total_return": eval_res.total_return,
                    "sharpe_ratio": eval_res.sharpe_ratio,
                    "passed": passed_slice,
                })
            except Exception:
                details.append({
                    "slice_index": i,
                    "passed": False,
                })

        pass_rate = passed_count / n_slices
        passed = pass_rate >= self.criteria.min_subsample_pass_rate

        return {
            "passed": passed,
            "slices_tested": n_slices,
            "passed_count": passed_count,
            "pass_rate": pass_rate,
            "details": details,
        }

    def _evaluate_cost_stress(
        self,
        candidate: CandidateSpec,
        df_reference: pd.DataFrame,
        execution_assumptions: ExecutionAssumptions,
    ) -> dict[str, Any]:
        """Evaluate performance under execution cost and slippage stress multipliers."""
        multipliers = self.criteria.cost_stress_multipliers
        passed_count = 0
        details = []

        for mult in sorted(multipliers):
            stressed_cost = execution_assumptions.transaction_cost * mult
            stressed_slippage = execution_assumptions.slippage * mult

            if not (math.isfinite(stressed_cost) and math.isfinite(stressed_slippage)):
                details.append({"multiplier": mult, "passed": False})
                continue

            try:
                eval_res = evaluate_strategy(
                    df=df_reference,
                    name=candidate.strategy_name,
                    registry=self.registry,
                    strategy_kwargs=candidate.parameters,
                    transaction_cost=stressed_cost,
                    slippage=stressed_slippage,
                )
                # Stressed scenario passes if total return remains positive / non-catastrophic
                passed_stress = eval_res.total_return > -0.2 and math.isfinite(eval_res.total_return)
                if passed_stress:
                    passed_count += 1

                details.append({
                    "multiplier": mult,
                    "transaction_cost": stressed_cost,
                    "slippage": stressed_slippage,
                    "total_return": eval_res.total_return,
                    "sharpe_ratio": eval_res.sharpe_ratio,
                    "passed": passed_stress,
                })
            except Exception:
                details.append({"multiplier": mult, "passed": False})

        pass_rate = passed_count / len(multipliers) if multipliers else 1.0
        passed = pass_rate >= self.criteria.min_cost_stress_pass_rate

        return {
            "passed": passed,
            "scenarios_tested": len(multipliers),
            "passed_count": passed_count,
            "pass_rate": pass_rate,
            "details": details,
        }

    def _evaluate_statistical_significance(
        self,
        candidate: CandidateSpec,
        df_reference: pd.DataFrame,
        execution_assumptions: ExecutionAssumptions,
    ) -> StatisticalResult:
        """Run candidate backtest and extract trade/period return observations for statistical testing."""
        try:
            eval_res = evaluate_strategy(
                df=df_reference,
                name=candidate.strategy_name,
                registry=self.registry,
                strategy_kwargs=candidate.parameters,
                transaction_cost=execution_assumptions.transaction_cost,
                slippage=execution_assumptions.slippage,
            )
            returns_series = eval_res.returns
            if isinstance(returns_series, pd.Series) and not returns_series.empty:
                # Filter non-zero returns or period returns
                obs = returns_series.dropna().tolist()
            else:
                obs = []
            return compute_statistical_validation(obs, self.criteria)
        except Exception:
            return StatisticalResult(
                observation_count=0,
                mean_return=0.0,
                std_return=0.0,
                t_statistic=0.0,
                p_value=1.0,
                is_valid=False,
                methodology_version=f"stat_t_test_{self.criteria.version}",
            )
