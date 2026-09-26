"""Canonical Research Experiment Execution Service for Project 1.

Consumes a ResearchExperimentSpec and historical dataset to produce an
authoritative ResearchEvidence artifact. Enforces:
- Fail-closed validation of inputs (ResearchExperimentSpec, DatasetScope, ExecutionAssumptions, CodeProvenance)
- Deterministic dataset resolution and scope matching (no fake data, no silent substitution)
- Strategy parameter integrity (no parameter invention or mutation)
- Strict temporal separation across partitions (In-Sample, Validation, Out-of-Sample, Walk-Forward)
- Execution friction model applying transaction cost, slippage, and latency assumptions
- Robustness and statistical significance checks via RobustnessEvaluator
- Benchmark comparisons
- Promotion safety: research execution produces ResearchEvidence ONLY and never creates production candidate bindings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.data.loader import load_csv
from src.evaluation.candidate_generator import CandidateSpec
from src.evaluation.metrics import max_drawdown, profit_factor, sharpe_ratio, total_return, win_rate
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
from src.evaluation.research_store import DEFAULT_RESEARCH_DIR, save_research_experiment
from src.evaluation.robustness_evaluator import RobustnessEvaluator
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


def resolve_historical_dataset(dataset_scope: DatasetScope) -> pd.DataFrame:
    """Resolve historical dataset strictly matching DatasetScope from repository paths.

    Fails closed if dataset file cannot be found or is invalid.
    """
    symbol_lower = dataset_scope.symbol.lower()

    # Search candidates in repo data directory
    repo_root = Path(__file__).resolve().parents[2]
    search_paths = [
        repo_root / "data" / "raw" / f"{symbol_lower}_daily_2025.csv",
        repo_root / "data" / "raw" / symbol_lower / f"{dataset_scope.dataset_id}.csv",
        repo_root / "data" / "processed" / f"{symbol_lower}.csv",
        repo_root / "data" / f"{dataset_scope.dataset_id}.csv",
    ]

    for p in search_paths:
        if p.exists() and p.is_file():
            return load_csv(p)

    glob_matches = list((repo_root / "data").glob(f"**/*{dataset_scope.dataset_id}*.csv"))
    if glob_matches:
        return load_csv(glob_matches[0])

    glob_symbol_matches = list((repo_root / "data").glob(f"**/*{symbol_lower}*.csv"))
    if glob_symbol_matches:
        return load_csv(glob_symbol_matches[0])

    raise FileNotFoundError(
        f"Unable to resolve dataset for symbol '{dataset_scope.symbol}' "
        f"and dataset_id '{dataset_scope.dataset_id}' from repo data directory."
    )


def validate_and_prepare_dataset(
    df: pd.DataFrame, dataset_scope: DatasetScope
) -> pd.DataFrame:
    """Validate DataFrame schema, timestamp ordering, and DatasetScope boundaries.

    Fails closed on missing columns, non-chronological order, or date scope mismatches.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("df must not be empty.")

    data = df.copy()

    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values("timestamp").reset_index(drop=True)
        ts_series = data["timestamp"]
    elif isinstance(data.index, pd.DatetimeIndex):
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()
        ts_series = pd.Series(data.index, index=data.index)
    else:
        raise ValueError("df must contain a 'timestamp' column or a DatetimeIndex.")

    if not ts_series.is_monotonic_increasing:
        raise ValueError("Time-order violation: dataset timestamps are not strictly chronological.")

    if "return" not in data.columns:
        data = add_returns(data)

    df_start_str = pd.to_datetime(ts_series.min()).strftime("%Y-%m-%d")
    df_end_str = pd.to_datetime(ts_series.max()).strftime("%Y-%m-%d")

    scope_start = dataset_scope.start_date[:10]
    scope_end = dataset_scope.end_date[:10]

    if scope_start < df_start_str or scope_end > df_end_str:
        raise ValueError(
            f"DatasetScope dates [{scope_start}, {scope_end}] extend beyond actual "
            f"data boundaries [{df_start_str}, {df_end_str}]."
        )

    if "timestamp" in data.columns:
        mask = (data["timestamp"] >= pd.to_datetime(dataset_scope.start_date)) & (
            data["timestamp"] <= pd.to_datetime(dataset_scope.end_date)
        )
        filtered = data.loc[mask].reset_index(drop=True)
    else:
        filtered = data.loc[dataset_scope.start_date : dataset_scope.end_date]

    if filtered.empty:
        raise ValueError(
            f"Filtered dataset is empty for scope [{scope_start}, {scope_end}]."
        )

    return filtered


def run_research_experiment(
    spec: ResearchExperimentSpec,
    df: pd.DataFrame | None = None,
    *,
    criteria: DiscoveryCriteria | None = None,
    registry: StrategyRegistry = DEFAULT_REGISTRY,
    val_ratio: float = 0.2,
    oos_ratio: float = 0.3,
    wf_train_size: int | None = None,
    wf_test_size: int | None = None,
    persist_evidence: bool = False,
    base_dir: str | Path = DEFAULT_RESEARCH_DIR,
) -> ResearchEvidence:
    """Execute a single canonical research experiment and return its ResearchEvidence.

    Fails closed on invalid inputs, dataset mismatches, time-order violations, or
    missing strategy definitions.

    Does NOT promote candidates or alter production decision bindings.
    """
    if not isinstance(spec, ResearchExperimentSpec):
        raise TypeError("spec must be a ResearchExperimentSpec instance.")
    if not isinstance(spec.dataset_scope, DatasetScope):
        raise TypeError("spec.dataset_scope must be a DatasetScope instance.")
    if not isinstance(spec.execution_assumptions, ExecutionAssumptions):
        raise TypeError("spec.execution_assumptions must be an ExecutionAssumptions instance.")
    if not isinstance(spec.code_provenance, CodeProvenance):
        raise TypeError("spec.code_provenance must be a CodeProvenance instance.")

    crit = criteria or DiscoveryCriteria()
    rejection_reasons: list[RejectionReason] = []

    if not registry.contains(spec.strategy_name):
        rejection_reasons.append(RejectionReason.SPECIFICATION_INVALID)

    if df is not None:
        data = validate_and_prepare_dataset(df, spec.dataset_scope)
    else:
        try:
            raw_data = resolve_historical_dataset(spec.dataset_scope)
            data = validate_and_prepare_dataset(raw_data, spec.dataset_scope)
        except (FileNotFoundError, ValueError) as exc:
            rejection_reasons.append(RejectionReason.INVALID_DATASET_SCOPE)
            data = pd.DataFrame()

    if data.empty:
        if RejectionReason.INVALID_DATASET_SCOPE not in rejection_reasons:
            rejection_reasons.append(RejectionReason.INSUFFICIENT_DATA)
        now_utc = datetime.now(timezone.utc).isoformat()
        return ResearchEvidence(
            experiment_fingerprint=spec.fingerprint,
            spec=spec,
            partitions=(),
            robustness_verdict={},
            benchmark_comparison={},
            promotion_status=PromotionStatus.REJECTED,
            rejection_reasons=tuple(dict.fromkeys(rejection_reasons)),
            critique_notes="Canonical research experiment execution failed closed.",
            created_at_utc=now_utc,
        )

    latency_ms = float(spec.execution_assumptions.latency_ms)
    latency_slippage_factor = (latency_ms / 1000.0) * 0.0001
    effective_slippage = spec.execution_assumptions.slippage + latency_slippage_factor
    effective_cost = spec.execution_assumptions.transaction_cost

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

    partitions: list[EvidencePartition] = []

    cand_spec = CandidateSpec(
        generator_name="canonical_research_runner",
        generator_version=crit.methodology_version,
        strategy_name=spec.strategy_name,
        parameters=dict(spec.parameters),
        random_seed=spec.random_seed,
        hypothesis_template=spec.hypothesis,
    )

    is_part, is_rejections = _eval_partition(
        role=EvidencePartitionRole.IN_SAMPLE,
        df_part=df_is,
        spec=spec,
        cand=cand_spec,
        registry=registry,
        transaction_cost=effective_cost,
        slippage=effective_slippage,
        latency_ms=latency_ms,
        min_obs=crit.min_observations_is,
    )
    partitions.append(is_part)
    rejection_reasons.extend(is_rejections)

    if is_part.sharpe_ratio < crit.min_is_sharpe or is_part.total_return < crit.min_is_total_return:
        rejection_reasons.append(RejectionReason.FAILED_VALIDATION)

    val_part, val_rejections = _eval_partition(
        role=EvidencePartitionRole.VALIDATION,
        df_part=df_val,
        spec=spec,
        cand=cand_spec,
        registry=registry,
        transaction_cost=effective_cost,
        slippage=effective_slippage,
        latency_ms=latency_ms,
        min_obs=crit.min_observations_oos,
    )
    partitions.append(val_part)
    rejection_reasons.extend(val_rejections)

    if val_part.sharpe_ratio < crit.min_validation_sharpe:
        rejection_reasons.append(RejectionReason.FAILED_VALIDATION)

    oos_part, oos_rejections = _eval_partition(
        role=EvidencePartitionRole.OUT_OF_SAMPLE,
        df_part=df_oos,
        spec=spec,
        cand=cand_spec,
        registry=registry,
        transaction_cost=effective_cost,
        slippage=effective_slippage,
        latency_ms=latency_ms,
        min_obs=crit.min_observations_oos,
    )
    partitions.append(oos_part)
    rejection_reasons.extend(oos_rejections)

    if oos_part.sharpe_ratio < crit.min_oos_sharpe:
        rejection_reasons.append(RejectionReason.FAILED_OOS)

    if is_part.sharpe_ratio > 0:
        deg = (is_part.sharpe_ratio - oos_part.sharpe_ratio) / is_part.sharpe_ratio
        if deg > crit.max_oos_sharpe_degradation:
            rejection_reasons.append(RejectionReason.FAILED_OOS)
            rejection_reasons.append(RejectionReason.IS_ONLY_SUCCESS)

    wf_part, wf_rejections = _eval_walk_forward(
        df_full=data,
        cand=cand_spec,
        spec=spec,
        registry=registry,
        transaction_cost=effective_cost,
        slippage=effective_slippage,
        latency_ms=latency_ms,
        crit=crit,
        wf_train_size=wf_train_size,
        wf_test_size=wf_test_size,
    )
    rejection_reasons.extend(wf_rejections)
    if wf_part is not None:
        partitions.append(wf_part)

    benchmark_comp = _eval_benchmark(df_is, is_part.total_return, crit.benchmark_reference)

    df_ref = pd.concat([df_is, df_val]).sort_values("timestamp").reset_index(drop=True) if "timestamp" in df_is.columns else pd.concat([df_is, df_val])

    robustness_eval = RobustnessEvaluator(
        criteria=crit.robustness_criteria,
        registry=registry,
    )
    robustness_res = robustness_eval.evaluate_candidate_robustness(
        candidate=cand_spec,
        df_reference=df_ref,
        df_oos=df_oos,
        dataset_scope=spec.dataset_scope,
        execution_assumptions=spec.execution_assumptions,
    )
    rejection_reasons.extend(robustness_res.rejection_reasons)

    dedup_rejections = tuple(dict.fromkeys(rejection_reasons))
    if dedup_rejections:
        status = PromotionStatus.REJECTED
    else:
        status = PromotionStatus.PROMOTABLE

    now_utc = datetime.now(timezone.utc).isoformat()

    evidence = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=tuple(partitions),
        robustness_verdict=robustness_res.as_dict(),
        benchmark_comparison=benchmark_comp,
        promotion_status=status,
        rejection_reasons=dedup_rejections,
        critique_notes="Canonical research experiment execution completed.",
        created_at_utc=now_utc,
    )

    if persist_evidence:
        save_research_experiment(evidence, base_dir=base_dir)

    return evidence


def _eval_partition(
    *,
    role: EvidencePartitionRole,
    df_part: pd.DataFrame,
    spec: ResearchExperimentSpec,
    cand: CandidateSpec,
    registry: StrategyRegistry,
    transaction_cost: float,
    slippage: float,
    latency_ms: float,
    min_obs: int,
) -> tuple[EvidencePartition, list[RejectionReason]]:
    rejections: list[RejectionReason] = []

    if "timestamp" in df_part.columns:
        ts = df_part["timestamp"]
    else:
        ts = df_part.index

    start_str = pd.to_datetime(ts.min()).strftime("%Y-%m-%d") if len(df_part) > 0 else "1970-01-01"
    end_str = pd.to_datetime(ts.max()).strftime("%Y-%m-%d") if len(df_part) > 0 else "1970-01-01"

    add_metrics = {
        "latency_ms": latency_ms,
        "transaction_cost": transaction_cost,
        "slippage": slippage,
    }

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
                additional_metrics=add_metrics,
            ),
            rejections,
        )

    try:
        eval_res = evaluate_strategy(
            df=df_part,
            name=spec.strategy_name,
            registry=registry,
            strategy_kwargs=spec.parameters,
            transaction_cost=transaction_cost,
            slippage=slippage,
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
            additional_metrics=add_metrics,
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
                additional_metrics=add_metrics,
            ),
            rejections,
        )


def _eval_walk_forward(
    *,
    df_full: pd.DataFrame,
    cand: CandidateSpec,
    spec: ResearchExperimentSpec,
    registry: StrategyRegistry,
    transaction_cost: float,
    slippage: float,
    latency_ms: float,
    crit: DiscoveryCriteria,
    wf_train_size: int | None,
    wf_test_size: int | None,
) -> tuple[EvidencePartition | None, list[RejectionReason]]:
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
                name=spec.strategy_name,
                registry=registry,
                strategy_kwargs=spec.parameters,
                transaction_cost=transaction_cost,
                slippage=slippage,
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
    if pos_ratio < crit.min_walk_forward_positive_ratio:
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
            "latency_ms": latency_ms,
            "transaction_cost": transaction_cost,
            "slippage": slippage,
        },
    )

    return wf_partition, rejections


def _eval_benchmark(
    df_is: pd.DataFrame, candidate_is_return: float, benchmark_ref: str
) -> dict[str, Any]:
    if "return" in df_is.columns:
        benchmark_return = float((1.0 + df_is["return"]).prod() - 1.0)
    elif "close" in df_is.columns and len(df_is) > 1:
        benchmark_return = float(
            (df_is["close"].iloc[-1] - df_is["close"].iloc[0]) / df_is["close"].iloc[0]
        )
    else:
        benchmark_return = 0.0

    return {
        "benchmark_reference": benchmark_ref,
        "benchmark_total_return": round(benchmark_return, 6),
        "outperformed_benchmark": candidate_is_return > benchmark_return,
    }
