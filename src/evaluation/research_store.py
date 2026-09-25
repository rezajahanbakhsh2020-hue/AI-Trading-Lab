"""Research experiment persistence store for Project 1.

Persists and loads research experiments and their evidence adhering to Project 1 conventions.
Fail-closed on corrupted or invalid research evidence objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
)

DEFAULT_RESEARCH_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "research_experiments"
)


def save_research_experiment(
    evidence: ResearchEvidence,
    base_dir: str | Path = DEFAULT_RESEARCH_DIR,
) -> Path:
    """Persist a ResearchEvidence object to disk as JSON.

    Saves under `base_dir / <experiment_fingerprint> / evidence.json`.
    Returns the file path where the evidence was saved.
    """
    if not isinstance(evidence, ResearchEvidence):
        raise TypeError("evidence must be a ResearchEvidence instance.")

    target_dir = Path(base_dir) / evidence.experiment_fingerprint
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / "evidence.json"
    content = json.dumps(evidence.as_dict(), indent=2)
    file_path.write_text(content, encoding="utf-8")

    return file_path


def load_research_experiment(
    fingerprint_or_path: str | Path,
    base_dir: str | Path = DEFAULT_RESEARCH_DIR,
) -> ResearchEvidence:
    """Load and reconstruct a ResearchEvidence object from disk.

    Fails closed if file is missing, malformed, or fails contract validation.
    """
    path = Path(fingerprint_or_path)
    if not path.is_file():
        path = Path(base_dir) / str(fingerprint_or_path) / "evidence.json"

    if not path.exists():
        raise FileNotFoundError(f"Research evidence file not found at: {path}")

    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse research evidence JSON from {path}: {exc}") from exc

    return reconstruct_research_evidence(raw_data)


def reconstruct_research_evidence(data: dict[str, Any]) -> ResearchEvidence:
    """Reconstruct a validated ResearchEvidence object from a dictionary representation."""
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary.")

    spec_data = data.get("spec")
    if not isinstance(spec_data, dict):
        raise ValueError("Missing or invalid 'spec' dictionary in evidence data.")

    ds_data = spec_data.get("dataset_scope", {})
    dataset_scope = DatasetScope(
        dataset_id=ds_data.get("dataset_id", ""),
        symbol=ds_data.get("symbol", ""),
        timeframe=ds_data.get("timeframe", ""),
        start_date=ds_data.get("start_date", ""),
        end_date=ds_data.get("end_date", ""),
    )

    ea_data = spec_data.get("execution_assumptions", {})
    execution_assumptions = ExecutionAssumptions(
        transaction_cost=float(ea_data.get("transaction_cost", 0.0)),
        slippage=float(ea_data.get("slippage", 0.0)),
        latency_ms=float(ea_data.get("latency_ms", 0.0)),
    )

    cp_data = spec_data.get("code_provenance", {})
    code_provenance = CodeProvenance(
        commit_sha=cp_data.get("commit_sha", ""),
        repository_status=cp_data.get("repository_status", "clean"),
        author=cp_data.get("author", ""),
    )

    spec = ResearchExperimentSpec(
        hypothesis=spec_data.get("hypothesis", ""),
        methodology_version=spec_data.get("methodology_version", ""),
        strategy_name=spec_data.get("strategy_name", ""),
        strategy_version=spec_data.get("strategy_version", ""),
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        parameters=spec_data.get("parameters", {}),
        benchmark_reference=spec_data.get("benchmark_reference", "BUY_AND_HOLD"),
        random_seed=spec_data.get("random_seed"),
    )

    partitions_data = data.get("partitions", [])
    partitions = []
    for p_data in partitions_data:
        partitions.append(
            EvidencePartition(
                role=EvidencePartitionRole(p_data["role"]),
                start_date=p_data["start_date"],
                end_date=p_data["end_date"],
                total_return=float(p_data["total_return"]),
                max_drawdown=float(p_data["max_drawdown"]),
                sharpe_ratio=float(p_data["sharpe_ratio"]),
                win_rate=float(p_data.get("win_rate", 0.0)),
                profit_factor=float(p_data.get("profit_factor", 0.0)),
                observations=int(p_data.get("observations", 0)),
                additional_metrics=p_data.get("additional_metrics", {}),
            )
        )

    rejection_reasons = tuple(
        RejectionReason(r) for r in data.get("rejection_reasons", [])
    )

    return ResearchEvidence(
        experiment_fingerprint=data.get("experiment_fingerprint", ""),
        spec=spec,
        partitions=tuple(partitions),
        robustness_verdict=data.get("robustness_verdict", {}),
        benchmark_comparison=data.get("benchmark_comparison", {}),
        promotion_status=PromotionStatus(data.get("promotion_status", "PROPOSED")),
        rejection_reasons=rejection_reasons,
        critique_notes=data.get("critique_notes", ""),
        created_at_utc=data.get("created_at_utc", ""),
    )
