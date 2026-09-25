"""Research experiment persistence store for Project 1.

Persists and loads research experiments and their evidence adhering to Project 1 conventions.
Fail-closed on corrupted, missing, or conflicting research evidence objects.
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
    If an evidence artifact already exists at the target path:
      - If the existing evidence has the exact same evidence_id / content, operation is idempotent.
      - If the existing evidence has conflicting content, fails closed with FileExistsError.

    Returns the file path where the evidence was saved.
    """
    if not isinstance(evidence, ResearchEvidence):
        raise TypeError("evidence must be a ResearchEvidence instance.")

    target_dir = Path(base_dir) / evidence.experiment_fingerprint
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / "evidence.json"

    if file_path.exists():
        existing_evidence = load_research_experiment(file_path)
        if existing_evidence.evidence_id == evidence.evidence_id:
            # Idempotent save for identical evidence
            return file_path
        raise FileExistsError(
            f"Cannot overwrite existing research evidence artifact at '{file_path}' "
            f"with conflicting evidence (existing evidence_id: '{existing_evidence.evidence_id}', "
            f"new evidence_id: '{evidence.evidence_id}')."
        )

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
    """Reconstruct a validated ResearchEvidence object from a dictionary representation.

    Strictly enforces presence of required fields without inventing silent defaults.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary.")

    spec_data = data.get("spec")
    if not isinstance(spec_data, dict):
        raise ValueError("Missing or invalid 'spec' dictionary in evidence data.")

    ds_data = spec_data.get("dataset_scope")
    if not isinstance(ds_data, dict):
        raise ValueError("Missing or invalid 'dataset_scope' in spec.")
    dataset_scope = DatasetScope(
        dataset_id=ds_data.get("dataset_id", ""),
        symbol=ds_data.get("symbol", ""),
        timeframe=ds_data.get("timeframe", ""),
        start_date=ds_data.get("start_date", ""),
        end_date=ds_data.get("end_date", ""),
    )

    ea_data = spec_data.get("execution_assumptions")
    if not isinstance(ea_data, dict):
        raise ValueError("Missing or invalid 'execution_assumptions' in spec.")
    for req_field in ("transaction_cost", "slippage", "latency_ms"):
        if req_field not in ea_data:
            raise ValueError(f"Missing required execution assumption field: '{req_field}'.")

    execution_assumptions = ExecutionAssumptions(
        transaction_cost=float(ea_data["transaction_cost"]),
        slippage=float(ea_data["slippage"]),
        latency_ms=float(ea_data["latency_ms"]),
    )

    cp_data = spec_data.get("code_provenance")
    if not isinstance(cp_data, dict):
        raise ValueError("Missing or invalid 'code_provenance' in spec.")
    code_provenance = CodeProvenance(
        commit_sha=cp_data.get("commit_sha", ""),
        repository_status=cp_data.get("repository_status", "clean"),
        author=cp_data.get("author", ""),
    )

    benchmark_ref = spec_data.get("benchmark_reference")
    if not benchmark_ref or not isinstance(benchmark_ref, str):
        raise ValueError("Missing or invalid 'benchmark_reference' in spec.")

    spec = ResearchExperimentSpec(
        hypothesis=spec_data.get("hypothesis", ""),
        methodology_version=spec_data.get("methodology_version", ""),
        strategy_name=spec_data.get("strategy_name", ""),
        strategy_version=spec_data.get("strategy_version", ""),
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        benchmark_reference=benchmark_ref,
        parameters=spec_data.get("parameters", {}),
        random_seed=spec_data.get("random_seed"),
    )

    partitions_data = data.get("partitions")
    if not isinstance(partitions_data, list):
        raise ValueError("Missing or invalid 'partitions' list in evidence data.")

    partitions = []
    for p_data in partitions_data:
        if not isinstance(p_data, dict):
            raise ValueError("Partition entry must be a dictionary.")
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

    status_str = data.get("promotion_status")
    if not status_str or not isinstance(status_str, str):
        raise ValueError("Missing or invalid 'promotion_status' in evidence data.")

    return ResearchEvidence(
        experiment_fingerprint=data.get("experiment_fingerprint", ""),
        spec=spec,
        partitions=tuple(partitions),
        robustness_verdict=data.get("robustness_verdict", {}),
        benchmark_comparison=data.get("benchmark_comparison", {}),
        promotion_status=PromotionStatus(status_str),
        rejection_reasons=rejection_reasons,
        critique_notes=data.get("critique_notes", ""),
        created_at_utc=data.get("created_at_utc", ""),
    )
