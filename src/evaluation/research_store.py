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

CANDIDATE_INDEX_DIRNAME = "by_candidate"
CANDIDATE_BINDING_FILENAME = "candidate.json"
EVIDENCE_FILENAME = "evidence.json"


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

    file_path = target_dir / EVIDENCE_FILENAME

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
        path = Path(base_dir) / str(fingerprint_or_path) / EVIDENCE_FILENAME

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


class PromotionUnavailable(LookupError):
    """Raised when a requested promoted candidate cannot be resolved from persisted research state."""


class PromotionIntegrityError(ValueError):
    """Raised when persisted candidate/evidence identity or fingerprint lineage is inconsistent."""


class PromotionEligibilityError(ValueError):
    """Raised when persisted evidence is present but not eligible for production execution."""


def _candidate_index_dir(base_dir: str | Path) -> Path:
    return Path(base_dir) / CANDIDATE_INDEX_DIRNAME


def _candidate_binding_path(candidate_id: str, base_dir: str | Path) -> Path:
    return _candidate_index_dir(base_dir) / candidate_id.strip() / CANDIDATE_BINDING_FILENAME


def _require_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PromotionIntegrityError(f"{field_name} must be a non-empty string.")
    return value.strip()


def persist_promoted_candidate_binding(
    *,
    candidate_id: str,
    evidence: ResearchEvidence,
    base_dir: str | Path = DEFAULT_RESEARCH_DIR,
    policy: Any | None = None,
) -> Path:
    """Persist an identity binding from a research candidate to already-saved evidence.

    Requires explicit research evidence qualification before binding. Fails closed
    if the evidence fails qualification criteria.
    """
    from src.evaluation.research_qualification import (
        ResearchQualificationPolicy,
        qualify_research_evidence,
    )

    if not isinstance(evidence, ResearchEvidence):
        raise TypeError("evidence must be a ResearchEvidence instance.")

    qualification = qualify_research_evidence(
        evidence,
        policy=policy if isinstance(policy, ResearchQualificationPolicy) else None,
    )

    if not qualification.qualified:
        reasons = [r.value for r in qualification.rejection_reasons]
        raise PromotionEligibilityError(
            f"Cannot bind candidate '{candidate_id}': evidence '{evidence.evidence_id}' "
            f"failed qualification ({qualification.qualification_notes}). Rejection reasons: {reasons}"
        )

    candidate_id = _require_non_empty_str(candidate_id, "candidate_id")
    spec = evidence.spec
    binding = {
        "candidate_id": candidate_id,
        "strategy_name": spec.strategy_name,
        "strategy_version": spec.strategy_version,
        "experiment_fingerprint": evidence.experiment_fingerprint,
        "evidence_id": evidence.evidence_id,
        "symbol": spec.dataset_scope.symbol,
        "timeframe": spec.dataset_scope.timeframe,
        "parameters": spec.parameters,
    }

    binding_path = _candidate_binding_path(candidate_id, base_dir)
    binding_path.parent.mkdir(parents=True, exist_ok=True)

    if binding_path.exists():
        existing = json.loads(binding_path.read_text(encoding="utf-8"))
        if existing != binding:
            raise FileExistsError(
                f"Cannot overwrite existing promoted candidate binding at '{binding_path}' "
                f"with conflicting identity (existing evidence_id: "
                f"'{existing.get('evidence_id')}', new evidence_id: '{evidence.evidence_id}')."
            )
        return binding_path

    binding_path.write_text(json.dumps(binding, indent=2), encoding="utf-8")
    return binding_path


def save_research_candidate(
    *,
    candidate_id: str,
    evidence: ResearchEvidence,
    base_dir: str | Path = DEFAULT_RESEARCH_DIR,
) -> Path:
    """Persist research evidence and the candidate identity that produced it."""
    save_research_experiment(evidence, base_dir=base_dir)
    return persist_promoted_candidate_binding(
        candidate_id=candidate_id,
        evidence=evidence,
        base_dir=base_dir,
    )


def load_candidate_binding(
    candidate_id: str,
    base_dir: str | Path = DEFAULT_RESEARCH_DIR,
) -> dict[str, Any]:
    """Load a persisted candidate identity binding. Does not invent defaults."""
    candidate_id = _require_non_empty_str(candidate_id, "candidate_id")
    binding_path = _candidate_binding_path(candidate_id, base_dir)
    if not binding_path.exists():
        raise FileNotFoundError(
            f"Promoted candidate binding not found for candidate_id '{candidate_id}' at: {binding_path}"
        )
    try:
        data = json.loads(binding_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PromotionIntegrityError(
            f"Failed to parse candidate binding JSON from {binding_path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise PromotionIntegrityError(f"Candidate binding at {binding_path} is not a dictionary.")
    return data


def list_research_evidence(
    base_dir: str | Path = DEFAULT_RESEARCH_DIR,
) -> list[ResearchEvidence]:
    """Load all persisted ResearchEvidence artifacts from the research store."""
    root = Path(base_dir)
    if not root.exists():
        return []
    artifacts: list[ResearchEvidence] = []
    for evidence_path in sorted(root.glob(f"*/{EVIDENCE_FILENAME}")):
        artifacts.append(load_research_experiment(evidence_path, base_dir=base_dir))
    return artifacts


def _reconstitute_promoted_candidate_from_binding(
    binding: dict[str, Any],
    evidence: ResearchEvidence,
    policy: Any | None = None,
) -> Any:
    """Reconstitute a PromotedCandidateArtifact solely from persisted research state."""
    from src.evaluation.live_production_decision import (
        ProductionPromotionPolicy,
        PromotedCandidateArtifact,
        validate_promotion_eligibility,
    )

    candidate_id = _require_non_empty_str(binding.get("candidate_id"), "candidate_id")
    strategy_name = _require_non_empty_str(binding.get("strategy_name"), "strategy_name")
    strategy_version = _require_non_empty_str(binding.get("strategy_version"), "strategy_version")
    fingerprint = _require_non_empty_str(
        binding.get("experiment_fingerprint"), "experiment_fingerprint"
    )
    evidence_id = _require_non_empty_str(binding.get("evidence_id"), "evidence_id")
    symbol = _require_non_empty_str(binding.get("symbol"), "symbol")
    timeframe = _require_non_empty_str(binding.get("timeframe"), "timeframe")
    parameters = binding.get("parameters")
    if parameters is None:
        parameters = evidence.spec.parameters
    if not isinstance(parameters, dict):
        raise PromotionIntegrityError("candidate binding parameters must be a dictionary.")

    if fingerprint != evidence.experiment_fingerprint:
        raise PromotionIntegrityError(
            f"Candidate '{candidate_id}' research fingerprint '{fingerprint}' does not match "
            f"persisted evidence fingerprint '{evidence.experiment_fingerprint}'."
        )
    if evidence_id != evidence.evidence_id:
        raise PromotionIntegrityError(
            f"Candidate '{candidate_id}' evidence_id '{evidence_id}' does not match "
            f"persisted evidence_id '{evidence.evidence_id}'."
        )
    if strategy_name != evidence.spec.strategy_name:
        raise PromotionIntegrityError(
            f"Candidate '{candidate_id}' strategy_name '{strategy_name}' does not match "
            f"persisted evidence strategy '{evidence.spec.strategy_name}'."
        )
    if strategy_version != evidence.spec.strategy_version:
        raise PromotionIntegrityError(
            f"Candidate '{candidate_id}' strategy_version '{strategy_version}' does not match "
            f"persisted evidence strategy_version '{evidence.spec.strategy_version}'."
        )

    reconstitution_policy = policy if policy is not None else ProductionPromotionPolicy()
    try:
        validate_promotion_eligibility(evidence, policy=reconstitution_policy)
        return PromotedCandidateArtifact.from_persisted_research(
            candidate_id=candidate_id,
            evidence=evidence,
            symbol=symbol,
            timeframe=timeframe,
            parameters=dict(parameters),
            policy=reconstitution_policy,
        )
    except PromotionEligibilityError:
        raise
    except ValueError as exc:
        message = str(exc)
        if "not allowed for production" in message or "rejection reasons" in message or "stale" in message:
            raise PromotionEligibilityError(message) from exc
        raise PromotionIntegrityError(message) from exc


def resolve_promoted_candidate(
    *,
    candidate_id: str | None = None,
    strategy_id: str | None = None,
    strategy_version: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    base_dir: str | Path = DEFAULT_RESEARCH_DIR,
    policy: Any | None = None,
) -> Any | None:
    """Resolve an already-persisted promoted candidate. Never manufactures promotion.

    Returns the reconstituted PromotedCandidateArtifact, or None when the
    requested identity is absent from persisted research state.
    """
    if candidate_id is not None and str(candidate_id).strip():
        requested_id = str(candidate_id).strip()
        try:
            binding = load_candidate_binding(requested_id, base_dir=base_dir)
        except FileNotFoundError:
            return None
        if str(binding.get("candidate_id", "")).strip() != requested_id:
            raise PromotionIntegrityError(
                f"Candidate binding identity '{binding.get('candidate_id')}' does not match "
                f"requested candidate_id '{requested_id}'."
            )

        fingerprint = binding.get("experiment_fingerprint")
        if not fingerprint:
            raise PromotionIntegrityError(
                f"Candidate '{requested_id}' binding is missing experiment_fingerprint."
            )
        evidence_path = Path(base_dir) / str(fingerprint) / EVIDENCE_FILENAME
        if not evidence_path.exists():
            raise PromotionIntegrityError(
                f"Evidence missing for candidate '{requested_id}' "
                f"(fingerprint '{fingerprint}') at: {evidence_path}"
            )
        evidence = load_research_experiment(evidence_path, base_dir=base_dir)
        artifact = _reconstitute_promoted_candidate_from_binding(binding, evidence, policy=policy)

        if strategy_id is not None and str(strategy_id).strip():
            requested_strategy = str(strategy_id).strip()
            if artifact.strategy_name != requested_strategy:
                raise PromotionIntegrityError(
                    f"Requested strategy_id '{requested_strategy}' does not match "
                    f"persisted candidate '{artifact.candidate_id}' strategy "
                    f"'{artifact.strategy_name}'."
                )
        if strategy_version is not None and str(strategy_version).strip():
            requested_version = str(strategy_version).strip()
            if artifact.strategy_version != requested_version:
                raise PromotionIntegrityError(
                    f"Requested strategy_version '{requested_version}' does not match "
                    f"persisted candidate '{artifact.candidate_id}' strategy_version "
                    f"'{artifact.strategy_version}'."
                )
        if symbol is not None and str(symbol).strip():
            requested_symbol = str(symbol).strip().upper()
            if artifact.symbol.upper() != requested_symbol:
                raise PromotionEligibilityError(
                    f"Requested symbol '{requested_symbol}' does not match "
                    f"persisted candidate '{artifact.candidate_id}' symbol '{artifact.symbol}'."
                )
        if timeframe is not None and str(timeframe).strip():
            requested_timeframe = str(timeframe).strip()
            if artifact.timeframe != requested_timeframe:
                raise PromotionEligibilityError(
                    f"Requested timeframe '{requested_timeframe}' does not match "
                    f"persisted candidate '{artifact.candidate_id}' timeframe '{artifact.timeframe}'."
                )

        return artifact

    if strategy_id is None or not str(strategy_id).strip():
        return None

    requested_strategy = str(strategy_id).strip()
    matches: list[Any] = []
    index_root = _candidate_index_dir(base_dir)
    if not index_root.exists():
        return None

    for binding_path in sorted(index_root.glob(f"*/{CANDIDATE_BINDING_FILENAME}")):
        try:
            binding = json.loads(binding_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise PromotionIntegrityError(
                f"Failed to parse candidate binding JSON from {binding_path}: {exc}"
            ) from exc
        if not isinstance(binding, dict):
            raise PromotionIntegrityError(f"Candidate binding at {binding_path} is not a dictionary.")
        if binding.get("strategy_name") != requested_strategy:
            continue
        if strategy_version is not None and str(strategy_version).strip():
            if binding.get("strategy_version") != str(strategy_version).strip():
                continue
        fingerprint = binding.get("experiment_fingerprint")
        if not fingerprint:
            raise PromotionIntegrityError(
                f"Candidate binding at {binding_path} is missing experiment_fingerprint."
            )
        evidence_path = Path(base_dir) / str(fingerprint) / EVIDENCE_FILENAME
        if not evidence_path.exists():
            raise PromotionIntegrityError(
                f"Evidence missing for candidate binding at {binding_path} "
                f"(fingerprint '{fingerprint}')."
            )
        evidence = load_research_experiment(evidence_path, base_dir=base_dir)
        try:
            artifact = _reconstitute_promoted_candidate_from_binding(binding, evidence, policy=policy)
        except PromotionEligibilityError:
            continue
        if symbol is not None and str(symbol).strip():
            if artifact.symbol.upper() != str(symbol).strip().upper():
                continue
        if timeframe is not None and str(timeframe).strip():
            if artifact.timeframe != str(timeframe).strip():
                continue
        matches.append(artifact)

    if not matches:
        return None
    if len(matches) > 1:
        ids = [m.candidate_id for m in matches]
        raise PromotionIntegrityError(
            f"Ambiguous promoted candidate resolution for strategy_id '{requested_strategy}': {ids}. "
            "Specify candidate_id to select an already-promoted artifact."
        )
    return matches[0]
