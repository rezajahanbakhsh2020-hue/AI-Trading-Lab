"""Research experiment persistence store for Project 1.

Persists and loads research experiments and their evidence adhering to Project 1 conventions.
Fail-closed on corrupted, missing, or conflicting research evidence objects.
"""

from __future__ import annotations

from enum import Enum
import hashlib
import json
from datetime import datetime, timezone
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
    ResearchCampaign,
    ResearchCampaignDefinition,
    ResearchCampaignStatus,
    ResearchEvidence,
    ResearchExperimentSpec,
    ResearchPlannedTrial,
    ResearchTrialCheckpoint,
    ResearchTrialPlan,
    WalkForwardProtocol,
    validate_campaign_state_transition,
)

DEFAULT_RESEARCH_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "research_experiments"
)

DEFAULT_CAMPAIGN_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "research_campaigns"
)

CANDIDATE_INDEX_DIRNAME = "by_candidate"
CANDIDATE_BINDING_FILENAME = "candidate.json"
EVIDENCE_FILENAME = "evidence.json"
CAMPAIGN_FILENAME = "campaign.json"


def save_research_campaign(
    campaign: ResearchCampaign,
    base_dir: str | Path = DEFAULT_CAMPAIGN_DIR,
) -> Path:
    """Persist a ResearchCampaign object to disk as JSON.

    Saves under `base_dir / <campaign_id> / campaign.json`.
    Fail-closed on conflicting contents for identical campaign IDs.
    """
    if not isinstance(campaign, ResearchCampaign):
        raise TypeError("campaign must be a ResearchCampaign instance.")

    target_dir = Path(base_dir) / campaign.campaign_id
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / CAMPAIGN_FILENAME

    if file_path.exists():
        existing_campaign = load_research_campaign(file_path)
        if existing_campaign.reproducibility_fingerprint == campaign.reproducibility_fingerprint:
            return file_path
        raise FileExistsError(
            f"Cannot overwrite existing research campaign artifact at '{file_path}' "
            f"with conflicting campaign data."
        )

    content = json.dumps(campaign.as_dict(), indent=2)
    file_path.write_text(content, encoding="utf-8")

    return file_path


def load_research_campaign(
    campaign_id_or_path: str | Path,
    base_dir: str | Path = DEFAULT_CAMPAIGN_DIR,
) -> ResearchCampaign:
    """Load and reconstruct a ResearchCampaign object from disk."""
    path = Path(campaign_id_or_path)
    if not path.is_file():
        path = Path(base_dir) / str(campaign_id_or_path) / CAMPAIGN_FILENAME

    if not path.exists():
        raise FileNotFoundError(f"Research campaign file not found at: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse research campaign JSON from {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise TypeError("Campaign data must be a dictionary.")

    ds_data = data.get("dataset_scope", {})
    dataset_scope = DatasetScope(
        dataset_id=ds_data.get("dataset_id", ""),
        symbol=ds_data.get("symbol", ""),
        timeframe=ds_data.get("timeframe", ""),
        start_date=ds_data.get("start_date", ""),
        end_date=ds_data.get("end_date", ""),
    )

    ea_data = data.get("execution_assumptions", {})
    execution_assumptions = ExecutionAssumptions(
        transaction_cost=float(ea_data.get("transaction_cost", 0.0)),
        slippage=float(ea_data.get("slippage", 0.0)),
        latency_ms=float(ea_data.get("latency_ms", 0.0)),
    )

    cp_data = data.get("code_provenance", {})
    code_provenance = CodeProvenance(
        commit_sha=cp_data.get("commit_sha", ""),
        repository_status=cp_data.get("repository_status", "clean"),
        author=cp_data.get("author", ""),
    )

    campaign = ResearchCampaign(
        campaign_id=data.get("campaign_id", ""),
        search_space_fingerprint=data.get("search_space_fingerprint", ""),
        search_policy_fingerprint=data.get("search_policy_fingerprint", ""),
        criteria_fingerprint=data.get("criteria_fingerprint", ""),
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        candidate_ids=tuple(data.get("candidate_ids", [])),
        evidence_fingerprints=tuple(data.get("evidence_fingerprints", [])),
        selected_candidate_ids=tuple(data.get("selected_candidate_ids", [])),
        status=ResearchCampaignStatus(data.get("status", "COMPLETED")),
        created_at_utc=data.get("created_at_utc", ""),
        definition_fingerprint=data.get("definition_fingerprint", ""),
        trial_plan_fingerprint=data.get("trial_plan_fingerprint", ""),
        executed_trial_count=data.get("executed_trial_count", 0),
        failed_trial_count=data.get("failed_trial_count", 0),
        blocked_trial_count=data.get("blocked_trial_count", 0),
    )

    if campaign.reproducibility_fingerprint != data.get("reproducibility_fingerprint"):
        raise ValueError(
            f"Loaded campaign reproducibility fingerprint mismatch: expected "
            f"'{campaign.reproducibility_fingerprint}', got '{data.get('reproducibility_fingerprint')}'."
        )

    return campaign


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

    wf_data = spec_data.get("walk_forward_protocol")
    wf_protocol = None
    if wf_data is not None:
        if not isinstance(wf_data, dict):
            raise ValueError("Missing or invalid 'walk_forward_protocol' dictionary in spec.")
        if "train_size" not in wf_data or "test_size" not in wf_data:
            raise ValueError("walk_forward_protocol dictionary missing 'train_size' or 'test_size'.")
        try:
            wf_protocol = WalkForwardProtocol(
                train_size=int(wf_data["train_size"]),
                test_size=int(wf_data["test_size"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid walk_forward_protocol data in spec: {exc}") from exc

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
        walk_forward_protocol=wf_protocol,
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
                start_timestamp_utc=p_data.get("start_timestamp_utc"),
                end_timestamp_utc=p_data.get("end_timestamp_utc"),
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
    canonical_stability: Any = None,
    operational_stability_score: float | None = None,
    base_dir: str | Path = DEFAULT_RESEARCH_DIR,
    policy: Any | None = None,
    robustness_assessment: Any | None = None,
    governance_decision: Any | None = None,
    campaign_selection_decision: Any | None = None,
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

    candidate_id = _require_non_empty_str(candidate_id, "candidate_id")

    if governance_decision is not None:
        if not getattr(governance_decision, "qualified", False):
            reasons = [r.value if isinstance(r, Enum) else str(r) for r in getattr(governance_decision, "rejection_reasons", ())]
            raise PromotionEligibilityError(
                f"Cannot bind candidate '{candidate_id}': governance decision failed qualification "
                f"({getattr(governance_decision, 'qualification_notes', '')}). Rejection reasons: {reasons}"
            )
        gov_fp = getattr(governance_decision, "decision_fingerprint", None)
        if not isinstance(gov_fp, str) or not gov_fp.strip():
            raise PromotionIntegrityError(
                f"Cannot bind candidate '{candidate_id}': governance decision missing valid decision_fingerprint."
            )
        gov_fp = gov_fp.strip()
    else:
        if robustness_assessment is None:
            try:
                from src.evaluation.research_robustness import (
                    assess_research_robustness,
                )
                robustness_assessment = assess_research_robustness(evidence)
            except Exception:
                robustness_assessment = None

        qualification = qualify_research_evidence(
            evidence,
            policy=policy if isinstance(policy, ResearchQualificationPolicy) else None,
            robustness_assessment=robustness_assessment,
        )
        if not qualification.qualified:
            reasons = [r.value if isinstance(r, Enum) else str(r) for r in qualification.rejection_reasons]
            raise PromotionEligibilityError(
                f"Cannot bind candidate '{candidate_id}': evidence '{evidence.evidence_id}' "
                f"failed qualification ({qualification.qualification_notes}). Rejection reasons: {reasons}"
            )
        gov_fp = qualification.decision_fingerprint

    campaign_sel_fp: str | None = None
    if campaign_selection_decision is not None:
        camp_id = getattr(campaign_selection_decision, "campaign_id", None)
        if not isinstance(camp_id, str) or not camp_id.strip():
            raise PromotionIntegrityError(
                f"Cannot bind candidate '{candidate_id}': campaign_selection_decision.campaign_id must be a non-empty string."
            )

        dec_status = getattr(campaign_selection_decision, "decision_status", None)
        if isinstance(dec_status, Enum):
            dec_status = dec_status.value
        dec_status_str = str(dec_status).strip() if dec_status is not None else ""
        if dec_status_str != "SELECTED":
            raise PromotionEligibilityError(
                f"Cannot bind candidate '{candidate_id}': campaign selection decision status "
                f"is '{dec_status_str}', expected 'SELECTED'."
            )

        selected_ids = getattr(campaign_selection_decision, "selected_candidate_ids", ())
        if candidate_id not in selected_ids:
            raise PromotionEligibilityError(
                f"Cannot bind candidate '{candidate_id}': candidate is not contained in "
                f"campaign selection decision selected_candidate_ids ({selected_ids})."
            )

        cs_fp = getattr(campaign_selection_decision, "decision_fingerprint", None)
        if not isinstance(cs_fp, str) or not cs_fp.strip():
            raise PromotionIntegrityError(
                f"Cannot bind candidate '{candidate_id}': campaign_selection_decision missing valid decision_fingerprint."
            )
        campaign_sel_fp = cs_fp.strip()

    spec = evidence.spec

    from src.evaluation.stability import CanonicalStabilityEvidence

    if canonical_stability is not None:
        if not isinstance(canonical_stability, CanonicalStabilityEvidence):
            raise PromotionIntegrityError("promoted binding requires canonical stability evidence (CanonicalStabilityEvidence).")
        if canonical_stability.strategy_name != spec.strategy_name:
            raise PromotionIntegrityError(
                f"canonical stability strategy '{canonical_stability.strategy_name}' does not match "
                f"evidence strategy '{spec.strategy_name}'."
            )
        f_stab_score = canonical_stability.stability_score
    else:
        if operational_stability_score is None or isinstance(operational_stability_score, bool) or not isinstance(operational_stability_score, (int, float)):
            raise PromotionIntegrityError("operational_stability_score must be a numeric float or CanonicalStabilityEvidence.")
        f_stab_score = float(operational_stability_score)
        import math
        if not math.isfinite(f_stab_score):
            raise PromotionIntegrityError("operational_stability_score must be finite (not NaN or infinity).")

    binding = {
        "candidate_id": candidate_id,
        "strategy_name": spec.strategy_name,
        "strategy_version": spec.strategy_version,
        "experiment_fingerprint": evidence.experiment_fingerprint,
        "evidence_id": evidence.evidence_id,
        "symbol": spec.dataset_scope.symbol,
        "timeframe": spec.dataset_scope.timeframe,
        "parameters": spec.parameters,
        "governance_decision_fingerprint": gov_fp,
        "campaign_selection_decision_fingerprint": campaign_sel_fp,
        "operational_stability_score": f_stab_score,
    }

    binding_path = _candidate_binding_path(candidate_id, base_dir)
    binding_path.parent.mkdir(parents=True, exist_ok=True)

    if binding_path.exists():
        existing = json.loads(binding_path.read_text(encoding="utf-8"))
        if existing != binding:
            raise FileExistsError(
                f"Cannot overwrite existing promoted candidate binding at '{binding_path}' "
                f"with conflicting identity or lineage."
            )
        return binding_path

    binding_path.write_text(json.dumps(binding, indent=2), encoding="utf-8")
    return binding_path


def save_research_candidate(
    *,
    candidate_id: str,
    evidence: ResearchEvidence,
    canonical_stability: Any = None,
    operational_stability_score: float | None = None,
    base_dir: str | Path = DEFAULT_RESEARCH_DIR,
    governance_decision: Any | None = None,
    campaign_selection_decision: Any | None = None,
) -> Path:
    """Persist research evidence and the candidate identity that produced it."""
    save_research_experiment(evidence, base_dir=base_dir)
    return persist_promoted_candidate_binding(
        candidate_id=candidate_id,
        evidence=evidence,
        canonical_stability=canonical_stability,
        operational_stability_score=operational_stability_score,
        base_dir=base_dir,
        governance_decision=governance_decision,
        campaign_selection_decision=campaign_selection_decision,
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

    if "operational_stability_score" not in binding:
        raise PromotionIntegrityError(
            f"Candidate '{candidate_id}' binding is missing required operational_stability_score."
        )
    raw_stab = binding["operational_stability_score"]
    if raw_stab is None or isinstance(raw_stab, bool) or not isinstance(raw_stab, (int, float)):
        raise PromotionIntegrityError(
            f"Candidate '{candidate_id}' binding has invalid operational_stability_score."
        )
    f_stab = float(raw_stab)
    import math
    if not math.isfinite(f_stab):
        raise PromotionIntegrityError(
            f"Candidate '{candidate_id}' operational_stability_score must be finite (got {raw_stab})."
        )

    reconstitution_policy = policy if policy is not None else ProductionPromotionPolicy()
    gov_fp = binding.get("governance_decision_fingerprint")
    campaign_sel_fp = binding.get("campaign_selection_decision_fingerprint")
    try:
        validate_promotion_eligibility(evidence, policy=reconstitution_policy, governance_decision_fingerprint=gov_fp)
        return PromotedCandidateArtifact.from_persisted_research(
            candidate_id=candidate_id,
            evidence=evidence,
            symbol=symbol,
            timeframe=timeframe,
            parameters=dict(parameters),
            policy=reconstitution_policy,
            governance_decision_fingerprint=gov_fp,
            campaign_selection_decision_fingerprint=campaign_sel_fp,
            operational_stability_score=f_stab,
        )
    except PromotionEligibilityError:
        raise
    except ValueError as exc:
        message = str(exc)
        if "not allowed for production" in message or "rejection reasons" in message or "stale" in message or "exceeds max allowed age" in message:
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
            try:
                from src.evaluation.mtf_intelligence import CanonicalTimeframe
                req_tf = CanonicalTimeframe.from_str(requested_timeframe).value
                art_tf = CanonicalTimeframe.from_str(artifact.timeframe).value
            except Exception:
                req_tf = requested_timeframe
                art_tf = artifact.timeframe

            if art_tf != req_tf:
                raise PromotionEligibilityError(
                    f"Requested timeframe '{requested_timeframe}' does not match "
                    f"persisted candidate '{artifact.candidate_id}' timeframe '{artifact.timeframe}'."
                )

        return artifact

    requested_strategy = str(strategy_id).strip() if strategy_id is not None and str(strategy_id).strip() else None
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
        if requested_strategy is not None and binding.get("strategy_name") != requested_strategy:
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
            try:
                from src.evaluation.mtf_intelligence import CanonicalTimeframe
                req_tf = CanonicalTimeframe.from_str(timeframe).value
                art_tf = CanonicalTimeframe.from_str(artifact.timeframe).value
            except Exception:
                req_tf = str(timeframe).strip()
                art_tf = artifact.timeframe

            if art_tf != req_tf:
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


class ResearchCampaignStore:
    """Canonical persistence store for durable Research Campaign lifecycle and checkpoints.

    Supports:
    - Campaign Definition persistence and loading
    - Trial Plan persistence and loading
    - Lifecycle state persistence and loading
    - Trial Checkpoint persistence, updating, loading, and listing
    - Identification of completed, pending, failed, and blocked trials
    - Integrity validation on load (failing closed on corrupt or mismatched fingerprint state)
    """

    def __init__(self, base_dir: str | Path = DEFAULT_CAMPAIGN_DIR) -> None:
        self.base_dir = Path(base_dir)

    def _campaign_dir(self, campaign_id: str) -> Path:
        return self.base_dir / campaign_id.strip()

    def _checkpoints_dir(self, campaign_id: str) -> Path:
        return self._campaign_dir(campaign_id) / "checkpoints"

    def save_definition(self, definition: ResearchCampaignDefinition) -> Path:
        if not isinstance(definition, ResearchCampaignDefinition):
            raise TypeError("definition must be a ResearchCampaignDefinition instance.")
        cdir = self._campaign_dir(definition.campaign_id)
        cdir.mkdir(parents=True, exist_ok=True)
        path = cdir / "definition.json"
        if path.exists():
            existing = self.load_definition(definition.campaign_id)
            if existing.definition_fingerprint != definition.definition_fingerprint:
                raise FileExistsError(
                    f"Cannot overwrite campaign definition at '{path}' with conflicting definition fingerprint."
                )
            return path
        content = json.dumps(definition.as_dict(), indent=2)
        path.write_text(content, encoding="utf-8")
        return path

    def load_definition(self, campaign_id: str) -> ResearchCampaignDefinition:
        path = self._campaign_dir(campaign_id) / "definition.json"
        if not path.exists():
            raise FileNotFoundError(f"Campaign definition not found at: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse campaign definition JSON at {path}: {exc}") from exc

        ds_data = data.get("dataset_scope", {})
        ds = DatasetScope(
            dataset_id=ds_data.get("dataset_id", ""),
            symbol=ds_data.get("symbol", ""),
            timeframe=ds_data.get("timeframe", ""),
            start_date=ds_data.get("start_date", ""),
            end_date=ds_data.get("end_date", ""),
        )
        ea_data = data.get("execution_assumptions", {})
        ea = ExecutionAssumptions(
            transaction_cost=float(ea_data.get("transaction_cost", 0.0)),
            slippage=float(ea_data.get("slippage", 0.0)),
            latency_ms=float(ea_data.get("latency_ms", 0.0)),
        )
        cp_data = data.get("code_provenance", {})
        cp = CodeProvenance(
            commit_sha=cp_data.get("commit_sha", ""),
            repository_status=cp_data.get("repository_status", "clean"),
            author=cp_data.get("author", ""),
        )
        wf_data = data.get("walk_forward_protocol")
        wf = None
        if wf_data is not None:
            wf = WalkForwardProtocol(
                train_size=int(wf_data["train_size"]),
                test_size=int(wf_data["test_size"]),
            )

        defn = ResearchCampaignDefinition(
            search_space_fingerprint=data.get("search_space_fingerprint", ""),
            search_policy_fingerprint=data.get("search_policy_fingerprint", ""),
            criteria_fingerprint=data.get("criteria_fingerprint", ""),
            dataset_scope=ds,
            execution_assumptions=ea,
            code_provenance=cp,
            methodology_version=data.get("methodology_version", ""),
            candidate_ids=tuple(data.get("candidate_ids", [])),
            trial_count=int(data.get("trial_count", 0)),
            walk_forward_protocol=wf,
            governance_constraints=data.get("governance_constraints", {}),
            memory_policy_id=data.get("memory_policy_id", "default_memory_policy_v1"),
        )
        if defn.definition_fingerprint != data.get("definition_fingerprint"):
            raise ValueError(
                f"Loaded definition fingerprint mismatch for campaign '{campaign_id}': "
                f"expected '{defn.definition_fingerprint}', got '{data.get('definition_fingerprint')}'."
            )
        return defn

    def save_trial_plan(self, plan: ResearchTrialPlan) -> Path:
        if not isinstance(plan, ResearchTrialPlan):
            raise TypeError("plan must be a ResearchTrialPlan instance.")
        cdir = self._campaign_dir(plan.campaign_id)
        cdir.mkdir(parents=True, exist_ok=True)
        path = cdir / "plan.json"
        if path.exists():
            existing = self.load_trial_plan(plan.campaign_id)
            if existing.plan_fingerprint != plan.plan_fingerprint:
                raise FileExistsError(
                    f"Cannot overwrite trial plan at '{path}' with conflicting plan fingerprint."
                )
            return path
        content = json.dumps(plan.as_dict(), indent=2)
        path.write_text(content, encoding="utf-8")
        return path

    def load_trial_plan(self, campaign_id: str) -> ResearchTrialPlan:
        path = self._campaign_dir(campaign_id) / "plan.json"
        if not path.exists():
            raise FileNotFoundError(f"Trial plan not found at: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse trial plan JSON at {path}: {exc}") from exc

        trials = []
        for t_data in data.get("trials", []):
            t = ResearchPlannedTrial(
                campaign_id=t_data.get("campaign_id", ""),
                trial_id=t_data.get("trial_id", ""),
                trial_index=int(t_data.get("trial_index", 0)),
                candidate_id=t_data.get("candidate_id", ""),
                candidate_fingerprint=t_data.get("candidate_fingerprint", ""),
                hypothesis_fingerprint=t_data.get("hypothesis_fingerprint", ""),
                strategy_name=t_data.get("strategy_name", ""),
                strategy_version=t_data.get("strategy_version", ""),
                dataset_id=t_data.get("dataset_id", ""),
                execution_assumptions_id=t_data.get("execution_assumptions_id", ""),
                planned_status=t_data.get("planned_status", "PENDING"),
            )
            trials.append(t)

        plan = ResearchTrialPlan(
            campaign_id=data.get("campaign_id", ""),
            definition_fingerprint=data.get("definition_fingerprint", ""),
            trials=tuple(trials),
        )
        if plan.plan_fingerprint != data.get("plan_fingerprint"):
            raise ValueError(
                f"Loaded plan fingerprint mismatch for campaign '{campaign_id}': "
                f"expected '{plan.plan_fingerprint}', got '{data.get('plan_fingerprint')}'."
            )
        return plan

    def save_lifecycle_state(
        self,
        campaign_id: str,
        status: ResearchCampaignStatus,
        reason: str = "",
        updated_at_utc: str = "",
    ) -> Path:
        if not isinstance(status, ResearchCampaignStatus):
            raise TypeError("status must be a ResearchCampaignStatus enum member.")
        cdir = self._campaign_dir(campaign_id)
        cdir.mkdir(parents=True, exist_ok=True)
        path = cdir / "state.json"

        if path.exists():
            curr_state = self.load_lifecycle_state(campaign_id)
            current_status = ResearchCampaignStatus(curr_state.get("status"))
            validate_campaign_state_transition(current_status, status)

        timestamp = updated_at_utc if updated_at_utc else datetime.now(timezone.utc).isoformat()
        state_data = {
            "campaign_id": campaign_id.strip(),
            "status": status.value,
            "reason": reason,
            "updated_at_utc": timestamp,
        }
        path.write_text(json.dumps(state_data, indent=2), encoding="utf-8")
        return path

    def load_lifecycle_state(self, campaign_id: str) -> dict[str, Any]:
        path = self._campaign_dir(campaign_id) / "state.json"
        if not path.exists():
            raise FileNotFoundError(f"Campaign lifecycle state not found at: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse lifecycle state JSON at {path}: {exc}") from exc

        if not isinstance(data, dict) or "status" not in data:
            raise ValueError(f"Invalid state data at {path}")
        return data

    def save_trial_checkpoint(self, checkpoint: ResearchTrialCheckpoint) -> Path:
        if not isinstance(checkpoint, ResearchTrialCheckpoint):
            raise TypeError("checkpoint must be a ResearchTrialCheckpoint instance.")
        cp_dir = self._checkpoints_dir(checkpoint.campaign_id)
        cp_dir.mkdir(parents=True, exist_ok=True)
        path = cp_dir / f"{checkpoint.trial_id}.json"

        timestamp = checkpoint.updated_at_utc or datetime.now(timezone.utc).isoformat()
        if checkpoint.updated_at_utc != timestamp:
            checkpoint = ResearchTrialCheckpoint(
                trial_id=checkpoint.trial_id,
                campaign_id=checkpoint.campaign_id,
                candidate_id=checkpoint.candidate_id,
                trial_index=checkpoint.trial_index,
                attempt_number=checkpoint.attempt_number,
                status=checkpoint.status,
                experiment_fingerprint=checkpoint.experiment_fingerprint,
                evidence_fingerprint=checkpoint.evidence_fingerprint,
                qualification_status=checkpoint.qualification_status,
                rejection_reasons=checkpoint.rejection_reasons,
                error_message=checkpoint.error_message,
                execution_history=checkpoint.execution_history,
                updated_at_utc=timestamp,
            )

        content = json.dumps(checkpoint.as_dict(), indent=2)
        path.write_text(content, encoding="utf-8")
        return path

    def load_trial_checkpoint(self, campaign_id: str, trial_id: str) -> ResearchTrialCheckpoint:
        path = self._checkpoints_dir(campaign_id) / f"{trial_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Trial checkpoint not found at: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse trial checkpoint JSON at {path}: {exc}") from exc

        return ResearchTrialCheckpoint(
            trial_id=data.get("trial_id", ""),
            campaign_id=data.get("campaign_id", ""),
            candidate_id=data.get("candidate_id", ""),
            trial_index=int(data.get("trial_index", 0)),
            attempt_number=int(data.get("attempt_number", 1)),
            status=data.get("status", "PENDING"),
            experiment_fingerprint=data.get("experiment_fingerprint"),
            evidence_fingerprint=data.get("evidence_fingerprint"),
            qualification_status=data.get("qualification_status"),
            rejection_reasons=tuple(data.get("rejection_reasons", [])),
            error_message=data.get("error_message", ""),
            execution_history=tuple(data.get("execution_history", [])),
            updated_at_utc=data.get("updated_at_utc", ""),
        )

    def list_trial_checkpoints(self, campaign_id: str) -> list[ResearchTrialCheckpoint]:
        cp_dir = self._checkpoints_dir(campaign_id)
        if not cp_dir.exists():
            return []
        checkpoints: list[ResearchTrialCheckpoint] = []
        for path in sorted(cp_dir.glob("*.json")):
            try:
                cp = self.load_trial_checkpoint(campaign_id, path.stem)
                checkpoints.append(cp)
            except Exception as exc:
                raise ValueError(f"Failed to load trial checkpoint at '{path}': {exc}") from exc
        return sorted(checkpoints, key=lambda c: c.trial_index)

    def get_completed_trials(self, campaign_id: str) -> list[ResearchTrialCheckpoint]:
        return [
            cp for cp in self.list_trial_checkpoints(campaign_id)
            if cp.status in ("COMPLETED", "QUALIFIED", "REJECTED")
        ]

    def get_pending_trials(self, campaign_id: str) -> list[ResearchTrialCheckpoint]:
        return [
            cp for cp in self.list_trial_checkpoints(campaign_id)
            if cp.status in ("PENDING", "RUNNING")
        ]

    def save_evidence_synthesis(self, synthesis: Any) -> Path:
        """Persist a ResearchCampaignEvidenceSynthesis object to disk as JSON."""
        cdir = self._campaign_dir(synthesis.campaign_id)
        cdir.mkdir(parents=True, exist_ok=True)
        path = cdir / "evidence_synthesis.json"
        if path.exists():
            existing = self.load_evidence_synthesis(synthesis.campaign_id)
            if existing.synthesis_fingerprint != synthesis.synthesis_fingerprint:
                raise FileExistsError(
                    f"Cannot overwrite evidence synthesis at '{path}' with conflicting synthesis fingerprint."
                )
            return path
        content = json.dumps(synthesis.as_dict(), indent=2)
        path.write_text(content, encoding="utf-8")
        return path

    def load_evidence_synthesis(self, campaign_id: str) -> Any:
        """Load and reconstruct a ResearchCampaignEvidenceSynthesis object from disk."""
        from src.evaluation.campaign_synthesis import (
            ResearchCampaignEvidenceSynthesis,
            ResearchCandidateComparison,
        )

        path = self._campaign_dir(campaign_id) / "evidence_synthesis.json"
        if not path.exists():
            raise FileNotFoundError(f"Evidence synthesis not found at: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse evidence synthesis JSON at {path}: {exc}") from exc

        candidate_comparisons = []
        for c_data in data.get("candidate_comparisons", []):
            comp = ResearchCandidateComparison(
                candidate_id=c_data.get("candidate_id", ""),
                candidate_fingerprint=c_data.get("candidate_fingerprint", ""),
                experiment_fingerprint=c_data.get("experiment_fingerprint", ""),
                evidence_fingerprint=c_data.get("evidence_fingerprint", ""),
                qualification_status=c_data.get("qualification_status", ""),
                qualification_fingerprint=c_data.get("qualification_fingerprint", ""),
                robustness_fingerprint=c_data.get("robustness_fingerprint", ""),
                robustness_status=c_data.get("robustness_status", ""),
                benchmark_evidence=c_data.get("benchmark_evidence", {}),
                regime_evidence=c_data.get("regime_evidence", {}),
                statistical_evidence=c_data.get("statistical_evidence", {}),
                oos_evidence=c_data.get("oos_evidence", {}),
                walk_forward_evidence=c_data.get("walk_forward_evidence", {}),
                execution_assumptions=c_data.get("execution_assumptions", {}),
                rejection_reasons=tuple(c_data.get("rejection_reasons", [])),
                selection_governance_result=c_data.get("selection_governance_result", {}),
                comparison_metrics=c_data.get("comparison_metrics", {}),
            )
            candidate_comparisons.append(comp)

        synthesis = ResearchCampaignEvidenceSynthesis(
            campaign_id=data.get("campaign_id", ""),
            campaign_definition_fingerprint=data.get("campaign_definition_fingerprint", ""),
            trial_plan_fingerprint=data.get("trial_plan_fingerprint", ""),
            search_space_fingerprint=data.get("search_space_fingerprint", ""),
            search_policy_fingerprint=data.get("search_policy_fingerprint", ""),
            criteria_fingerprint=data.get("criteria_fingerprint", ""),
            dataset_identity=data.get("dataset_identity", {}),
            execution_assumptions=data.get("execution_assumptions", {}),
            code_provenance=data.get("code_provenance", {}),
            methodology_version=data.get("methodology_version", ""),
            ordered_trial_identities=tuple(data.get("ordered_trial_identities", [])),
            ordered_evidence_fingerprints=tuple(data.get("ordered_evidence_fingerprints", [])),
            completed_trial_count=int(data.get("completed_trial_count", 0)),
            failed_trial_count=int(data.get("failed_trial_count", 0)),
            blocked_trial_count=int(data.get("blocked_trial_count", 0)),
            qualified_candidate_count=int(data.get("qualified_candidate_count", 0)),
            rejected_candidate_count=int(data.get("rejected_candidate_count", 0)),
            selection_governance_status=data.get("selection_governance_status", ""),
            selection_policy_fingerprint=data.get("selection_policy_fingerprint", ""),
            candidate_comparisons=tuple(candidate_comparisons),
            synthesis_methodology_version=data.get("synthesis_methodology_version", "1.0"),
        )

        if synthesis.synthesis_fingerprint != data.get("synthesis_fingerprint"):
            raise ValueError(
                f"Loaded synthesis fingerprint mismatch for campaign '{campaign_id}': "
                f"expected '{synthesis.synthesis_fingerprint}', got '{data.get('synthesis_fingerprint')}'."
            )

        return synthesis

    def save_selection_decision(self, decision: Any) -> Path:
        """Persist a ResearchCampaignSelectionDecision object to disk as JSON."""
        cdir = self._campaign_dir(decision.campaign_id)
        cdir.mkdir(parents=True, exist_ok=True)
        path = cdir / "selection_decision.json"
        if path.exists():
            existing = self.load_selection_decision(decision.campaign_id)
            if existing.decision_fingerprint != decision.decision_fingerprint:
                raise FileExistsError(
                    f"Cannot overwrite selection decision at '{path}' with conflicting decision fingerprint."
                )
            return path
        content = json.dumps(decision.as_dict(), indent=2)
        path.write_text(content, encoding="utf-8")
        return path

    def load_selection_decision(self, campaign_id: str) -> Any:
        """Load and reconstruct a ResearchCampaignSelectionDecision object from disk."""
        from src.evaluation.campaign_synthesis import ResearchCampaignSelectionDecision

        path = self._campaign_dir(campaign_id) / "selection_decision.json"
        if not path.exists():
            raise FileNotFoundError(f"Selection decision not found at: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse selection decision JSON at {path}: {exc}") from exc

        decision = ResearchCampaignSelectionDecision(
            campaign_id=data.get("campaign_id", ""),
            synthesis_fingerprint=data.get("synthesis_fingerprint", ""),
            selection_policy_fingerprint=data.get("selection_policy_fingerprint", ""),
            selected_candidate_ids=tuple(data.get("selected_candidate_ids", [])),
            eligible_candidate_ids=tuple(data.get("eligible_candidate_ids", [])),
            rejected_candidate_ids=tuple(data.get("rejected_candidate_ids", [])),
            blocked_candidate_ids=tuple(data.get("blocked_candidate_ids", [])),
            candidate_comparison_fingerprints=tuple(data.get("candidate_comparison_fingerprints", [])),
            selection_governance_fingerprints=tuple(data.get("selection_governance_fingerprints", [])),
            qualification_fingerprints=tuple(data.get("qualification_fingerprints", [])),
            robustness_fingerprints=tuple(data.get("robustness_fingerprints", [])),
            evidence_fingerprints=tuple(data.get("evidence_fingerprints", [])),
            decision_status=data.get("decision_status", ""),
            decision_reason=data.get("decision_reason", ""),
            deterministic_ordering=tuple(data.get("deterministic_ordering", [])),
            decision_methodology_version=data.get("decision_methodology_version", "1.0"),
        )

        if decision.decision_fingerprint != data.get("decision_fingerprint"):
            raise ValueError(
                f"Loaded selection decision fingerprint mismatch for campaign '{campaign_id}': "
                f"expected '{decision.decision_fingerprint}', got '{data.get('decision_fingerprint')}'."
            )

        return decision

    def save_campaign_learning(self, artifact: Any) -> Path:
        """Persist a GovernedCampaignLearningArtifact object to disk as JSON."""
        cdir = self._campaign_dir(artifact.campaign_id)
        cdir.mkdir(parents=True, exist_ok=True)
        path = cdir / "campaign_learning.json"
        if path.exists():
            existing = self.load_campaign_learning(artifact.campaign_id)
            if existing.artifact_fingerprint != artifact.artifact_fingerprint:
                raise FileExistsError(
                    f"Cannot overwrite campaign learning artifact at '{path}' with conflicting artifact fingerprint."
                )
            return path
        content = json.dumps(artifact.as_dict(), indent=2)
        path.write_text(content, encoding="utf-8")
        return path

    def load_campaign_learning(self, campaign_id: str) -> Any:
        """Load and reconstruct a GovernedCampaignLearningArtifact object from disk."""
        from src.evaluation.campaign_learning import GovernedCampaignLearningArtifact

        path = self._campaign_dir(campaign_id) / "campaign_learning.json"
        if not path.exists():
            raise FileNotFoundError(f"Campaign learning artifact not found at: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse campaign learning JSON at {path}: {exc}") from exc

        return GovernedCampaignLearningArtifact.from_dict(data)
