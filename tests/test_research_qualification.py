"""Unit and AST structural tests for Research Qualification Service and Safe Promotion Gate."""

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

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
    compute_experiment_fingerprint,
)
from src.evaluation.research_qualification import (
    ResearchQualificationPolicy,
    ResearchQualificationResult,
    qualify_research_evidence,
)
from src.evaluation.research_store import PromotionEligibilityError, save_research_candidate


def make_valid_evidence(
    *,
    hypothesis: str = "Test research hypothesis",
    methodology_version: str = "1.0",
    strategy_name: str = "momentum",
    strategy_version: str = "1.0.0",
    dataset_id: str = "ds_xauusd",
    symbol: str = "XAUUSD",
    timeframe: str = "5m",
    start_date: str = "2025-01-01",
    end_date: str = "2025-01-02",
    transaction_cost: float = 0.001,
    slippage: float = 0.001,
    latency_ms: float = 10.0,
    commit_sha: str = "843dfa76cf86a9057dba0a127541d7093fb15e42",
    benchmark_reference: str = "buy_and_hold",
    parameters: dict | None = None,
    robustness_passed: bool = True,
    promotion_status: PromotionStatus = PromotionStatus.PROMOTABLE,
    rejection_reasons: tuple[RejectionReason, ...] = (),
    created_at_utc: str = "2025-01-01T00:00:00+00:00",
    include_is: bool = True,
    include_oos: bool = True,
    include_wf: bool = True,
    is_obs: int = 50,
    oos_obs: int = 30,
    wf_obs: int = 30,
) -> ResearchEvidence:
    ds = DatasetScope(
        dataset_id=dataset_id,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )
    ea = ExecutionAssumptions(
        transaction_cost=transaction_cost,
        slippage=slippage,
        latency_ms=latency_ms,
    )
    cp = CodeProvenance(commit_sha=commit_sha)
    spec = ResearchExperimentSpec(
        hypothesis=hypothesis,
        methodology_version=methodology_version,
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        benchmark_reference=benchmark_reference,
        parameters=parameters or {"momentum_window": 10, "stop_loss_pct": 0.01, "take_profit_pct": 0.02},
    )

    partitions = []
    if include_is:
        partitions.append(
            EvidencePartition(
                role=EvidencePartitionRole.IN_SAMPLE,
                start_date=start_date,
                end_date=end_date,
                total_return=0.20,
                max_drawdown=0.05,
                sharpe_ratio=2.0,
                observations=is_obs,
            )
        )
    if include_oos:
        partitions.append(
            EvidencePartition(
                role=EvidencePartitionRole.OUT_OF_SAMPLE,
                start_date=start_date,
                end_date=end_date,
                total_return=0.15,
                max_drawdown=0.05,
                sharpe_ratio=1.8,
                observations=oos_obs,
            )
        )
    if include_wf:
        partitions.append(
            EvidencePartition(
                role=EvidencePartitionRole.WALK_FORWARD,
                start_date=start_date,
                end_date=end_date,
                total_return=0.10,
                max_drawdown=0.05,
                sharpe_ratio=1.5,
                observations=wf_obs,
            )
        )

    return ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=tuple(partitions),
        robustness_verdict={"passed": robustness_passed},
        promotion_status=promotion_status,
        rejection_reasons=rejection_reasons,
        created_at_utc=created_at_utc,
    )


def test_valid_evidence_qualifies() -> None:
    evidence = make_valid_evidence()
    res = qualify_research_evidence(evidence)

    assert res.qualified is True
    assert res.status == PromotionStatus.PROMOTABLE
    assert res.rejection_reasons == ()
    assert res.evidence_fingerprint == evidence.experiment_fingerprint


def test_missing_oos_rejected() -> None:
    evidence = make_valid_evidence(include_oos=False)
    res = qualify_research_evidence(evidence)

    assert res.qualified is False
    assert RejectionReason.FAILED_OOS in res.rejection_reasons


def test_missing_walk_forward_rejected() -> None:
    evidence = make_valid_evidence(include_wf=False)
    res = qualify_research_evidence(evidence)

    assert res.qualified is False
    assert RejectionReason.FAILED_WALK_FORWARD in res.rejection_reasons


def test_missing_robustness_rejected() -> None:
    evidence = make_valid_evidence(robustness_passed=False)
    res = qualify_research_evidence(evidence)

    assert res.qualified is False
    assert RejectionReason.FAILED_ROBUSTNESS in res.rejection_reasons


def test_missing_statistical_sample_rejected() -> None:
    evidence = make_valid_evidence(is_obs=5, oos_obs=5, wf_obs=5)
    res = qualify_research_evidence(evidence)

    assert res.qualified is False
    assert RejectionReason.INSUFFICIENT_STATISTICAL_SAMPLE in res.rejection_reasons


def test_tampered_evidence_fingerprint_mismatch_rejected() -> None:
    evidence = make_valid_evidence()
    # Mutate strategy_name after fingerprint computation
    object.__setattr__(evidence.spec, "strategy_name", "tampered_strategy")

    res = qualify_research_evidence(evidence)

    assert res.qualified is False
    assert RejectionReason.FAILED_REPRODUCIBILITY in res.rejection_reasons


def test_unpromoted_status_rejected() -> None:
    evidence = make_valid_evidence(promotion_status=PromotionStatus.PROPOSED)
    res = qualify_research_evidence(evidence)

    assert res.qualified is False
    assert RejectionReason.CRITIQUE_REJECTED in res.rejection_reasons


def test_prior_rejection_reasons_rejected() -> None:
    evidence = make_valid_evidence(
        promotion_status=PromotionStatus.REJECTED,
        rejection_reasons=(RejectionReason.DATA_LEAKAGE_CONCERN,),
    )
    res = qualify_research_evidence(evidence)

    assert res.qualified is False
    assert RejectionReason.DATA_LEAKAGE_CONCERN in res.rejection_reasons


def test_stale_evidence_rejected() -> None:
    evidence = make_valid_evidence(created_at_utc="2020-01-01T00:00:00+00:00")
    policy = ResearchQualificationPolicy(max_evidence_age_days=1.0)
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)

    res = qualify_research_evidence(evidence, policy=policy, now=now)

    assert res.qualified is False
    assert RejectionReason.STALE_INVALID_DATA in res.rejection_reasons


def test_unqualified_evidence_cannot_be_saved_to_candidate_store(tmp_path: Path) -> None:
    evidence = make_valid_evidence(include_oos=False)

    with pytest.raises(PromotionEligibilityError, match="failed qualification"):
        save_research_candidate(candidate_id="cand_bad", evidence=evidence, base_dir=tmp_path)


# --- AST Structural Invariant Tests ---

def test_ast_production_decision_code_does_not_calculate_qualification() -> None:
    """Verify production decision logic delegates qualification check and does not calculate custom qualification."""
    prod_file = Path("src/evaluation/live_production_decision.py")
    tree = ast.parse(prod_file.read_text(encoding="utf-8"), filename=str(prod_file))

    found_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.evaluation.research_qualification":
            if any(alias.name == "qualify_research_evidence" for alias in node.names):
                found_import = True

    assert found_import is True, "live_production_decision.py must delegate through qualify_research_evidence"


def test_ast_qualification_service_does_not_depend_on_production_decision_chain() -> None:
    """Verify research qualification is isolated from production decision and signal creation."""
    qual_file = Path("src/evaluation/research_qualification.py")
    content = qual_file.read_text(encoding="utf-8")
    tree = ast.parse(content, filename=str(qual_file))

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in (
                "src.evaluation.live_production_decision",
                "src.evaluation.live_execution_runtime",
                "src.integration.project2_publisher",
            ), f"Forbidden import '{node.module}' found in research_qualification.py"

    assert "ProductionDecision" not in content
    assert "ProductionSignal" not in content
    assert "ProductionRiskLevels" not in content
