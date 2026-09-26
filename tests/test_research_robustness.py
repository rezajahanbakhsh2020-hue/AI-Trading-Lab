"""Comprehensive unit, integration, property, and AST tests for
Canonical Research Robustness, Benchmark, and Regime-Aware Evidence Layer.
"""

import ast
from datetime import datetime, timezone
from pathlib import Path
import pytest
import pandas as pd

from src.evaluation.candidate_generator import CandidateSpec
from src.evaluation.discovery_engine import DiscoveryEngine, DiscoveryRunResult
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
from src.evaluation.research_qualification import qualify_research_evidence
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
from src.evaluation.selection_governance import assess_research_selection


def make_sample_evidence(
    *,
    hypothesis: str = "Test hypothesis for momentum candidate",
    methodology_version: str = "1.0",
    strategy_name: str = "momentum",
    strategy_version: str = "1.0.0",
    dataset_id: str = "ds_xauusd_2025",
    symbol: str = "XAUUSD",
    timeframe: str = "5m",
    start_date: str = "2025-01-01",
    end_date: str = "2025-01-02",
    transaction_cost: float = 0.001,
    slippage: float = 0.001,
    latency_ms: float = 10.0,
    commit_sha: str = "abc1234567890def1234567890def1234567890d",
    benchmark_reference: str = "buy_and_hold",
    parameters: dict | None = None,
    robustness_verdict: dict | None = None,
    benchmark_comparison: dict | None = None,
    promotion_status: PromotionStatus = PromotionStatus.PROMOTABLE,
    rejection_reasons: tuple[RejectionReason, ...] = (),
    include_is: bool = True,
    include_val: bool = True,
    include_oos: bool = True,
    include_wf: bool = True,
    is_obs: int = 50,
    val_obs: int = 30,
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
        parameters=parameters or {"momentum_window": 10, "threshold": 0.01},
    )

    partitions = []
    if include_is:
        partitions.append(
            EvidencePartition(
                role=EvidencePartitionRole.IN_SAMPLE,
                start_date="2025-01-01",
                end_date="2025-01-01",
                total_return=0.10,
                max_drawdown=0.02,
                sharpe_ratio=1.5,
                observations=is_obs,
            )
        )
    if include_val:
        partitions.append(
            EvidencePartition(
                role=EvidencePartitionRole.VALIDATION,
                start_date="2025-01-01",
                end_date="2025-01-02",
                total_return=0.08,
                max_drawdown=0.02,
                sharpe_ratio=1.2,
                observations=val_obs,
            )
        )
    if include_oos:
        partitions.append(
            EvidencePartition(
                role=EvidencePartitionRole.OUT_OF_SAMPLE,
                start_date="2025-01-02",
                end_date="2025-01-02",
                total_return=0.06,
                max_drawdown=0.03,
                sharpe_ratio=1.1,
                observations=oos_obs,
            )
        )
    if include_wf:
        partitions.append(
            EvidencePartition(
                role=EvidencePartitionRole.WALK_FORWARD,
                start_date="2025-01-01",
                end_date="2025-01-02",
                total_return=0.07,
                max_drawdown=0.02,
                sharpe_ratio=1.3,
                observations=wf_obs,
            )
        )

    if robustness_verdict is None:
        robustness_verdict = {
            "is_robust": True,
            "parameter_sensitivity": {"passed": True},
            "subsample_stability": {"passed": True},
            "execution_cost_stress": {"passed": True},
            "statistical_validation": {"passed": True},
            "anti_overfitting": {"passed": True},
            "verdict_summary": "Passed all robustness checks.",
        }

    if benchmark_comparison is None:
        benchmark_comparison = {
            "benchmark_reference": benchmark_reference,
            "benchmark_total_return": 0.02,
            "outperformed_benchmark": True,
        }

    return ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=tuple(partitions),
        robustness_verdict=robustness_verdict,
        benchmark_comparison=benchmark_comparison,
        promotion_status=promotion_status,
        rejection_reasons=rejection_reasons,
        created_at_utc="2025-01-01T00:00:00+00:00",
    )


# --- Test Matrix A-D: Robustness Status Classifications ---

def test_A_robustness_not_evaluated() -> None:
    ev = make_sample_evidence(robustness_verdict={})
    assessment = assess_research_robustness(ev)

    assert assessment.status == RobustnessStatus.NOT_EVALUATED
    assert assessment.is_robust is False
    assert len(assessment.dimensions_evaluated) == 0
    assert len(assessment.dimensions_unavailable) == 5
    assert "not executed" in assessment.limitations[0]


def test_B_robustness_evaluated() -> None:
    ev = make_sample_evidence()
    assessment = assess_research_robustness(ev)

    assert assessment.status == RobustnessStatus.EVALUATED
    assert assessment.is_robust is True
    assert len(assessment.dimensions_evaluated) == 5
    assert len(assessment.dimensions_unavailable) == 0


def test_C_partial_robustness_evaluation() -> None:
    partial_verdict = {
        "is_robust": False,
        "parameter_sensitivity": {"passed": True},
        "verdict_summary": "Partial checks.",
    }
    ev = make_sample_evidence(robustness_verdict=partial_verdict)
    assessment = assess_research_robustness(ev)

    assert assessment.status == RobustnessStatus.PARTIALLY_EVALUATED
    assert "parameter_sensitivity" in assessment.dimensions_evaluated
    assert "subsample_stability" in assessment.dimensions_unavailable


def test_D_insufficient_robustness_data() -> None:
    insufficient_verdict = {
        "is_robust": False,
        "rejection_reasons": ["INSUFFICIENT_STATISTICAL_SAMPLE"],
        "parameter_sensitivity": {"passed": False},
        "verdict_summary": "Insufficient statistical sample.",
    }
    ev = make_sample_evidence(robustness_verdict=insufficient_verdict)
    assessment = assess_research_robustness(ev)

    assert assessment.status == RobustnessStatus.INSUFFICIENT_DATA
    assert any("Insufficient statistical observations" in lim for lim in assessment.limitations)


# --- Test Matrix E-H: Fingerprints and Input Sensitivity ---

def test_E_deterministic_robustness_fingerprint() -> None:
    ev1 = make_sample_evidence()
    ev2 = make_sample_evidence()

    ass1 = assess_research_robustness(ev1)
    ass2 = assess_research_robustness(ev2)

    assert ass1.robustness_fingerprint == ass2.robustness_fingerprint
    assert len(ass1.robustness_fingerprint) == 64


def test_F_changed_source_evidence_changes_fingerprint() -> None:
    ev1 = make_sample_evidence(hypothesis="Hypothesis 1")
    ev2 = make_sample_evidence(hypothesis="Hypothesis 2")

    ass1 = assess_research_robustness(ev1)
    ass2 = assess_research_robustness(ev2)

    assert ass1.robustness_fingerprint != ass2.robustness_fingerprint


def test_G_changed_partition_changes_fingerprint() -> None:
    ev1 = make_sample_evidence(is_obs=50)
    ev2 = make_sample_evidence(is_obs=100)

    ass1 = assess_research_robustness(ev1)
    ass2 = assess_research_robustness(ev2)

    assert ass1.robustness_fingerprint != ass2.robustness_fingerprint


def test_H_changed_execution_assumptions_changes_fingerprint() -> None:
    ev1 = make_sample_evidence(slippage=0.001)
    ev2 = make_sample_evidence(slippage=0.005)

    ass1 = assess_research_robustness(ev1)
    ass2 = assess_research_robustness(ev2)

    assert ass1.robustness_fingerprint != ass2.robustness_fingerprint


def test_changed_robustness_criteria_changes_fingerprint() -> None:
    ev = make_sample_evidence()
    crit1 = RobustnessCriteria(min_t_stat=1.65)
    crit2 = RobustnessCriteria(min_t_stat=2.0)

    ass1 = assess_research_robustness(ev, robustness_criteria=crit1)
    ass2 = assess_research_robustness(ev, robustness_criteria=crit2)

    assert ass1.robustness_fingerprint != ass2.robustness_fingerprint


# --- Test Matrix I-K: Regime Analysis ---

def test_I_regime_analysis_available_only_when_authoritative_regime_data_exists() -> None:
    regime_info = {
        "methodology": "volatility_regime_v1",
        "evaluated_regimes": ["low_volatility", "normal_volatility", "high_volatility"],
        "regime_stability_score": 0.85,
        "notes": "Evaluated using historical 20-period volatility ratio.",
    }
    verdict = {
        "is_robust": True,
        "parameter_sensitivity": {"passed": True},
        "regime_analysis": regime_info,
    }
    ev = make_sample_evidence(robustness_verdict=verdict)
    assessment = assess_research_robustness(ev)

    assert assessment.regime_assessment.status == RegimeStatus.REGIME_EVALUATED
    assert assessment.regime_assessment.regime_methodology == "volatility_regime_v1"
    assert assessment.regime_assessment.evaluated_regimes == ("low_volatility", "normal_volatility", "high_volatility")
    assert assessment.regime_assessment.regime_stability_score == 0.85


def test_J_regime_analysis_unavailable_is_explicit() -> None:
    ev = make_sample_evidence()
    assessment = assess_research_robustness(ev)

    assert assessment.regime_assessment.status == RegimeStatus.REGIME_UNAVAILABLE
    assert assessment.regime_assessment.regime_methodology == "none"
    assert "not configured or evaluated" in assessment.regime_assessment.notes


def test_K_no_fabricated_regime_labels() -> None:
    ev = make_sample_evidence()
    assessment = assess_research_robustness(ev)

    # Must NOT fabricate arbitrary regime labels
    assert assessment.regime_assessment.evaluated_regimes == ()
    assert assessment.regime_assessment.regime_stability_score is None


# --- Test Matrix L-N: Benchmark Evidence ---

def test_L_benchmark_available_when_authoritative_benchmark_exists() -> None:
    ev = make_sample_evidence(
        benchmark_reference="buy_and_hold",
        benchmark_comparison={
            "benchmark_reference": "buy_and_hold",
            "benchmark_total_return": 0.05,
            "outperformed_benchmark": True,
        },
    )
    assessment = assess_research_robustness(ev)

    assert assessment.benchmark_assessment.status == BenchmarkStatus.BENCHMARK_EVALUATED
    assert assessment.benchmark_assessment.benchmark_reference == "buy_and_hold"
    assert assessment.benchmark_assessment.benchmark_total_return == 0.05
    assert assessment.benchmark_assessment.outperformed_benchmark is True


def test_M_benchmark_unavailable_is_explicit() -> None:
    ev = make_sample_evidence(
        benchmark_reference="none",
        benchmark_comparison={},
    )
    assessment = assess_research_robustness(ev)

    assert assessment.benchmark_assessment.status == BenchmarkStatus.BENCHMARK_UNAVAILABLE
    assert assessment.benchmark_assessment.benchmark_reference == "none"
    assert "No authoritative benchmark" in assessment.benchmark_assessment.notes


def test_N_no_fabricated_benchmark() -> None:
    ev = make_sample_evidence(
        benchmark_reference="none",
        benchmark_comparison={},
    )
    assessment = assess_research_robustness(ev)

    assert assessment.benchmark_assessment.benchmark_total_return is None
    assert assessment.benchmark_assessment.outperformed_benchmark is None


# --- Test Matrix O-Q: Multi-Partition Provenance & Separation ---

def test_O_multi_partition_evidence_preserves_provenance() -> None:
    ev = make_sample_evidence()
    assessment = assess_research_robustness(ev)

    assert len(assessment.partition_records) == 4
    roles = [p.role for p in assessment.partition_records]
    assert roles == ["IN_SAMPLE", "VALIDATION", "OUT_OF_SAMPLE", "WALK_FORWARD"]

    for prec in assessment.partition_records:
        assert prec.evidence_fingerprint == ev.evidence_id
        assert len(prec.partition_fingerprint) == 64


def test_P_OOS_partition_remains_distinct_from_IS() -> None:
    ev = make_sample_evidence()
    assessment = assess_research_robustness(ev)

    is_rec = next(p for p in assessment.partition_records if p.role == "IN_SAMPLE")
    oos_rec = next(p for p in assessment.partition_records if p.role == "OUT_OF_SAMPLE")

    assert is_rec.role != oos_rec.role
    assert is_rec.partition_fingerprint != oos_rec.partition_fingerprint


def test_Q_walk_forward_evidence_remains_distinct() -> None:
    ev = make_sample_evidence()
    assessment = assess_research_robustness(ev)

    wf_rec = next(p for p in assessment.partition_records if p.role == "WALK_FORWARD")
    assert wf_rec.role == "WALK_FORWARD"
    assert len(wf_rec.partition_fingerprint) == 64


# --- Test Matrix R-V: Governance, Qualification, and Auto-Promotion Safety ---

def test_R_selection_governance_remains_canonical() -> None:
    ev = make_sample_evidence()
    sel_ass = assess_research_selection(ev)
    assert sel_ass.selection_fingerprint is not None
    assert len(sel_ass.selection_fingerprint) == 64


def test_S_qualification_remains_canonical() -> None:
    ev = make_sample_evidence()
    qual_res = qualify_research_evidence(ev)
    assert qual_res.qualified is True
    assert qual_res.evidence_fingerprint == ev.experiment_fingerprint


def test_V_no_automatic_promotion() -> None:
    ev = make_sample_evidence()
    rob_ass = assess_research_robustness(ev)

    # Robustness assessment must NOT alter promotion status
    assert ev.promotion_status == PromotionStatus.PROMOTABLE
    assert getattr(rob_ass, "promoted", False) is False


# --- Test Matrix Z: Full Discovery -> Evidence -> Qualification -> Selection -> Robustness -> Finding ---

def test_Z_full_discovery_pipeline_integration() -> None:
    dates = pd.date_range("2025-01-01", periods=100, freq="1h")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 2000.0,
            "high": 2010.0,
            "low": 1990.0,
            "close": 2005.0,
            "volume": 100,
        }
    )

    ds = DatasetScope(
        dataset_id="ds_test",
        symbol="XAUUSD",
        timeframe="1h",
        start_date="2025-01-01",
        end_date="2025-01-05",
    )
    ea = ExecutionAssumptions(transaction_cost=0.001, slippage=0.001, latency_ms=10.0)
    cp = CodeProvenance(commit_sha="abc1234567890def1234567890def1234567890d")

    candidates = [
        CandidateSpec(
            generator_name="test_gen",
            generator_version="1.0",
            strategy_name="momentum",
            parameters={"momentum_window": 5},
        )
    ]

    engine = DiscoveryEngine()
    result: DiscoveryRunResult = engine.run_discovery(
        df=df,
        candidates=candidates,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
    )

    assert result.candidates_evaluated == 1
    assert len(result.trial_ledger) == 1
    assert len(result.selection_assessments) == 1
    assert len(result.robustness_assessments) == 1

    rob_ass = result.robustness_assessments[0]
    assert isinstance(rob_ass, ResearchRobustnessAssessment)
    assert rob_ass.status in (RobustnessStatus.EVALUATED, RobustnessStatus.INSUFFICIENT_DATA, RobustnessStatus.PARTIALLY_EVALUATED)
    assert len(rob_ass.robustness_fingerprint) == 64


# --- Test Matrix W-Y: AST / Structural Guards ---

def test_W_X_Y_ast_structural_guards() -> None:
    rob_file = Path("src/evaluation/research_robustness.py")
    tree = ast.parse(rob_file.read_text(encoding="utf-8"), filename=str(rob_file))

    forbidden_imports = {
        "src.evaluation.live_production_decision",
        "src.evaluation.live_execution_runtime",
        "src.integration.project2_publisher",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert (
                node.module not in forbidden_imports
            ), f"Forbidden import '{node.module}' found in research_robustness.py"

    content = rob_file.read_text(encoding="utf-8")
    assert "ProductionDecision" not in content
    assert "ProductionSignal" not in content
    assert "ProductionRiskLevels" not in content
    assert "project2" not in content.lower()
