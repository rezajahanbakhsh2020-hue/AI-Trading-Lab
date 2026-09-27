"""Unit tests for Research Campaign governance layer in Project 1."""

import os
from pathlib import Path
import tempfile
import pytest
import pandas as pd

from src.evaluation.candidate_generator import CandidateSpec, ResearchSearchSpace
from src.evaluation.discovery_engine import (
    DiscoveryCriteria,
    DiscoveryEngine,
    DiscoveryRunResult,
    ResearchSearchPolicy,
)
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
    PromotionStatus,
    RejectionReason,
    ResearchCampaign,
    ResearchCampaignStatus,
    compute_campaign_fingerprint,
)
from src.evaluation.research_store import load_research_campaign, save_research_campaign


def make_test_data() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": [100.0 + i * 0.1 for i in range(100)],
            "high": [101.0 + i * 0.1 for i in range(100)],
            "low": [99.0 + i * 0.1 for i in range(100)],
            "close": [100.5 + i * 0.1 for i in range(100)],
            "volume": [1000 + i for i in range(100)],
        }
    )
    return df


def test_campaign_identity_is_deterministic():
    ds = DatasetScope("d1", "XAUUSD", "1h", "2024-01-01", "2024-01-05")
    ea = ExecutionAssumptions(0.0001, 0.0001, 5.0)
    cp = CodeProvenance("sha123")

    fp1 = compute_campaign_fingerprint(
        search_space_fingerprint="space1",
        search_policy_fingerprint="policy1",
        criteria_fingerprint="crit1",
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        candidate_ids=("cand1", "cand2"),
    )

    fp2 = compute_campaign_fingerprint(
        search_space_fingerprint="space1",
        search_policy_fingerprint="policy1",
        criteria_fingerprint="crit1",
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        candidate_ids=("cand2", "cand1"),  # Order shouldn't affect fingerprint
    )

    assert fp1 == fp2
    assert len(fp1) == 24


def test_campaign_preserves_search_space_and_policy_fingerprints():
    df = make_test_data()
    cand1 = CandidateSpec(
        generator_name="grid",
        generator_version="1.0",
        strategy_name="baseline_sma",
        parameters={"short_period": 5, "long_period": 20},
    )
    cand2 = CandidateSpec(
        generator_name="grid",
        generator_version="1.0",
        strategy_name="baseline_sma",
        parameters={"short_period": 10, "long_period": 30},
    )
    space = ResearchSearchSpace(candidate_definitions=(cand1, cand2), search_id="s1")
    policy = ResearchSearchPolicy(max_trials=2)

    ds = DatasetScope("d1", "XAUUSD", "1h", "2024-01-01", "2024-01-05")
    ea = ExecutionAssumptions(0.0001, 0.0001, 5.0)
    cp = CodeProvenance("sha123")

    engine = DiscoveryEngine()
    res = engine.run_discovery(
        df,
        space,
        ds,
        ea,
        cp,
        search_policy=policy,
    )

    assert res.campaign is not None
    assert res.campaign.search_space_fingerprint == space.search_fingerprint
    assert res.campaign.candidate_ids == (cand1.candidate_id, cand2.candidate_id)


def test_campaign_links_trials_to_campaign():
    df = make_test_data()
    cand = CandidateSpec(
        generator_name="grid",
        generator_version="1.0",
        strategy_name="baseline_sma",
        parameters={"short_period": 5, "long_period": 20},
    )
    space = ResearchSearchSpace(candidate_definitions=(cand,), search_id="s1")
    ds = DatasetScope("d1", "XAUUSD", "1h", "2024-01-01", "2024-01-05")
    ea = ExecutionAssumptions(0.0001, 0.0001, 5.0)
    cp = CodeProvenance("sha123")

    engine = DiscoveryEngine()
    res = engine.run_discovery(df, space, ds, ea, cp)

    assert res.campaign is not None
    assert len(res.trial_ledger) == 1
    assert res.trial_ledger[0].campaign_id == res.campaign.campaign_id


def test_campaign_selection_cannot_bypass_qualification():
    df = make_test_data()
    cand = CandidateSpec(
        generator_name="grid",
        generator_version="1.0",
        strategy_name="baseline_sma",
        parameters={"short_period": 5, "long_period": 20},
    )
    space = ResearchSearchSpace(candidate_definitions=(cand,), search_id="s1")
    ds = DatasetScope("d1", "XAUUSD", "1h", "2024-01-01", "2024-01-05")
    ea = ExecutionAssumptions(0.0001, 0.0001, 5.0)
    cp = CodeProvenance("sha123")

    # Criteria requiring impossible Sharpe ratio to force rejection
    strict_criteria = DiscoveryCriteria(min_is_sharpe=100.0)
    engine = DiscoveryEngine(criteria=strict_criteria)
    res = engine.run_discovery(df, space, ds, ea, cp)

    assert res.campaign is not None
    assert len(res.campaign.selected_candidate_ids) == 0
    assert len(res.promoted_evidence) == 0


def test_campaign_reconstruction_is_lossless():
    ds = DatasetScope("d1", "XAUUSD", "1h", "2024-01-01", "2024-01-05")
    ea = ExecutionAssumptions(0.0001, 0.0001, 5.0)
    cp = CodeProvenance("sha123")

    cid = compute_campaign_fingerprint(
        search_space_fingerprint="space1",
        search_policy_fingerprint="policy1",
        criteria_fingerprint="crit1",
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        candidate_ids=("c1", "c2"),
    )

    campaign = ResearchCampaign(
        campaign_id=cid,
        search_space_fingerprint="space1",
        search_policy_fingerprint="policy1",
        criteria_fingerprint="crit1",
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        candidate_ids=("c1", "c2"),
        evidence_fingerprints=("ev1",),
        selected_candidate_ids=("c1",),
        status=ResearchCampaignStatus.COMPLETED,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_research_campaign(campaign, base_dir=tmpdir)
        loaded = load_research_campaign(cid, base_dir=tmpdir)

        assert loaded.campaign_id == campaign.campaign_id
        assert loaded.reproducibility_fingerprint == campaign.reproducibility_fingerprint
        assert loaded.selected_candidate_ids == campaign.selected_candidate_ids


def test_campaign_corrupt_persisted_state_fails_closed():
    ds = DatasetScope("d1", "XAUUSD", "1h", "2024-01-01", "2024-01-05")
    ea = ExecutionAssumptions(0.0001, 0.0001, 5.0)
    cp = CodeProvenance("sha123")

    cid = compute_campaign_fingerprint(
        search_space_fingerprint="space1",
        search_policy_fingerprint="policy1",
        criteria_fingerprint="crit1",
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        candidate_ids=("c1",),
    )

    campaign = ResearchCampaign(
        campaign_id=cid,
        search_space_fingerprint="space1",
        search_policy_fingerprint="policy1",
        criteria_fingerprint="crit1",
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        candidate_ids=("c1",),
        evidence_fingerprints=(),
        selected_candidate_ids=(),
        status=ResearchCampaignStatus.COMPLETED,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_research_campaign(campaign, base_dir=tmpdir)

        # Corrupt file content
        data = path.read_text()
        corrupted_data = data.replace('"COMPLETED"', '"FAILED"')
        path.write_text(corrupted_data)

        with pytest.raises(ValueError, match="reproducibility fingerprint mismatch"):
            load_research_campaign(cid, base_dir=tmpdir)


def test_campaign_does_not_create_production_artifact():
    df = make_test_data()
    cand = CandidateSpec(
        generator_name="grid",
        generator_version="1.0",
        strategy_name="baseline_sma",
        parameters={"short_period": 5, "long_period": 20},
    )
    space = ResearchSearchSpace(candidate_definitions=(cand,), search_id="s1")
    ds = DatasetScope("d1", "XAUUSD", "1h", "2024-01-01", "2024-01-05")
    ea = ExecutionAssumptions(0.0001, 0.0001, 5.0)
    cp = CodeProvenance("sha123")

    engine = DiscoveryEngine(criteria=DiscoveryCriteria(min_observations_is=5, min_observations_oos=5, min_is_sharpe=-10.0, min_oos_sharpe=-10.0))
    res = engine.run_discovery(df, space, ds, ea, cp)

    assert res.campaign is not None
    # Ensure campaign is strictly a research container and has no production methods/artifact bindings
    assert not hasattr(res.campaign, "to_production_signal")
    assert not hasattr(res.campaign, "evaluate_production_decision")
