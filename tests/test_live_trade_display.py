import pandas as pd
import pytest

from src.evaluation.live_runtime import build_live_runtime
from src.evaluation.live_trade_display import (
    DEFAULT_TP1_MULTIPLIER,
    DEFAULT_TP2_MULTIPLIER,
    DEFAULT_TP3_MULTIPLIER,
    build_live_trade_display,
)


@pytest.fixture(autouse=True)
def mock_candidate_resolution(monkeypatch):
    from tests.test_production_decision_integrity import make_promoted_evidence
    from src.evaluation.research_qualification import qualify_research_evidence
    from src.evaluation.research_robustness import assess_research_robustness
    from src.evaluation.live_production_decision import PromotedCandidateArtifact

    def synth_resolve(candidate_id=None, strategy_id=None, symbol="XAUUSD", timeframe="5m", base_dir=None, **kwargs):
        strat = strategy_id or "momentum"
        tf = timeframe or "5m"
        cid = candidate_id or f"cand_{strat}_{tf}"
        ev = make_promoted_evidence(timeframe=tf, symbol=symbol, strategy_name=strat)
        rob = assess_research_robustness(ev)
        gov = qualify_research_evidence(ev, robustness_assessment=rob)
        params = dict(ev.spec.parameters)
        return PromotedCandidateArtifact(
            candidate_id=cid,
            strategy_name=strat,
            strategy_version="1.0",
            evidence=ev,
            symbol=symbol,
            timeframe=tf,
            operational_stability_score=0.7458282289664787,
            parameters=params,
            governance_decision=gov,
            governance_decision_fingerprint=gov.decision_fingerprint,
        )

    monkeypatch.setattr("src.evaluation.research_store.resolve_promoted_candidate", synth_resolve)
    monkeypatch.setattr("src.evaluation.live_runtime.resolve_promoted_candidate", synth_resolve)


def _rising_data(rows: int = 80) -> pd.DataFrame:
    timestamps = pd.date_range(
        end=pd.Timestamp.now(tz="UTC"),
        periods=rows,
        freq="5min",
    )

    close = pd.Series(
        [2000.0 + index for index in range(rows)],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "isOpen": False,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        }
    )


def _get_canonical_decision(data: pd.DataFrame, strategy: str = "momentum", score: float | None = None, min_score: float = 0.50):
    res = build_live_runtime(data, stable_strategy=strategy, stability_score=score, min_stability_score=min_score, persist=False)
    return res.canonical_decision




def test_buy_trade_display_contains_entry_sl_and_three_targets():
    data = _rising_data()
    cld = _get_canonical_decision(data, strategy="momentum", score=None)
    result = build_live_trade_display(
        data,
        canonical_decision=cld,
        stable_strategy="momentum",
        stability_score=None,
    )

    assert result["decision"] == "BUY"
    assert result["entry_price"] is not None
    assert result["stop_loss"] is not None
    assert result["tp1"] is not None
    assert result["tp2"] is not None
    assert result["tp3"] is not None

    assert result["stop_loss"] < result["entry_price"]
    assert result["entry_price"] < result["tp1"]
    assert result["tp1"] < result["tp2"]
    assert result["tp2"] < result["tp3"]


def test_default_targets_have_one_two_three_risk_structure():
    data = _rising_data()
    cld = _get_canonical_decision(data, strategy="momentum", score=None)
    result = build_live_trade_display(
        data,
        canonical_decision=cld,
        stable_strategy="momentum",
        stability_score=None,
    )

    assert result["tp1_multiplier"] == DEFAULT_TP1_MULTIPLIER
    assert result["tp2_multiplier"] == DEFAULT_TP2_MULTIPLIER
    assert result["tp3_multiplier"] == DEFAULT_TP3_MULTIPLIER

    assert result["risk_reward_tp1"] == pytest.approx(1.0)
    assert result["risk_reward_tp2"] == pytest.approx(2.0)
    assert result["risk_reward_tp3"] == pytest.approx(3.0)


def test_custom_target_multipliers_are_applied():
    from dataclasses import replace
    from src.evaluation.research_store import resolve_promoted_candidate
    from src.evaluation.live_production_decision import calculate_production_risk_levels

    data = _rising_data()
    cld = _get_canonical_decision(data, strategy="momentum", score=None)
    cand = resolve_promoted_candidate(strategy_id="momentum", symbol="XAUUSD", timeframe="5m")

    # Re-calculate canonical risk levels with custom multipliers to test canonical projection
    custom_risk = calculate_production_risk_levels(
        decision=cld.decision,
        candidate=cand,
        tp1_multiplier=0.5,
        tp2_multiplier=1.5,
        tp3_multiplier=2.5,
    )
    custom_cld = replace(cld, risk_levels=custom_risk)

    result = build_live_trade_display(
        data,
        canonical_decision=custom_cld,
        stable_strategy="momentum",
        stability_score=None,
        tp1_multiplier=0.5,
        tp2_multiplier=1.5,
        tp3_multiplier=2.5,
    )

    assert result["tp1"] == custom_cld.risk_levels.tp1
    assert result["tp2"] == custom_cld.risk_levels.tp2
    assert result["tp3"] == custom_cld.risk_levels.tp3
    assert result["risk_reward_tp1"] == pytest.approx(0.5)
    assert result["risk_reward_tp2"] == pytest.approx(1.5)
    assert result["risk_reward_tp3"] == pytest.approx(2.5)


def test_no_trade_has_no_trade_levels():
    data = _rising_data()
    cld = _get_canonical_decision(data, strategy="momentum", score=None, min_score=0.90)
    result = build_live_trade_display(
        data,
        canonical_decision=cld,
        stable_strategy="momentum",
        stability_score=None,
    )

    assert result["decision"] == "NO TRADE"
    assert result["entry_price"] is None
    assert result["stop_loss"] is None
    assert result["tp1"] is None
    assert result["tp2"] is None
    assert result["tp3"] is None
    assert result["risk_distance"] is None


def test_unsupported_strategy_has_no_trade_levels(monkeypatch):
    data = _rising_data()

    from unittest.mock import MagicMock
    from src.evaluation.research_store import resolve_promoted_candidate
    mom_cand = resolve_promoted_candidate(strategy_id="momentum", symbol="XAUUSD", timeframe="5m")

    mock_cand = MagicMock(spec=mom_cand)
    mock_cand.candidate_id = "cand_unsupported_5m"
    mock_cand.strategy_name = "unsupported_strategy"
    mock_cand.strategy_version = "1.0"
    mock_cand.symbol = "XAUUSD"
    mock_cand.timeframe = "5m"
    mock_cand.evidence = mom_cand.evidence
    mock_cand.policy = mom_cand.policy
    mock_cand.governance_decision = None
    mock_cand.parameters = mom_cand.parameters
    mock_cand.operational_stability_score = 0.7458282289664787
    mock_cand.governance_decision_fingerprint = "gov_dec_unsupported"
    mock_cand.artifact_fingerprint = "art_unsupported"
    mock_cand.campaign_selection_decision_fingerprint = None

    monkeypatch.setattr(
        "src.evaluation.live_runtime.resolve_promoted_candidate",
        lambda *args, **kwargs: mock_cand,
    )
    monkeypatch.setattr(
        "src.evaluation.research_store.resolve_promoted_candidate",
        lambda *args, **kwargs: mock_cand,
    )

    cld = _get_canonical_decision(data, strategy="unsupported_strategy", score=None)
    result = build_live_trade_display(
        data,
        canonical_decision=cld,
        stable_strategy="unsupported_strategy",
        stability_score=None,
    )

    assert result["decision"] == "NO TRADE"
    assert result["strategy_supported"] is False
    assert result["tp1"] is None
    assert result["tp2"] is None
    assert result["tp3"] is None


def test_moving_average_strategy_supported_in_display():
    data = _rising_data()
    cld = _get_canonical_decision(data, strategy="moving_average", score=None)
    result = build_live_trade_display(
        data,
        canonical_decision=cld,
        stable_strategy="moving_average",
        stability_score=None,
    )

    assert result["decision"] == "BUY"
    assert result["strategy_supported"] is True
    assert result["tp1"] is not None
    assert result["tp2"] is not None
    assert result["tp3"] is not None


def test_tp_multipliers_must_be_strictly_increasing():
    data = _rising_data()
    cld = _get_canonical_decision(data)
    with pytest.raises(ValueError, match="TP1 < TP2 < TP3"):
        build_live_trade_display(
            data,
            canonical_decision=cld,
            stable_strategy="momentum",
            stability_score=None,
            tp1_multiplier=2.0,
            tp2_multiplier=1.0,
            tp3_multiplier=3.0,
        )


def test_tp_multipliers_must_be_positive():
    data = _rising_data()
    cld = _get_canonical_decision(data)
    with pytest.raises(
        ValueError,
        match="tp1_multiplier must be greater than 0",
    ):
        build_live_trade_display(
            data,
            canonical_decision=cld,
            stable_strategy="momentum",
            stability_score=None,
            tp1_multiplier=0.0,
        )


def test_no_sell_decision_is_invented():
    data = _rising_data()
    cld = _get_canonical_decision(data, strategy="momentum", score=None)
    result = build_live_trade_display(
        data,
        canonical_decision=cld,
        stable_strategy="momentum",
        stability_score=None,
    )

    assert result["decision"] in {"BUY", "NO TRADE"}
    assert result["decision"] != "SELL"


def test_missing_canonical_decision_fails_closed_without_calling_runtime(monkeypatch):
    data = _rising_data()

    mock_build_runtime = pytest.fail
    mock_build_prod_dec = pytest.fail
    mock_authorize = pytest.fail
    mock_resolve = pytest.fail

    monkeypatch.setattr("src.evaluation.live_runtime.build_live_runtime", mock_build_runtime, raising=False)
    monkeypatch.setattr("src.evaluation.live_production_decision.build_live_production_decision", mock_build_prod_dec, raising=False)
    monkeypatch.setattr("src.evaluation.live_production_decision.authorize_production_runtime", mock_authorize, raising=False)
    monkeypatch.setattr("src.evaluation.research_store.resolve_promoted_candidate", mock_resolve, raising=False)

    result = build_live_trade_display(data, canonical_decision=None, stable_strategy="momentum", stability_score=None)

    assert result["decision"] == "BLOCKED"
    assert result["reason"] == "missing_canonical_live_decision"
    assert result["entry_price"] is None
    assert result["stop_loss"] is None
    assert result["tp1"] is None


def test_valid_canonical_decision_preserves_identities_without_runtime_calls(monkeypatch):
    data = _rising_data()
    cld = _get_canonical_decision(data, strategy="momentum", score=None)

    monkeypatch.setattr("src.evaluation.live_runtime.build_live_runtime", pytest.fail, raising=False)
    monkeypatch.setattr("src.evaluation.live_production_decision.build_live_production_decision", pytest.fail, raising=False)
    monkeypatch.setattr("src.evaluation.live_production_decision.authorize_production_runtime", pytest.fail, raising=False)
    monkeypatch.setattr("src.evaluation.research_store.resolve_promoted_candidate", pytest.fail, raising=False)

    result = build_live_trade_display(data, canonical_decision=cld)

    assert result["decision"] == "BUY"
    assert result["decision_id"] == cld.decision.decision_id
    assert result["canonical_live_decision_fingerprint"] == cld.canonical_live_decision_fingerprint
    assert result["authorization_fingerprint"] == cld.authorization_receipt.authorization_fingerprint


def test_end_to_end_candidate_tp_multipliers_propagate_to_display(monkeypatch):
    from src.evaluation.research_store import resolve_promoted_candidate
    from src.evaluation.research_qualification import qualify_research_evidence
    from src.evaluation.research_robustness import assess_research_robustness
    from src.evaluation.live_production_decision import PromotedCandidateArtifact
    from dataclasses import replace

    mom_cand = resolve_promoted_candidate(strategy_id="momentum", symbol="XAUUSD", timeframe="5m")
    custom_params = {
        "momentum_window": 10,
        "stop_loss_pct": 0.01,
        "take_profit_pct": 0.02,
        "tp1_multiplier": 0.5,
        "tp2_multiplier": 1.5,
        "tp3_multiplier": 2.5,
    }
    custom_spec = replace(mom_cand.evidence.spec, parameters=custom_params)
    custom_evidence = replace(mom_cand.evidence, spec=custom_spec, experiment_fingerprint=custom_spec.fingerprint)
    rob = assess_research_robustness(custom_evidence)
    gov = qualify_research_evidence(custom_evidence, robustness_assessment=rob)
    custom_cand = PromotedCandidateArtifact(
        candidate_id=mom_cand.candidate_id,
        strategy_name=mom_cand.strategy_name,
        strategy_version=mom_cand.strategy_version,
        evidence=custom_evidence,
        symbol=mom_cand.symbol,
        timeframe=mom_cand.timeframe,
        operational_stability_score=mom_cand.operational_stability_score,
        governance_decision=gov,
        governance_decision_fingerprint=gov.decision_fingerprint,
        parameters=custom_params,
    )

    monkeypatch.setattr("src.evaluation.live_runtime.resolve_promoted_candidate", lambda *a, **k: custom_cand)
    monkeypatch.setattr("src.evaluation.research_store.resolve_promoted_candidate", lambda *a, **k: custom_cand)

    data = _rising_data()
    res = build_live_runtime(data, stable_strategy="momentum", stability_score=None, persist=False)
    cld = res.canonical_decision

    display = build_live_trade_display(data, canonical_decision=cld)

    assert display["tp1_multiplier"] == 0.5
    assert display["tp2_multiplier"] == 1.5
    assert display["tp3_multiplier"] == 2.5
    assert display["tp1"] == cld.risk_levels.tp1
    assert display["tp2"] == cld.risk_levels.tp2
    assert display["tp3"] == cld.risk_levels.tp3
    assert display["risk_reward_tp1"] == pytest.approx(0.5)
    assert display["risk_reward_tp2"] == pytest.approx(1.5)
    assert display["risk_reward_tp3"] == pytest.approx(2.5)


def test_canonical_risk_levels_are_authoritative_for_display():
    from dataclasses import replace
    from src.evaluation.live_production_decision import ProductionRiskLevels, Direction

    data = _rising_data()
    cld = _get_canonical_decision(data, strategy="momentum", score=None)

    # Construct discriminating custom risk levels artifact distinct from default formula
    # Default formula for entry=2079, sl=2058.21 would be tp1=2099.79, tp2=2120.58, tp3=2141.37
    discriminated_risk = ProductionRiskLevels(
        decision_id=cld.decision.decision_id,
        candidate_id=cld.decision.candidate_id,
        evidence_id=cld.decision.evidence_id,
        symbol=cld.decision.symbol,
        timeframe=cld.decision.timeframe,
        direction=Direction.BUY,
        entry_price=2079.0,
        stop_loss=2050.0,
        tp1=2100.0,
        tp2=2150.0,
        tp3=2200.0,
        trailing_stop=2040.0,
        risk_reward_ratio=2.448275862,
    )
    custom_cld = replace(cld, risk_levels=discriminated_risk)

    display = build_live_trade_display(data, canonical_decision=custom_cld)

    assert display["entry_price"] == 2079.0
    assert display["stop_loss"] == 2050.0
    assert display["tp1"] == 2100.0
    assert display["tp2"] == 2150.0
    assert display["tp3"] == 2200.0
    assert display["take_profit"] == 2150.0
    assert display["trailing_stop"] == 2040.0
    assert display["risk_reward_ratio"] == 2.448275862


def test_no_trade_and_invalid_risk_data_remain_safe():
    data = _rising_data()
    cld = _get_canonical_decision(data, strategy="momentum", score=None, min_score=0.99)
    display = build_live_trade_display(data, canonical_decision=cld)

    assert display["decision"] == "NO TRADE"
    assert display["entry_price"] is None
    assert display["stop_loss"] is None
    assert display["tp1"] is None
    assert display["tp2"] is None
    assert display["tp3"] is None
    assert display["risk_reward_ratio"] is None


@pytest.mark.parametrize("missing_field", ["tp1", "tp2", "tp3", "risk_reward_ratio"])
def test_missing_canonical_risk_field_fails_closed(missing_field):
    from dataclasses import replace
    data = _rising_data()
    cld = _get_canonical_decision(data, strategy="momentum", score=None)

    incomplete_risk = replace(cld.risk_levels, **{missing_field: None})
    custom_cld = replace(cld, risk_levels=incomplete_risk)

    expected_match = "tp1, tp2, and tp3" if "tp" in missing_field else "risk_reward_ratio"
    with pytest.raises(ValueError, match=expected_match):
        build_live_trade_display(data, canonical_decision=custom_cld)
