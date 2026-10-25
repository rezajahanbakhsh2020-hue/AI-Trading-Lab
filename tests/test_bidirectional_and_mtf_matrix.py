"""Tests for end-to-end signal direction preservation (BUY, SELL, NO_TRADE) and seven-timeframe matrix isolation."""

import json
import pytest
import pandas as pd

from src.strategies.baseline import baseline_signal
from src.strategies.momentum import momentum_signal
from src.evaluation.live_production_decision import (
    Direction,
    ProductionDecision,
    ProductionSignal,
    ProductionRiskLevels,
    ProductionAuthorizationReceipt,
    ProductionIntelligencePublication,
    calculate_production_risk_levels,
    evaluate_production_decision,
)
from src.evaluation.live_runtime import evaluate_authorized_live_runtime
from src.evaluation.live_decision_lifecycle import create_canonical_live_decision
from src.evaluation.live_runtime_context import create_authorized_runtime_context
from src.evaluation.live_market_evaluation import create_live_market_evaluation
from src.evaluation.live_decision_store import append_live_decision_to_store, load_live_decision_history
from src.evaluation.live_publication_store import publish_canonical_live_decision, load_publication_history
from src.evaluation.mtf_intelligence import CanonicalTimeframe


def make_market_data_for_trend(trend: str = "UP", rows: int = 80) -> pd.DataFrame:
    now_dt = pd.Timestamp.now(tz="UTC")
    dates = [now_dt - pd.Timedelta(minutes=5 * (rows - i)) for i in range(rows)]

    if trend == "UP":
        closes = [100.0 + idx * 0.5 for idx in range(rows)]
    elif trend == "DOWN":
        closes = [200.0 - idx * 0.5 for idx in range(rows)]
    else:
        closes = [100.0] * rows

    return pd.DataFrame({
        "timestamp": dates,
        "open": closes,
        "high": [c + 1.0 for c in closes],
        "low": [c - 1.0 for c in closes],
        "close": closes,
    })


@pytest.mark.parametrize("direction_type", ["BUY", "SELL", "NO_TRADE"])
def test_end_to_end_direction_preservation(direction_type, tmp_path):
    """Verify that direction is perfectly preserved from strategy through decision, risk, runtime, store persistence, and publication."""
    from tests.test_production_decision_integrity import make_promoted_evidence
    from src.evaluation.research_qualification import qualify_research_evidence
    from src.evaluation.research_robustness import assess_research_robustness
    from src.evaluation.live_production_decision import PromotedCandidateArtifact, authorize_production_runtime

    if direction_type == "BUY":
        data = make_market_data_for_trend("UP")
        expected_dir = Direction.BUY
    elif direction_type == "SELL":
        data = make_market_data_for_trend("DOWN")
        expected_dir = Direction.SELL
    else:
        data = make_market_data_for_trend("FLAT")
        expected_dir = Direction.NO_TRADE

    ev = make_promoted_evidence(timeframe="5m", symbol="XAUUSD", strategy_name="momentum")
    rob = assess_research_robustness(ev)
    gov = qualify_research_evidence(ev, robustness_assessment=rob)
    cand = PromotedCandidateArtifact(
        candidate_id="cand_momentum_5m",
        strategy_name="momentum",
        strategy_version="1.0",
        evidence=ev,
        symbol="XAUUSD",
        timeframe="5m",
        operational_stability_score=0.75,
        parameters=dict(ev.spec.parameters),
        governance_decision=gov,
        governance_decision_fingerprint=gov.decision_fingerprint,
    )

    auth = authorize_production_runtime(cand, symbol="XAUUSD", timeframe="5m")
    receipt = ProductionAuthorizationReceipt.from_authorization(auth)
    ctx = create_authorized_runtime_context(
        candidate=cand,
        authorization=auth,
        authorization_receipt=receipt,
    )
    eval_mkt = create_live_market_evaluation(data=data, context=ctx)

    store_file = tmp_path / "decision_store.json"
    pub_file = tmp_path / "publication_store.json"

    result = evaluate_authorized_live_runtime(
        data=data,
        evaluation=eval_mkt,
        context=ctx,
        stable_strategy="momentum",
        store_path=store_file,
        persist=True,
        publish=False,
    )

    cld = result.canonical_decision
    assert cld.decision.direction == expected_dir
    assert cld.signal.direction == expected_dir
    assert cld.risk_levels.direction == expected_dir
    assert result.decision["decision"] == expected_dir.value

    # Read back from decision store file on disk and verify persisted direction
    history = load_live_decision_history(store_file)
    assert len(history) == 1
    persisted_rec = history[0]
    assert persisted_rec["signal_label"] == expected_dir.value
    if expected_dir == Direction.BUY:
        assert persisted_rec["signal"] == 1
        assert persisted_rec["trend"] == "UP"
    elif expected_dir == Direction.SELL:
        assert persisted_rec["signal"] == -1
        assert persisted_rec["trend"] == "DOWN"
    else:
        assert persisted_rec["signal"] == 0
        assert persisted_rec["trend"] == "NEUTRAL"

    # Publication contract payload conversion and boundary test
    pub = ProductionIntelligencePublication.from_artifacts(
        decision=cld.decision,
        signal=cld.signal,
        risk=cld.risk_levels,
        candidate=cand,
        authorization=auth,
    )
    payload = pub.to_contract_v1_payload()

    assert payload["signal"]["decision"] == expected_dir.value
    if expected_dir == Direction.BUY:
        assert payload["signal"]["trend"] == "BULLISH"
        assert payload["trade_setup"]["stop_loss"] < payload["trade_setup"]["entry_price"]
        assert payload["trade_setup"]["take_profit"] > payload["trade_setup"]["entry_price"]
    elif expected_dir == Direction.SELL:
        assert payload["signal"]["trend"] == "BEARISH"
        assert payload["trade_setup"]["stop_loss"] > payload["trade_setup"]["entry_price"]
        assert payload["trade_setup"]["take_profit"] < payload["trade_setup"]["entry_price"]
    else:
        assert payload["signal"]["trend"] == "NEUTRAL"
        assert payload["trade_setup"]["stop_loss"] is None
        assert payload["trade_setup"]["take_profit"] is None


@pytest.mark.parametrize("tf", ["1m", "5m", "15m", "30m", "1H", "4H", "1D"])
def test_seven_timeframe_matrix_isolation(tf):
    """Verify that every canonical timeframe resolves independently without cross-timeframe fallback."""
    from src.evaluation.research_store import resolve_promoted_candidate, DEFAULT_RESEARCH_DIR

    cand = resolve_promoted_candidate(symbol="XAUUSD", timeframe=tf, base_dir=DEFAULT_RESEARCH_DIR)

    # 1D is the only timeframe with a verified promoted candidate in research store
    if tf == "1D":
        assert cand is not None
        assert cand.candidate_id == "cand_moving_average_1d_verified_20261020"
        assert cand.timeframe == "1D"
    else:
        assert cand is None
