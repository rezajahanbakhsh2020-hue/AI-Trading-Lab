import pandas as pd
import pytest

from src.evaluation.live_production_decision import (
    AuthorizedLiveDecision,
    Direction,
    ProductionAuthorizationReceipt,
    ProductionDecision,
    ProductionRiskLevels,
    PromotedCandidateArtifact,
    authorize_production_runtime,
    calculate_production_risk_levels,
    evaluate_production_decision,
)
from src.evaluation.live_trade_display import (
    DEFAULT_TP1_MULTIPLIER,
    DEFAULT_TP2_MULTIPLIER,
    DEFAULT_TP3_MULTIPLIER,
    build_live_trade_display,
)
from tests.test_production_decision_integrity import make_promoted_evidence


def _rising_data(rows: int = 80) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    timestamps = [now - pd.Timedelta(minutes=5 * (rows - 1 - i)) for i in range(rows)]

    close = pd.Series(
        [2000.0 + index for index in range(rows)],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        }
    )


def _make_authorized_decision(
    data: pd.DataFrame,
    strategy_name: str = "momentum",
    stability_score: float = 0.80,
) -> AuthorizedLiveDecision:
    ev = make_promoted_evidence()
    if strategy_name != ev.spec.strategy_name:
        object.__setattr__(ev.spec, "strategy_name", strategy_name)

    cand = PromotedCandidateArtifact(
        candidate_id="cand_test_display",
        strategy_name=strategy_name,
        strategy_version="1.0",
        evidence=ev,
        symbol="XAUUSD",
        timeframe="5m",
        governance_decision_fingerprint="gov_fp_display_test",
    )
    auth = authorize_production_runtime(cand, symbol="XAUUSD", timeframe="5m")
    receipt = ProductionAuthorizationReceipt.from_authorization(auth)

    dec_obj = evaluate_production_decision(cand, data, authorization=receipt)

    if stability_score < 0.50 or strategy_name.lower() != "momentum":
        final_dir = Direction.NO_TRADE
        entry_p = None
        inval = None
        reason = "stability_score_below_threshold" if stability_score < 0.50 else "unsupported_live_strategy"
    else:
        final_dir = dec_obj.direction
        entry_p = dec_obj.entry_price
        inval = dec_obj.invalidation_condition
        reason = dec_obj.reason

    final_dec = ProductionDecision(
        candidate_id=cand.candidate_id,
        evidence_id=cand.evidence.evidence_id,
        experiment_fingerprint=cand.evidence.experiment_fingerprint,
        symbol=cand.symbol,
        timeframe=cand.timeframe,
        decision_timestamp=dec_obj.decision_timestamp,
        market_timestamp=dec_obj.market_timestamp,
        direction=final_dir,
        reason=reason,
        entry_price=entry_p,
        invalidation_condition=inval,
        confidence=stability_score,
        parameters=cand.parameters,
    )

    risk_obj = calculate_production_risk_levels(final_dec, cand)

    return AuthorizedLiveDecision.create(
        authorization=receipt,
        candidate=cand,
        decision=final_dec,
        risk_levels=risk_obj,
    )


def test_buy_trade_display_contains_entry_sl_and_three_targets():
    auth_dec = _make_authorized_decision(_rising_data(), strategy_name="momentum", stability_score=0.80)
    result = build_live_trade_display(auth_dec)

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
    auth_dec = _make_authorized_decision(_rising_data(), strategy_name="momentum", stability_score=0.80)
    result = build_live_trade_display(auth_dec)

    assert result["tp1_multiplier"] == DEFAULT_TP1_MULTIPLIER
    assert result["tp2_multiplier"] == DEFAULT_TP2_MULTIPLIER
    assert result["tp3_multiplier"] == DEFAULT_TP3_MULTIPLIER

    assert result["risk_reward_tp1"] == pytest.approx(1.0)
    assert result["risk_reward_tp2"] == pytest.approx(2.0)
    assert result["risk_reward_tp3"] == pytest.approx(3.0)


def test_custom_target_multipliers_are_applied():
    auth_dec = _make_authorized_decision(_rising_data(), strategy_name="momentum", stability_score=0.80)
    result = build_live_trade_display(
        auth_dec,
        tp1_multiplier=0.5,
        tp2_multiplier=1.5,
        tp3_multiplier=2.5,
    )

    assert result["risk_reward_tp1"] == pytest.approx(0.5)
    assert result["risk_reward_tp2"] == pytest.approx(1.5)
    assert result["risk_reward_tp3"] == pytest.approx(2.5)


def test_no_trade_has_no_trade_levels():
    auth_dec = _make_authorized_decision(_rising_data(), strategy_name="momentum", stability_score=0.20)
    result = build_live_trade_display(auth_dec)

    assert result["decision"] == "NO TRADE"
    assert result["entry_price"] is None
    assert result["stop_loss"] is None
    assert result["tp1"] is None
    assert result["tp2"] is None
    assert result["tp3"] is None
    assert result["risk_distance"] is None


def test_unsupported_strategy_has_no_trade_levels():
    auth_dec = _make_authorized_decision(_rising_data(), strategy_name="moving_average", stability_score=0.80)
    result = build_live_trade_display(auth_dec)

    assert result["decision"] == "NO TRADE"
    assert result["strategy_supported"] is False
    assert result["tp1"] is None
    assert result["tp2"] is None
    assert result["tp3"] is None


def test_tp_multipliers_must_be_strictly_increasing():
    auth_dec = _make_authorized_decision(_rising_data(), strategy_name="momentum", stability_score=0.80)
    with pytest.raises(ValueError, match="TP1 < TP2 < TP3"):
        build_live_trade_display(
            auth_dec,
            tp1_multiplier=2.0,
            tp2_multiplier=1.0,
            tp3_multiplier=3.0,
        )


def test_tp_multipliers_must_be_positive():
    auth_dec = _make_authorized_decision(_rising_data(), strategy_name="momentum", stability_score=0.80)
    with pytest.raises(
        ValueError,
        match="tp1_multiplier must be greater than 0",
    ):
        build_live_trade_display(
            auth_dec,
            tp1_multiplier=0.0,
        )


def test_no_sell_decision_is_invented():
    auth_dec = _make_authorized_decision(_rising_data(), strategy_name="momentum", stability_score=0.80)
    result = build_live_trade_display(auth_dec)

    assert result["decision"] in {"BUY", "NO TRADE"}
    assert result["decision"] != "SELL"
