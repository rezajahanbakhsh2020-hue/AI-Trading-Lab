"""Tests for LiveMarketEvaluation artifact, factory, and context lineage validation."""

import datetime
import pandas as pd
import pytest

from src.evaluation.live_market_evaluation import (
    LiveMarketEvaluation,
    MarketEvaluationValidationError,
    create_live_market_evaluation,
    validate_market_evaluation_context_lineage,
)
from src.evaluation.live_production_decision import (
    ProductionAuthorizationReceipt,
    authorize_production_runtime,
)
from src.evaluation.live_runtime_context import (
    create_authorized_runtime_context,
)
from tests.test_live_execution_runtime import (
    make_buy_market_data,
    make_dummy_df,
    persist_momentum_candidate,
)
from src.evaluation.research_store import resolve_promoted_candidate


def build_test_context(base_dir, candidate_id="cand_eval_test", symbol="XAUUSD", timeframe="5m"):
    persist_momentum_candidate(base_dir, candidate_id=candidate_id, symbol=symbol)
    candidate = resolve_promoted_candidate(
        candidate_id=candidate_id,
        symbol=symbol,
        timeframe=timeframe,
        base_dir=base_dir,
    )
    ref_now = datetime.datetime(2025, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    auth = authorize_production_runtime(candidate, symbol=symbol, timeframe=timeframe, now=ref_now)
    receipt = ProductionAuthorizationReceipt.from_authorization(auth)
    context = create_authorized_runtime_context(candidate, auth, receipt)
    return context, ref_now


def test_fresh_market_evaluation_creation(tmp_path):
    """A. Fresh evaluation test."""
    context, ref_now = build_test_context(tmp_path)
    df = make_buy_market_data()
    # Align latest timestamp close to ref_now
    df_ts = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime()
    evaluation = create_live_market_evaluation(df, context, reference_now=df_ts + datetime.timedelta(minutes=5, seconds=10))

    assert evaluation.fresh is True
    assert evaluation.freshness_status is True
    assert evaluation.freshness_reason == "fresh"
    assert evaluation.symbol == "XAUUSD"
    assert evaluation.timeframe == "5m"
    assert evaluation.candidate_id == context.candidate_id
    assert evaluation.authorized_runtime_context_fingerprint == context.context_fingerprint
    assert evaluation.promoted_artifact_fingerprint == context.promoted_artifact_fingerprint
    assert evaluation.governance_decision_fingerprint == context.governance_decision_fingerprint
    assert evaluation.authorization_fingerprint == context.authorization_fingerprint

    # Test immutability
    with pytest.raises(AttributeError):
        evaluation.freshness_status = False


def test_closed_market_evaluation_remains_valid_after_300_seconds(tmp_path):
    context, _ = build_test_context(tmp_path)
    df = make_buy_market_data()
    df_ts = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime()

    evaluation = create_live_market_evaluation(
        df, context, reference_now=df_ts + datetime.timedelta(seconds=1000), max_age_seconds=300.0
    )

    assert evaluation.fresh is True
    assert evaluation.freshness_status is True
    assert evaluation.freshness_reason == "fresh"
    assert evaluation.age_seconds == 1000.0
    assert evaluation.authorized_runtime_context_fingerprint == context.context_fingerprint


def test_per_field_fingerprint_sensitivity(tmp_path):
    """Prove that changing each material input field independently changes evaluation_fingerprint."""
    context, ref_now = build_test_context(tmp_path)
    df = make_buy_market_data()
    df_ts = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime()

    base_eval = create_live_market_evaluation(df, context, reference_now=df_ts + datetime.timedelta(minutes=5))
    base_fp = base_eval.evaluation_fingerprint

    base_kwargs = {
        "symbol": base_eval.symbol,
        "timeframe": base_eval.timeframe,
        "candle_timestamp_utc": base_eval.candle_timestamp_utc,
        "freshness_status": base_eval.freshness_status,
        "freshness_reason": base_eval.freshness_reason,
        "age_seconds": base_eval.age_seconds,
        "reference_timestamp_utc": base_eval.reference_timestamp_utc,
        "authorized_runtime_context_fingerprint": base_eval.authorized_runtime_context_fingerprint,
        "promoted_artifact_fingerprint": base_eval.promoted_artifact_fingerprint,
        "governance_decision_fingerprint": base_eval.governance_decision_fingerprint,
        "campaign_selection_decision_fingerprint": base_eval.campaign_selection_decision_fingerprint,
        "authorization_fingerprint": base_eval.authorization_fingerprint,
        "authorization_policy_version": base_eval.authorization_policy_version,
        "candidate_id": base_eval.candidate_id,
        "strategy_id": base_eval.strategy_id,
        "strategy_version": base_eval.strategy_version,
    }

    # 1. symbol
    k1 = dict(base_kwargs, symbol="EURUSD")
    assert LiveMarketEvaluation(**k1).evaluation_fingerprint != base_fp

    # 2. timeframe
    k2 = dict(base_kwargs, timeframe="1h")
    assert LiveMarketEvaluation(**k2).evaluation_fingerprint != base_fp

    # 3. candle_timestamp_utc
    k3 = dict(base_kwargs, candle_timestamp_utc="2025-01-01T11:59:00+00:00")
    assert LiveMarketEvaluation(**k3).evaluation_fingerprint != base_fp

    # 4. freshness_status & reason
    k4 = dict(base_kwargs, freshness_status=False, freshness_reason="stale_market_data", candle_timestamp_utc=None)
    assert LiveMarketEvaluation(**k4).evaluation_fingerprint != base_fp

    # 5. freshness_reason
    k5 = dict(base_kwargs, freshness_reason="custom_reason")
    assert LiveMarketEvaluation(**k5).evaluation_fingerprint != base_fp

    # 6. age_seconds
    k6 = dict(base_kwargs, age_seconds=99.0)
    assert LiveMarketEvaluation(**k6).evaluation_fingerprint != base_fp

    # 7. reference_timestamp_utc
    k7 = dict(base_kwargs, reference_timestamp_utc="2025-01-01T12:05:00+00:00")
    assert LiveMarketEvaluation(**k7).evaluation_fingerprint != base_fp

    # 8. authorized_runtime_context_fingerprint
    k8 = dict(base_kwargs, authorized_runtime_context_fingerprint="ctx_fp_diff")
    assert LiveMarketEvaluation(**k8).evaluation_fingerprint != base_fp

    # 9. promoted_artifact_fingerprint
    k9 = dict(base_kwargs, promoted_artifact_fingerprint="art_fp_diff")
    assert LiveMarketEvaluation(**k9).evaluation_fingerprint != base_fp

    # 10. governance_decision_fingerprint
    k10 = dict(base_kwargs, governance_decision_fingerprint="gov_fp_diff")
    assert LiveMarketEvaluation(**k10).evaluation_fingerprint != base_fp

    # 11. campaign_selection_decision_fingerprint
    k11 = dict(base_kwargs, campaign_selection_decision_fingerprint="camp_fp_diff")
    assert LiveMarketEvaluation(**k11).evaluation_fingerprint != base_fp

    # 12. authorization_fingerprint
    k12 = dict(base_kwargs, authorization_fingerprint="auth_fp_diff")
    assert LiveMarketEvaluation(**k12).evaluation_fingerprint != base_fp

    # 13. authorization_policy_version
    k13 = dict(base_kwargs, authorization_policy_version="v2.0")
    assert LiveMarketEvaluation(**k13).evaluation_fingerprint != base_fp

    # 14. candidate_id
    k14 = dict(base_kwargs, candidate_id="cand_diff")
    assert LiveMarketEvaluation(**k14).evaluation_fingerprint != base_fp

    # 15. strategy_id
    k15 = dict(base_kwargs, strategy_id="strat_diff")
    assert LiveMarketEvaluation(**k15).evaluation_fingerprint != base_fp

    # 16. strategy_version
    k16 = dict(base_kwargs, strategy_version="2.0")
    assert LiveMarketEvaluation(**k16).evaluation_fingerprint != base_fp


def test_context_evaluation_mismatch_fails_closed(tmp_path):
    """H. Context/evaluation mismatch checks fail closed."""
    context, ref_now = build_test_context(tmp_path)
    df = make_buy_market_data()
    df_ts = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime()

    evaluation = create_live_market_evaluation(df, context, reference_now=df_ts + datetime.timedelta(minutes=5))

    # 1. Symbol mismatch
    bad_eval = LiveMarketEvaluation(
        symbol="EURUSD",
        timeframe=evaluation.timeframe,
        candle_timestamp_utc=evaluation.candle_timestamp_utc,
        freshness_status=evaluation.freshness_status,
        freshness_reason=evaluation.freshness_reason,
        age_seconds=evaluation.age_seconds,
        reference_timestamp_utc=evaluation.reference_timestamp_utc,
        authorized_runtime_context_fingerprint=evaluation.authorized_runtime_context_fingerprint,
        promoted_artifact_fingerprint=evaluation.promoted_artifact_fingerprint,
        governance_decision_fingerprint=evaluation.governance_decision_fingerprint,
        campaign_selection_decision_fingerprint=evaluation.campaign_selection_decision_fingerprint,
        authorization_fingerprint=evaluation.authorization_fingerprint,
        authorization_policy_version=evaluation.authorization_policy_version,
        candidate_id=evaluation.candidate_id,
        strategy_id=evaluation.strategy_id,
        strategy_version=evaluation.strategy_version,
    )
    with pytest.raises(MarketEvaluationValidationError, match="symbol"):
        validate_market_evaluation_context_lineage(bad_eval, context)

    # 2. Candidate ID mismatch
    bad_eval = LiveMarketEvaluation(
        symbol=evaluation.symbol,
        timeframe=evaluation.timeframe,
        candle_timestamp_utc=evaluation.candle_timestamp_utc,
        freshness_status=evaluation.freshness_status,
        freshness_reason=evaluation.freshness_reason,
        age_seconds=evaluation.age_seconds,
        reference_timestamp_utc=evaluation.reference_timestamp_utc,
        authorized_runtime_context_fingerprint=evaluation.authorized_runtime_context_fingerprint,
        promoted_artifact_fingerprint=evaluation.promoted_artifact_fingerprint,
        governance_decision_fingerprint=evaluation.governance_decision_fingerprint,
        campaign_selection_decision_fingerprint=evaluation.campaign_selection_decision_fingerprint,
        authorization_fingerprint=evaluation.authorization_fingerprint,
        authorization_policy_version=evaluation.authorization_policy_version,
        candidate_id="other_candidate",
        strategy_id=evaluation.strategy_id,
        strategy_version=evaluation.strategy_version,
    )
    with pytest.raises(MarketEvaluationValidationError, match="candidate_id"):
        validate_market_evaluation_context_lineage(bad_eval, context)

    # 3. Context fingerprint mismatch
    bad_eval = LiveMarketEvaluation(
        symbol=evaluation.symbol,
        timeframe=evaluation.timeframe,
        candle_timestamp_utc=evaluation.candle_timestamp_utc,
        freshness_status=evaluation.freshness_status,
        freshness_reason=evaluation.freshness_reason,
        age_seconds=evaluation.age_seconds,
        reference_timestamp_utc=evaluation.reference_timestamp_utc,
        authorized_runtime_context_fingerprint="0000000000000000000000000000000000000000000000000000000000000000",
        promoted_artifact_fingerprint=evaluation.promoted_artifact_fingerprint,
        governance_decision_fingerprint=evaluation.governance_decision_fingerprint,
        campaign_selection_decision_fingerprint=evaluation.campaign_selection_decision_fingerprint,
        authorization_fingerprint=evaluation.authorization_fingerprint,
        authorization_policy_version=evaluation.authorization_policy_version,
        candidate_id=evaluation.candidate_id,
        strategy_id=evaluation.strategy_id,
        strategy_version=evaluation.strategy_version,
    )
    with pytest.raises(MarketEvaluationValidationError, match="authorized_runtime_context_fingerprint"):
        validate_market_evaluation_context_lineage(bad_eval, context)


def test_strict_utc_and_age_validation(tmp_path):
    """Part 3 & 4: Strict UTC parsing, naive timestamp rejection, non-zero offset rejection, and age bounds."""
    context, ref_now = build_test_context(tmp_path)

    base_kwargs = {
        "symbol": "XAUUSD",
        "timeframe": "5m",
        "candle_timestamp_utc": "2025-01-01T12:00:00+00:00",
        "freshness_status": True,
        "freshness_reason": "fresh",
        "age_seconds": 10.0,
        "reference_timestamp_utc": "2025-01-01T12:00:10+00:00",
        "authorized_runtime_context_fingerprint": context.context_fingerprint,
        "promoted_artifact_fingerprint": context.promoted_artifact_fingerprint,
        "governance_decision_fingerprint": context.governance_decision_fingerprint,
        "campaign_selection_decision_fingerprint": context.campaign_selection_decision_fingerprint,
        "authorization_fingerprint": context.authorization_fingerprint,
        "authorization_policy_version": context.authorization_policy_version,
        "candidate_id": context.candidate_id,
        "strategy_id": context.strategy_id,
        "strategy_version": context.strategy_version,
    }

    # 1. Valid UTC ISO-8601 with Z
    eval_z = LiveMarketEvaluation(**dict(base_kwargs, reference_timestamp_utc="2025-01-01T12:00:10Z", candle_timestamp_utc="2025-01-01T12:00:00Z"))
    assert eval_z.reference_timestamp_utc == "2025-01-01T12:00:10Z"

    # 2. Naive reference timestamp rejected
    with pytest.raises(MarketEvaluationValidationError, match="zero offset"):
        LiveMarketEvaluation(**dict(base_kwargs, reference_timestamp_utc="2025-01-01T12:00:10"))

    # 3. Non-zero offset rejected
    with pytest.raises(MarketEvaluationValidationError, match="zero offset"):
        LiveMarketEvaluation(**dict(base_kwargs, reference_timestamp_utc="2025-01-01T13:00:10+01:00"))

    # 4. Naive candle timestamp rejected
    with pytest.raises(MarketEvaluationValidationError, match="zero offset"):
        LiveMarketEvaluation(**dict(base_kwargs, candle_timestamp_utc="2025-01-01T12:00:00"))

    # 5. Non-zero candle offset rejected
    with pytest.raises(MarketEvaluationValidationError, match="zero offset"):
        LiveMarketEvaluation(**dict(base_kwargs, candle_timestamp_utc="2025-01-01T07:00:00-05:00"))

    # 6. Negative age_seconds=-0.001 rejected
    with pytest.raises(MarketEvaluationValidationError, match="non-negative"):
        LiveMarketEvaluation(**dict(base_kwargs, age_seconds=-0.001))

    # 7. Negative age_seconds=-100.0 rejected
    with pytest.raises(MarketEvaluationValidationError, match="non-negative"):
        LiveMarketEvaluation(**dict(base_kwargs, age_seconds=-100.0))

    # 8. NaN age rejected
    with pytest.raises(MarketEvaluationValidationError, match="non-negative"):
        LiveMarketEvaluation(**dict(base_kwargs, age_seconds=float("nan")))

    # 7. +inf age rejected
    with pytest.raises(MarketEvaluationValidationError, match="finite float"):
        LiveMarketEvaluation(**dict(base_kwargs, age_seconds=float("inf")))

    # 8. -inf age rejected
    with pytest.raises(MarketEvaluationValidationError, match="finite float"):
        LiveMarketEvaluation(**dict(base_kwargs, age_seconds=float("-inf")))


def test_missing_candle_timestamp_non_fabrication(tmp_path):
    """Part 2 & 10: Missing candle timestamp remains None without substituting reference timestamp."""
    context, ref_now = build_test_context(tmp_path)
    df = make_dummy_df()
    df_no_ts = df.drop(columns=["timestamp", "openTime"])

    evaluation = create_live_market_evaluation(df_no_ts, context, reference_now=ref_now)

    assert evaluation.fresh is False
    assert evaluation.candle_timestamp_utc is None
    assert evaluation.candle_timestamp_utc != evaluation.reference_timestamp_utc
