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
    evaluation = create_live_market_evaluation(df, context, reference_now=df_ts + datetime.timedelta(seconds=10))

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


def test_stale_market_evaluation_creation(tmp_path):
    """B. Stale evaluation test."""
    context, ref_now = build_test_context(tmp_path)
    df = make_buy_market_data()
    df_ts = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime()
    stale_ref_now = df_ts + datetime.timedelta(seconds=1000)

    evaluation = create_live_market_evaluation(df, context, reference_now=stale_ref_now, max_age_seconds=300.0)

    assert evaluation.fresh is False
    assert evaluation.freshness_status is False
    assert evaluation.freshness_reason == "stale_market_data"
    assert evaluation.age_seconds == 1000.0
    assert evaluation.authorized_runtime_context_fingerprint == context.context_fingerprint


def test_fingerprint_determinism_and_sensitivity(tmp_path):
    """I. Fingerprint determinism and material input sensitivity."""
    context, ref_now = build_test_context(tmp_path)
    df = make_buy_market_data()
    df_ts = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime()

    eval1 = create_live_market_evaluation(df, context, reference_now=df_ts)
    eval2 = create_live_market_evaluation(df, context, reference_now=df_ts)

    # Same semantic inputs -> same fingerprint
    assert eval1.evaluation_fingerprint == eval2.evaluation_fingerprint

    # Material change -> different fingerprint
    eval_stale = create_live_market_evaluation(df, context, reference_now=df_ts + datetime.timedelta(seconds=1000))
    assert eval_stale.evaluation_fingerprint != eval1.evaluation_fingerprint


def test_context_evaluation_mismatch_fails_closed(tmp_path):
    """H. Context/evaluation mismatch checks fail closed."""
    context, ref_now = build_test_context(tmp_path)
    df = make_dummy_df()

    evaluation = create_live_market_evaluation(df, context, reference_now=ref_now)

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
