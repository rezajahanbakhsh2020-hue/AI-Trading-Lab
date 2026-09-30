"""Canonical Live Market Evaluation artifact representing authoritative market-data evaluation for Project 1 live execution."""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.evaluation.live_production_decision import ProductionRuntimeAuthorizationError
from src.evaluation.live_runtime_context import AuthorizedProductionRuntimeContext


class MarketEvaluationValidationError(ProductionRuntimeAuthorizationError):
    """Raised when LiveMarketEvaluation validation fails or lineage mismatches context."""


@dataclass(frozen=True)
class LiveMarketEvaluation:
    """Canonical, immutable market-data evaluation artifact for one execution cycle.

    Binds market data freshness, candle timestamps, and reference timestamps
    strictly to the AuthorizedProductionRuntimeContext lineage.
    """

    symbol: str
    timeframe: str
    candle_timestamp_utc: str | None
    freshness_status: bool
    freshness_reason: str
    age_seconds: float | None
    reference_timestamp_utc: str
    authorized_runtime_context_fingerprint: str
    promoted_artifact_fingerprint: str
    governance_decision_fingerprint: str
    campaign_selection_decision_fingerprint: str | None
    authorization_fingerprint: str
    authorization_policy_version: str
    candidate_id: str
    strategy_id: str
    strategy_version: str
    evaluation_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        req_symbol = str(self.symbol).strip().upper() if self.symbol else ""
        req_timeframe = str(self.timeframe).strip() if self.timeframe else ""
        req_reason = str(self.freshness_reason).strip() if self.freshness_reason else ""
        req_ref_ts = str(self.reference_timestamp_utc).strip() if self.reference_timestamp_utc else ""
        req_ctx_fp = (
            str(self.authorized_runtime_context_fingerprint).strip()
            if self.authorized_runtime_context_fingerprint
            else ""
        )
        req_art_fp = (
            str(self.promoted_artifact_fingerprint).strip()
            if self.promoted_artifact_fingerprint
            else ""
        )
        req_gov_fp = (
            str(self.governance_decision_fingerprint).strip()
            if self.governance_decision_fingerprint
            else ""
        )
        req_auth_fp = (
            str(self.authorization_fingerprint).strip()
            if self.authorization_fingerprint
            else ""
        )
        req_auth_pol_ver = (
            str(self.authorization_policy_version).strip()
            if self.authorization_policy_version
            else ""
        )
        req_cand_id = str(self.candidate_id).strip() if self.candidate_id else ""
        req_strat_id = str(self.strategy_id).strip() if self.strategy_id else ""
        req_strat_ver = str(self.strategy_version).strip() if self.strategy_version else ""

        if not req_symbol:
            raise MarketEvaluationValidationError("symbol must be a non-empty string.")
        if not req_timeframe:
            raise MarketEvaluationValidationError("timeframe must be a non-empty string.")
        if not req_reason:
            raise MarketEvaluationValidationError("freshness_reason must be a non-empty string.")
        if not req_ref_ts:
            raise MarketEvaluationValidationError("reference_timestamp_utc must be a non-empty string.")
        if not req_ctx_fp:
            raise MarketEvaluationValidationError("authorized_runtime_context_fingerprint must be a non-empty string.")
        if not req_art_fp:
            raise MarketEvaluationValidationError("promoted_artifact_fingerprint must be a non-empty string.")
        if not req_gov_fp:
            raise MarketEvaluationValidationError("governance_decision_fingerprint must be a non-empty string.")
        if not req_auth_fp:
            raise MarketEvaluationValidationError("authorization_fingerprint must be a non-empty string.")
        if not req_auth_pol_ver:
            raise MarketEvaluationValidationError("authorization_policy_version must be a non-empty string.")
        if not req_cand_id:
            raise MarketEvaluationValidationError("candidate_id must be a non-empty string.")
        if not req_strat_id:
            raise MarketEvaluationValidationError("strategy_id must be a non-empty string.")
        if not req_strat_ver:
            raise MarketEvaluationValidationError("strategy_version must be a non-empty string.")

        object.__setattr__(self, "symbol", req_symbol)
        object.__setattr__(self, "timeframe", req_timeframe)
        object.__setattr__(self, "freshness_reason", req_reason)
        object.__setattr__(self, "reference_timestamp_utc", req_ref_ts)
        object.__setattr__(self, "authorized_runtime_context_fingerprint", req_ctx_fp)
        object.__setattr__(self, "promoted_artifact_fingerprint", req_art_fp)
        object.__setattr__(self, "governance_decision_fingerprint", req_gov_fp)
        object.__setattr__(self, "authorization_fingerprint", req_auth_fp)
        object.__setattr__(self, "authorization_policy_version", req_auth_pol_ver)
        object.__setattr__(self, "candidate_id", req_cand_id)
        object.__setattr__(self, "strategy_id", req_strat_id)
        object.__setattr__(self, "strategy_version", req_strat_ver)

        if self.candle_timestamp_utc is not None:
            c_ts = str(self.candle_timestamp_utc).strip()
            if not c_ts:
                raise MarketEvaluationValidationError("candle_timestamp_utc cannot be whitespace.")
            object.__setattr__(self, "candle_timestamp_utc", c_ts)

        if self.campaign_selection_decision_fingerprint is not None:
            csdf = str(self.campaign_selection_decision_fingerprint).strip()
            if not csdf:
                raise MarketEvaluationValidationError("campaign_selection_decision_fingerprint cannot be empty if provided.")
            object.__setattr__(self, "campaign_selection_decision_fingerprint", csdf)

        if self.age_seconds is not None:
            object.__setattr__(self, "age_seconds", float(self.age_seconds))

        # Calculate deterministic evaluation fingerprint
        payload = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "candle_timestamp_utc": self.candle_timestamp_utc,
            "freshness_status": bool(self.freshness_status),
            "freshness_reason": self.freshness_reason,
            "age_seconds": self.age_seconds,
            "reference_timestamp_utc": self.reference_timestamp_utc,
            "authorized_runtime_context_fingerprint": self.authorized_runtime_context_fingerprint,
            "promoted_artifact_fingerprint": self.promoted_artifact_fingerprint,
            "governance_decision_fingerprint": self.governance_decision_fingerprint,
            "campaign_selection_decision_fingerprint": self.campaign_selection_decision_fingerprint,
            "authorization_fingerprint": self.authorization_fingerprint,
            "authorization_policy_version": self.authorization_policy_version,
            "candidate_id": self.candidate_id,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        object.__setattr__(
            self,
            "evaluation_fingerprint",
            hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        )

    @property
    def fresh(self) -> bool:
        """Convenience property matching freshness_status."""
        return self.freshness_status

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "candle_timestamp_utc": self.candle_timestamp_utc,
            "freshness_status": self.freshness_status,
            "freshness_reason": self.freshness_reason,
            "age_seconds": self.age_seconds,
            "reference_timestamp_utc": self.reference_timestamp_utc,
            "authorized_runtime_context_fingerprint": self.authorized_runtime_context_fingerprint,
            "promoted_artifact_fingerprint": self.promoted_artifact_fingerprint,
            "governance_decision_fingerprint": self.governance_decision_fingerprint,
            "campaign_selection_decision_fingerprint": self.campaign_selection_decision_fingerprint,
            "authorization_fingerprint": self.authorization_fingerprint,
            "authorization_policy_version": self.authorization_policy_version,
            "candidate_id": self.candidate_id,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "evaluation_fingerprint": self.evaluation_fingerprint,
        }


def validate_market_evaluation_context_lineage(
    evaluation: LiveMarketEvaluation,
    context: AuthorizedProductionRuntimeContext,
) -> None:
    """Validate that a LiveMarketEvaluation matches the supplied AuthorizedProductionRuntimeContext fail-closed."""
    if not isinstance(evaluation, LiveMarketEvaluation):
        raise MarketEvaluationValidationError(
            f"evaluation must be a LiveMarketEvaluation, got {type(evaluation).__name__}"
        )
    if not isinstance(context, AuthorizedProductionRuntimeContext):
        raise MarketEvaluationValidationError(
            f"context must be an AuthorizedProductionRuntimeContext, got {type(context).__name__}"
        )

    if evaluation.symbol.upper() != context.symbol.upper():
        raise MarketEvaluationValidationError(
            f"Evaluation symbol '{evaluation.symbol}' does not match context symbol '{context.symbol}'."
        )
    if evaluation.timeframe != context.timeframe:
        raise MarketEvaluationValidationError(
            f"Evaluation timeframe '{evaluation.timeframe}' does not match context timeframe '{context.timeframe}'."
        )
    if evaluation.candidate_id != context.candidate_id:
        raise MarketEvaluationValidationError(
            f"Evaluation candidate_id '{evaluation.candidate_id}' does not match context candidate_id '{context.candidate_id}'."
        )
    if evaluation.strategy_id != context.strategy_id:
        raise MarketEvaluationValidationError(
            f"Evaluation strategy_id '{evaluation.strategy_id}' does not match context strategy_id '{context.strategy_id}'."
        )
    if evaluation.strategy_version != context.strategy_version:
        raise MarketEvaluationValidationError(
            f"Evaluation strategy_version '{evaluation.strategy_version}' does not match context strategy_version '{context.strategy_version}'."
        )
    if evaluation.promoted_artifact_fingerprint != context.promoted_artifact_fingerprint:
        raise MarketEvaluationValidationError(
            f"Evaluation promoted_artifact_fingerprint '{evaluation.promoted_artifact_fingerprint}' does not match context artifact fingerprint '{context.promoted_artifact_fingerprint}'."
        )
    if evaluation.governance_decision_fingerprint != context.governance_decision_fingerprint:
        raise MarketEvaluationValidationError(
            f"Evaluation governance_decision_fingerprint '{evaluation.governance_decision_fingerprint}' does not match context governance fingerprint '{context.governance_decision_fingerprint}'."
        )
    if evaluation.campaign_selection_decision_fingerprint != context.campaign_selection_decision_fingerprint:
        raise MarketEvaluationValidationError(
            f"Evaluation campaign_selection_decision_fingerprint '{evaluation.campaign_selection_decision_fingerprint}' does not match context campaign selection fingerprint '{context.campaign_selection_decision_fingerprint}'."
        )
    if evaluation.authorization_fingerprint != context.authorization_fingerprint:
        raise MarketEvaluationValidationError(
            f"Evaluation authorization_fingerprint '{evaluation.authorization_fingerprint}' does not match context authorization fingerprint '{context.authorization_fingerprint}'."
        )
    if evaluation.authorized_runtime_context_fingerprint != context.context_fingerprint:
        raise MarketEvaluationValidationError(
            f"Evaluation authorized_runtime_context_fingerprint '{evaluation.authorized_runtime_context_fingerprint}' does not match actual context fingerprint '{context.context_fingerprint}'."
        )


def create_live_market_evaluation(
    data: pd.DataFrame,
    context: AuthorizedProductionRuntimeContext,
    reference_now: datetime.datetime | None = None,
    max_age_seconds: float = 300.0,
) -> LiveMarketEvaluation:
    """Factory constructing LiveMarketEvaluation strictly derived from validate_market_data_freshness and AuthorizedProductionRuntimeContext."""
    from src.evaluation.live_execution_runtime import validate_market_data_freshness

    if not isinstance(context, AuthorizedProductionRuntimeContext):
        raise MarketEvaluationValidationError(
            f"context must be an AuthorizedProductionRuntimeContext, got {type(context).__name__}"
        )

    ref_now = reference_now if reference_now is not None else datetime.datetime.now(datetime.timezone.utc)
    if ref_now.tzinfo is None:
        ref_now = ref_now.replace(tzinfo=datetime.timezone.utc)
    ref_now_iso = ref_now.isoformat()

    freshness = validate_market_data_freshness(
        data=data,
        max_age_seconds=max_age_seconds,
        reference_now=ref_now,
    )

    evaluation = LiveMarketEvaluation(
        symbol=context.symbol,
        timeframe=context.timeframe,
        candle_timestamp_utc=freshness.get("candle_timestamp"),
        freshness_status=bool(freshness.get("fresh", False)),
        freshness_reason=str(freshness.get("reason", "unknown")),
        age_seconds=freshness.get("age_seconds"),
        reference_timestamp_utc=ref_now_iso,
        authorized_runtime_context_fingerprint=context.context_fingerprint,
        promoted_artifact_fingerprint=context.promoted_artifact_fingerprint,
        governance_decision_fingerprint=context.governance_decision_fingerprint,
        campaign_selection_decision_fingerprint=context.campaign_selection_decision_fingerprint,
        authorization_fingerprint=context.authorization_fingerprint,
        authorization_policy_version=context.authorization_policy_version,
        candidate_id=context.candidate_id,
        strategy_id=context.strategy_id,
        strategy_version=context.strategy_version,
    )

    validate_market_evaluation_context_lineage(evaluation, context)
    return evaluation
