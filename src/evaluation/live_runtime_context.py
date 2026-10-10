"""Canonical Authorized Production Runtime Context handoff artifact for Project 1 live runtime execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from src.evaluation.live_production_decision import (
    ProductionAuthorizationReceipt,
    ProductionRuntimeAuthorization,
    ProductionRuntimeAuthorizationError,
    PromotedCandidateArtifact,
)


class RuntimeContextValidationError(ProductionRuntimeAuthorizationError):
    """Raised when AuthorizedProductionRuntimeContext validation fails."""


@dataclass(frozen=True)
class AuthorizedProductionRuntimeContext:
    """Canonical, immutable handoff artifact binding a PromotedCandidateArtifact to its ProductionRuntimeAuthorization.

    Ensures candidate resolution and production runtime authorization happen EXACTLY ONCE per execution cycle.
    """

    candidate: PromotedCandidateArtifact
    authorization: ProductionRuntimeAuthorization
    authorization_receipt: ProductionAuthorizationReceipt
    symbol: str
    timeframe: str
    candidate_id: str
    strategy_id: str
    strategy_version: str
    promoted_artifact_fingerprint: str
    governance_decision_fingerprint: str
    campaign_selection_decision_fingerprint: str | None
    authorization_fingerprint: str
    authorization_policy_version: str
    context_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, PromotedCandidateArtifact):
            raise RuntimeContextValidationError(
                f"candidate must be a PromotedCandidateArtifact, got {type(self.candidate).__name__}"
            )
        if not isinstance(self.authorization, ProductionRuntimeAuthorization):
            raise RuntimeContextValidationError(
                f"authorization must be a ProductionRuntimeAuthorization, got {type(self.authorization).__name__}"
            )
        if not isinstance(self.authorization_receipt, ProductionAuthorizationReceipt):
            raise RuntimeContextValidationError(
                f"authorization_receipt must be a ProductionAuthorizationReceipt, got {type(self.authorization_receipt).__name__}"
            )

        req_symbol = str(self.symbol).strip().upper() if self.symbol else ""
        raw_timeframe = str(self.timeframe).strip() if self.timeframe else ""
        req_cand_id = str(self.candidate_id).strip() if self.candidate_id else ""
        req_strat_id = str(self.strategy_id).strip() if self.strategy_id else ""
        req_strat_ver = str(self.strategy_version).strip() if self.strategy_version else ""

        if not req_symbol:
            raise RuntimeContextValidationError("symbol must be a non-empty string.")
        if not raw_timeframe:
            raise RuntimeContextValidationError("timeframe must be a non-empty string.")
        if not req_cand_id:
            raise RuntimeContextValidationError("candidate_id must be a non-empty string.")
        if not req_strat_id:
            raise RuntimeContextValidationError("strategy_id must be a non-empty string.")
        if not req_strat_ver:
            raise RuntimeContextValidationError("strategy_version must be a non-empty string.")

        try:
            from src.evaluation.mtf_intelligence import CanonicalTimeframe
            req_timeframe = CanonicalTimeframe.from_str(raw_timeframe).value
            cand_timeframe = CanonicalTimeframe.from_str(self.candidate.timeframe).value
            auth_timeframe = CanonicalTimeframe.from_str(self.authorization.timeframe).value
        except (ValueError, TypeError) as exc:
            raise RuntimeContextValidationError(f"Invalid timeframe: {exc}")

        # Normalize string attributes
        object.__setattr__(self, "symbol", req_symbol)
        object.__setattr__(self, "timeframe", req_timeframe)
        object.__setattr__(self, "candidate_id", req_cand_id)
        object.__setattr__(self, "strategy_id", req_strat_id)
        object.__setattr__(self, "strategy_version", req_strat_ver)

        # 1. Candidate identity matching authorization identity
        if self.candidate.candidate_id != self.authorization.candidate_id:
            raise RuntimeContextValidationError(
                f"Candidate ID '{self.candidate.candidate_id}' does not match authorization candidate ID '{self.authorization.candidate_id}'."
            )
        if self.candidate_id != self.candidate.candidate_id:
            raise RuntimeContextValidationError(
                f"Context candidate_id '{self.candidate_id}' does not match candidate candidate_id '{self.candidate.candidate_id}'."
            )

        # 2. Symbol and timeframe matching
        if self.symbol != self.candidate.symbol.upper():
            raise RuntimeContextValidationError(
                f"Context symbol '{self.symbol}' does not match candidate symbol '{self.candidate.symbol}'."
            )
        if self.authorization.symbol.upper() != self.symbol:
            raise RuntimeContextValidationError(
                f"Authorization symbol '{self.authorization.symbol}' does not match context symbol '{self.symbol}'."
            )
        if req_timeframe != cand_timeframe:
            raise RuntimeContextValidationError(
                f"Context timeframe '{self.timeframe}' does not match candidate timeframe '{self.candidate.timeframe}'."
            )
        if auth_timeframe != req_timeframe:
            raise RuntimeContextValidationError(
                f"Authorization timeframe '{self.authorization.timeframe}' does not match context timeframe '{self.timeframe}'."
            )

        # 3. Strategy identity and version consistency
        if self.strategy_id != self.candidate.strategy_name:
            raise RuntimeContextValidationError(
                f"Context strategy_id '{self.strategy_id}' does not match candidate strategy_name '{self.candidate.strategy_name}'."
            )
        if self.authorization.strategy_name != self.strategy_id:
            raise RuntimeContextValidationError(
                f"Authorization strategy_name '{self.authorization.strategy_name}' does not match context strategy_id '{self.strategy_id}'."
            )
        if self.strategy_version != self.candidate.strategy_version:
            raise RuntimeContextValidationError(
                f"Context strategy_version '{self.strategy_version}' does not match candidate strategy_version '{self.candidate.strategy_version}'."
            )
        if self.authorization.strategy_version != self.strategy_version:
            raise RuntimeContextValidationError(
                f"Authorization strategy_version '{self.authorization.strategy_version}' does not match context strategy_version '{self.strategy_version}'."
            )

        # 4. Fingerprint matching
        art_fp = str(self.promoted_artifact_fingerprint).strip() if self.promoted_artifact_fingerprint else ""
        gov_fp = str(self.governance_decision_fingerprint).strip() if self.governance_decision_fingerprint else ""
        auth_fp = str(self.authorization_fingerprint).strip() if self.authorization_fingerprint else ""
        auth_pol_ver = str(self.authorization_policy_version).strip() if self.authorization_policy_version else ""

        if not art_fp:
            raise RuntimeContextValidationError("promoted_artifact_fingerprint must be a non-empty string.")
        if not gov_fp:
            raise RuntimeContextValidationError("governance_decision_fingerprint must be a non-empty string.")
        if not auth_fp:
            raise RuntimeContextValidationError("authorization_fingerprint must be a non-empty string.")
        if not auth_pol_ver:
            raise RuntimeContextValidationError("authorization_policy_version must be a non-empty string.")

        object.__setattr__(self, "promoted_artifact_fingerprint", art_fp)
        object.__setattr__(self, "governance_decision_fingerprint", gov_fp)
        object.__setattr__(self, "authorization_fingerprint", auth_fp)
        object.__setattr__(self, "authorization_policy_version", auth_pol_ver)

        if art_fp != self.candidate.artifact_fingerprint:
            raise RuntimeContextValidationError(
                f"promoted_artifact_fingerprint '{art_fp}' does not match candidate artifact fingerprint '{self.candidate.artifact_fingerprint}'."
            )
        if art_fp != self.authorization.promoted_artifact_fingerprint:
            raise RuntimeContextValidationError(
                f"promoted_artifact_fingerprint '{art_fp}' does not match authorization promoted_artifact_fingerprint '{self.authorization.promoted_artifact_fingerprint}'."
            )

        if gov_fp != self.candidate.governance_decision_fingerprint:
            raise RuntimeContextValidationError(
                f"governance_decision_fingerprint '{gov_fp}' does not match candidate governance decision fingerprint '{self.candidate.governance_decision_fingerprint}'."
            )
        if gov_fp != self.authorization.governance_decision_fingerprint:
            raise RuntimeContextValidationError(
                f"governance_decision_fingerprint '{gov_fp}' does not match authorization governance_decision_fingerprint '{self.authorization.governance_decision_fingerprint}'."
            )

        # Campaign selection decision fingerprint check
        csdf = self.campaign_selection_decision_fingerprint
        if csdf is not None:
            csdf_str = str(csdf).strip()
            if not csdf_str:
                raise RuntimeContextValidationError("campaign_selection_decision_fingerprint cannot be empty if provided.")
            object.__setattr__(self, "campaign_selection_decision_fingerprint", csdf_str)
        else:
            object.__setattr__(self, "campaign_selection_decision_fingerprint", None)

        if self.campaign_selection_decision_fingerprint != self.candidate.campaign_selection_decision_fingerprint:
            raise RuntimeContextValidationError(
                f"campaign_selection_decision_fingerprint '{self.campaign_selection_decision_fingerprint}' does not match candidate campaign selection fingerprint '{self.candidate.campaign_selection_decision_fingerprint}'."
            )
        if self.campaign_selection_decision_fingerprint != self.authorization.campaign_selection_decision_fingerprint:
            raise RuntimeContextValidationError(
                f"campaign_selection_decision_fingerprint '{self.campaign_selection_decision_fingerprint}' does not match authorization campaign selection fingerprint '{self.authorization.campaign_selection_decision_fingerprint}'."
            )

        # Operational stability score matching across candidate, authorization, and receipt
        if abs(self.candidate.operational_stability_score - self.authorization.operational_stability_score) > 1e-9:
            raise RuntimeContextValidationError(
                f"Candidate operational_stability_score ({self.candidate.operational_stability_score}) "
                f"does not match authorization operational_stability_score ({self.authorization.operational_stability_score})."
            )
        if abs(self.authorization_receipt.operational_stability_score - self.authorization.operational_stability_score) > 1e-9:
            raise RuntimeContextValidationError(
                f"Receipt operational_stability_score ({self.authorization_receipt.operational_stability_score}) "
                f"does not match authorization operational_stability_score ({self.authorization.operational_stability_score})."
            )

        # Receipt matching
        if self.authorization_receipt.authorization_fingerprint != self.authorization.authorization_fingerprint:
            raise RuntimeContextValidationError(
                f"Receipt authorization_fingerprint '{self.authorization_receipt.authorization_fingerprint}' does not match authorization fingerprint '{self.authorization.authorization_fingerprint}'."
            )
        if auth_fp != self.authorization.authorization_fingerprint:
            raise RuntimeContextValidationError(
                f"Context authorization_fingerprint '{auth_fp}' does not match authorization fingerprint '{self.authorization.authorization_fingerprint}'."
            )
        if auth_pol_ver != self.authorization.authorization_policy_version:
            raise RuntimeContextValidationError(
                f"Context authorization_policy_version '{auth_pol_ver}' does not match authorization policy version '{self.authorization.authorization_policy_version}'."
            )

        # 5. Calculate deterministic context fingerprint over canonical lineage inputs
        payload = {
            "candidate_id": self.candidate_id,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "promoted_artifact_fingerprint": self.promoted_artifact_fingerprint,
            "governance_decision_fingerprint": self.governance_decision_fingerprint,
            "campaign_selection_decision_fingerprint": self.campaign_selection_decision_fingerprint,
            "authorization_fingerprint": self.authorization_fingerprint,
            "authorization_policy_version": self.authorization_policy_version,
            "operational_stability_score": self.candidate.operational_stability_score,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        object.__setattr__(
            self,
            "context_fingerprint",
            hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "promoted_artifact_fingerprint": self.promoted_artifact_fingerprint,
            "governance_decision_fingerprint": self.governance_decision_fingerprint,
            "campaign_selection_decision_fingerprint": self.campaign_selection_decision_fingerprint,
            "authorization_fingerprint": self.authorization_fingerprint,
            "authorization_policy_version": self.authorization_policy_version,
            "operational_stability_score": self.candidate.operational_stability_score,
            "context_fingerprint": self.context_fingerprint,
        }


def create_authorized_runtime_context(
    candidate: PromotedCandidateArtifact,
    authorization: ProductionRuntimeAuthorization,
    authorization_receipt: ProductionAuthorizationReceipt,
) -> AuthorizedProductionRuntimeContext:
    """Factory creating an immutable AuthorizedProductionRuntimeContext from candidate, authorization, and receipt artifacts."""
    if not isinstance(authorization_receipt, ProductionAuthorizationReceipt):
        raise RuntimeContextValidationError(
            f"authorization_receipt must be a ProductionAuthorizationReceipt, got {type(authorization_receipt).__name__}"
        )
    return AuthorizedProductionRuntimeContext(
        candidate=candidate,
        authorization=authorization,
        authorization_receipt=authorization_receipt,
        symbol=authorization.symbol,
        timeframe=authorization.timeframe,
        candidate_id=authorization.candidate_id,
        strategy_id=authorization.strategy_name,
        strategy_version=authorization.strategy_version,
        promoted_artifact_fingerprint=authorization.promoted_artifact_fingerprint,
        governance_decision_fingerprint=authorization.governance_decision_fingerprint,
        campaign_selection_decision_fingerprint=authorization.campaign_selection_decision_fingerprint,
        authorization_fingerprint=authorization.authorization_fingerprint,
        authorization_policy_version=authorization.authorization_policy_version,
    )
