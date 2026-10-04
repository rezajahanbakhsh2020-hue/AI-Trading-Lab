"""Production Decision domain models, promotion eligibility validation, authoritative decision evaluation,
signal lineage projection, and risk geometry validation for Project 1.

Connects promoted research candidates and evidence to the live execution decision path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
from numbers import Real
from typing import Any, Optional

import pandas as pd

from src.evaluation.research_constitution import (
    PromotionStatus,
    ResearchEvidence,
)


class ProductionRuntimeAuthorizationError(ValueError):
    """Raised when Production Runtime Authorization fails validation or invariant checks."""


class Direction(str, Enum):
    """Authoritative trading directions."""

    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO TRADE"


@dataclass(frozen=True)
class ProductionAuthorizationReceipt:
    """Canonical immutable projection of ProductionRuntimeAuthorization for durable execution lineage."""

    candidate_id: str
    strategy_name: str
    strategy_version: str
    symbol: str
    timeframe: str
    promoted_artifact_fingerprint: str
    governance_decision_fingerprint: str
    campaign_selection_decision_fingerprint: Optional[str]
    authorization_policy_version: str
    authorized_at_utc: str
    authorization_fingerprint: str
    operational_stability_score: float

    def __post_init__(self) -> None:
        if isinstance(self.operational_stability_score, bool) or not isinstance(
            self.operational_stability_score, Real
        ):
            raise ProductionRuntimeAuthorizationError(
                "operational_stability_score must be a numeric float."
            )

        f_stab = float(self.operational_stability_score)
        if not math.isfinite(f_stab):
            raise ProductionRuntimeAuthorizationError(
                "operational_stability_score must be finite (not NaN or infinity)."
            )

        object.__setattr__(self, "operational_stability_score", f_stab)

        if not self.candidate_id or not str(self.candidate_id).strip():
            raise ProductionRuntimeAuthorizationError(
                "candidate_id must be a non-empty string."
            )
        if not self.strategy_name or not str(self.strategy_name).strip():
            raise ProductionRuntimeAuthorizationError(
                "strategy_name must be a non-empty string."
            )
        if not self.strategy_version or not str(self.strategy_version).strip():
            raise ProductionRuntimeAuthorizationError(
                "strategy_version must be a non-empty string."
            )
        if not self.symbol or not str(self.symbol).strip():
            raise ProductionRuntimeAuthorizationError(
                "symbol must be a non-empty string."
            )
        if not self.timeframe or not str(self.timeframe).strip():
            raise ProductionRuntimeAuthorizationError(
                "timeframe must be a non-empty string."
            )

        if (
            not self.promoted_artifact_fingerprint
            or not str(self.promoted_artifact_fingerprint).strip()
        ):
            raise ProductionRuntimeAuthorizationError(
                "promoted_artifact_fingerprint must be a non-empty string."
            )

        if (
            not self.governance_decision_fingerprint
            or not str(self.governance_decision_fingerprint).strip()
        ):
            raise ProductionRuntimeAuthorizationError(
                "governance_decision_fingerprint must be a non-empty string."
            )

        csdf = self.campaign_selection_decision_fingerprint
        if csdf is not None:
            csdf_str = str(csdf).strip()
            if not csdf_str:
                raise ProductionRuntimeAuthorizationError(
                    "campaign_selection_decision_fingerprint cannot be empty if provided."
                )
            object.__setattr__(
                self,
                "campaign_selection_decision_fingerprint",
                csdf_str,
            )

        if (
            not self.authorization_policy_version
            or not str(self.authorization_policy_version).strip()
        ):
            raise ProductionRuntimeAuthorizationError(
                "authorization_policy_version must be a non-empty string."
            )

        if not self.authorized_at_utc or not str(self.authorized_at_utc).strip():
            raise ProductionRuntimeAuthorizationError(
                "authorized_at_utc must be a non-empty string."
            )

        if (
            not self.authorization_fingerprint
            or not str(self.authorization_fingerprint).strip()
        ):
            raise ProductionRuntimeAuthorizationError(
                "authorization_fingerprint must be a non-empty string."
            )

        object.__setattr__(self, "candidate_id", str(self.candidate_id).strip())
        object.__setattr__(self, "strategy_name", str(self.strategy_name).strip())
        object.__setattr__(
            self,
            "strategy_version",
            str(self.strategy_version).strip(),
        )
        object.__setattr__(
            self,
            "symbol",
            str(self.symbol).strip().upper(),
        )
        object.__setattr__(
            self,
            "timeframe",
            str(self.timeframe).strip(),
        )
        object.__setattr__(
            self,
            "promoted_artifact_fingerprint",
            str(self.promoted_artifact_fingerprint).strip(),
        )
        object.__setattr__(
            self,
            "governance_decision_fingerprint",
            str(self.governance_decision_fingerprint).strip(),
        )
        object.__setattr__(
            self,
            "authorization_policy_version",
            str(self.authorization_policy_version).strip(),
        )
        object.__setattr__(
            self,
            "authorized_at_utc",
            str(self.authorized_at_utc).strip(),
        )
        object.__setattr__(
            self,
            "authorization_fingerprint",
            str(self.authorization_fingerprint).strip(),
        )

    @classmethod
    def from_authorization(
        cls,
        authorization: ProductionRuntimeAuthorization,
    ) -> ProductionAuthorizationReceipt:
        """Construct a receipt projection from an authoritative ProductionRuntimeAuthorization."""
        if not isinstance(authorization, ProductionRuntimeAuthorization):
            raise ProductionRuntimeAuthorizationError(
                "authorization must be a ProductionRuntimeAuthorization, "
                f"got {type(authorization).__name__}"
            )

        return cls(
            candidate_id=authorization.candidate_id,
            strategy_name=authorization.strategy_name,
            strategy_version=authorization.strategy_version,
            symbol=authorization.symbol,
            timeframe=authorization.timeframe,
            promoted_artifact_fingerprint=authorization.promoted_artifact_fingerprint,
            governance_decision_fingerprint=authorization.governance_decision_fingerprint,
            campaign_selection_decision_fingerprint=(
                authorization.campaign_selection_decision_fingerprint
            ),
            authorization_policy_version=authorization.authorization_policy_version,
            authorized_at_utc=authorization.authorized_at_utc,
            authorization_fingerprint=authorization.authorization_fingerprint,
            operational_stability_score=authorization.operational_stability_score,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "promoted_artifact_fingerprint": self.promoted_artifact_fingerprint,
            "governance_decision_fingerprint": self.governance_decision_fingerprint,
            "campaign_selection_decision_fingerprint": (
                self.campaign_selection_decision_fingerprint
            ),
            "authorization_policy_version": self.authorization_policy_version,
            "authorized_at_utc": self.authorized_at_utc,
            "authorization_fingerprint": self.authorization_fingerprint,
            "operational_stability_score": self.operational_stability_score,
        }


@dataclass(frozen=True)
class ProductionRuntimeAuthorization:
    """Explicit, immutable authorization artifact required for live runtime execution."""

    candidate_id: str
    strategy_name: str
    strategy_version: str
    symbol: str
    timeframe: str

    promoted_artifact_fingerprint: str
    governance_decision_fingerprint: str
    campaign_selection_decision_fingerprint: Optional[str]

    authorization_policy_version: str
    authorized_at_utc: str
    operational_stability_score: float

    authorization_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.operational_stability_score, bool) or not isinstance(
            self.operational_stability_score, Real
        ):
            raise ProductionRuntimeAuthorizationError(
                "operational_stability_score must be a numeric float."
            )

        f_stab = float(self.operational_stability_score)
        if not math.isfinite(f_stab):
            raise ProductionRuntimeAuthorizationError(
                "operational_stability_score must be finite (not NaN or infinity)."
            )

        object.__setattr__(self, "operational_stability_score", f_stab)

        if not self.candidate_id or not str(self.candidate_id).strip():
            raise ProductionRuntimeAuthorizationError(
                "candidate_id must be a non-empty string."
            )
        if not self.strategy_name or not str(self.strategy_name).strip():
            raise ProductionRuntimeAuthorizationError(
                "strategy_name must be a non-empty string."
            )
        if not self.strategy_version or not str(self.strategy_version).strip():
            raise ProductionRuntimeAuthorizationError(
                "strategy_version must be a non-empty string."
            )
        if not self.symbol or not str(self.symbol).strip():
            raise ProductionRuntimeAuthorizationError(
                "symbol must be a non-empty string."
            )
        if not self.timeframe or not str(self.timeframe).strip():
            raise ProductionRuntimeAuthorizationError(
                "timeframe must be a non-empty string."
            )

        if (
            not self.promoted_artifact_fingerprint
            or not str(self.promoted_artifact_fingerprint).strip()
        ):
            raise ProductionRuntimeAuthorizationError(
                "promoted_artifact_fingerprint must be a non-empty string."
            )

        if (
            not self.governance_decision_fingerprint
            or not str(self.governance_decision_fingerprint).strip()
        ):
            raise ProductionRuntimeAuthorizationError(
                "governance_decision_fingerprint must be a non-empty string."
            )

        csdf = self.campaign_selection_decision_fingerprint
        if csdf is not None:
            csdf_str = str(csdf).strip()
            if not csdf_str:
                raise ProductionRuntimeAuthorizationError(
                    "campaign_selection_decision_fingerprint cannot be empty if provided."
                )
            object.__setattr__(
                self,
                "campaign_selection_decision_fingerprint",
                csdf_str,
            )

        if (
            not self.authorization_policy_version
            or not str(self.authorization_policy_version).strip()
        ):
            raise ProductionRuntimeAuthorizationError(
                "authorization_policy_version must be a non-empty string."
            )

        if not self.authorized_at_utc or not str(self.authorized_at_utc).strip():
            raise ProductionRuntimeAuthorizationError(
                "authorized_at_utc must be a non-empty string."
            )

        object.__setattr__(self, "candidate_id", str(self.candidate_id).strip())
        object.__setattr__(self, "strategy_name", str(self.strategy_name).strip())
        object.__setattr__(
            self,
            "strategy_version",
            str(self.strategy_version).strip(),
        )
        object.__setattr__(
            self,
            "symbol",
            str(self.symbol).strip().upper(),
        )
        object.__setattr__(
            self,
            "timeframe",
            str(self.timeframe).strip(),
        )
        object.__setattr__(
            self,
            "promoted_artifact_fingerprint",
            str(self.promoted_artifact_fingerprint).strip(),
        )
        object.__setattr__(
            self,
            "governance_decision_fingerprint",
            str(self.governance_decision_fingerprint).strip(),
        )
        object.__setattr__(
            self,
            "authorization_policy_version",
            str(self.authorization_policy_version).strip(),
        )
        object.__setattr__(
            self,
            "authorized_at_utc",
            str(self.authorized_at_utc).strip(),
        )

        payload = {
            "candidate_id": self.candidate_id,
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "promoted_artifact_fingerprint": self.promoted_artifact_fingerprint,
            "governance_decision_fingerprint": self.governance_decision_fingerprint,
            "campaign_selection_decision_fingerprint": (
                self.campaign_selection_decision_fingerprint
            ),
            "authorization_policy_version": self.authorization_policy_version,
            "authorized_at_utc": self.authorized_at_utc,
            "operational_stability_score": self.operational_stability_score,
        }

        serialized = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
        )

        object.__setattr__(
            self,
            "authorization_fingerprint",
            hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        )


def authorize_production_runtime(
    promoted_candidate: PromotedCandidateArtifact,
    *,
    symbol: str,
    timeframe: str,
    now: Optional[datetime] = None,
    authorization_policy_version: str = "runtime_auth_v1.0",
) -> ProductionRuntimeAuthorization:
    """Authorize a promoted candidate artifact for production runtime execution."""

    if not isinstance(promoted_candidate, PromotedCandidateArtifact):
        raise ProductionRuntimeAuthorizationError(
            "promoted_candidate must be a PromotedCandidateArtifact, "
            f"got {type(promoted_candidate).__name__}"
        )

    # 1. Validate promotion eligibility via canonical path
    validate_promotion_eligibility(
        promoted_candidate.evidence,
        policy=promoted_candidate.policy,
        governance_decision=promoted_candidate.governance_decision,
        governance_decision_fingerprint=(
            promoted_candidate.governance_decision_fingerprint
        ),
    )

    # 2. Scope validation
    req_symbol = str(symbol).strip().upper() if symbol else ""
    req_timeframe = str(timeframe).strip() if timeframe else ""

    if not req_symbol:
        raise ProductionRuntimeAuthorizationError(
            "symbol must be a non-empty string."
        )
    if not req_timeframe:
        raise ProductionRuntimeAuthorizationError(
            "timeframe must be a non-empty string."
        )

    if promoted_candidate.symbol.upper() != req_symbol:
        raise ProductionRuntimeAuthorizationError(
            f"Candidate symbol '{promoted_candidate.symbol}' does not match "
            f"requested symbol '{req_symbol}'."
        )

    if promoted_candidate.timeframe != req_timeframe:
        raise ProductionRuntimeAuthorizationError(
            f"Candidate timeframe '{promoted_candidate.timeframe}' does not match "
            f"requested timeframe '{req_timeframe}'."
        )

    # 3. Governance decision fingerprint mandatory check
    gov_fp = promoted_candidate.governance_decision_fingerprint
    if not gov_fp or not str(gov_fp).strip():
        raise ProductionRuntimeAuthorizationError(
            f"Promoted candidate '{promoted_candidate.candidate_id}' is missing "
            "mandatory governance_decision_fingerprint."
        )

    # 4. Artifact fingerprint check
    art_fp = promoted_candidate.artifact_fingerprint
    if not art_fp or not str(art_fp).strip():
        raise ProductionRuntimeAuthorizationError(
            f"Promoted candidate '{promoted_candidate.candidate_id}' is missing "
            "mandatory artifact_fingerprint."
        )

    if now is None:
        authorized_at = datetime.now(timezone.utc)
    else:
        authorized_at = now

    if authorized_at.tzinfo is None:
        authorized_at = authorized_at.replace(tzinfo=timezone.utc)
    else:
        authorized_at = authorized_at.astimezone(timezone.utc)

    return ProductionRuntimeAuthorization(
        candidate_id=promoted_candidate.candidate_id,
        strategy_name=promoted_candidate.strategy_name,
        strategy_version=promoted_candidate.strategy_version,
        symbol=req_symbol,
        timeframe=req_timeframe,
        promoted_artifact_fingerprint=art_fp,
        governance_decision_fingerprint=str(gov_fp).strip(),
        campaign_selection_decision_fingerprint=(
            promoted_candidate.campaign_selection_decision_fingerprint
        ),
        authorization_policy_version=authorization_policy_version,
        authorized_at_utc=authorized_at.isoformat(),
        operational_stability_score=(
            promoted_candidate.operational_stability_score
        ),
    )


@dataclass(frozen=True)
class ProductionPromotionPolicy:
    """Versioned promotion policy requirements for production eligibility."""

    policy_version: str = "promotion_v1.0"
    allowed_statuses: tuple[PromotionStatus, ...] = (
        PromotionStatus.PROMOTABLE,
        PromotionStatus.VALIDATED,
    )
    max_evidence_age_days: Optional[float] = None
    require_robustness_pass: bool = True

    def __post_init__(self) -> None:
        if not self.policy_version or not self.policy_version.strip():
            raise ValueError("policy_version must be a non-empty string.")
        if not self.allowed_statuses:
            raise ValueError("allowed_statuses must not be empty.")


@dataclass(frozen=True)
class PromotedCandidateArtifact:
    """Authoritative representation of a promoted research candidate eligible for production execution."""

    candidate_id: str
    strategy_name: str
    strategy_version: str
    evidence: ResearchEvidence
    symbol: str
    timeframe: str
    operational_stability_score: float
    parameters: dict[str, Any] = field(default_factory=dict)
    policy: ProductionPromotionPolicy = field(
        default_factory=ProductionPromotionPolicy
    )
    governance_decision: Any | None = None
    governance_decision_fingerprint: str | None = None
    campaign_selection_decision_fingerprint: str | None = None
    artifact_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.operational_stability_score, bool) or not isinstance(
            self.operational_stability_score, Real
        ):
            raise ValueError(
               
