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
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from src.evaluation.mtf_intelligence import MTFIntelligence

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
        if isinstance(self.operational_stability_score, bool) or not isinstance(self.operational_stability_score, Real):
            raise ProductionRuntimeAuthorizationError("operational_stability_score must be a numeric float.")
        f_stab = float(self.operational_stability_score)
        if not math.isfinite(f_stab):
            raise ProductionRuntimeAuthorizationError("operational_stability_score must be finite (not NaN or infinity).")
        object.__setattr__(self, "operational_stability_score", f_stab)
        if not self.candidate_id or not str(self.candidate_id).strip():
            raise ProductionRuntimeAuthorizationError("candidate_id must be a non-empty string.")
        if not self.strategy_name or not str(self.strategy_name).strip():
            raise ProductionRuntimeAuthorizationError("strategy_name must be a non-empty string.")
        if not self.strategy_version or not str(self.strategy_version).strip():
            raise ProductionRuntimeAuthorizationError("strategy_version must be a non-empty string.")
        if not self.symbol or not str(self.symbol).strip():
            raise ProductionRuntimeAuthorizationError("symbol must be a non-empty string.")
        if not self.timeframe or not str(self.timeframe).strip():
            raise ProductionRuntimeAuthorizationError("timeframe must be a non-empty string.")

        if not self.promoted_artifact_fingerprint or not str(self.promoted_artifact_fingerprint).strip():
            raise ProductionRuntimeAuthorizationError("promoted_artifact_fingerprint must be a non-empty string.")
        if not self.governance_decision_fingerprint or not str(self.governance_decision_fingerprint).strip():
            raise ProductionRuntimeAuthorizationError("governance_decision_fingerprint must be a non-empty string.")

        csdf = self.campaign_selection_decision_fingerprint
        if csdf is not None:
            csdf_str = str(csdf).strip()
            if not csdf_str:
                raise ProductionRuntimeAuthorizationError("campaign_selection_decision_fingerprint cannot be empty if provided.")
            object.__setattr__(self, "campaign_selection_decision_fingerprint", csdf_str)

        if not self.authorization_policy_version or not str(self.authorization_policy_version).strip():
            raise ProductionRuntimeAuthorizationError("authorization_policy_version must be a non-empty string.")
        if not self.authorized_at_utc or not str(self.authorized_at_utc).strip():
            raise ProductionRuntimeAuthorizationError("authorized_at_utc must be a non-empty string.")
        if not self.authorization_fingerprint or not str(self.authorization_fingerprint).strip():
            raise ProductionRuntimeAuthorizationError("authorization_fingerprint must be a non-empty string.")

        object.__setattr__(self, "candidate_id", str(self.candidate_id).strip())
        object.__setattr__(self, "strategy_name", str(self.strategy_name).strip())
        object.__setattr__(self, "strategy_version", str(self.strategy_version).strip())
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        object.__setattr__(self, "timeframe", str(self.timeframe).strip())
        object.__setattr__(self, "promoted_artifact_fingerprint", str(self.promoted_artifact_fingerprint).strip())
        object.__setattr__(self, "governance_decision_fingerprint", str(self.governance_decision_fingerprint).strip())
        object.__setattr__(self, "authorization_policy_version", str(self.authorization_policy_version).strip())
        object.__setattr__(self, "authorized_at_utc", str(self.authorized_at_utc).strip())
        object.__setattr__(self, "authorization_fingerprint", str(self.authorization_fingerprint).strip())

    @classmethod
    def from_authorization(
        cls, authorization: ProductionRuntimeAuthorization
    ) -> ProductionAuthorizationReceipt:
        """Construct a receipt projection from an authoritative ProductionRuntimeAuthorization."""
        if not isinstance(authorization, ProductionRuntimeAuthorization):
            raise ProductionRuntimeAuthorizationError(
                f"authorization must be a ProductionRuntimeAuthorization, got {type(authorization).__name__}"
            )

        return cls(
            candidate_id=authorization.candidate_id,
            strategy_name=authorization.strategy_name,
            strategy_version=authorization.strategy_version,
            symbol=authorization.symbol,
            timeframe=authorization.timeframe,
            promoted_artifact_fingerprint=authorization.promoted_artifact_fingerprint,
            governance_decision_fingerprint=authorization.governance_decision_fingerprint,
            campaign_selection_decision_fingerprint=authorization.campaign_selection_decision_fingerprint,
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
            "campaign_selection_decision_fingerprint": self.campaign_selection_decision_fingerprint,
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
        if isinstance(self.operational_stability_score, bool) or not isinstance(self.operational_stability_score, Real):
            raise ProductionRuntimeAuthorizationError("operational_stability_score must be a numeric float.")
        f_stab = float(self.operational_stability_score)
        if not math.isfinite(f_stab):
            raise ProductionRuntimeAuthorizationError("operational_stability_score must be finite (not NaN or infinity).")
        object.__setattr__(self, "operational_stability_score", f_stab)
        if not self.candidate_id or not str(self.candidate_id).strip():
            raise ProductionRuntimeAuthorizationError("candidate_id must be a non-empty string.")
        if not self.strategy_name or not str(self.strategy_name).strip():
            raise ProductionRuntimeAuthorizationError("strategy_name must be a non-empty string.")
        if not self.strategy_version or not str(self.strategy_version).strip():
            raise ProductionRuntimeAuthorizationError("strategy_version must be a non-empty string.")
        if not self.symbol or not str(self.symbol).strip():
            raise ProductionRuntimeAuthorizationError("symbol must be a non-empty string.")
        if not self.timeframe or not str(self.timeframe).strip():
            raise ProductionRuntimeAuthorizationError("timeframe must be a non-empty string.")

        if not self.promoted_artifact_fingerprint or not str(self.promoted_artifact_fingerprint).strip():
            raise ProductionRuntimeAuthorizationError("promoted_artifact_fingerprint must be a non-empty string.")
        if not self.governance_decision_fingerprint or not str(self.governance_decision_fingerprint).strip():
            raise ProductionRuntimeAuthorizationError("governance_decision_fingerprint must be a non-empty string.")

        csdf = self.campaign_selection_decision_fingerprint
        if csdf is not None:
            csdf_str = str(csdf).strip()
            if not csdf_str:
                raise ProductionRuntimeAuthorizationError("campaign_selection_decision_fingerprint cannot be empty if provided.")
            object.__setattr__(self, "campaign_selection_decision_fingerprint", csdf_str)

        if not self.authorization_policy_version or not str(self.authorization_policy_version).strip():
            raise ProductionRuntimeAuthorizationError("authorization_policy_version must be a non-empty string.")
        if not self.authorized_at_utc or not str(self.authorized_at_utc).strip():
            raise ProductionRuntimeAuthorizationError("authorized_at_utc must be a non-empty string.")

        object.__setattr__(self, "candidate_id", str(self.candidate_id).strip())
        object.__setattr__(self, "strategy_name", str(self.strategy_name).strip())
        object.__setattr__(self, "strategy_version", str(self.strategy_version).strip())
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        object.__setattr__(self, "timeframe", str(self.timeframe).strip())
        object.__setattr__(self, "promoted_artifact_fingerprint", str(self.promoted_artifact_fingerprint).strip())
        object.__setattr__(self, "governance_decision_fingerprint", str(self.governance_decision_fingerprint).strip())
        object.__setattr__(self, "authorization_policy_version", str(self.authorization_policy_version).strip())
        object.__setattr__(self, "authorized_at_utc", str(self.authorized_at_utc).strip())

        payload = {
            "candidate_id": self.candidate_id,
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "promoted_artifact_fingerprint": self.promoted_artifact_fingerprint,
            "governance_decision_fingerprint": self.governance_decision_fingerprint,
            "campaign_selection_decision_fingerprint": self.campaign_selection_decision_fingerprint,
            "authorization_policy_version": self.authorization_policy_version,
            "authorized_at_utc": self.authorized_at_utc,
            "operational_stability_score": self.operational_stability_score,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
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
            f"promoted_candidate must be a PromotedCandidateArtifact, got {type(promoted_candidate).__name__}"
        )

    # 1. Validate promotion eligibility via canonical path
    validate_promotion_eligibility(
        promoted_candidate.evidence,
        policy=promoted_candidate.policy,
        governance_decision=promoted_candidate.governance_decision,
        governance_decision_fingerprint=promoted_candidate.governance_decision_fingerprint,
    )

    # 2. Scope validation
    req_symbol = str(symbol).strip().upper() if symbol else ""
    req_timeframe = str(timeframe).strip() if timeframe else ""
    if not req_symbol:
        raise ProductionRuntimeAuthorizationError("symbol must be a non-empty string.")
    if not req_timeframe:
        raise ProductionRuntimeAuthorizationError("timeframe must be a non-empty string.")

    if promoted_candidate.symbol.upper() != req_symbol:
        raise ProductionRuntimeAuthorizationError(
            f"Candidate symbol '{promoted_candidate.symbol}' does not match requested symbol '{req_symbol}'."
        )
    if promoted_candidate.timeframe != req_timeframe:
        raise ProductionRuntimeAuthorizationError(
            f"Candidate timeframe '{promoted_candidate.timeframe}' does not match requested timeframe '{req_timeframe}'."
        )

    # 3. Governance decision fingerprint mandatory check
    gov_fp = promoted_candidate.governance_decision_fingerprint
    if not gov_fp or not str(gov_fp).strip():
        raise ProductionRuntimeAuthorizationError(
            f"Promoted candidate '{promoted_candidate.candidate_id}' is missing mandatory governance_decision_fingerprint."
        )

    # 4. Artifact fingerprint check
    art_fp = promoted_candidate.artifact_fingerprint
    if not art_fp or not str(art_fp).strip():
        raise ProductionRuntimeAuthorizationError(
            f"Promoted candidate '{promoted_candidate.candidate_id}' is missing mandatory artifact_fingerprint."
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
        campaign_selection_decision_fingerprint=promoted_candidate.campaign_selection_decision_fingerprint,
        authorization_policy_version=authorization_policy_version,
        authorized_at_utc=authorized_at.isoformat(),
        operational_stability_score=promoted_candidate.operational_stability_score,
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
    policy: ProductionPromotionPolicy = field(default_factory=ProductionPromotionPolicy)
    governance_decision: Any | None = None
    governance_decision_fingerprint: str | None = None
    campaign_selection_decision_fingerprint: str | None = None
    artifact_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.operational_stability_score, bool) or not isinstance(self.operational_stability_score, Real):
            raise ValueError("operational_stability_score must be a numeric float.")
        f_stab = float(self.operational_stability_score)
        if not math.isfinite(f_stab):
            raise ValueError("operational_stability_score must be finite (not NaN or infinity).")
        object.__setattr__(self, "operational_stability_score", f_stab)
        if not self.candidate_id or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string.")
        if not self.strategy_name or not self.strategy_name.strip():
            raise ValueError("strategy_name must be a non-empty string.")
        if not self.strategy_version or not self.strategy_version.strip():
            raise ValueError("strategy_version must be a non-empty string.")
        if not isinstance(self.evidence, ResearchEvidence):
            raise TypeError("evidence must be a ResearchEvidence instance.")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string.")
        if not self.timeframe or not self.timeframe.strip():
            raise ValueError("timeframe must be a non-empty string.")


        merged_params = dict(self.evidence.spec.parameters) if (self.evidence and self.evidence.spec.parameters) else {}
        if self.parameters:
            merged_params.update(self.parameters)
        object.__setattr__(self, "parameters", merged_params)

        if self.governance_decision_fingerprint is None and self.governance_decision is not None:
            object.__setattr__(self, "governance_decision_fingerprint", getattr(self.governance_decision, "decision_fingerprint", None))
        validate_promotion_eligibility(self.evidence, policy=self.policy, governance_decision=self.governance_decision, governance_decision_fingerprint=self.governance_decision_fingerprint)

        if self.evidence.experiment_fingerprint != self.evidence.spec.fingerprint:
            raise ValueError(
                f"Candidate '{self.candidate_id}' research fingerprint "
                f"'{self.evidence.experiment_fingerprint}' does not match "
                f"evidence spec fingerprint '{self.evidence.spec.fingerprint}'."
            )
        if self.strategy_name != self.evidence.spec.strategy_name:
            raise ValueError(
                f"Candidate '{self.candidate_id}' strategy_name '{self.strategy_name}' does not match "
                f"evidence strategy '{self.evidence.spec.strategy_name}'."
            )
        if self.strategy_version != self.evidence.spec.strategy_version:
            raise ValueError(
                f"Candidate '{self.candidate_id}' strategy_version '{self.strategy_version}' does not match "
                f"evidence strategy_version '{self.evidence.spec.strategy_version}'."
            )

        # Check scope match
        ds = self.evidence.spec.dataset_scope
        if ds.symbol.upper() != self.symbol.upper():
            raise ValueError(
                f"Candidate symbol '{self.symbol}' does not match evidence dataset symbol '{ds.symbol}'."
            )
        if ds.timeframe != self.timeframe:
            raise ValueError(
                f"Candidate timeframe '{self.timeframe}' does not match evidence dataset timeframe '{ds.timeframe}'."
            )

        payload = {
            "candidate_id": self.candidate_id,
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "evidence_id": self.evidence.evidence_id,
            "evidence_fingerprint": self.evidence.experiment_fingerprint,
            "symbol": self.symbol.upper(),
            "timeframe": self.timeframe,
            "parameters": self.parameters,
            "policy_version": self.policy.policy_version,
            "governance_decision_fingerprint": self.governance_decision_fingerprint,
            "campaign_selection_decision_fingerprint": self.campaign_selection_decision_fingerprint,
            "operational_stability_score": self.operational_stability_score,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        object.__setattr__(
            self,
            "artifact_fingerprint",
            hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        )

    @classmethod
    def from_persisted_research(
        cls,
        *,
        candidate_id: str,
        evidence: ResearchEvidence,
        symbol: str,
        timeframe: str,
        operational_stability_score: float,
        parameters: Optional[dict[str, Any]] = None,
        policy: Optional[ProductionPromotionPolicy] = None,
        governance_decision: Any | None = None,
        governance_decision_fingerprint: str | None = None,
        campaign_selection_decision_fingerprint: str | None = None,
    ) -> "PromotedCandidateArtifact":
        """Reconstitute a candidate solely from persisted research evidence.

        This path cannot establish promotion. Eligibility is derived from the
        persisted ResearchEvidence object, not from live configuration.
        """
        if not isinstance(evidence, ResearchEvidence):
            raise TypeError("evidence must be a ResearchEvidence instance.")
        return cls(
            candidate_id=candidate_id,
            strategy_name=evidence.spec.strategy_name,
            strategy_version=evidence.spec.strategy_version,
            evidence=evidence,
            symbol=symbol,
            timeframe=timeframe,
            parameters=dict(parameters) if parameters is not None else dict(evidence.spec.parameters),
            policy=policy if policy is not None else ProductionPromotionPolicy(),
            governance_decision=governance_decision,
            governance_decision_fingerprint=governance_decision_fingerprint,
            campaign_selection_decision_fingerprint=campaign_selection_decision_fingerprint,
            operational_stability_score=operational_stability_score,
        )


def validate_production_scope(
    candidate: PromotedCandidateArtifact,
    *,
    current_symbol: Optional[str] = None,
    current_timeframe: Optional[str] = None,
    current_strategy_id: Optional[str] = None,
    current_strategy_version: Optional[str] = None,
    current_candidate_id: Optional[str] = None,
    now: Optional[datetime] = None,
) -> bool:
    """Verify production scope against the persisted promoted artifact. Fail closed on mismatch."""
    if not isinstance(candidate, PromotedCandidateArtifact):
        raise TypeError("candidate must be a PromotedCandidateArtifact instance.")

    validate_promotion_eligibility(candidate.evidence, policy=candidate.policy, now=now)

    if candidate.evidence.experiment_fingerprint != candidate.evidence.spec.fingerprint:
        raise ValueError(
            f"Candidate '{candidate.candidate_id}' research fingerprint does not match persisted evidence fingerprint."
        )
    if current_candidate_id is not None and str(current_candidate_id).strip():
        if candidate.candidate_id != str(current_candidate_id).strip():
            raise ValueError(
                f"Configured candidate_id '{current_candidate_id}' does not match "
                f"resolved candidate '{candidate.candidate_id}'."
            )
    if current_strategy_id is not None and str(current_strategy_id).strip():
        if candidate.strategy_name != str(current_strategy_id).strip():
            raise ValueError(
                f"Configured strategy_id '{current_strategy_id}' does not match "
                f"resolved strategy '{candidate.strategy_name}'."
            )
    if current_strategy_version is not None and str(current_strategy_version).strip():
        if candidate.strategy_version != str(current_strategy_version).strip():
            raise ValueError(
                f"Configured strategy_version '{current_strategy_version}' does not match "
                f"resolved strategy_version '{candidate.strategy_version}'."
            )
    if current_symbol is not None and str(current_symbol).strip():
        if candidate.symbol.upper() != str(current_symbol).strip().upper():
            raise ValueError(
                f"Configured symbol '{current_symbol}' does not match "
                f"resolved candidate symbol '{candidate.symbol}'."
            )
    if current_timeframe is not None and str(current_timeframe).strip():
        try:
            from src.evaluation.mtf_intelligence import CanonicalTimeframe
            curr_tf = CanonicalTimeframe.from_str(current_timeframe).value
            cand_tf = CanonicalTimeframe.from_str(candidate.timeframe).value
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid canonical timeframe identity: "
                f"configured={current_timeframe!r}, "
                f"candidate={candidate.timeframe!r}"
            ) from exc

        if cand_tf != curr_tf:
            raise ValueError(
                f"Configured timeframe '{current_timeframe}' does not match "
                f"resolved candidate timeframe '{candidate.timeframe}'."
            )
    return True


def validate_promotion_eligibility(
    evidence: ResearchEvidence,
    policy: Optional[ProductionPromotionPolicy] = None,
    governance_decision: Any | None = None,
    governance_decision_fingerprint: str | None = None,
    now: Optional[datetime] = None,
) -> bool:
    """Validate that research evidence satisfies production promotion criteria.

    Consumes canonical governance decision. Fails closed if evidence is missing,
    unpromoted, rejected, or invalid. Does NOT recompute qualification.
    """
    if policy is None:
        policy = ProductionPromotionPolicy()

    if not isinstance(evidence, ResearchEvidence):
        raise TypeError("evidence must be a ResearchEvidence instance.")

    if governance_decision is not None:
        if not getattr(governance_decision, "qualified", False):
            raise ValueError(
                f"Evidence '{evidence.evidence_id}' supplied governance decision is rejected."
            )
        if getattr(governance_decision, "experiment_fingerprint", "") != evidence.experiment_fingerprint:
            raise ValueError(
                f"Governance decision experiment fingerprint '{getattr(governance_decision, 'experiment_fingerprint', '')}' "
                f"does not match evidence fingerprint '{evidence.experiment_fingerprint}'."
            )
        if governance_decision_fingerprint is not None:
            dec_fp = getattr(governance_decision, "decision_fingerprint", None)
            if dec_fp != governance_decision_fingerprint:
                raise ValueError(
                    f"Governance decision fingerprint '{dec_fp}' does not match "
                    f"expected fingerprint '{governance_decision_fingerprint}'."
                )

    if evidence.promotion_status not in policy.allowed_statuses:
        raise ValueError(
            f"Evidence '{evidence.evidence_id}' has status '{evidence.promotion_status.value}' "
            f"which is not allowed for production (allowed: {[s.value for s in policy.allowed_statuses]})."
        )

    if evidence.rejection_reasons:
        orig_reasons = [r.value for r in evidence.rejection_reasons]
        raise ValueError(
            f"Evidence '{evidence.evidence_id}' contains rejection reasons: {orig_reasons}"
        )

    if policy.max_evidence_age_days is not None and evidence.created_at_utc:
        try:
            created = datetime.fromisoformat(str(evidence.created_at_utc).replace("Z", "+00:00"))
            ref_now = now or datetime.now(timezone.utc)
            if ref_now.tzinfo is None:
                ref_now = ref_now.replace(tzinfo=timezone.utc)
            age_days = (ref_now - created).total_seconds() / 86400.0
            if age_days > policy.max_evidence_age_days:
                raise ValueError(
                    f"Evidence '{evidence.evidence_id}' is stale: age ({age_days:.1f} days) "
                    f"exceeds max allowed age ({policy.max_evidence_age_days} days)."
                )
        except (ValueError, TypeError) as exc:
            if "exceeds max allowed age" in str(exc):
                raise
            pass

    return True


def validate_market_data_for_production(
    data: pd.DataFrame,
    symbol: str,
    timeframe: str,
    max_age_seconds: float = 300.0,
    reference_now: Optional[Any] = None,
) -> pd.Series:
    """Validate live market data for production decision evaluation.

    Checks symbol, timeframe, timestamp presence, ordering, freshness, non-future timestamps,
    and valid OHLC values. Returns the latest valid bar as a Series.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Market data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("Market data DataFrame is empty.")

    if "timestamp" not in data.columns and "openTime" in data.columns:
        data = data.copy()
        data["timestamp"] = data["openTime"]

    required_cols = {"timestamp", "open", "high", "low", "close"}
    missing = required_cols.difference(data.columns)
    if missing:
        raise ValueError(f"Market data missing required columns: {sorted(missing)}")

    # Check for NaNs or non-finite OHLC values
    for col in ("open", "high", "low", "close"):
        s = pd.to_numeric(data[col], errors="coerce")
        if s.isna().any() or not math.isfinite(s.iloc[-1]):
            raise ValueError(f"Market data contains invalid or non-finite values in column '{col}'.")
        if (s <= 0).any():
            raise ValueError(f"Market data contains non-positive price values in column '{col}'.")

    # High/Low sanity check
    if (data["high"] < data["low"]).any() or (data["high"] < data["open"]).any() or (data["high"] < data["close"]).any():
        raise ValueError("Market data contains impossible OHLC high/low relationships.")

    latest_bar = data.iloc[-1]
    ts = pd.to_datetime(latest_bar["timestamp"], utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError("Latest bar timestamp is invalid.")

    # Validate freshness against reference_now
    if reference_now is None:
        now_dt = pd.Timestamp.now(tz="UTC")
    else:
        now_dt = pd.to_datetime(reference_now, utc=True)

    age_seconds = (now_dt - ts).total_seconds()
    if age_seconds < 0:
        raise ValueError(f"Market data timestamp '{ts}' is in the future relative to '{now_dt}'.")

    from src.evaluation.mtf_intelligence import CanonicalTimeframe
    from src.evaluation.live_execution_runtime import is_candle_closed

    canonical_tf = CanonicalTimeframe.from_str(timeframe).value
    if not is_candle_closed(latest_bar, canonical_tf, now_dt.to_pydatetime()):
        raise ValueError(
            f"Market data candle for timeframe '{canonical_tf}' is not closed yet."
        )

    return latest_bar


@dataclass(frozen=True)
class ProductionDecision:
    """Authoritative trading decision produced by strategy evaluation against market data."""

    candidate_id: str
    evidence_id: str
    experiment_fingerprint: str
    symbol: str
    timeframe: str
    decision_timestamp: str
    market_timestamp: str | None
    direction: Direction
    reason: str
    entry_price: Optional[float]
    invalidation_condition: Optional[str]
    confidence: Optional[float] = None
    parameters: dict[str, Any] = field(default_factory=dict)
    decision_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string.")
        if not self.evidence_id or not self.evidence_id.strip():
            raise ValueError("evidence_id must be a non-empty string.")
        if not self.experiment_fingerprint or not self.experiment_fingerprint.strip():
            raise ValueError("experiment_fingerprint must be a non-empty string.")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string.")
        if not self.timeframe or not self.timeframe.strip():
            raise ValueError("timeframe must be a non-empty string.")
        if not self.decision_timestamp or not self.decision_timestamp.strip():
            raise ValueError("decision_timestamp must be a non-empty string.")
        if self.market_timestamp is not None:
            mts = str(self.market_timestamp).strip()
            if not mts:
                raise ValueError("market_timestamp cannot be whitespace if provided.")
            object.__setattr__(self, "market_timestamp", mts)

        if not isinstance(self.direction, Direction):
            raise TypeError("direction must be a Direction enum member.")
        if self.direction in (Direction.BUY, Direction.SELL) and self.market_timestamp is None:
            raise ValueError("Executable decision requires a non-empty market_timestamp.")

        if self.direction in (Direction.BUY, Direction.SELL):
            if self.entry_price is None or not math.isfinite(self.entry_price) or self.entry_price <= 0:
                raise ValueError(
                    f"Executable decision '{self.direction.value}' requires a positive finite entry_price."
                )
        else:
            if self.entry_price is not None and (not math.isfinite(self.entry_price) or self.entry_price <= 0):
                raise ValueError("entry_price must be positive and finite if provided.")

        payload = {
            "candidate_id": self.candidate_id,
            "evidence_id": self.evidence_id,
            "experiment_fingerprint": self.experiment_fingerprint,
            "symbol": self.symbol.upper(),
            "timeframe": self.timeframe,
            "market_timestamp": self.market_timestamp,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "parameters": self.parameters,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        dec_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        object.__setattr__(self, "decision_id", f"dec_{dec_hash}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "candidate_id": self.candidate_id,
            "evidence_id": self.evidence_id,
            "experiment_fingerprint": self.experiment_fingerprint,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "decision_timestamp": self.decision_timestamp,
            "market_timestamp": self.market_timestamp,
            "direction": self.direction.value,
            "reason": self.reason,
            "entry_price": self.entry_price,
            "invalidation_condition": self.invalidation_condition,
            "confidence": self.confidence,
            "parameters": self.parameters,
        }


def evaluate_production_decision(
    candidate: PromotedCandidateArtifact,
    data: pd.DataFrame,
    reference_now: Optional[Any] = None,
    max_age_seconds: float = float("inf"),
) -> ProductionDecision:
    """Evaluate promoted candidate strategy against live market data to produce an authoritative ProductionDecision."""
    if not isinstance(candidate, PromotedCandidateArtifact):
        raise TypeError("candidate must be a PromotedCandidateArtifact instance.")

    latest_bar = validate_market_data_for_production(
        data=data,
        symbol=candidate.symbol,
        timeframe=candidate.timeframe,
        max_age_seconds=max_age_seconds,
        reference_now=reference_now,
    )

    now_ts = (
        pd.Timestamp.now(tz="UTC").isoformat()
        if reference_now is None
        else pd.to_datetime(reference_now, utc=True).isoformat()
    )
    market_ts = pd.to_datetime(latest_bar["timestamp"], utc=True).isoformat()

    # Execute strategy evaluation based on candidate strategy name
    strat_name = candidate.strategy_name.lower()
    close_price = float(latest_bar["close"])

    if strat_name == "momentum":
        from live_signal import generate_live_signal
        from live_trend import build_live_trend_snapshot

        window = candidate.parameters.get("momentum_window", candidate.parameters.get("window"))
        if window is None:
            raise ValueError(
                f"Candidate '{candidate.candidate_id}' parameters missing required "
                f"'momentum_window' or 'window' parameter for momentum strategy."
            )
        sig_df = generate_live_signal(data, window=int(window))
        sig_val = int(sig_df["signal"].iloc[-1])

        trend_snap = build_live_trend_snapshot(data)
        trend_val = trend_snap["trend"]

        if sig_val == 1 and trend_val == "UP":
            direction = Direction.BUY
            reason = "momentum_signal_1_and_trend_UP_confirmed"
            invalidation = "Close below stop_loss or trend turns DOWN"
        else:
            direction = Direction.NO_TRADE
            reason = f"signal={sig_val}, trend={trend_val} (unconfirmed setup)"
            invalidation = None

        return ProductionDecision(
            candidate_id=candidate.candidate_id,
            evidence_id=candidate.evidence.evidence_id,
            experiment_fingerprint=candidate.evidence.experiment_fingerprint,
            symbol=candidate.symbol,
            timeframe=candidate.timeframe,
            decision_timestamp=now_ts,
            market_timestamp=market_ts,
            direction=direction,
            reason=reason,
            entry_price=close_price if direction != Direction.NO_TRADE else None,
            invalidation_condition=invalidation,
            confidence=None,  # Do not manufacture confidence unless calculated
            parameters=candidate.parameters,
        )

    else:
        # Unsupported strategy returns NO_TRADE
        return ProductionDecision(
            candidate_id=candidate.candidate_id,
            evidence_id=candidate.evidence.evidence_id,
            experiment_fingerprint=candidate.evidence.experiment_fingerprint,
            symbol=candidate.symbol,
            timeframe=candidate.timeframe,
            decision_timestamp=now_ts,
            market_timestamp=market_ts,
            direction=Direction.NO_TRADE,
            reason=f"strategy_{candidate.strategy_name}_unsupported_in_live_evaluator",
            entry_price=None,
            invalidation_condition=None,
            confidence=None,
            parameters=candidate.parameters,
        )


@dataclass(frozen=True)
class ProductionSignal:
    """Operational signal projected directly from an authoritative ProductionDecision."""

    decision_id: str
    candidate_id: str
    evidence_id: str
    symbol: str
    timeframe: str
    market_timestamp: str | None
    direction: Direction
    entry_price: Optional[float]
    signal_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.decision_id or not self.decision_id.strip():
            raise ValueError("decision_id must be a non-empty string.")
        if not self.candidate_id or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string.")
        if not self.evidence_id or not self.evidence_id.strip():
            raise ValueError("evidence_id must be a non-empty string.")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string.")
        if not self.timeframe or not self.timeframe.strip():
            raise ValueError("timeframe must be a non-empty string.")
        if self.market_timestamp is not None:
            mts = str(self.market_timestamp).strip()
            if not mts:
                raise ValueError("market_timestamp cannot be whitespace if provided.")
            object.__setattr__(self, "market_timestamp", mts)

        if not isinstance(self.direction, Direction):
            raise TypeError("direction must be a Direction enum member.")
        if self.direction in (Direction.BUY, Direction.SELL) and self.market_timestamp is None:
            raise ValueError("Executable signal requires a non-empty market_timestamp.")

        payload = {
            "decision_id": self.decision_id,
            "candidate_id": self.candidate_id,
            "evidence_id": self.evidence_id,
            "symbol": self.symbol.upper(),
            "timeframe": self.timeframe,
            "market_timestamp": self.market_timestamp,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        sig_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        object.__setattr__(self, "signal_id", f"sig_{sig_hash}")

    @classmethod
    def from_decision(cls, decision: ProductionDecision) -> ProductionSignal:
        if not isinstance(decision, ProductionDecision):
            raise TypeError("decision must be a ProductionDecision instance.")
        return cls(
            decision_id=decision.decision_id,
            candidate_id=decision.candidate_id,
            evidence_id=decision.evidence_id,
            symbol=decision.symbol,
            timeframe=decision.timeframe,
            market_timestamp=decision.market_timestamp,
            direction=decision.direction,
            entry_price=decision.entry_price,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "decision_id": self.decision_id,
            "candidate_id": self.candidate_id,
            "evidence_id": self.evidence_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "market_timestamp": self.market_timestamp,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
        }


@dataclass(frozen=True)
class ProductionRiskLevels:
    """Authoritative risk levels calculated strictly from strategy/risk logic and entry geometry."""

    decision_id: str
    candidate_id: str
    evidence_id: str
    symbol: str
    timeframe: str
    direction: Direction
    entry_price: Optional[float]
    stop_loss: Optional[float]
    tp1: Optional[float]
    tp2: Optional[float]
    tp3: Optional[float]
    trailing_stop: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    risk_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.decision_id or not self.decision_id.strip():
            raise ValueError("decision_id must be a non-empty string.")
        if not self.candidate_id or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string.")
        if not self.evidence_id or not self.evidence_id.strip():
            raise ValueError("evidence_id must be a non-empty string.")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string.")
        if not self.timeframe or not self.timeframe.strip():
            raise ValueError("timeframe must be a non-empty string.")
        if not isinstance(self.direction, Direction):
            raise TypeError("direction must be a Direction enum member.")

        # Validate geometry if active trade direction
        if self.direction in (Direction.BUY, Direction.SELL):
            if self.entry_price is None or not math.isfinite(self.entry_price) or self.entry_price <= 0:
                raise ValueError(f"Active risk for {self.direction.value} requires positive finite entry_price.")

            # Validate Stop Loss geometry
            if self.stop_loss is not None:
                if not math.isfinite(self.stop_loss) or self.stop_loss <= 0:
                    raise ValueError("stop_loss must be positive and finite if provided.")
                if self.direction == Direction.BUY and self.stop_loss >= self.entry_price:
                    raise ValueError(
                        f"BUY stop_loss ({self.stop_loss}) must be strictly below entry_price ({self.entry_price})."
                    )
                if self.direction == Direction.SELL and self.stop_loss <= self.entry_price:
                    raise ValueError(
                        f"SELL stop_loss ({self.stop_loss}) must be strictly above entry_price ({self.entry_price})."
                    )

            # Validate Take Profit geometry
            tps = [tp for tp in (self.tp1, self.tp2, self.tp3) if tp is not None]
            for idx, tp in enumerate(tps, 1):
                if not math.isfinite(tp) or tp <= 0:
                    raise ValueError(f"take_profit level must be positive and finite, got: {tp}")
                if self.direction == Direction.BUY and tp <= self.entry_price:
                    raise ValueError(f"BUY take_profit ({tp}) must be strictly above entry_price ({self.entry_price}).")
                if self.direction == Direction.SELL and tp >= self.entry_price:
                    raise ValueError(f"SELL take_profit ({tp}) must be strictly below entry_price ({self.entry_price}).")

            # Validate TP ordering
            if self.direction == Direction.BUY:
                if self.tp1 is not None and self.tp2 is not None and self.tp1 >= self.tp2:
                    raise ValueError(f"BUY TP1 ({self.tp1}) must be less than TP2 ({self.tp2}).")
                if self.tp2 is not None and self.tp3 is not None and self.tp2 >= self.tp3:
                    raise ValueError(f"BUY TP2 ({self.tp2}) must be less than TP3 ({self.tp3}).")
            elif self.direction == Direction.SELL:
                if self.tp1 is not None and self.tp2 is not None and self.tp1 <= self.tp2:
                    raise ValueError(f"SELL TP1 ({self.tp1}) must be greater than TP2 ({self.tp2}).")
                if self.tp2 is not None and self.tp3 is not None and self.tp2 <= self.tp3:
                    raise ValueError(f"SELL TP2 ({self.tp2}) must be greater than TP3 ({self.tp3}).")

            # Validate trailing stop
            if self.trailing_stop is not None:
                if not math.isfinite(self.trailing_stop) or self.trailing_stop <= 0:
                    raise ValueError("trailing_stop must be positive and finite if provided.")
        else:
            # NO_TRADE
            if self.stop_loss is not None or self.tp1 is not None:
                raise ValueError("NO_TRADE decision must not carry active stop_loss or take_profit levels.")

        payload = {
            "decision_id": self.decision_id,
            "candidate_id": self.candidate_id,
            "evidence_id": self.evidence_id,
            "symbol": self.symbol.upper(),
            "timeframe": self.timeframe,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "trailing_stop": self.trailing_stop,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        risk_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        object.__setattr__(self, "risk_id", f"risk_{risk_hash}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "risk_id": self.risk_id,
            "decision_id": self.decision_id,
            "candidate_id": self.candidate_id,
            "evidence_id": self.evidence_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "trailing_stop": self.trailing_stop,
            "risk_reward_ratio": self.risk_reward_ratio,
        }


def calculate_production_risk_levels(
    decision: ProductionDecision,
    candidate: PromotedCandidateArtifact,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    tp1_multiplier: float = 1.0,
    tp2_multiplier: float = 2.0,
    tp3_multiplier: float = 3.0,
) -> ProductionRiskLevels:
    """Calculate risk levels from authoritative decision and candidate configuration."""
    if not isinstance(decision, ProductionDecision):
        raise TypeError("decision must be a ProductionDecision instance.")
    if not isinstance(candidate, PromotedCandidateArtifact):
        raise TypeError("candidate must be a PromotedCandidateArtifact instance.")

    if decision.direction == Direction.NO_TRADE or decision.entry_price is None:
        return ProductionRiskLevels(
            decision_id=decision.decision_id,
            candidate_id=decision.candidate_id,
            evidence_id=decision.evidence_id,
            symbol=decision.symbol,
            timeframe=decision.timeframe,
            direction=Direction.NO_TRADE,
            entry_price=None,
            stop_loss=None,
            tp1=None,
            tp2=None,
            tp3=None,
            trailing_stop=None,
            risk_reward_ratio=None,
        )

    # Obtain stop_loss_pct and take_profit_pct from candidate parameters or arguments
    sl_pct = (
        stop_loss_pct
        if stop_loss_pct is not None
        else candidate.parameters.get("stop_loss_pct")
    )
    tp_pct = (
        take_profit_pct
        if take_profit_pct is not None
        else candidate.parameters.get("take_profit_pct")
    )

    if sl_pct is None:
        raise ValueError(
            f"Candidate '{candidate.candidate_id}' parameters missing required 'stop_loss_pct' risk parameter."
        )
    if tp_pct is None:
        raise ValueError(
            f"Candidate '{candidate.candidate_id}' parameters missing required 'take_profit_pct' risk parameter."
        )

    if not math.isfinite(sl_pct) or sl_pct <= 0:
        raise ValueError("stop_loss_pct must be a positive finite float.")
    if not math.isfinite(tp_pct) or tp_pct <= 0:
        raise ValueError("take_profit_pct must be a positive finite float.")

    tp1_mult = (
        candidate.parameters.get("tp1_multiplier", tp1_multiplier)
        if tp1_multiplier == 1.0
        else tp1_multiplier
    )
    tp2_mult = (
        candidate.parameters.get("tp2_multiplier", tp2_multiplier)
        if tp2_multiplier == 2.0
        else tp2_multiplier
    )
    tp3_mult = (
        candidate.parameters.get("tp3_multiplier", tp3_multiplier)
        if tp3_multiplier == 3.0
        else tp3_multiplier
    )

    entry = decision.entry_price

    if decision.direction == Direction.BUY:
        stop_loss = entry * (1.0 - sl_pct)
        take_profit = entry * (1.0 + tp_pct)
        risk_dist = entry - stop_loss
        reward_dist = take_profit - entry
        rr_ratio = reward_dist / risk_dist if risk_dist > 0 else None

        tp1 = entry + (risk_dist * tp1_mult)
        tp2 = entry + (risk_dist * tp2_mult)
        tp3 = entry + (risk_dist * tp3_mult)
        trailing = candidate.parameters.get("trailing_stop_level")

    elif decision.direction == Direction.SELL:
        stop_loss = entry * (1.0 + sl_pct)
        take_profit = entry * (1.0 - tp_pct)
        risk_dist = stop_loss - entry
        reward_dist = entry - take_profit
        rr_ratio = reward_dist / risk_dist if risk_dist > 0 else None

        tp1 = entry - (risk_dist * tp1_mult)
        tp2 = entry - (risk_dist * tp2_mult)
        tp3 = entry - (risk_dist * tp3_mult)
        trailing = candidate.parameters.get("trailing_stop_level")

    else:
        raise ValueError(f"Unsupported direction for risk calculation: {decision.direction}")

    return ProductionRiskLevels(
        decision_id=decision.decision_id,
        candidate_id=decision.candidate_id,
        evidence_id=decision.evidence_id,
        symbol=decision.symbol,
        timeframe=decision.timeframe,
        direction=decision.direction,
        entry_price=entry,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        trailing_stop=trailing,
        risk_reward_ratio=rr_ratio,
    )


@dataclass(frozen=True)
class ProductionIntelligencePublication:
    """Canonical, versioned, immutable publication artifact for Project 2 delivery."""

    schema_version: str
    publication_id: str
    signal_id: str
    decision_id: str
    strategy_id: str
    candidate_id: str
    research_evidence_id: str
    research_fingerprint: str
    symbol: str
    timeframe: str
    decision_timestamp: str
    market_data_timestamp: str | None
    decision: str
    confidence: Optional[float]
    entry: Optional[float]
    invalidation: Optional[str]
    stop_loss: Optional[float]
    tp1: Optional[float]
    tp2: Optional[float]
    tp3: Optional[float]
    trailing_stop: Optional[float]
    risk_reward_ratio: Optional[float]
    operational_stability_score: float
    provenance: dict[str, Any]
    mtf_intelligence: Optional[MTFIntelligence] = None

    def __post_init__(self) -> None:
        if not self.schema_version or not self.schema_version.strip():
            raise ValueError("schema_version must be a non-empty string.")
        if not self.publication_id or not self.publication_id.strip():
            raise ValueError("publication_id must be a non-empty string.")
        if not self.signal_id or not self.signal_id.strip():
            raise ValueError("signal_id must be a non-empty string.")
        if not self.decision_id or not self.decision_id.strip():
            raise ValueError("decision_id must be a non-empty string.")
        if not self.strategy_id or not self.strategy_id.strip():
            raise ValueError("strategy_id must be a non-empty string.")
        if not self.candidate_id or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string.")
        if not self.research_evidence_id or not self.research_evidence_id.strip():
            raise ValueError("research_evidence_id must be a non-empty string.")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string.")
        if not self.timeframe or not self.timeframe.strip():
            raise ValueError("timeframe must be a non-empty string.")

        if isinstance(self.operational_stability_score, bool) or not isinstance(self.operational_stability_score, Real):
            raise ValueError("operational_stability_score must be a numeric float.")
        f_stab = float(self.operational_stability_score)
        if not math.isfinite(f_stab):
            raise ValueError("operational_stability_score must be finite (not NaN or infinity).")
        object.__setattr__(self, "operational_stability_score", f_stab)

        # Validate decision_timestamp as timezone-aware ISO-8601 string
        ts_dec_str = str(self.decision_timestamp).strip()
        if not ts_dec_str:
            raise ValueError("decision_timestamp must be a non-empty string.")
        try:
            dt_dec = datetime.fromisoformat(ts_dec_str)
            if dt_dec.tzinfo is None or dt_dec.tzinfo.utcoffset(dt_dec) is None:
                raise ValueError("decision_timestamp must be timezone-aware ISO-8601.")
        except Exception as exc:
            raise ValueError(f"Invalid decision_timestamp '{self.decision_timestamp}': {exc}")

        # Validate market_data_timestamp if present as timezone-aware ISO-8601 string
        if self.market_data_timestamp is not None and str(self.market_data_timestamp).strip():
            ts_mkt_str = str(self.market_data_timestamp).strip()
            try:
                dt_mkt = datetime.fromisoformat(ts_mkt_str)
                if dt_mkt.tzinfo is None or dt_mkt.tzinfo.utcoffset(dt_mkt) is None:
                    raise ValueError("market_data_timestamp must be timezone-aware ISO-8601.")
            except Exception as exc:
                raise ValueError(f"Invalid market_data_timestamp '{self.market_data_timestamp}': {exc}")

    @classmethod
    def from_artifacts(
        cls,
        decision: ProductionDecision,
        signal: ProductionSignal,
        risk: ProductionRiskLevels,
        candidate: PromotedCandidateArtifact,
        authorization: Optional[ProductionRuntimeAuthorization | ProductionAuthorizationReceipt] = None,
        confidence: Optional[float] = None,
        schema_version: str = "1.0",
        mtf_intelligence: Optional[MTFIntelligence] = None,
    ) -> ProductionIntelligencePublication:
        if not isinstance(decision, ProductionDecision):
            raise TypeError("decision must be a ProductionDecision instance.")
        if not isinstance(signal, ProductionSignal):
            raise TypeError("signal must be a ProductionSignal instance.")
        if not isinstance(risk, ProductionRiskLevels):
            raise TypeError("risk must be a ProductionRiskLevels instance.")
        if not isinstance(candidate, PromotedCandidateArtifact):
            raise TypeError("candidate must be a PromotedCandidateArtifact instance.")

        if signal.decision_id != decision.decision_id:
            raise ValueError(f"Signal decision_id '{signal.decision_id}' does not match decision ID '{decision.decision_id}'.")
        if risk.decision_id != decision.decision_id:
            raise ValueError(f"Risk decision_id '{risk.decision_id}' does not match decision ID '{decision.decision_id}'.")

        conf = confidence if confidence is not None else decision.confidence
        stab_score = candidate.operational_stability_score

        receipt = None
        if authorization is not None:
            receipt = (
                authorization
                if isinstance(authorization, ProductionAuthorizationReceipt)
                else ProductionAuthorizationReceipt.from_authorization(authorization)
            )

        cld_fp = getattr(decision, "canonical_live_decision_fingerprint", None) or getattr(signal, "canonical_live_decision_fingerprint", None)
        cld_state = getattr(decision, "current_lifecycle_state", None) or getattr(signal, "current_lifecycle_state", None)

        pub_raw = {
            "schema_version": schema_version,
            "signal_id": signal.signal_id,
            "decision_id": decision.decision_id,
            "strategy_id": candidate.strategy_name,
            "candidate_id": candidate.candidate_id,
            "research_evidence_id": candidate.evidence.evidence_id,
            "research_fingerprint": candidate.evidence.experiment_fingerprint,
            "symbol": decision.symbol.upper(),
            "timeframe": decision.timeframe,
            "market_data_timestamp": decision.market_timestamp,
            "decision": decision.direction.value,
            "confidence": conf,
            "entry": decision.entry_price,
            "invalidation": decision.invalidation_condition,
            "stop_loss": risk.stop_loss,
            "tp1": risk.tp1,
            "tp2": risk.tp2,
            "tp3": risk.tp3,
            "trailing_stop": risk.trailing_stop,
            "risk_reward_ratio": risk.risk_reward_ratio,
            "strategy_name": candidate.strategy_name,
            "strategy_version": candidate.strategy_version,
            "operational_stability_score": stab_score,
            "canonical_live_decision_fingerprint": str(cld_fp) if cld_fp else None,
            "runtime_authorization_fingerprint": receipt.authorization_fingerprint if receipt else None,
        }
        serialized = json.dumps(pub_raw, sort_keys=True, ensure_ascii=True)
        pub_id = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:32]

        provenance = {
            "source": "AI-Trading-Lab",
            "provenance_type": "live_signal",
            "is_live": True,
            "produced_at": decision.decision_timestamp,
            "candidate_id": candidate.candidate_id,
            "evidence_id": candidate.evidence.evidence_id,
            "experiment_fingerprint": candidate.evidence.experiment_fingerprint,
            "artifact_fingerprint": candidate.artifact_fingerprint,
            "policy_version": candidate.policy.policy_version,
            "strategy_name": candidate.strategy_name,
            "strategy_version": candidate.strategy_version,
            "operational_stability_score": stab_score,
            "campaign_selection_decision_fingerprint": candidate.campaign_selection_decision_fingerprint,
            "publication_contract_version": "1.0",
        }

        if candidate.governance_decision_fingerprint:
            provenance["governance_decision_fingerprint"] = candidate.governance_decision_fingerprint

        if receipt is not None:
            provenance.update({
                "runtime_authorization_fingerprint": receipt.authorization_fingerprint,
                "promoted_artifact_fingerprint": receipt.promoted_artifact_fingerprint,
                "governance_decision_fingerprint": receipt.governance_decision_fingerprint,
                "campaign_selection_decision_fingerprint": receipt.campaign_selection_decision_fingerprint,
                "authorization_policy_version": receipt.authorization_policy_version,
                "authorized_at_utc": receipt.authorized_at_utc,
            })

        if cld_fp:
            provenance["canonical_live_decision_fingerprint"] = str(cld_fp)
        if cld_state:
            provenance["current_lifecycle_state"] = str(cld_state)

        return cls(
            schema_version=schema_version,
            publication_id=pub_id,
            signal_id=signal.signal_id,
            decision_id=decision.decision_id,
            strategy_id=candidate.strategy_name,
            candidate_id=candidate.candidate_id,
            research_evidence_id=candidate.evidence.evidence_id,
            research_fingerprint=candidate.evidence.experiment_fingerprint,
            symbol=decision.symbol.upper(),
            timeframe=decision.timeframe,
            decision_timestamp=decision.decision_timestamp,
            market_data_timestamp=decision.market_timestamp,
            decision=decision.direction.value,
            confidence=conf,
            entry=decision.entry_price,
            invalidation=decision.invalidation_condition,
            stop_loss=risk.stop_loss,
            tp1=risk.tp1,
            tp2=risk.tp2,
            tp3=risk.tp3,
            trailing_stop=risk.trailing_stop,
            risk_reward_ratio=risk.risk_reward_ratio,
            operational_stability_score=stab_score,
            provenance=provenance,
            mtf_intelligence=mtf_intelligence,
        )

    def as_dict(self) -> dict[str, Any]:
        res = {
            "schema_version": self.schema_version,
            "publication_id": self.publication_id,
            "signal_id": self.signal_id,
            "decision_id": self.decision_id,
            "strategy_id": self.strategy_id,
            "candidate_id": self.candidate_id,
            "research_evidence_id": self.research_evidence_id,
            "research_fingerprint": self.research_fingerprint,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "decision_timestamp": self.decision_timestamp,
            "market_data_timestamp": self.market_data_timestamp,
            "decision": self.decision,
            "confidence": self.confidence,
            "entry": self.entry,
            "invalidation": self.invalidation,
            "stop_loss": self.stop_loss,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "trailing_stop": self.trailing_stop,
            "risk_reward_ratio": self.risk_reward_ratio,
            "operational_stability_score": self.operational_stability_score,
            "provenance": dict(self.provenance),
        }
        if self.mtf_intelligence is not None:
            res["mtf"] = (
                self.mtf_intelligence.as_dict()
                if hasattr(self.mtf_intelligence, "as_dict") and callable(self.mtf_intelligence.as_dict)
                else self.mtf_intelligence
            )
        return res

    def _as_dict_legacy(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "publication_id": self.publication_id,
            "signal_id": self.signal_id,
            "decision_id": self.decision_id,
            "strategy_id": self.strategy_id,
            "candidate_id": self.candidate_id,
            "research_evidence_id": self.research_evidence_id,
            "research_fingerprint": self.research_fingerprint,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "decision_timestamp": self.decision_timestamp,
            "market_data_timestamp": self.market_data_timestamp,
            "decision": self.decision,
            "confidence": self.confidence,
            "entry": self.entry,
            "invalidation": self.invalidation,
            "stop_loss": self.stop_loss,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "trailing_stop": self.trailing_stop,
            "risk_reward_ratio": self.risk_reward_ratio,
            "operational_stability_score": self.operational_stability_score,
            "provenance": dict(self.provenance),
        }

    def to_contract_v1_payload(self) -> dict[str, Any]:
        """Convert publication artifact into Contract v1.0 payload dict for Project 2."""
        if not self.market_data_timestamp or not str(self.market_data_timestamp).strip():
            raise ValueError("market_data_timestamp is required for Contract v1.0 payload delivery.")

        # Validate market_data_timestamp as timezone-aware ISO-8601
        ts_mkt_str = str(self.market_data_timestamp).strip()
        try:
            dt_ts = datetime.fromisoformat(ts_mkt_str)
            if dt_ts.tzinfo is None or dt_ts.tzinfo.utcoffset(dt_ts) is None:
                raise ValueError("market_data_timestamp must be timezone-aware ISO-8601.")
        except Exception as exc:
            raise ValueError(f"Invalid market_data_timestamp '{self.market_data_timestamp}': {exc}")

        prov = dict(self.provenance)
        prov["publication_contract_version"] = "1.0"

        # Deterministic derivation of take_profit from canonical TP values:
        # Uses TP2 if set, else TP1, else None.
        tp_take = self.tp2 if self.tp2 is not None else (self.tp1 if self.tp1 is not None else None)

        payload = {
            "contract_version": self.schema_version,
            "event_id": self.publication_id,
            "event_type": "TRADING_SIGNAL",
            "timestamp": self.market_data_timestamp,
            "instrument": {
                "symbol": self.symbol,
                "interval": self.timeframe,
            },
            "signal": {
                "publication_id": self.publication_id,
                "signal_id": self.signal_id,
                "decision_id": self.decision_id,
                "decision": self.decision,
                "strategy": self.strategy_id,
                "candidate_id": self.candidate_id,
                "confidence": self.confidence,
                "stability_score": self.operational_stability_score,
                "invalidation": self.invalidation,
                "signal_label": self.decision,
                "trend": "BULLISH" if self.decision == "BUY" else ("BEARISH" if self.decision == "SELL" else "NEUTRAL"),
            },
            "trade_setup": {
                "entry_price": self.entry,
                "stop_loss": self.stop_loss,
                "tp1": self.tp1,
                "tp2": self.tp2,
                "tp3": self.tp3,
                "take_profit": tp_take,
                "risk_reward_ratio": self.risk_reward_ratio,
                "trailing_stop": self.trailing_stop,
            },
            "provenance": prov,
        }
        if self.mtf_intelligence is not None:
            payload["mtf"] = (
                self.mtf_intelligence.as_dict()
                if hasattr(self.mtf_intelligence, "as_dict") and callable(self.mtf_intelligence.as_dict)
                else self.mtf_intelligence
            )
        return payload


# Legacy wrapper function for backward compatibility with existing codebase/tests
DEFAULT_MIN_STABILITY_SCORE = 0.50
DEFAULT_SYMBOL = "XAUUSD"
DEFAULT_INTERVAL = "5m"


def _validate_score(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number.")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return value


def _validate_strategy(stable_strategy: str) -> str:
    if not isinstance(stable_strategy, str):
        raise ValueError("stable_strategy must be a string.")
    strategy = stable_strategy.strip()
    if not strategy:
        raise ValueError("stable_strategy must not be empty.")
    return strategy


def build_live_production_decision(
    data: pd.DataFrame,
    *,
    reference_now: Optional[Any] = None,
    stable_strategy: str,
    stability_score: float,
    min_stability_score: float = DEFAULT_MIN_STABILITY_SCORE,
    momentum_window: Optional[int] = None,
    fast_window: int = 20,
    slow_window: int = 50,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    candidate_id: Optional[str] = None,
    research_dir: Optional[Any] = None,
) -> dict[str, Any]:
    """Operational wrapper delegating directly to authoritative candidate resolution, decision evaluation, and risk calculation."""
    from live_trend import build_live_trend_snapshot
    from src.evaluation.research_store import DEFAULT_RESEARCH_DIR, resolve_promoted_candidate

    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("data must not be empty.")

    stable_strategy = _validate_strategy(stable_strategy)
    stability_score = _validate_score("stability_score", stability_score)
    min_stability_score = _validate_score("min_stability_score", min_stability_score)

    trend_snapshot = build_live_trend_snapshot(
        data,
        fast_window=fast_window,
        slow_window=slow_window,
    )

    r_dir = research_dir if research_dir is not None else DEFAULT_RESEARCH_DIR
    resolved_candidate = resolve_promoted_candidate(
        candidate_id=candidate_id,
        strategy_id=stable_strategy,
        symbol=symbol,
        timeframe=interval,
        base_dir=r_dir,
    )

    if resolved_candidate is None:
        raise ValueError(
            f"No authoritative promoted candidate resolved for strategy '{stable_strategy}' "
            f"(candidate_id={candidate_id!r}, symbol={symbol!r}, timeframe={interval!r}). "
            f"Operational production path fails closed."
        )

    # Strategy and stability score caller assertions against authoritative candidate lineage
    if stable_strategy != resolved_candidate.strategy_name:
        raise ValueError(
            f"stable_strategy '{stable_strategy}' conflicts with candidate's authoritative strategy_name '{resolved_candidate.strategy_name}'."
        )

    if abs(stability_score - resolved_candidate.operational_stability_score) > 1e-9:
        raise ValueError(
            f"Caller-supplied stability_score ({stability_score}) conflicts with candidate operational_stability_score ({resolved_candidate.operational_stability_score})."
        )

    ref_now = reference_now

    # Authorize runtime execution through canonical path
    authorization = authorize_production_runtime(
        resolved_candidate,
        symbol=symbol,
        timeframe=interval,
        now=ref_now,
    )
    receipt = ProductionAuthorizationReceipt.from_authorization(authorization)

    # Authoritative evaluation through promoted candidate
    decision_obj = evaluate_production_decision(
        candidate=resolved_candidate,
        data=data,
        reference_now=ref_now,
        max_age_seconds=float("inf"),
    )

    from live_signal import generate_live_signal
    eff_window = resolved_candidate.parameters.get("momentum_window", resolved_candidate.parameters.get("window", momentum_window or 10))
    sig_df = generate_live_signal(data, window=int(eff_window))
    sig_val = int(sig_df["signal"].iloc[-1])

    # Apply operational gating criteria (stability threshold, unsupported strategy, trend)
    if stability_score < min_stability_score:
        reason = "stability_score_below_threshold"
        final_direction = Direction.NO_TRADE
    elif stable_strategy != "momentum":
        reason = "stable_strategy_not_supported_by_live_signal"
        final_direction = Direction.NO_TRADE
    elif trend_snapshot["trend"] != "UP":
        reason = "trend_not_confirmed"
        final_direction = Direction.NO_TRADE
    elif sig_val != 1:
        reason = "live_signal_not_active"
        final_direction = Direction.NO_TRADE
    else:
        reason = "stable_strategy_live_signal_and_trend_confirmed"
        final_direction = Direction.BUY

    now_ts = decision_obj.decision_timestamp
    market_ts = decision_obj.market_timestamp
    close_price = float(data["close"].iloc[-1]) if "close" in data.columns else None

    final_decision_obj = ProductionDecision(
        candidate_id=resolved_candidate.candidate_id,
        evidence_id=resolved_candidate.evidence.evidence_id,
        experiment_fingerprint=resolved_candidate.evidence.experiment_fingerprint,
        symbol=resolved_candidate.symbol,
        timeframe=resolved_candidate.timeframe,
        decision_timestamp=now_ts,
        market_timestamp=market_ts,
        direction=final_direction,
        reason=reason,
        entry_price=close_price if final_direction == Direction.BUY else None,
        invalidation_condition="Close below stop_loss or trend turns DOWN" if final_direction == Direction.BUY else None,
        confidence=stability_score,
        parameters=resolved_candidate.parameters,
    )

    risk_obj = calculate_production_risk_levels(
        decision=final_decision_obj,
        candidate=resolved_candidate,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )

    effective_momentum_window = resolved_candidate.parameters.get(
        "momentum_window",
        resolved_candidate.parameters.get("window", momentum_window or 10),
    )
    effective_sl_pct = resolved_candidate.parameters.get("stop_loss_pct", stop_loss_pct)
    effective_tp_pct = resolved_candidate.parameters.get("take_profit_pct", take_profit_pct)

    return {
        "symbol": symbol,
        "interval": interval,
        "decision": final_decision_obj.direction.value,
        "reason": final_decision_obj.reason,
        "stable_strategy": stable_strategy,
        "stability_score": stability_score,
        "min_stability_score": min_stability_score,
        "strategy_supported": stable_strategy == "momentum",
        "signal": sig_val if stable_strategy == "momentum" else 0,
        "signal_label": "BUY" if (sig_val == 1 and stable_strategy == "momentum") else "NO TRADE",
        "trend": str(trend_snapshot["trend"]),
        "momentum": float(data["close"].iloc[-1]),
        "entry_price": risk_obj.entry_price,
        "stop_loss": risk_obj.stop_loss,
        "take_profit": risk_obj.tp2 if risk_obj.tp2 is not None else risk_obj.tp1,
        "risk_reward_ratio": risk_obj.risk_reward_ratio,
        "stop_loss_pct": effective_sl_pct,
        "take_profit_pct": effective_tp_pct,
        "momentum_window": effective_momentum_window,
        "fast_window": fast_window,
        "slow_window": slow_window,
        "timestamp": final_decision_obj.market_timestamp,
        "candidate_id": resolved_candidate.candidate_id,
        "evidence_id": resolved_candidate.evidence.evidence_id,
        "experiment_fingerprint": resolved_candidate.evidence.experiment_fingerprint,
        "runtime_authorization_fingerprint": receipt.authorization_fingerprint,
        "authorization_policy_version": receipt.authorization_policy_version,
        "authorized_at_utc": receipt.authorized_at_utc,
        "promoted_artifact_fingerprint": receipt.promoted_artifact_fingerprint,
        "governance_decision_fingerprint": receipt.governance_decision_fingerprint,
        "campaign_selection_decision_fingerprint": receipt.campaign_selection_decision_fingerprint,
    }
