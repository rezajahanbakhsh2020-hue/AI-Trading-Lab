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
            object.__setattr__(
                self,
                "governance_decision_fingerprint",
                getattr(self.governance_decision, "decision_fingerprint", None),
            )

        validate_promotion_eligibility(
            self.evidence,
            policy=self.policy,
            governance_decision=self.governance_decision,
            governance_decision_fingerprint=self.governance_decision_fingerprint,
        )

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
                f"Configured candidate_id '{current_candidate_id}'
