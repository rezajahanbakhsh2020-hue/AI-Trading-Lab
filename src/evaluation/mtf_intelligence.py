"""Canonical Multi-Timeframe (MTF) Intelligence Module for Project 1.

Provides authoritative multi-timeframe signal aggregation, alignment coverage,
and classification without mutating lower-timeframe decision authority.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from src.evaluation.live_decision_lifecycle import CanonicalLiveDecision
from src.evaluation.live_production_decision import (
    Direction,
    ProductionAuthorizationReceipt,
    ProductionDecision,
    ProductionIntelligencePublication,
    ProductionRiskLevels,
    ProductionRuntimeAuthorization,
    ProductionSignal,
    PromotedCandidateArtifact,
    authorize_production_runtime,
)

logger = logging.getLogger(__name__)


class CanonicalTimeframe(str, Enum):
    """Canonical strongly typed domain representation for supported timeframes in Project 1."""

    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1H"
    FOUR_HOURS = "4H"
    ONE_DAY = "1D"

    @property
    def order_rank(self) -> int:
        """Deterministic ordering index across the six supported timeframe identities."""
        ranks = {
            CanonicalTimeframe.FIVE_MINUTES: 1,
            CanonicalTimeframe.FIFTEEN_MINUTES: 2,
            CanonicalTimeframe.THIRTY_MINUTES: 3,
            CanonicalTimeframe.ONE_HOUR: 4,
            CanonicalTimeframe.FOUR_HOURS: 5,
            CanonicalTimeframe.ONE_DAY: 6,
        }
        return ranks[self]

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, CanonicalTimeframe):
            try:
                other = CanonicalTimeframe.from_str(other)
            except Exception:
                return NotImplemented
        return self.order_rank < other.order_rank

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, CanonicalTimeframe):
            try:
                other = CanonicalTimeframe.from_str(other)
            except Exception:
                return NotImplemented
        return self.order_rank <= other.order_rank

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, CanonicalTimeframe):
            try:
                other = CanonicalTimeframe.from_str(other)
            except Exception:
                return NotImplemented
        return self.order_rank > other.order_rank

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, CanonicalTimeframe):
            try:
                other = CanonicalTimeframe.from_str(other)
            except Exception:
                return NotImplemented
        return self.order_rank >= other.order_rank

    @classmethod
    def from_str(cls, value: Union[str, CanonicalTimeframe]) -> CanonicalTimeframe:
        """Parse string or CanonicalTimeframe into a canonical timeframe. Fails closed on unknown values."""
        if isinstance(value, CanonicalTimeframe):
            return value
        if value is None or not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid timeframe value: {value!r}. Must be a non-empty string.")

        s = value.strip()
        m = {
            "5m": cls.FIVE_MINUTES,
            "15m": cls.FIFTEEN_MINUTES,
            "30m": cls.THIRTY_MINUTES,
            "1h": cls.ONE_HOUR,
            "1H": cls.ONE_HOUR,
            "4h": cls.FOUR_HOURS,
            "4H": cls.FOUR_HOURS,
            "1d": cls.ONE_DAY,
            "1D": cls.ONE_DAY,
        }
        if s in m:
            return m[s]
        raise ValueError(
            f"Unknown or unsupported timeframe '{value}'. Canonical timeframes are: 5m, 15m, 30m, 1H, 4H, 1D."
        )

    @classmethod
    def canonical_ladder(cls) -> Tuple[CanonicalTimeframe, ...]:
        """Return the complete canonical six-timeframe ladder in ascending order."""
        return (
            cls.FIVE_MINUTES,
            cls.FIFTEEN_MINUTES,
            cls.THIRTY_MINUTES,
            cls.ONE_HOUR,
            cls.FOUR_HOURS,
            cls.ONE_DAY,
        )


@dataclass(frozen=True)
class PerTimeframeSignal:
    """Immutable projection of an authoritative per-timeframe live decision/signal.

    Preserves full lineage and timeframe signal identity. Distinct timeframes produce
    distinct signal identities even if all other decision metadata is identical.
    """

    symbol: str
    timeframe: CanonicalTimeframe
    direction: Direction
    decision_id: str
    signal_id: Optional[str]
    decision_timestamp: str
    market_timestamp: Optional[str]
    strategy_name: str
    strategy_version: str
    candidate_id: str
    evidence_id: str
    experiment_fingerprint: str
    canonical_live_decision_fingerprint: str
    authorization_fingerprint: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    constituent_fingerprint: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if not self.symbol or not str(self.symbol).strip():
            raise ValueError("symbol must be a non-empty string.")
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())

        tf = CanonicalTimeframe.from_str(self.timeframe)
        object.__setattr__(self, "timeframe", tf)

        if not isinstance(self.direction, Direction):
            if isinstance(self.direction, str):
                dir_str = self.direction.strip().upper()
                if dir_str == "BUY":
                    object.__setattr__(self, "direction", Direction.BUY)
                elif dir_str == "SELL":
                    object.__setattr__(self, "direction", Direction.SELL)
                elif dir_str in ("NO_TRADE", "NO TRADE"):
                    object.__setattr__(self, "direction", Direction.NO_TRADE)
                else:
                    raise ValueError(f"Invalid direction '{self.direction}'.")
            else:
                raise TypeError(f"direction must be a Direction enum, got {type(self.direction).__name__}")

        if not self.decision_id or not str(self.decision_id).strip():
            raise ValueError("decision_id must be a non-empty string.")
        if not self.candidate_id or not str(self.candidate_id).strip():
            raise ValueError("candidate_id must be a non-empty string.")
        if not self.strategy_name or not str(self.strategy_name).strip():
            raise ValueError("strategy_name must be a non-empty string.")
        if not self.strategy_version or not str(self.strategy_version).strip():
            raise ValueError("strategy_version must be a non-empty string.")
        if not self.evidence_id or not str(self.evidence_id).strip():
            raise ValueError("evidence_id must be a non-empty string.")
        if not self.experiment_fingerprint or not str(self.experiment_fingerprint).strip():
            raise ValueError("experiment_fingerprint must be a non-empty string.")
        if not self.canonical_live_decision_fingerprint or not str(self.canonical_live_decision_fingerprint).strip():
            raise ValueError("canonical_live_decision_fingerprint must be a non-empty string.")
        if not self.authorization_fingerprint or not str(self.authorization_fingerprint).strip():
            raise ValueError("authorization_fingerprint must be a non-empty string.")
        if not self.decision_timestamp or not str(self.decision_timestamp).strip():
            raise ValueError("decision_timestamp must be a non-empty string.")

        # Ensure provenance is immutable via MappingProxyType
        if isinstance(self.provenance, Mapping):
            object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

        # Compute deterministic constituent_fingerprint incorporating timeframe and full signal identity
        fp_payload = {
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "direction": self.direction.value,
            "decision_id": self.decision_id,
            "signal_id": self.signal_id,
            "decision_timestamp": self.decision_timestamp,
            "market_timestamp": self.market_timestamp,
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "candidate_id": self.candidate_id,
            "evidence_id": self.evidence_id,
            "experiment_fingerprint": self.experiment_fingerprint,
            "canonical_live_decision_fingerprint": self.canonical_live_decision_fingerprint,
            "authorization_fingerprint": self.authorization_fingerprint,
        }
        serialized = json.dumps(fp_payload, sort_keys=True, ensure_ascii=True)
        fp = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        object.__setattr__(self, "constituent_fingerprint", fp)

    @classmethod
    def from_canonical_live_decision(
        cls,
        cld: CanonicalLiveDecision,
        candidate: Optional[PromotedCandidateArtifact] = None,
    ) -> PerTimeframeSignal:
        """Construct PerTimeframeSignal projection strictly from an authoritative CanonicalLiveDecision."""
        if not isinstance(cld, CanonicalLiveDecision):
            raise TypeError(f"cld must be a CanonicalLiveDecision instance, got {type(cld).__name__}")

        decision = cld.decision
        signal = cld.signal
        receipt = cld.authorization_receipt

        tf = CanonicalTimeframe.from_str(decision.timeframe)
        prov = {
            "source": "AI-Trading-Lab",
            "provenance_type": "per_timeframe_signal",
            "decision_id": decision.decision_id,
            "signal_id": signal.signal_id,
            "canonical_live_decision_fingerprint": cld.canonical_live_decision_fingerprint,
            "authorization_fingerprint": receipt.authorization_fingerprint,
            "candidate_id": receipt.candidate_id,
            "governance_decision_fingerprint": receipt.governance_decision_fingerprint,
            "campaign_selection_decision_fingerprint": receipt.campaign_selection_decision_fingerprint,
            "current_lifecycle_state": cld.current_state.value,
        }
        if candidate is not None:
            prov["artifact_fingerprint"] = candidate.artifact_fingerprint

        return cls(
            symbol=decision.symbol.upper(),
            timeframe=tf,
            direction=decision.direction,
            decision_id=decision.decision_id,
            signal_id=signal.signal_id,
            decision_timestamp=decision.decision_timestamp,
            market_timestamp=decision.market_timestamp,
            strategy_name=receipt.strategy_name,
            strategy_version=receipt.strategy_version,
            candidate_id=receipt.candidate_id,
            evidence_id=decision.evidence_id,
            experiment_fingerprint=decision.experiment_fingerprint,
            canonical_live_decision_fingerprint=cld.canonical_live_decision_fingerprint,
            authorization_fingerprint=receipt.authorization_fingerprint,
            provenance=prov,
        )

    def as_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary representation."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "direction": self.direction.value,
            "decision_id": self.decision_id,
            "signal_id": self.signal_id,
            "decision_timestamp": self.decision_timestamp,
            "market_timestamp": self.market_timestamp,
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "candidate_id": self.candidate_id,
            "evidence_id": self.evidence_id,
            "experiment_fingerprint": self.experiment_fingerprint,
            "canonical_live_decision_fingerprint": self.canonical_live_decision_fingerprint,
            "authorization_fingerprint": self.authorization_fingerprint,
            "constituent_fingerprint": self.constituent_fingerprint,
            "provenance": dict(self.provenance),
        }


class MTFClassification(str, Enum):
    """Strongly typed classification for MTF intelligence signal context."""

    ALIGNED = "ALIGNED"
    COUNTER_TREND = "COUNTER_TREND"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"


@dataclass(frozen=True)
class HigherTimeframeContext:
    """Explicit, lineage-bearing higher-timeframe context representation."""

    local_timeframe: CanonicalTimeframe
    present_higher_timeframes: Tuple[CanonicalTimeframe, ...]
    missing_higher_timeframes: Tuple[CanonicalTimeframe, ...]
    agreed_higher_timeframes: Tuple[CanonicalTimeframe, ...]
    disagreed_higher_timeframes: Tuple[CanonicalTimeframe, ...]
    context_direction: str

    def __post_init__(self) -> None:
        ltf = CanonicalTimeframe.from_str(self.local_timeframe)
        object.__setattr__(self, "local_timeframe", ltf)
        object.__setattr__(
            self,
            "present_higher_timeframes",
            tuple(sorted(set(CanonicalTimeframe.from_str(tf) for tf in self.present_higher_timeframes))),
        )
        object.__setattr__(
            self,
            "missing_higher_timeframes",
            tuple(sorted(set(CanonicalTimeframe.from_str(tf) for tf in self.missing_higher_timeframes))),
        )
        object.__setattr__(
            self,
            "agreed_higher_timeframes",
            tuple(sorted(set(CanonicalTimeframe.from_str(tf) for tf in self.agreed_higher_timeframes))),
        )
        object.__setattr__(
            self,
            "disagreed_higher_timeframes",
            tuple(sorted(set(CanonicalTimeframe.from_str(tf) for tf in self.disagreed_higher_timeframes))),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "local_timeframe": self.local_timeframe.value,
            "present_higher_timeframes": [tf.value for tf in self.present_higher_timeframes],
            "missing_higher_timeframes": [tf.value for tf in self.missing_higher_timeframes],
            "agreed_higher_timeframes": [tf.value for tf in self.agreed_higher_timeframes],
            "disagreed_higher_timeframes": [tf.value for tf in self.disagreed_higher_timeframes],
            "context_direction": self.context_direction,
        }


@dataclass(frozen=True)
class MTFIntelligence:
    """Immutable domain model for authoritative Project 1 Multi-Timeframe Intelligence."""

    symbol: str
    local_timeframe: CanonicalTimeframe
    participating_timeframes: Tuple[CanonicalTimeframe, ...]
    signals: Tuple[PerTimeframeSignal, ...]
    alignment_count: int
    alignment_coverage: int
    classification: MTFClassification
    higher_timeframe_context: HigherTimeframeContext
    constituent_fingerprints: Tuple[str, ...]
    matching_signal_count: int = 0
    available_signal_count: int = 0
    intelligence_fingerprint: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if not self.symbol or not str(self.symbol).strip():
            raise ValueError("symbol must be a non-empty string.")
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())

        ltf = CanonicalTimeframe.from_str(self.local_timeframe)
        object.__setattr__(self, "local_timeframe", ltf)

        ptf = tuple(CanonicalTimeframe.from_str(tf) for tf in self.participating_timeframes)
        object.__setattr__(self, "participating_timeframes", ptf)

        if not isinstance(self.signals, tuple):
            object.__setattr__(self, "signals", tuple(self.signals))

        for sig in self.signals:
            if not isinstance(sig, PerTimeframeSignal):
                raise TypeError(f"All items in signals must be PerTimeframeSignal, got {type(sig).__name__}")

        if not isinstance(self.classification, MTFClassification):
            if isinstance(self.classification, str):
                object.__setattr__(self, "classification", MTFClassification(self.classification.strip().upper()))
            else:
                raise TypeError("classification must be a MTFClassification enum instance.")

        if not isinstance(self.higher_timeframe_context, HigherTimeframeContext):
            raise TypeError("higher_timeframe_context must be a HigherTimeframeContext instance.")

        if not (1 <= self.alignment_coverage <= 6):
            raise ValueError(f"alignment_coverage must be between 1 and 6, got {self.alignment_coverage}.")

        if self.alignment_count < 0:
            raise ValueError(f"alignment_count must be non-negative, got {self.alignment_count}.")

        cfps = tuple(sig.constituent_fingerprint for sig in self.signals)
        object.__setattr__(self, "constituent_fingerprints", cfps)

        # Compute deterministic intelligence fingerprint
        fp_payload = {
            "symbol": self.symbol,
            "local_timeframe": self.local_timeframe.value,
            "participating_timeframes": [tf.value for tf in self.participating_timeframes],
            "alignment_count": self.alignment_count,
            "alignment_coverage": self.alignment_coverage,
            "matching_signal_count": self.matching_signal_count,
            "available_signal_count": self.available_signal_count,
            "classification": self.classification.value,
            "higher_timeframe_context": self.higher_timeframe_context.as_dict(),
            "constituent_fingerprints": list(cfps),
        }
        serialized = json.dumps(fp_payload, sort_keys=True, ensure_ascii=True)
        fp = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        object.__setattr__(self, "intelligence_fingerprint", fp)

    @property
    def star_representation(self) -> str:
        """Authoritative star representation of contiguous MTF alignment coverage (1..6)."""
        return "⭐" * self.alignment_coverage

    def as_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary boundary representation."""
        return {
            "symbol": self.symbol,
            "local_timeframe": self.local_timeframe.value,
            "participating_timeframes": [tf.value for tf in self.participating_timeframes],
            "signals": [sig.as_dict() for sig in self.signals],
            "alignment_count": self.alignment_count,
            "alignment_coverage": self.alignment_coverage,
            "matching_signal_count": self.matching_signal_count,
            "available_signal_count": self.available_signal_count,
            "star_representation": self.star_representation,
            "classification": self.classification.value,
            "higher_timeframe_context": self.higher_timeframe_context.as_dict(),
            "constituent_fingerprints": list(self.constituent_fingerprints),
            "intelligence_fingerprint": self.intelligence_fingerprint,
        }

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Alias for as_dict for consistency with project canonical serialization."""
        return self.as_dict()


def build_mtf_intelligence(
    signals: Sequence[PerTimeframeSignal],
    *,
    local_timeframe: Union[str, CanonicalTimeframe],
) -> MTFIntelligence:
    """Pure, deterministic function computing multi-timeframe intelligence.

    Validation rules:
    - Fails closed on empty signals list.
    - Fails closed on duplicate timeframes in signals.
    - Fails closed on mismatched symbols across signals.
    - Fails closed if local_timeframe is missing from signals.
    - Contiguous alignment coverage starts from local_timeframe along 5m->15m->30m->1H->4H->1D.
    - Missing intermediate timeframes break contiguous alignment coverage.
    - Higher-timeframe disagreement NEVER invalidates or erases lower-timeframe BUY/SELL decision.
    """
    if not signals:
        raise ValueError("signals sequence must not be empty.")

    local_tf = CanonicalTimeframe.from_str(local_timeframe)

    timeframe_map: Dict[CanonicalTimeframe, PerTimeframeSignal] = {}
    symbols_seen: List[str] = []

    for sig in signals:
        if not isinstance(sig, PerTimeframeSignal):
            raise TypeError(f"All elements in signals must be PerTimeframeSignal, got {type(sig).__name__}")

        if sig.timeframe in timeframe_map:
            raise ValueError(
                f"Duplicate timeframe signal provided for timeframe '{sig.timeframe.value}'."
            )
        timeframe_map[sig.timeframe] = sig
        symbols_seen.append(sig.symbol)

    if len(set(symbols_seen)) > 1:
        raise ValueError(
            f"Mismatched symbols across signals: {sorted(set(symbols_seen))}."
        )

    if local_tf not in timeframe_map:
        raise ValueError(
            f"Local timeframe '{local_tf.value}' signal not found in provided signals."
        )

    local_signal = timeframe_map[local_tf]
    local_direction = local_signal.direction
    symbol = local_signal.symbol

    participating_tfs = tuple(sorted(timeframe_map.keys()))
    ordered_signals = tuple(timeframe_map[tf] for tf in participating_tfs)

    canonical_ladder = CanonicalTimeframe.canonical_ladder()
    canonical_htfs = tuple(tf for tf in canonical_ladder if tf > local_tf)
    present_htfs = tuple(tf for tf in participating_tfs if tf > local_tf)
    missing_htfs = tuple(tf for tf in canonical_htfs if tf not in present_htfs)

    agreed_htfs: List[CanonicalTimeframe] = []
    disagreed_htfs: List[CanonicalTimeframe] = []

    for tf in present_htfs:
        sig = timeframe_map[tf]
        if local_direction in (Direction.BUY, Direction.SELL) and sig.direction == local_direction:
            agreed_htfs.append(tf)
        else:
            disagreed_htfs.append(tf)

    agreed_htfs_tuple = tuple(sorted(agreed_htfs))
    disagreed_htfs_tuple = tuple(sorted(disagreed_htfs))

    if not present_htfs:
        context_direction = "INSUFFICIENT"
    else:
        htf_directions = set(timeframe_map[tf].direction for tf in present_htfs)
        if len(htf_directions) == 1:
            context_direction = next(iter(htf_directions)).value
        else:
            context_direction = "MIXED"

    htf_context = HigherTimeframeContext(
        local_timeframe=local_tf,
        present_higher_timeframes=present_htfs,
        missing_higher_timeframes=missing_htfs,
        agreed_higher_timeframes=agreed_htfs_tuple,
        disagreed_higher_timeframes=disagreed_htfs_tuple,
        context_direction=context_direction,
    )

    # Calculate matching_signal_count across all available signals
    matching_signal_count = (
        sum(1 for sig in ordered_signals if sig.direction == local_direction)
        if local_direction in (Direction.BUY, Direction.SELL)
        else 0
    )
    available_signal_count = len(ordered_signals)

    # Contiguous Ladder Coverage Semantics:
    # Start at local_tf in canonical ladder and count contiguous matching timeframes.
    # Any gap (missing timeframe or direction mismatch) breaks the contiguous prefix chain.
    start_index = canonical_ladder.index(local_tf)
    ladder_from_local = canonical_ladder[start_index:]

    contiguous_coverage = 0
    if local_direction in (Direction.BUY, Direction.SELL):
        for tf in ladder_from_local:
            if tf in timeframe_map and timeframe_map[tf].direction == local_direction:
                contiguous_coverage += 1
            else:
                break

    alignment_coverage = max(1, min(6, contiguous_coverage if contiguous_coverage >= 1 else 1))
    alignment_count = matching_signal_count

    if local_direction not in (Direction.BUY, Direction.SELL) or not present_htfs:
        classification = MTFClassification.INSUFFICIENT_CONTEXT
    else:
        if len(agreed_htfs_tuple) == len(present_htfs):
            classification = MTFClassification.ALIGNED
        else:
            classification = MTFClassification.COUNTER_TREND

    return MTFIntelligence(
        symbol=symbol,
        local_timeframe=local_tf,
        participating_timeframes=participating_tfs,
        signals=ordered_signals,
        alignment_count=alignment_count,
        alignment_coverage=alignment_coverage,
        matching_signal_count=matching_signal_count,
        available_signal_count=available_signal_count,
        classification=classification,
        higher_timeframe_context=htf_context,
        constituent_fingerprints=tuple(sig.constituent_fingerprint for sig in ordered_signals),
    )


@dataclass(frozen=True)
class MTFLiveRuntimeResult:
    """Immutable result from multi-timeframe live execution containing per-timeframe and aggregated MTF outputs."""

    local_result: Any
    mtf_intelligence: MTFIntelligence
    per_timeframe_results: Mapping[CanonicalTimeframe, Any]

    def __post_init__(self) -> None:
        if isinstance(self.per_timeframe_results, Mapping):
            object.__setattr__(
                self,
                "per_timeframe_results",
                MappingProxyType(dict(self.per_timeframe_results)),
            )


def evaluate_mtf_live_runtime(
    data_by_timeframe: Dict[Union[str, CanonicalTimeframe], pd.DataFrame],
    *,
    symbol: str = "XAUUSD",
    local_timeframe: Union[str, CanonicalTimeframe] = "5m",
    stable_strategy: str = "momentum",
    candidate_id: Optional[str] = None,
    research_dir: Any = None,
    store_path: Optional[Union[Path, str]] = None,
    publisher: Any = None,
    publish: bool = False,
    skip_if_no_trade: bool = False,
    persist: bool = True,
    reference_now: Optional[datetime] = None,
    actor: str = "mtf_live_runtime",
) -> MTFLiveRuntimeResult:
    """Orchestrates authoritative per-timeframe live evaluations and constructs MTF intelligence.

    Lifecycle semantics:
    Phase A (Evaluation):
      For each requested timeframe:
        -> resolve_promoted_candidate(timeframe)
        -> authorize_production_runtime(candidate, timeframe)
        -> create_authorized_runtime_context()
        -> create_live_market_evaluation()
        -> evaluate_authorized_live_runtime(..., persist=False, publish=False) [EXACTLY ONCE PER TIMEFRAME]
        -> PerTimeframeSignal projection

    Phase B (MTF Composition):
      -> build_mtf_intelligence(signals, local_timeframe) [PURE DETERMINISTIC DOMAIN CALCULATION]

    Phase C (Finalization):
      -> finalize_authorized_live_runtime(local_res, context=local_context, mtf_intelligence=mtf_intel)
         [FINALIZES THE EXISTING PHASE-A RESULT WITHOUT RE-EVALUATING MARKET OR STRATEGY LOGIC]
    """
    from src.evaluation.live_execution_runtime import validate_and_prepare_market_snapshot
    from src.evaluation.live_market_evaluation import create_live_market_evaluation
    from src.evaluation.live_production_decision import (
        ProductionAuthorizationReceipt,
        authorize_production_runtime,
    )
    from src.evaluation.live_runtime import (
        evaluate_authorized_live_runtime,
        finalize_authorized_live_runtime,
    )
    from src.evaluation.live_runtime_context import create_authorized_runtime_context
    from src.evaluation.research_store import (
        DEFAULT_RESEARCH_DIR,
        resolve_promoted_candidate,
    )

    if not data_by_timeframe or not isinstance(data_by_timeframe, dict):
        raise ValueError("data_by_timeframe must be a non-empty dict mapping timeframes to pandas DataFrames.")

    r_dir = research_dir if research_dir is not None else DEFAULT_RESEARCH_DIR
    local_tf = CanonicalTimeframe.from_str(local_timeframe)
    ref_now = reference_now if reference_now is not None else datetime.now(timezone.utc)
    if ref_now.tzinfo is None:
        ref_now = ref_now.replace(tzinfo=timezone.utc)

    per_tf_results: Dict[CanonicalTimeframe, Any] = {}
    per_tf_signals: List[PerTimeframeSignal] = []
    per_tf_contexts: Dict[CanonicalTimeframe, Any] = {}
    per_tf_evaluations: Dict[CanonicalTimeframe, Any] = {}
    per_tf_candidates: Dict[CanonicalTimeframe, Any] = {}

    # PHASE A: EVALUATION (EXACTLY ONCE PER TIMEFRAME)
    for tf_key, raw_df in data_by_timeframe.items():
        tf = CanonicalTimeframe.from_str(tf_key)
        prepared_df = validate_and_prepare_market_snapshot(raw_df, symbol=symbol)

        # 1. Resolve candidate FOR THAT EXACT TIMEFRAME
        candidate = resolve_promoted_candidate(
            candidate_id=candidate_id,
            strategy_id=stable_strategy,
            symbol=symbol,
            timeframe=tf.value,
            base_dir=r_dir,
        )
        if candidate is None:
            raise ValueError(
                f"No authoritative promoted candidate resolved for timeframe '{tf.value}'."
            )

        # 2. Authorize production runtime FOR THAT EXACT TIMEFRAME
        authorization = authorize_production_runtime(
            candidate,
            symbol=symbol,
            timeframe=tf.value,
            now=ref_now,
        )
        receipt = ProductionAuthorizationReceipt.from_authorization(authorization)

        # 3. Create authorized runtime context
        context = create_authorized_runtime_context(
            candidate=candidate,
            authorization=authorization,
            authorization_receipt=receipt,
        )

        # 4. Create live market evaluation
        evaluation = create_live_market_evaluation(
            data=prepared_df,
            context=context,
            reference_now=ref_now,
            max_age_seconds=300.0,
        )

        # 5. Evaluate authorized live runtime (Phase A evaluation: persist=False, publish=False)
        res = evaluate_authorized_live_runtime(
            data=prepared_df,
            evaluation=evaluation,
            context=context,
            stable_strategy=candidate.strategy_name,
            stability_score=candidate.operational_stability_score,
            min_stability_score=0.50,
            store_path=store_path,
            publisher=None,
            publish=False,
            skip_if_no_trade=skip_if_no_trade,
            persist=False,
            actor=actor,
        )

        per_tf_results[tf] = res
        per_tf_contexts[tf] = context
        per_tf_evaluations[tf] = evaluation
        per_tf_candidates[tf] = candidate

        cld = res.canonical_decision
        if cld is not None:
            ptf_sig = PerTimeframeSignal.from_canonical_live_decision(cld, candidate=candidate)
            per_tf_signals.append(ptf_sig)

    # PHASE B: MTF COMPOSITION
    mtf_intel = build_mtf_intelligence(per_tf_signals, local_timeframe=local_tf)

    # PHASE C: FINALIZATION (FINALIZE EXISTING PHASE-A LOCAL RESULT WITHOUT RE-EVALUATION)
    local_initial_res = per_tf_results[local_tf]
    local_context = per_tf_contexts[local_tf]

    final_local_res = finalize_authorized_live_runtime(
        local_initial_res,
        context=local_context,
        mtf_intelligence=mtf_intel,
        store_path=store_path,
        publisher=publisher,
        publish=publish,
        skip_if_no_trade=skip_if_no_trade,
        persist=persist,
        actor=actor,
    )

    per_tf_results[local_tf] = final_local_res

    return MTFLiveRuntimeResult(
        local_result=final_local_res,
        mtf_intelligence=mtf_intel,
        per_timeframe_results=per_tf_results,
    )
