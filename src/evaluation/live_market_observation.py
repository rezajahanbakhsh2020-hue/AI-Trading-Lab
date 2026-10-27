"""Truthful per-timeframe market observation publication for timeframes without an authorized production candidate.

Each canonical timeframe is first-class. A timeframe whose production-candidate
resolution fails still owns its own real market-data slice, its own evaluation
result, and its own publication outcome. This module provides the publication
artifact for that outcome: a NO_TRADE market observation carrying real candle
timestamps, real decision timestamps, and explicit provenance that no authorized
candidate exists for the timeframe.

Contract rules:
- The decision is always exactly "NO TRADE". Without a qualified candidate there
  is no authorization to trade, so BUY/SELL can never be produced by this path.
- Nothing about the market state is fabricated: every price/level field is None
  and the event timestamp is the provider's real closed-candle timestamp.
- The missing authorization is disclosed in the provenance block, never concealed.
- Stale, unclosed, invalid, or missing market data must never be laundered into a
  releasable observation; the caller (ContinuousLiveRuntime) fail-closes such
  cycles before constructing this artifact.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

OBSERVATION_DECISION = "NO TRADE"
OBSERVATION_PROVENANCE_TYPE = "market_observation_no_trade"


def validate_utc_iso8601(name: str, raw: object) -> str:
    """Validate a timezone-aware UTC ISO-8601 timestamp string and return it normalized."""
    ts_str = str(raw).strip() if raw is not None else ""
    if not ts_str:
        raise ValueError(f"{name} must be a non-empty timezone-aware UTC ISO-8601 string.")
    try:
        parsed = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception as exc:
        raise ValueError(f"Invalid {name} '{ts_str}': {exc}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset() != datetime.timedelta(0):
        raise ValueError(f"{name} must be timezone-aware with a zero UTC offset.")
    if ts_str.endswith("+00:00"):
        ts_str = ts_str.replace("+00:00", "Z")
    return ts_str


@dataclass(frozen=True)
class MarketObservationPublication:
    """Immutable Contract v1.0 observation event for a candidate-less timeframe cycle.

    Event time is the provider's real closed-candle timestamp; the observation
    (decision) time is separate and explicit. Nothing else is published.
    """

    symbol: str
    timeframe: str
    observation_id: str = field(init=False)
    decision: str = field(init=False)
    reason: str
    detail: str
    candle_timestamp_utc: str
    observation_timestamp_utc: str

    def __post_init__(self) -> None:
        req_symbol = str(self.symbol).strip().upper() if self.symbol else ""
        req_timeframe = str(self.timeframe).strip() if self.timeframe else ""
        req_reason = str(self.reason).strip() if self.reason else ""
        req_detail = str(self.detail).strip() if self.detail else ""

        if not req_symbol:
            raise ValueError("symbol must be a non-empty string.")
        if not req_timeframe:
            raise ValueError("timeframe must be a non-empty string.")
        if not req_reason:
            raise ValueError("reason must be a non-empty string.")
        if not req_detail:
            raise ValueError("detail must be a non-empty string.")

        object.__setattr__(self, "symbol", req_symbol)
        object.__setattr__(self, "timeframe", req_timeframe)
        object.__setattr__(self, "reason", req_reason)
        object.__setattr__(self, "detail", req_detail)

        candle_ts = validate_utc_iso8601("candle_timestamp_utc", self.candle_timestamp_utc)
        observation_ts = validate_utc_iso8601("observation_timestamp_utc", self.observation_timestamp_utc)
        object.__setattr__(self, "candle_timestamp_utc", candle_ts)
        object.__setattr__(self, "observation_timestamp_utc", observation_ts)

        # Observations place no trade: only the truthfully untradeable outcome is valid.
        object.__setattr__(self, "decision", OBSERVATION_DECISION)
        object.__setattr__(self, "observation_id", self._derive_observation_id())

    def _derive_observation_id(self) -> str:
        raw = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "decision": self.decision,
            "reason": self.reason,
            "detail": self.detail,
            "candle_timestamp_utc": self.candle_timestamp_utc,
            "observation_timestamp_utc": self.observation_timestamp_utc,
        }
        serialized = json.dumps(raw, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:32]

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "decision": self.decision,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "reason": self.reason,
            "detail": self.detail,
            "candle_timestamp_utc": self.candle_timestamp_utc,
            "observation_timestamp_utc": self.observation_timestamp_utc,
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "source": "AI-Trading-Lab",
            "provenance_type": OBSERVATION_PROVENANCE_TYPE,
            "is_live": True,
            "produced_at": self.observation_timestamp_utc,
            "candidate_id": None,
            "strategy_name": None,
            "operational_stability_score": None,
            "authorized": False,
            "authorization_state": "no_authorized_candidate_for_timeframe",
            "blocked_reason_code": self.reason,
            "blocked_detail": self.detail,
            "publication_contract_version": "1.0",
        }

    def to_contract_v1_payload(self) -> dict[str, Any]:
        """Convert the observation into a Contract v1.0 payload for Project 2.

        Mirrors the canonical signal payload shape while truthfully carrying no
        trade identity, no trade levels, and no stability score.
        """
        observation_id = self.observation_id
        payload = {
            "contract_version": "1.0",
            "event_id": observation_id,
            "event_type": "TRADING_SIGNAL",
            "timestamp": self.candle_timestamp_utc,
            "instrument": {
                "symbol": self.symbol,
                "interval": self.timeframe,
            },
            "signal": {
                "publication_id": observation_id,
                "signal_id": observation_id,
                "decision_id": observation_id,
                "decision": self.decision,
                "strategy": None,
                "candidate_id": None,
                "confidence": None,
                "stability_score": None,
                "invalidation": None,
                "signal_label": self.decision,
                "trend": "NEUTRAL",
            },
            "trade_setup": {
                "entry_price": None,
                "stop_loss": None,
                "tp1": None,
                "tp2": None,
                "tp3": None,
                "take_profit": None,
                "risk_reward_ratio": None,
                "trailing_stop": None,
            },
            "provenance": self.provenance(),
        }
        return payload
