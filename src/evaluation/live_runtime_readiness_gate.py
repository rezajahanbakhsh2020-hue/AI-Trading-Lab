from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.evaluation.live_runtime_history import LiveRuntimeHistory
from src.evaluation.live_runtime_status import (
    LiveRuntimeStatus,
    build_live_runtime_status,
)


@dataclass(frozen=True)
class LiveRuntimeReadinessGate:
    """
    Gate for evaluating live runtime readiness.
    
    Provides a final readiness decision for live deployment based on
    the existing LiveRuntimeStatus. Does not re-evaluate conditions
    already checked by the status layer.
    """
    is_ready: bool
    status: str
    reason: str
    recommendation: str


def build_live_runtime_readiness_gate(
    history: LiveRuntimeHistory,
) -> LiveRuntimeReadinessGate:
    """
    Evaluate live runtime readiness and produce a gate decision.
    
    Uses the existing live runtime status to determine if the system
    is ready for live deployment. The gate provides a clear decision
    and actionable recommendation.
    
    Args:
        history: LiveRuntimeHistory instance containing runtime snapshots
        
    Returns:
        LiveRuntimeReadinessGate with readiness decision and details
    """
    if not isinstance(history, LiveRuntimeHistory):
        raise TypeError("history must be a LiveRuntimeHistory.")
    
    status = build_live_runtime_status(history)
    
    # Gate decision: only READY state passes
    is_ready = status.state == "READY"
    
    # Build reason and recommendation based on status state
    reason, recommendation = _build_reason_and_recommendation(status)
    
    return LiveRuntimeReadinessGate(
        is_ready=is_ready,
        status=status.state,
        reason=reason,
        recommendation=recommendation,
    )


def _build_reason_and_recommendation(
    status: LiveRuntimeStatus,
) -> tuple[str, str]:
    """
    Generate reason and recommendation based on status state.
    
    Args:
        status: LiveRuntimeStatus object
        
    Returns:
        Tuple of (reason, recommendation) strings
    """
    reasons_and_recommendations = {
        "EMPTY": (
            "No runtime snapshots available. "
            "System has not been exercised yet.",
            "Collect runtime snapshots by running the system in staging mode. "
            "Ensure adequate baseline data before live deployment.",
        ),
        "DEGRADED": (
            f"System health is degraded. "
            f"Health score: {status.health_score:.2f}. "
            f"Data integrity or consistency issues detected.",
            f"Investigate and resolve health issues. "
            f"Review system diagnostics and repair any "
            f"timestamp, data ordering, or consistency problems.",
        ),
        "WARNING": (
            "System status is warning. "
            "Warnings detected in runtime alerts. "
            "Not all readiness criteria are fully satisfied.",
            "Review and address the active runtime alerts. "
            "Resolve warning conditions before attempting live deployment.",
        ),
        "READY": (
            "All readiness criteria satisfied. "
            "System is ready for live deployment.",
            "Proceed with live deployment. "
            "Monitor system performance and alerts continuously.",
        ),
    }
    
    return reasons_and_recommendations.get(
        status.state,
        (
            f"Unknown status state: {status.state}",
            "Review system state and diagnostics.",
        ),
    )


def is_live_runtime_ready(
    history: LiveRuntimeHistory,
) -> bool:
    """
    Quick check: is the live runtime ready for deployment?
    
    Args:
        history: LiveRuntimeHistory instance
        
    Returns:
        Boolean readiness decision
    """
    gate = build_live_runtime_readiness_gate(history)
    return gate.is_ready


def live_runtime_readiness_gate_message(
    history: LiveRuntimeHistory,
) -> str:
    """
    Generate a human-readable readiness gate status message.
    
    Args:
        history: LiveRuntimeHistory instance
        
    Returns:
        Formatted status message
    """
    gate = build_live_runtime_readiness_gate(history)
    
    status_indicator = "READY ✓" if gate.is_ready else "BLOCKED ✗"
    
    return (
        f"LIVE RUNTIME READINESS GATE: {status_indicator} | "
        f"{gate.reason}"
    )


def live_runtime_readiness_gate_dict(
    history: LiveRuntimeHistory,
) -> dict[str, Any]:
    """
    Return readiness gate evaluation as a dictionary.
    
    Useful for integration with production workflows and logging.
    
    Args:
        history: LiveRuntimeHistory instance
        
    Returns:
        Dictionary with gate evaluation details
    """
    gate = build_live_runtime_readiness_gate(history)
    
    return {
        "is_ready": gate.is_ready,
        "status": gate.status,
        "reason": gate.reason,
        "recommendation": gate.recommendation,
    }
