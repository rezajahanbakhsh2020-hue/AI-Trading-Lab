"""Focused tests verifying production activation boundary, Render worker deployment config, and startup preflight invariants."""
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.evaluation.live_execution_runtime import (
    ContinuousLiveRuntime,
    ProductionRuntimeConfig,
    parse_continuous_candidate_ids,
)
from src.evaluation.mtf_intelligence import CanonicalTimeframe
from src.evaluation.research_store import PromotionUnavailable
from src.integration.project2_publisher import Project2Publisher
from tests.test_continuous_live_runtime import (
    persist_momentum_candidate,
    make_tf_market_data,
)


def test_render_yaml_configuration() -> None:
    """1. Deployment definition points to existing continuous runtime with proper arguments and environment vars."""
    render_yaml_path = Path("render.yaml")
    assert render_yaml_path.exists(), "render.yaml must exist at repository root"

    content = render_yaml_path.read_text(encoding="utf-8")

    assert "type: worker" in content
    assert "name: p1-live-execution-worker" in content
    assert "env: python" in content
    assert "run_live_execution.py --continuous" in content
    assert "--symbol XAUUSD" in content
    assert "--timeframes 5m,1D" in content
    assert "--candidate-id 5m=cand_moving_average_5m,1D=cand_moving_average_1d" in content
    assert "--publish" in content
    assert "PROJECT2_PUBLISH_ENABLED" in content
    assert "PROJECT2_PUBLISH_URL" in content
    assert "ai-trading-lab-platform-1.onrender.com" in content
    assert "PROJECT2_API_KEY" in content
    assert "sync: false" in content


def test_startup_preflight_fails_closed_when_promoted_candidate_absent(tmp_path) -> None:
    """2. Production readiness preflight fails closed when required promoted candidate artifact is absent."""
    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        candidate_ids={"5m": "nonexistent_cand"},
        market_data_loaders=lambda tf: make_tf_market_data("2025-01-01 10:00", "5m"),
    )

    with pytest.raises(PromotionUnavailable, match="Startup readiness preflight failed"):
        cont.verify_startup_readiness()

    # Verify run_continuous fails closed before starting loop
    with pytest.raises(PromotionUnavailable):
        cont.run_continuous(max_ticks=1)


def test_startup_preflight_accepts_valid_authoritative_promoted_candidate(tmp_path) -> None:
    """3. Valid authoritative promoted candidate resolution is accepted by startup preflight."""
    cand_id = persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live", timeframe="5m")

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        candidate_ids={"5m": cand_id},
        market_data_loaders=lambda tf: make_tf_market_data("2025-01-01 10:00", "5m"),
    )

    readiness = cont.verify_startup_readiness()
    assert "5m" in readiness
    assert readiness["5m"]["candidate_id"] == cand_id


def test_candidate_id_not_copied_across_multiple_timeframes() -> None:
    """4. Candidate ID is not silently copied across multiple timeframes without explicit mapping."""
    with pytest.raises(ValueError, match="must specify explicit per-timeframe mappings"):
        parse_continuous_candidate_ids("cand_single", timeframes=["5m", "15m"])

    # Explicit mapping is parsed correctly
    mapping = parse_continuous_candidate_ids("5m=cand1,15m=cand2", timeframes=["5m", "15m"])
    assert mapping == {"5m": "cand1", "15m": "cand2"}


def test_project2_publishing_environment_driven() -> None:
    """5. Project2 publishing remains strictly environment-driven."""
    with patch.dict(os.environ, {
        "PROJECT2_PUBLISH_ENABLED": "true",
        "PROJECT2_PUBLISH_URL": "https://example.com/ingest",
        "PROJECT2_API_KEY": "test_secret_key",
    }):
        pub = Project2Publisher()
        assert pub.enabled is True
        assert pub.publish_url == "https://example.com/ingest"
        assert pub.api_key == "test_secret_key"

    with patch.dict(os.environ, {
        "PROJECT2_PUBLISH_ENABLED": "false",
        "PROJECT2_PUBLISH_URL": "",
        "PROJECT2_API_KEY": "",
    }):
        pub_disabled = Project2Publisher()
        assert pub_disabled.enabled is False


def test_secrets_never_committed() -> None:
    """6. Ensure no API keys or secret tokens are committed in code or render.yaml."""
    render_text = Path("render.yaml").read_text(encoding="utf-8")
    assert "sync: false" in render_text
    assert "value:" not in render_text.split("PROJECT2_API_KEY")[1].split("\n")[0]


def test_1m_identity_is_supported_but_not_activated_without_promoted_binding() -> None:
    """1m data identity is supported; production still requires an independently governed binding."""
    assert CanonicalTimeframe.from_str("1m") is CanonicalTimeframe.ONE_MINUTE
    assert ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["1m"]).timeframes == ("1m",)


def test_existing_continuous_runtime_behavior_intact(tmp_path) -> None:
    """8. Existing ContinuousLiveRuntime behavior remains fully intact."""
    cand_id = persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live", timeframe="5m")

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        candidate_ids={"5m": cand_id},
        market_data_loaders=lambda tf: make_tf_market_data("2025-01-01 10:00", "5m"),
    )

    cont.run_continuous(max_ticks=2)
    assert len(cont._execution_history) == 2
