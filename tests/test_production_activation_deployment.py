"""Focused tests verifying production activation boundary, Render worker deployment config, and startup preflight invariants."""
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import yaml

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

    with open(render_yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "services" in config
    services = config["services"]
    assert len(services) == 1
    worker = services[0]

    assert worker["type"] == "worker"
    assert worker["name"] == "p1-live-execution-worker"
    assert worker["env"] == "python"
    assert "run_live_execution.py --continuous" in worker["startCommand"]
    assert "--symbol XAUUSD" in worker["startCommand"]
    assert "--interval 5m" in worker["startCommand"]
    assert "--candidate-id cand_moving_average_5m" in worker["startCommand"]
    assert "--publish" in worker["startCommand"]

    env_vars = {env["key"]: env for env in worker["envVars"]}
    assert env_vars["PYTHONPATH"]["value"] == "."
    assert env_vars["PROJECT2_PUBLISH_ENABLED"]["value"] == "true"
    assert "ai-trading-lab-platform-1.onrender.com" in env_vars["PROJECT2_PUBLISH_URL"]["value"]
    assert env_vars["PROJECT2_API_KEY"]["sync"] is False


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
    assert readiness["5m"].candidate_id == cand_id


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


def test_unsupported_1m_rejected() -> None:
    """7. Unsupported timeframe '1m' remains strictly rejected."""
    with pytest.raises(ValueError, match="Unknown or unsupported timeframe '1m'"):
        CanonicalTimeframe.from_str("1m")

    with pytest.raises(ValueError, match="Unknown or unsupported timeframe '1m'"):
        ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["1m"])


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
