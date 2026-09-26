"""Comprehensive End-to-End Integration Test for Project 1 -> Contract v1.0 -> Project 2 Flow."""
import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.evaluation.live_execution_runtime import LiveExecutionRuntime, ProductionRuntimeConfig
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    EvidencePartition,
    EvidencePartitionRole,
    ExecutionAssumptions,
    PromotionStatus,
    ResearchEvidence,
    ResearchExperimentSpec,
)
from src.evaluation.research_store import save_research_candidate
from src.integration.project2_publisher import Project2Publisher, build_contract_v1_payload


def make_market_data() -> pd.DataFrame:
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    timestamps = [(now_dt - datetime.timedelta(minutes=5 * (100 - i))).isoformat() for i in range(100)]
    df = pd.DataFrame({
        "openTime": timestamps,
        "open": [2000.0 + i for i in range(100)],
        "high": [2005.0 + i for i in range(100)],
        "low": [1995.0 + i for i in range(100)],
        "close": [2002.0 + i for i in range(100)],
    })
    return df


@patch("src.evaluation.live_execution_runtime.load_production_selection")
@patch("src.evaluation.live_execution_runtime.load_live_market_data")
@patch("urllib.request.urlopen")
def test_end_to_end_pipeline(mock_urlopen, mock_load_data, mock_load_selection, tmp_path: Path) -> None:
    # Setup mocks
    mock_load_data.return_value = make_market_data()
    mock_load_selection.return_value = {
        "stable_strategy": "momentum",
        "stability_score": 0.88,
    }

    mock_resp = MagicMock()
    mock_resp.getcode.return_value = 200
    mock_resp.read.return_value = b'{"status": "ACKNOWLEDGED", "received_event_id": "test"}'
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    store_path = tmp_path / "live_history.json"
    snapshot_path = tmp_path / "latest_snapshot.json"

    publisher = Project2Publisher(
        publish_url="https://project2.example.com/api/v1/signals",
        api_key="secret-api-token",
        enabled=True,
    )

    ds = DatasetScope(
        dataset_id="ds_xauusd_5m",
        symbol="XAUUSD",
        timeframe="5m",
        start_date="2025-01-01",
        end_date="2025-01-02",
    )
    spec = ResearchExperimentSpec(
        hypothesis="Persisted e2e momentum candidate",
        methodology_version="1.0",
        strategy_name="momentum",
        strategy_version="1.0",
        dataset_scope=ds,
        execution_assumptions=ExecutionAssumptions(
            transaction_cost=0.001, slippage=0.001, latency_ms=10.0
        ),
        code_provenance=CodeProvenance(commit_sha="e52d95d1ede22cf3c8ce07dc216763ace4a4359c"),
        benchmark_reference="buy_and_hold",
        parameters={"momentum_window": 10, "stop_loss_pct": 0.01, "take_profit_pct": 0.02},
    )
    evidence = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(
            EvidencePartition(
                role=EvidencePartitionRole.OUT_OF_SAMPLE,
                start_date="2025-01-01",
                end_date="2025-01-02",
                total_return=0.15,
                max_drawdown=0.05,
                sharpe_ratio=1.8,
            ),
        ),
        robustness_verdict={"passed": True},
        promotion_status=PromotionStatus.PROMOTABLE,
        rejection_reasons=(),
    )
    save_research_candidate(candidate_id="cand_momentum_e2e", evidence=evidence, base_dir=tmp_path)

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        publisher=publisher,
        store_path=store_path,
        snapshot_path=snapshot_path,
        research_dir=tmp_path,
        production_config=ProductionRuntimeConfig(
            symbol="XAUUSD",
            timeframe="5m",
            candidate_id="cand_momentum_e2e",
            strategy_id="momentum",
            research_dir=tmp_path,
        ),
    )

    execution_result = runtime.run_once(publish=True, persist=True)

    # 1. Verify runtime outputs
    assert execution_result["symbol"] == "XAUUSD"
    assert execution_result["interval"] == "5m"
    assert execution_result["strategy"] == "momentum"
    assert execution_result["stability_score"] == 0.88

    # 2. Verify Contract v1.0 payload
    payload = execution_result["contract_payload"]
    assert payload["contract_version"] == "1.0"
    assert payload["event_type"] == "TRADING_SIGNAL"
    assert payload["instrument"]["symbol"] == "XAUUSD"
    assert payload["instrument"]["interval"] == "5m"
    assert payload["signal"]["strategy"] == "momentum"
    assert "event_id" in payload

    # 3. Verify Publisher delivery
    pub_res = execution_result["publish_result"]
    assert pub_res["status"] == "PUBLISHED"
    assert pub_res["published"] is True
    assert pub_res["http_code"] == 200

    # 4. Verify Persistence
    assert store_path.exists()
    assert snapshot_path.exists()

    history = json.loads(store_path.read_text())
    assert isinstance(history, list)
    assert len(history) == 1
    assert history[0]["symbol"] == "XAUUSD"
    assert history[0]["interval"] == "5m"
