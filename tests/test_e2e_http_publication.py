"""End-to-end local HTTP integration test for Project 1 -> Project 2 delivery boundary.

Constructs a local HTTP server simulating Project 2 Contract-v1 ingestion endpoint,
and verifies the entire pipeline:
Authorized Runtime Context -> Live Market Evaluation -> Canonical Live Decision
-> PERSISTED -> ProductionIntelligencePublication -> Contract v1 -> Project2Publisher
-> Local HTTP server POST -> Verified Acknowledgement -> DELIVERED receipt -> PUBLISHED lifecycle state.
"""

from datetime import datetime, timezone
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import threading
from unittest.mock import patch

import pandas as pd
import pytest

from src.evaluation.live_execution_runtime import LiveExecutionRuntime, ProductionRuntimeConfig
from src.evaluation.live_production_decision import PromotedCandidateArtifact
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
from src.integration.project2_publisher import Project2Publisher


class MockProject2IngestHandler(BaseHTTPRequestHandler):
    received_requests = []

    def log_message(self, format, *args):
        pass  # Quiet logging

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length)
        payload = json.loads(body_bytes.decode("utf-8"))

        MockProject2IngestHandler.received_requests.append({
            "headers": dict(self.headers),
            "payload": payload,
        })

        event_id = payload.get("event_id") or payload.get("signal", {}).get("publication_id")
        api_key = self.headers.get("X-API-Key")

        if api_key != "test-e2e-api-key":
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Unauthorized"}).encode("utf-8"))
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response_body = {
            "status": "INGESTED",
            "event_id": event_id,
            "publication_id": event_id,
            "received_at": datetime.now(timezone.utc).isoformat(),
        }
        self.wfile.write(json.dumps(response_body).encode("utf-8"))


def make_test_promoted_evidence() -> ResearchEvidence:
    ds = DatasetScope(
        dataset_id="ds_e2e_test",
        symbol="XAUUSD",
        timeframe="5m",
        start_date="2025-01-01",
        end_date="2025-01-10",
    )
    ea = ExecutionAssumptions(transaction_cost=0.001, slippage=0.001, latency_ms=10.0)
    cp = CodeProvenance(commit_sha="e52d95d1ede22cf3c8ce07dc216763ace4a4359c")
    spec = ResearchExperimentSpec(
        hypothesis="E2E test hypothesis",
        methodology_version="1.0",
        strategy_name="momentum",
        strategy_version="1.0",
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        benchmark_reference="buy_and_hold",
        parameters={"momentum_window": 10, "stop_loss_pct": 0.01, "take_profit_pct": 0.02},
    )
    part_is = EvidencePartition(
        role=EvidencePartitionRole.IN_SAMPLE,
        start_date="2025-01-01",
        end_date="2025-01-02",
        total_return=0.20,
        max_drawdown=0.05,
        sharpe_ratio=2.0,
        observations=50,
        start_timestamp_utc="2025-01-01T00:00:00+00:00",
        end_timestamp_utc="2025-01-02T00:00:00+00:00",
    )
    part_oos = EvidencePartition(
        role=EvidencePartitionRole.OUT_OF_SAMPLE,
        start_date="2025-01-03",
        end_date="2025-01-05",
        total_return=0.15,
        max_drawdown=0.05,
        sharpe_ratio=1.8,
        observations=30,
        start_timestamp_utc="2025-01-03T00:00:00+00:00",
        end_timestamp_utc="2025-01-05T00:00:00+00:00",
    )
    part_wf = EvidencePartition(
        role=EvidencePartitionRole.WALK_FORWARD,
        start_date="2025-01-01",
        end_date="2025-01-05",
        total_return=0.10,
        max_drawdown=0.05,
        sharpe_ratio=1.5,
        observations=30,
        start_timestamp_utc="2025-01-01T00:00:00+00:00",
        end_timestamp_utc="2025-01-05T00:00:00+00:00",
    )
    return ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(part_is, part_oos, part_wf),
        robustness_verdict={
            "passed": True,
            "parameter_sensitivity": {"passed": True},
            "subsample_stability": {"passed": True},
            "execution_cost_stress": {"passed": True},
            "statistical_validation": {"passed": True},
            "anti_overfitting": {"passed": True},
        },
        promotion_status=PromotionStatus.PROMOTABLE,
        rejection_reasons=(),
    )


def test_full_local_http_e2e_pipeline(tmp_path: Path) -> None:
    MockProject2IngestHandler.received_requests.clear()

    server = HTTPServer(("127.0.0.1", 0), MockProject2IngestHandler)
    server_port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    try:
        publish_url = f"http://127.0.0.1:{server_port}/api/v1/integration/project1/ingest"
        api_key = "test-e2e-api-key"

        # 1. Setup promoted candidate binding
        ev = make_test_promoted_evidence()
        save_research_candidate(
            candidate_id="cand_e2e_01",
            evidence=ev,
            operational_stability_score=0.88,
            base_dir=tmp_path,
        )

        # 2. Setup Market Data
        now_dt = datetime.now(timezone.utc)
        timestamps = [now_dt - pd.Timedelta(minutes=5)]
        for i in range(1, 20):
            timestamps.append(now_dt - pd.Timedelta(minutes=5 * i))
        timestamps.reverse()

        df = pd.DataFrame({
            "openTime": [ts.isoformat() for ts in timestamps],
            "timestamp": pd.to_datetime([ts.isoformat() for ts in timestamps], utc=True),
            "open": [2000.0 + i * 0.5 for i in range(20)],
            "high": [2005.0 + i * 0.5 for i in range(20)],
            "low": [1995.0 + i * 0.5 for i in range(20)],
            "close": [2004.0 + i * 0.5 for i in range(20)],
            "volume": [100.0] * 20,
        })

        publisher = Project2Publisher(
            publish_url=publish_url,
            api_key=api_key,
            enabled=True,
        )

        config = ProductionRuntimeConfig(
            symbol="XAUUSD",
            timeframe="5m",
            candidate_id="cand_e2e_01",
            strategy_id="momentum",
            research_dir=tmp_path,
        )

        store_file = tmp_path / "decision_history.json"
        snapshot_file = tmp_path / "latest_execution.json"

        runtime = LiveExecutionRuntime(
            symbol="XAUUSD",
            interval="5m",
            publisher=publisher,
            store_path=store_file,
            snapshot_path=snapshot_file,
            research_dir=tmp_path,
            production_config=config,
        )

        with patch("src.evaluation.live_execution_runtime.load_live_market_data", return_value=df):
            result = runtime.run_once(publish=True, persist=True, reference_now=now_dt)

        assert result["blocked"] is False
        assert result["current_lifecycle_state"] == "PUBLISHED"
        assert result["delivery_status"] == "DELIVERED"
        assert result["publish_result"]["status"] == "PUBLISHED"
        assert result["publish_result"]["published"] is True

        # Assert local HTTP server received the request
        assert len(MockProject2IngestHandler.received_requests) == 1
        req = MockProject2IngestHandler.received_requests[0]

        headers_lower = {k.lower(): v for k, v in req["headers"].items()}
        assert headers_lower.get("x-api-key") == api_key
        assert headers_lower.get("authorization") == f"Bearer {api_key}"
        assert req["headers"].get("X-Idempotency-Key") == result["publication"]["publication_id"]

        payload = req["payload"]
        assert payload["contract_version"] == "1.0"
        assert payload["event_id"] == result["publication"]["publication_id"]
        assert payload["instrument"]["symbol"] == "XAUUSD"
        assert payload["instrument"]["interval"] == "5m"
        assert payload["signal"]["strategy"] == "momentum"
        assert payload["trade_setup"]["entry_price"] == result["publication"]["entry"]
        assert payload["trade_setup"]["stop_loss"] == result["publication"]["stop_loss"]
        assert payload["provenance"]["governance_decision_fingerprint"] == result["publication"]["provenance"]["governance_decision_fingerprint"]
        assert payload["provenance"]["runtime_authorization_fingerprint"] == result["publication"]["provenance"]["runtime_authorization_fingerprint"]

    finally:
        server.shutdown()
        server.server_close()
