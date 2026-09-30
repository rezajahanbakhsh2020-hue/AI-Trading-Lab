"""Tests verifying architectural invariants, convergence, and AST structure for PR #50."""

import ast
import datetime
import inspect
import textwrap
from unittest.mock import patch

import pandas as pd

from src.evaluation.live_decision_lifecycle import LiveDecisionLifecycleState
from src.evaluation.live_execution_runtime import (
    LiveExecutionRuntime,
    resolve_authoritative_promoted_candidate,
)
from src.evaluation.live_runtime import evaluate_authorized_live_runtime
from tests.test_live_execution_runtime import (
    make_buy_market_data,
    production_config_for,
)


def test_exact_convergence_fresh_and_stale(tmp_path):
    """C & J. Fresh and stale market data converge through exact same downstream lifecycle functions."""
    config = production_config_for(tmp_path)

    df = make_buy_market_data()
    df_ts = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime()

    # 1. Fresh execution
    runtime_fresh = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "store_fresh.json",
        snapshot_path=tmp_path / "snap_fresh.json",
        research_dir=tmp_path,
        production_config=config,
    )
    with (
        patch("src.evaluation.live_execution_runtime.load_live_market_data", return_value=df),
        patch(
            "src.evaluation.live_execution_runtime.load_production_selection",
            return_value={"stability_score": 0.85, "stable_strategy": "momentum"},
        ),
    ):
        res_buy = runtime_fresh.run_once(publish=False, persist=True, reference_now=df_ts)
    assert res_buy["decision"] == "BUY"
    assert res_buy["current_lifecycle_state"] == LiveDecisionLifecycleState.PERSISTED.value

    # 2. Stale execution
    runtime_stale = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "store_stale.json",
        snapshot_path=tmp_path / "snap_stale.json",
        research_dir=tmp_path,
        production_config=config,
    )
    with (
        patch("src.evaluation.live_execution_runtime.load_live_market_data", return_value=df),
        patch(
            "src.evaluation.live_execution_runtime.load_production_selection",
            return_value={"stability_score": 0.85, "stable_strategy": "momentum"},
        ),
    ):
        res_stale = runtime_stale.run_once(
            publish=False, persist=True, reference_now=df_ts + datetime.timedelta(seconds=1000)
        )
    assert res_stale["decision"] == "NO TRADE"
    assert res_stale["current_lifecycle_state"] == LiveDecisionLifecycleState.PERSISTED.value
    assert res_stale["record"]["signal_label"] == "NO TRADE"
    assert res_stale["record"]["entry_price"] is None


def test_single_candidate_resolution_and_authorization(tmp_path):
    """D, E, F, G. Single resolution, single authorization, exact receipt object, no re-resolution."""
    config = production_config_for(tmp_path)
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    df = make_buy_market_data()
    ref_now = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime()

    # Pre-build candidate, authorization, and receipt OUTSIDE patch blocks
    from src.evaluation.live_production_decision import ProductionAuthorizationReceipt, authorize_production_runtime
    pre_candidate = resolve_authoritative_promoted_candidate(config)
    pre_auth = authorize_production_runtime(pre_candidate, symbol="XAUUSD", timeframe="5m", now=ref_now)
    exact_receipt = ProductionAuthorizationReceipt.from_authorization(pre_auth)

    with (
        patch(
            "src.evaluation.live_execution_runtime.resolve_authoritative_promoted_candidate",
            return_value=pre_candidate,
        ) as mock_resolve,
        patch(
            "src.evaluation.live_execution_runtime.authorize_production_runtime",
            return_value=pre_auth,
        ) as mock_auth,
        patch(
            "src.evaluation.live_execution_runtime.ProductionAuthorizationReceipt.from_authorization",
            return_value=exact_receipt,
        ) as mock_receipt,
        patch("src.evaluation.live_execution_runtime.load_live_market_data", return_value=df),
        patch(
            "src.evaluation.live_execution_runtime.evaluate_authorized_live_runtime",
            side_effect=evaluate_authorized_live_runtime,
        ) as mock_eval_downstream,
    ):
        runtime.run_once(publish=False, persist=True, reference_now=ref_now)

        # Assert resolve, authorize, receipt factory, and downstream evaluate called exactly ONCE during run_once()
        assert mock_resolve.call_count == 1
        assert mock_auth.call_count == 1
        assert mock_receipt.call_count == 1
        assert mock_eval_downstream.call_count == 1

        # Check EXACT RECEIPT OBJECT IDENTITY passed to downstream
        context_arg = mock_eval_downstream.call_args.kwargs["context"]
        assert context_arg.authorization_receipt is exact_receipt


def test_stale_path_bypasses_signal_and_trend_generators(tmp_path):
    """Part 9: Stale evaluation path must NOT invoke fresh signal or trend snapshot generators."""
    config = production_config_for(tmp_path)
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    df = make_buy_market_data()
    df_ts = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime()
    stale_ref_now = df_ts + datetime.timedelta(seconds=1000)

    import live_signal
    import live_trend

    with (
        patch("src.evaluation.live_execution_runtime.load_live_market_data", return_value=df),
        patch.object(live_signal, "generate_live_signal") as mock_sig,
        patch.object(live_trend, "build_live_trend_snapshot") as mock_trend,
    ):
        res = runtime.run_once(publish=False, persist=True, reference_now=stale_ref_now)

        assert res["decision"] == "NO TRADE"
        assert res["record"]["quote_stale"] is True
        # Signal and trend generators MUST NOT BE CALLED for stale data
        assert mock_sig.call_count == 0
        assert mock_trend.call_count == 0


def test_ast_static_checks_live_execution_runtime():
    """L. Structural AST checks proving LiveExecutionRuntime.run_once is orchestration-only."""
    source = textwrap.dedent(inspect.getsource(LiveExecutionRuntime.run_once))
    tree = ast.parse(source)

    class FunctionCallVisitor(ast.NodeVisitor):
        def __init__(self):
            self.calls = []

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                self.calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                self.calls.append(node.func.attr)
            self.generic_visit(node)

    visitor = FunctionCallVisitor()
    visitor.visit(tree)

    forbidden_direct_calls = {
        "ProductionDecision",
        "ProductionSignal",
        "calculate_production_risk_levels",
        "create_canonical_live_decision",
        "transition_live_decision",
        "persist_canonical_live_decision",
        "publish_canonical_live_decision",
        "publish",
    }

    found_forbidden = forbidden_direct_calls.intersection(visitor.calls)
    assert not found_forbidden, (
        f"LiveExecutionRuntime.run_once directly calls forbidden lifecycle functions: {found_forbidden}"
    )
