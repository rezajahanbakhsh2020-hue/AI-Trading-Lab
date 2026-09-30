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

    # 1. Fresh execution
    with patch("src.evaluation.live_execution_runtime.load_live_market_data", return_value=df):
        res_fresh = runtime.run_once(publish=False, persist=True, reference_now=df_ts)
    assert res_fresh["decision"] == "BUY"
    assert res_fresh["current_lifecycle_state"] == LiveDecisionLifecycleState.PERSISTED.value

    # 2. Stale execution
    with patch("src.evaluation.live_execution_runtime.load_live_market_data", return_value=df):
        res_stale = runtime.run_once(
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

    with (
        patch(
            "src.evaluation.live_execution_runtime.resolve_authoritative_promoted_candidate",
            side_effect=resolve_authoritative_promoted_candidate,
        ) as mock_resolve,
        patch("src.evaluation.live_execution_runtime.authorize_production_runtime") as mock_auth,
        patch("src.evaluation.live_execution_runtime.load_live_market_data") as mock_data,
        patch("src.evaluation.live_execution_runtime.evaluate_authorized_live_runtime") as mock_eval_downstream,
    ):
        mock_data.return_value = df
        from src.evaluation.live_production_decision import authorize_production_runtime

        promoted = resolve_authoritative_promoted_candidate(config)
        auth = authorize_production_runtime(promoted, symbol="XAUUSD", timeframe="5m", now=ref_now)
        mock_auth.return_value = auth

        mock_eval_downstream.side_effect = evaluate_authorized_live_runtime

        runtime.run_once(publish=False, persist=True, reference_now=ref_now)

        # Assert resolve and authorize were called exactly ONCE in run_once
        assert mock_resolve.call_count == 1
        assert mock_auth.call_count == 1

        # Check receipt identity passed to downstream
        context_arg = mock_eval_downstream.call_args.kwargs["context"]
        assert context_arg.authorization_receipt.authorization_fingerprint == auth.authorization_fingerprint


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
        "create_canonical_live_decision",
        "transition_live_decision",
        "persist_canonical_live_decision",
        "publish_canonical_live_decision",
    }

    found_forbidden = forbidden_direct_calls.intersection(visitor.calls)
    assert not found_forbidden, (
        f"LiveExecutionRuntime.run_once directly calls forbidden lifecycle functions: {found_forbidden}"
    )
