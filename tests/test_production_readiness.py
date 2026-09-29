from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.production_readiness import (
    build_production_readiness,
    is_production_ready,
    production_readiness_message,
)


def make_valid_authoritative_results(tmp_path: Path) -> Path:
    """
    Create a mock walk-forward experiment directory structure with experiment run folders
    so select_production_strategy() can discover experiments and return a valid strategy.
    """
    results_dir = tmp_path / "walk_forward"
    exp_dir = results_dir / "exp_001"
    exp_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "strategy": "momentum",
                "rank": 1,
                "total_return": 0.20,
                "max_drawdown": -0.05,
                "sharpe_ratio": 1.5,
                "sortino_ratio": 2.0,
                "calmar_ratio": 4.0,
                "positive_window_rate": 0.8,
            },
            {
                "strategy": "baseline",
                "rank": 2,
                "total_return": 0.10,
                "max_drawdown": -0.10,
                "sharpe_ratio": 0.8,
                "sortino_ratio": 1.0,
                "calmar_ratio": 1.0,
                "positive_window_rate": 0.5,
            },
        ]
    ).to_csv(
        exp_dir / "final_report.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {"strategy": "momentum"},
            {"strategy": "baseline"},
        ]
    ).to_csv(
        exp_dir / "eligible_strategies.csv",
        index=False,
    )

    (exp_dir / "metadata.json").write_text(
        '{"created_at_utc": "2025-01-01T00:00:00Z", "best_strategy": "momentum"}',
        encoding="utf-8",
    )

    return results_dir


def test_production_readiness_passes_with_stable_portfolio(
    tmp_path,
):
    results_dir = make_valid_authoritative_results(tmp_path)

    result = build_production_readiness(
        portfolio_stability_score=0.90,
        results_dir=results_dir,
    )

    assert result["strategy"] == "momentum"
    assert result["strategy_stability_score"] is not None
    assert result["portfolio_stability_score"] == pytest.approx(
        0.90
    )
    assert result["strategy_gate"] is True
    assert result["portfolio_gate"] is True
    assert result["readiness_gate"] is True
    assert result["ready"] is True
    assert result["status"] == "READY"
    assert result["failed_gates"] == []


def test_production_readiness_blocks_unstable_portfolio(
    tmp_path,
):
    results_dir = make_valid_authoritative_results(tmp_path)

    result = build_production_readiness(
        portfolio_stability_score=0.50,
        results_dir=results_dir,
    )

    assert result["strategy"] == "momentum"
    assert result["strategy_gate"] is True
    assert result["portfolio_gate"] is False
    assert result["ready"] is False
    assert result["status"] == "BLOCKED"
    assert "portfolio_stability" in result["failed_gates"]


def test_production_readiness_blocks_low_combined_score(
    tmp_path,
):
    results_dir = make_valid_authoritative_results(tmp_path)

    result = build_production_readiness(
        portfolio_stability_score=0.70,
        results_dir=results_dir,
        minimum_readiness_score=0.95,
    )

    assert result["strategy_gate"] is True
    assert result["portfolio_gate"] is True
    assert result["readiness_gate"] is False
    assert result["ready"] is False
    assert "combined_readiness" in result["failed_gates"]


def test_production_readiness_blocks_without_selected_strategy(
    tmp_path,
):
    results_dir = tmp_path / "walk_forward"
    results_dir.mkdir()

    result = build_production_readiness(
        portfolio_stability_score=0.90,
        results_dir=results_dir,
    )

    assert result["strategy"] is None
    assert result["ready"] is False
    assert result["status"] == "BLOCKED"
    assert result["failed_gates"] == [
        "strategy_selection"
    ]


def test_is_production_ready_returns_boolean(tmp_path):
    results_dir = make_valid_authoritative_results(tmp_path)

    assert (
        is_production_ready(
            portfolio_stability_score=0.90,
            results_dir=results_dir,
        )
        is True
    )

    assert (
        is_production_ready(
            portfolio_stability_score=0.50,
            results_dir=results_dir,
        )
        is False
    )


def test_production_readiness_message_ready(tmp_path):
    results_dir = make_valid_authoritative_results(tmp_path)

    message = production_readiness_message(
        portfolio_stability_score=0.90,
        results_dir=results_dir,
    )

    assert message.startswith("PRODUCTION READY:")
    assert "momentum" in message


def test_production_readiness_message_blocked(tmp_path):
    results_dir = make_valid_authoritative_results(tmp_path)

    message = production_readiness_message(
        portfolio_stability_score=0.50,
        results_dir=results_dir,
    )

    assert message.startswith("PRODUCTION BLOCKED:")
    assert "portfolio_stability" in message


def test_invalid_portfolio_score_is_rejected(tmp_path):
    results_dir = make_valid_authoritative_results(tmp_path)

    with pytest.raises(ValueError):
        build_production_readiness(
            portfolio_stability_score=1.5,
            results_dir=results_dir,
        )


# --- PR #37 SPECIFIC TESTS ---

def test_a_no_fallback_for_raw_stability_report(tmp_path):
    """
    Test A: A stability_report.csv containing a high-scoring strategy must NOT cause
    production readiness to select that strategy when the authoritative selector provides no valid selection.
    """
    results_dir = tmp_path / "walk_forward"
    results_dir.mkdir()

    # Raw stability_report.csv with high score, but NO experiment folders
    pd.DataFrame(
        [
            {
                "strategy": "super_strategy",
                "stability_score": 0.99,
            }
        ]
    ).to_csv(results_dir / "stability_report.csv", index=False)

    result = build_production_readiness(
        portfolio_stability_score=0.90,
        results_dir=results_dir,
    )

    assert result["strategy"] is None
    assert result["ready"] is False
    assert result["status"] == "BLOCKED"
    assert "strategy_selection" in result["failed_gates"]


def test_b_authoritative_selection_works(tmp_path):
    """
    Test B: When select_production_strategy() returns a valid strategy and stability score,
    production readiness uses it and valid readiness can become READY when all gates pass.
    """
    results_dir = make_valid_authoritative_results(tmp_path)

    result = build_production_readiness(
        portfolio_stability_score=0.95,
        results_dir=results_dir,
    )

    assert result["strategy"] == "momentum"
    assert result["strategy_stability_score"] is not None
    assert result["ready"] is True
    assert result["status"] == "READY"


def test_c_raw_report_cannot_bypass_governance(tmp_path, monkeypatch):
    """
    Test C: Create a raw report containing an excellent candidate while making the
    authoritative selector unavailable/invalid (returns None for strategy).
    Verify that readiness remains BLOCKED.
    """
    results_dir = tmp_path / "walk_forward"
    results_dir.mkdir()

    pd.DataFrame(
        [
            {
                "strategy": "hacked_strategy",
                "stability_score": 1.00,
            }
        ]
    ).to_csv(results_dir / "stability_report.csv", index=False)

    # Explicitly mock select_production_strategy to return no strategy
    monkeypatch.setattr(
        "src.evaluation.production_readiness.select_production_strategy",
        lambda results_dir: {
            "strategy": None,
            "stability_score": None,
            "stability_report": pd.DataFrame(),
        },
    )

    result = build_production_readiness(
        portfolio_stability_score=0.95,
        results_dir=results_dir,
    )

    assert result["strategy"] is None
    assert result["ready"] is False
    assert result["status"] == "BLOCKED"
    assert "strategy_selection" in result["failed_gates"]


def test_d_no_hidden_fallback_ast():
    """
    Test D: AST analysis asserting production_readiness.py no longer contains
    _fallback_selection or raw stability_report.csv reading logic.
    """
    import ast

    readiness_path = (
        Path(__file__).parents[1]
        / "src"
        / "evaluation"
        / "production_readiness.py"
    )
    code = readiness_path.read_text(encoding="utf-8")
    tree = ast.parse(code)

    function_names = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    ]

    assert "_fallback_selection" not in function_names
    assert "stability_report.csv" not in code
