from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.production_readiness import (
    build_production_readiness,
    is_production_ready,
    production_readiness_message,
)


def make_results(tmp_path: Path) -> Path:
    results_dir = tmp_path / "walk_forward"
    results_dir.mkdir()

    pd.DataFrame(
        [
            {
                "strategy": "momentum",
                "stability_score": 0.90,
            },
            {
                "strategy": "baseline",
                "stability_score": 0.60,
            },
        ]
    ).to_csv(
        results_dir / "stability_report.csv",
        index=False,
    )

    return results_dir


def test_production_readiness_passes_with_stable_portfolio(
    tmp_path,
):
    results_dir = make_results(tmp_path)

    result = build_production_readiness(
        portfolio_stability_score=0.90,
        results_dir=results_dir,
    )

    assert result["strategy"] == "momentum"
    assert result["strategy_stability_score"] == pytest.approx(
        0.90
    )
    assert result["portfolio_stability_score"] == pytest.approx(
        0.90
    )
    assert result["readiness_score"] == pytest.approx(
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
    results_dir = make_results(tmp_path)

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
    results_dir = make_results(tmp_path)

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
    results_dir = make_results(tmp_path)

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
    results_dir = make_results(tmp_path)

    message = production_readiness_message(
        portfolio_stability_score=0.90,
        results_dir=results_dir,
    )

    assert message.startswith("PRODUCTION READY:")
    assert "momentum" in message


def test_production_readiness_message_blocked(tmp_path):
    results_dir = make_results(tmp_path)

    message = production_readiness_message(
        portfolio_stability_score=0.50,
        results_dir=results_dir,
    )

    assert message.startswith("PRODUCTION BLOCKED:")
    assert "portfolio_stability" in message


def test_invalid_portfolio_score_is_rejected(tmp_path):
    results_dir = make_results(tmp_path)

    with pytest.raises(ValueError):
        build_production_readiness(
            portfolio_stability_score=1.5,
            results_dir=results_dir,
        )
