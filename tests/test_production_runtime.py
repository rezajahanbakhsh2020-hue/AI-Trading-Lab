from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.production_runtime import (
    build_production_runtime_decision,
    is_production_runtime_allowed,
    production_runtime_decision_message,
)


def _write_stability_report(
    results_dir: Path,
) -> None:
    exp_dir = results_dir / "exp_001"
    exp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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
                "strategy": "mean_reversion",
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
            {"strategy": "mean_reversion"},
        ]
    ).to_csv(
        exp_dir / "eligible_strategies.csv",
        index=False,
    )

    (exp_dir / "metadata.json").write_text(
        '{"created_at_utc": "2025-01-01T00:00:00Z", "best_strategy": "momentum"}',
        encoding="utf-8",
    )


def test_runtime_decision_allows_ready_strategy(
    tmp_path,
):
    _write_stability_report(tmp_path)

    decision = build_production_runtime_decision(
        0.90,
        results_dir=tmp_path,
    )

    assert decision["runtime_allowed"] is True
    assert decision["status"] == "READY"
    assert decision["strategy"] == "momentum"

    assert decision["readiness"]["status"] == "READY"
    assert (
        decision["runtime_gate"]["runtime_ready"]
        is True
    )


def test_runtime_decision_blocks_low_portfolio_stability(
    tmp_path,
):
    _write_stability_report(tmp_path)

    decision = build_production_runtime_decision(
        0.40,
        results_dir=tmp_path,
    )

    assert decision["runtime_allowed"] is False
    assert decision["status"] == "BLOCKED"
    assert "portfolio_stability" in decision[
        "runtime_gate"
    ]["failed_gates"]


def test_runtime_decision_blocks_low_strategy_stability(
    tmp_path,
):
    frame = pd.DataFrame(
        [
            {
                "strategy": "momentum",
                "stability_score": 0.50,
            }
        ]
    )

    tmp_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        tmp_path / "stability_report.csv",
        index=False,
    )

    decision = build_production_runtime_decision(
        0.90,
        results_dir=tmp_path,
    )

    assert decision["runtime_allowed"] is False
    assert decision["status"] == "BLOCKED"


def test_runtime_permission_returns_boolean(
    tmp_path,
):
    _write_stability_report(tmp_path)

    allowed = is_production_runtime_allowed(
        0.90,
        results_dir=tmp_path,
    )

    blocked = is_production_runtime_allowed(
        0.40,
        results_dir=tmp_path,
    )

    assert allowed is True
    assert blocked is False


def test_runtime_message_allowed(
    tmp_path,
):
    _write_stability_report(tmp_path)

    message = production_runtime_decision_message(
        0.90,
        results_dir=tmp_path,
    )

    assert message == (
        "PRODUCTION RUNTIME ALLOWED: momentum"
    )


def test_runtime_message_blocked(
    tmp_path,
):
    _write_stability_report(tmp_path)

    message = production_runtime_decision_message(
        0.40,
        results_dir=tmp_path,
    )

    assert message.startswith(
        "PRODUCTION RUNTIME BLOCKED:"
    )
    assert "portfolio_stability" in message


def test_invalid_portfolio_score_is_rejected(
    tmp_path,
):
    _write_stability_report(tmp_path)

    with pytest.raises(ValueError):
        build_production_runtime_decision(
            1.5,
            results_dir=tmp_path,
        )
