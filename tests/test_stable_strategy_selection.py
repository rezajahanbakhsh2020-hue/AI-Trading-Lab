import pandas as pd
import pytest

from src.evaluation.stable_strategy_selection import (
    rank_stable_strategies,
    select_best_stable_strategy,
    select_stable_strategies,
)


def make_report():
    return pd.DataFrame(
        {
            "strategy": [
                "stable",
                "fragile",
                "weak",
            ],
            "total_return": [
                0.35,
                0.60,
                0.10,
            ],
            "max_drawdown": [
                -0.10,
                -0.18,
                -0.08,
            ],
            "sharpe_ratio": [
                1.40,
                1.80,
                0.40,
            ],
            "positive_window_rate": [
                0.80,
                0.55,
                0.75,
            ],
            "stability_score": [
                0.82,
                0.58,
                0.40,
            ],
            "worst_window_return": [
                -0.06,
                -0.30,
                -0.02,
            ],
            "max_consecutive_losses": [
                2,
                5,
                1,
            ],
        }
    )


def test_stable_selection_applies_stability_gate():
    report = make_report()

    selected = select_stable_strategies(
        report,
        min_stability_score=0.70,
    )

    assert list(selected["strategy"]) == [
        "stable"
    ]


def test_worst_window_gate_removes_fragile_strategy():
    report = make_report()

    selected = select_stable_strategies(
        report,
        min_stability_score=0.50,
        min_worst_window_return=-0.10,
    )

    assert "stable" in selected["strategy"].tolist()
    assert "fragile" not in selected["strategy"].tolist()


def test_consecutive_loss_gate():
    report = make_report()

    selected = select_stable_strategies(
        report,
        min_stability_score=0.30,
        max_consecutive_losses=2,
    )

    assert "stable" in selected["strategy"].tolist()
    assert "fragile" not in selected["strategy"].tolist()


def test_ranking_prefers_stability():
    report = make_report()

    ranked = rank_stable_strategies(report)

    assert ranked.iloc[0]["strategy"] == "stable"


def test_best_strategy():
    report = make_report()

    best = select_best_stable_strategy(
        report,
        min_stability_score=0.70,
    )

    assert best == "stable"


def test_no_eligible_strategy():
    report = make_report()

    best = select_best_stable_strategy(
        report,
        min_stability_score=0.95,
    )

    assert best is None


def test_empty_report():
    report = make_report().iloc[0:0]

    selected = select_stable_strategies(report)

    assert selected.empty


def test_missing_column():
    report = make_report().drop(
        columns=["stability_score"]
    )

    with pytest.raises(ValueError):
        select_stable_strategies(report)


def test_invalid_stability_threshold():
    report = make_report()

    with pytest.raises(ValueError):
        select_stable_strategies(
            report,
            min_stability_score=1.5,
        )


def test_invalid_loss_threshold():
    report = make_report()

    with pytest.raises(ValueError):
        select_stable_strategies(
            report,
            max_consecutive_losses=-1,
        )
