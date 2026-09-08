import pandas as pd
import pytest

from src.evaluation.risk_aware_selection import (
    apply_risk_limits,
    calculate_risk_adjusted_score,
    rank_risk_adjusted_strategies,
    select_best_risk_adjusted_strategy,
    select_with_risk_limits,
)


def make_report():
    return pd.DataFrame(
        {
            "strategy": [
                "robust",
                "high_return_risky",
                "weak",
            ],
            "total_return": [
                0.30,
                0.60,
                0.08,
            ],
            "max_drawdown": [
                -0.08,
                -0.35,
                -0.05,
            ],
            "sharpe_ratio": [
                1.50,
                1.30,
                0.40,
            ],
            "stability_score": [
                0.90,
                0.45,
                0.35,
            ],
            "worst_window_return": [
                -0.04,
                -0.30,
                -0.03,
            ],
            "max_consecutive_losses": [
                1,
                6,
                2,
            ],
        }
    )


def test_risk_adjusted_score_is_created():
    report = make_report()

    scored = calculate_risk_adjusted_score(
        report
    )

    assert "risk_adjusted_score" in scored.columns
    assert scored["risk_adjusted_score"].notna().all()


def test_input_is_not_modified():
    report = make_report()
    original = report.copy(deep=True)

    calculate_risk_adjusted_score(report)

    pd.testing.assert_frame_equal(
        report,
        original,
    )


def test_ranking_returns_expected_columns():
    ranked = rank_risk_adjusted_strategies(
        make_report()
    )

    assert "risk_adjusted_score" in ranked.columns
    assert list(ranked["strategy"]) == [
        "robust",
        "high_return_risky",
        "weak",
    ]


def test_best_risk_adjusted_strategy():
    best = select_best_risk_adjusted_strategy(
        make_report()
    )

    assert best == "robust"


def test_drawdown_limit():
    selected = apply_risk_limits(
        make_report(),
        max_drawdown=0.20,
    )

    assert "robust" in selected["strategy"].tolist()
    assert "high_return_risky" not in selected[
        "strategy"
    ].tolist()


def test_stability_limit():
    selected = apply_risk_limits(
        make_report(),
        min_stability_score=0.70,
    )

    assert list(selected["strategy"]) == [
        "robust"
    ]


def test_worst_window_limit():
    selected = apply_risk_limits(
        make_report(),
        min_worst_window_return=-0.10,
    )

    assert "high_return_risky" not in selected[
        "strategy"
    ].tolist()


def test_loss_streak_limit():
    selected = apply_risk_limits(
        make_report(),
        max_consecutive_losses=3,
    )

    assert "high_return_risky" not in selected[
        "strategy"
    ].tolist()


def test_combined_risk_selection():
    best = select_with_risk_limits(
        make_report(),
        max_drawdown=0.20,
        min_sharpe_ratio=1.0,
        min_stability_score=0.70,
        min_worst_window_return=-0.10,
        max_consecutive_losses=3,
    )

    assert best == "robust"


def test_no_strategy_passes_risk_limits():
    best = select_with_risk_limits(
        make_report(),
        max_drawdown=0.01,
        min_stability_score=0.95,
    )

    assert best is None


def test_empty_report():
    report = make_report().iloc[0:0]

    ranked = rank_risk_adjusted_strategies(
        report
    )

    assert ranked.empty
    assert "risk_adjusted_score" in ranked.columns


def test_missing_column():
    report = make_report().drop(
        columns=["sharpe_ratio"]
    )

    with pytest.raises(ValueError):
        calculate_risk_adjusted_score(report)


def test_invalid_drawdown_limit():
    with pytest.raises(ValueError):
        apply_risk_limits(
            make_report(),
            max_drawdown=-0.1,
        )


def test_invalid_stability_limit():
    with pytest.raises(ValueError):
        apply_risk_limits(
            make_report(),
            min_stability_score=1.5,
        )
