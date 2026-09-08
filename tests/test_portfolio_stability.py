import pandas as pd
import pytest

from src.evaluation.portfolio_stability import (
    build_stability_summary,
    calculate_max_weight_change,
    calculate_mean_weight_change,
    calculate_stability_score,
    count_changed_strategies,
    find_changed_strategies,
    summarize_allocation_stability,
    validate_stability,
)


def make_report(weights):
    return pd.DataFrame(
        {
            "strategy": list(weights.keys()),
            "portfolio_weight": list(weights.values()),
        }
    )


def test_calculate_max_weight_change():
    current = make_report({"a": 0.6, "b": 0.4})
    previous = make_report({"a": 0.5, "b": 0.5})

    assert calculate_max_weight_change(current, previous) == pytest.approx(
        0.1
    )


def test_calculate_mean_weight_change():
    current = make_report({"a": 0.6, "b": 0.4})
    previous = make_report({"a": 0.5, "b": 0.5})

    assert calculate_mean_weight_change(current, previous) == pytest.approx(
        0.1
    )


def test_count_changed_strategies():
    current = make_report({"a": 0.6, "b": 0.4})
    previous = make_report({"a": 0.5, "b": 0.5})

    assert count_changed_strategies(current, previous) == 2


def test_find_changed_strategies():
    current = make_report({"a": 0.6, "b": 0.4})
    previous = make_report({"a": 0.5, "b": 0.5})

    assert find_changed_strategies(current, previous) == ["a", "b"]


def test_stability_score():
    current = make_report({"a": 0.6, "b": 0.4})
    previous = make_report({"a": 0.5, "b": 0.5})

    assert calculate_stability_score(current, previous) == pytest.approx(
        0.9
    )


def test_validate_stability():
    current = make_report({"a": 0.6, "b": 0.4})
    previous = make_report({"a": 0.5, "b": 0.5})

    assert validate_stability(current, previous, 0.1)
    assert not validate_stability(current, previous, 0.05)


def test_build_stability_summary():
    current = make_report({"a": 0.6, "b": 0.4})
    previous = make_report({"a": 0.5, "b": 0.5})

    summary = build_stability_summary(current, previous)

    assert summary["max_weight_change"] == pytest.approx(0.1)
    assert summary["mean_weight_change"] == pytest.approx(0.1)
    assert summary["changed_strategy_count"] == 2
    assert summary["changed_strategies"] == ["a", "b"]
    assert summary["stability_score"] == pytest.approx(0.9)


def test_summarize_allocation_stability_matches_builder():
    current = make_report({"a": 0.7, "b": 0.3})
    previous = make_report({"a": 0.7, "b": 0.3})

    assert summarize_allocation_stability(
        current, previous
    ) == build_stability_summary(current, previous)


def test_new_strategy_is_detected_as_changed():
    current = make_report({"a": 0.7, "b": 0.2, "c": 0.1})
    previous = make_report({"a": 0.7, "b": 0.3})

    assert find_changed_strategies(current, previous) == ["b", "c"]


def test_missing_required_column_is_rejected():
    current = pd.DataFrame(
        {"strategy": ["a"], "weight": [1.0]}
    )
    previous = make_report({"a": 1.0})

    with pytest.raises(ValueError):
        calculate_max_weight_change(current, previous)


def test_negative_tolerance_is_rejected():
    current = make_report({"a": 1.0})
    previous = make_report({"a": 1.0})

    with pytest.raises(ValueError):
        count_changed_strategies(current, previous, tolerance=-0.1)
