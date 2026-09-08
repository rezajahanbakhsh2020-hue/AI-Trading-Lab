import pandas as pd
import pytest

from src.evaluation.portfolio_constraints import (
    apply_weight_constraints,
    cap_strategy_weights,
    check_weight_constraints,
    enforce_max_strategy_count,
    normalize_weights,
    validate_portfolio_constraints,
)


def make_report():
    return pd.DataFrame(
        {
            "strategy": [
                "trend",
                "momentum",
                "mean_reversion",
            ],
            "portfolio_weight": [
                0.50,
                0.30,
                0.10,
            ],
        }
    )


def test_weight_constraints_pass():
    result = check_weight_constraints(
        make_report(),
        max_strategy_weight=0.60,
        max_strategies=3,
    )

    assert result["passed"] is True
    assert result["max_weight_ok"] is True
    assert result["strategy_count_ok"] is True


def test_max_weight_constraint_fails():
    result = check_weight_constraints(
        make_report(),
        max_strategy_weight=0.40,
    )

    assert result["passed"] is False
    assert result["max_weight_ok"] is False


def test_min_weight_constraint_fails():
    result = check_weight_constraints(
        make_report(),
        min_strategy_weight=0.20,
    )

    assert result["passed"] is False
    assert result["min_weight_ok"] is False


def test_strategy_count_constraint_fails():
    result = check_weight_constraints(
        make_report(),
        max_strategies=2,
    )

    assert result["passed"] is False
    assert result["strategy_count_ok"] is False
    assert result["strategy_count"] == 3


def test_apply_weight_constraints():
    result = apply_weight_constraints(
        make_report(),
        max_strategy_weight=0.40,
        min_strategy_weight=0.20,
    )

    assert result["strategy"].tolist() == [
        "momentum",
    ]

    assert result["portfolio_weight"].tolist() == [
        pytest.approx(0.30),
    ]


def test_enforce_max_strategy_count():
    result = enforce_max_strategy_count(
        make_report(),
        max_strategies=2,
    )

    assert result["strategy"].tolist() == [
        "trend",
        "momentum",
    ]

    assert result["portfolio_weight"].tolist() == [
        pytest.approx(0.50),
        pytest.approx(0.30),
    ]


def test_cap_strategy_weights():
    result = cap_strategy_weights(
        make_report(),
        max_strategy_weight=0.35,
    )

    assert result["portfolio_weight"].tolist() == [
        pytest.approx(0.35),
        pytest.approx(0.30),
        pytest.approx(0.10),
    ]


def test_normalize_weights():
    result = normalize_weights(
        {
            "trend": 2.0,
            "momentum": 1.0,
            "mean_reversion": 1.0,
        }
    )

    assert result["trend"] == pytest.approx(0.50)
    assert result["momentum"] == pytest.approx(0.25)
    assert result["mean_reversion"] == pytest.approx(0.25)
    assert sum(result.values()) == pytest.approx(1.0)


def test_zero_weights_normalize_to_zero():
    result = normalize_weights(
        {
            "trend": 0.0,
            "momentum": 0.0,
        }
    )

    assert result == {
        "trend": 0.0,
        "momentum": 0.0,
    }


def test_empty_weights():
    assert normalize_weights({}) == {}


def test_negative_weight_rejected():
    with pytest.raises(ValueError):
        normalize_weights(
            {
                "trend": -0.10,
            }
        )


def test_invalid_weight_rejected():
    with pytest.raises(ValueError):
        normalize_weights(
            {
                "trend": "invalid",
            }
        )


def test_report_negative_weight_rejected():
    report = make_report()
    report.loc[0, "portfolio_weight"] = -0.10

    with pytest.raises(ValueError):
        check_weight_constraints(report)


def test_report_weight_above_one_rejected():
    report = make_report()
    report.loc[0, "portfolio_weight"] = 0.80
    report.loc[1, "portfolio_weight"] = 0.40

    with pytest.raises(ValueError):
        check_weight_constraints(report)


def test_invalid_report_weight_rejected():
    report = make_report()

    values = report["portfolio_weight"].astype(object)
    values.iloc[0] = "invalid"
    report["portfolio_weight"] = values

    with pytest.raises(ValueError):
        check_weight_constraints(report)


def test_invalid_limits_rejected():
    with pytest.raises(ValueError):
        check_weight_constraints(
            make_report(),
            max_strategy_weight=1.2,
        )

    with pytest.raises(ValueError):
        check_weight_constraints(
            make_report(),
            min_strategy_weight=-0.1,
        )

    with pytest.raises(ValueError):
        check_weight_constraints(
            make_report(),
            min_strategy_weight=0.70,
            max_strategy_weight=0.60,
        )


def test_validate_portfolio_constraints():
    assert validate_portfolio_constraints(
        make_report(),
        max_strategy_weight=0.60,
        max_strategies=3,
    ) is True

    assert validate_portfolio_constraints(
        make_report(),
        max_strategy_weight=0.40,
    ) is False


def test_empty_report_passes_constraints():
    report = pd.DataFrame(
        {
            "strategy": pd.Series(dtype=str),
            "portfolio_weight": pd.Series(dtype=float),
        }
    )

    assert validate_portfolio_constraints(
        report
    ) is True
