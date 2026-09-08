import pandas as pd
import pytest

from src.evaluation.portfolio_validation import (
    assert_valid_portfolio,
    validate_no_nan_values,
    validate_portfolio_report,
    validate_strategy_names,
    validate_target_weights,
    validate_weight_bounds,
    validate_weight_sum,
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


def test_strategy_names_are_valid():
    assert validate_strategy_names(
        make_report()
    ) is True


def test_duplicate_strategy_names_rejected():
    report = make_report()
    report.loc[1, "strategy"] = "trend"

    assert validate_strategy_names(
        report
    ) is False


def test_empty_strategy_name_rejected():
    report = make_report()
    report.loc[0, "strategy"] = "   "

    assert validate_strategy_names(
        report
    ) is False


def test_weight_sum_allows_cash():
    assert validate_weight_sum(
        make_report()
    ) is True


def test_full_weight_sum_required():
    assert validate_weight_sum(
        make_report(),
        require_full_allocation=True,
    ) is False

    report = make_report()
    report.loc[2, "portfolio_weight"] = 0.20

    assert validate_weight_sum(
        report,
        require_full_allocation=True,
    ) is True


def test_weight_bounds():
    assert validate_weight_bounds(
        make_report(),
        min_weight=0.0,
        max_weight=0.60,
    ) is True


def test_weight_above_bound_rejected():
    report = make_report()
    report.loc[0, "portfolio_weight"] = 0.70

    assert validate_weight_bounds(
        report,
        max_weight=0.60,
    ) is False


def test_negative_weight_rejected():
    report = make_report()
    report.loc[0, "portfolio_weight"] = -0.10

    with pytest.raises(ValueError):
        validate_weight_bounds(report)


def test_nan_values_rejected():
    report = make_report()

    values = report["portfolio_weight"].astype(object)
    values.iloc[0] = None
    report["portfolio_weight"] = values

    assert validate_no_nan_values(
        report
    ) is False


def test_invalid_weight_sum_rejected():
    report = make_report()
    report.loc[0, "portfolio_weight"] = 0.80
    report.loc[1, "portfolio_weight"] = 0.40

    with pytest.raises(ValueError):
        validate_weight_sum(report)


def test_portfolio_report_passes():
    result = validate_portfolio_report(
        make_report()
    )

    assert result["passed"] is True
    assert result["strategy_names_valid"] is True
    assert result["weights_valid"] is True
    assert result["weight_bounds_valid"] is True
    assert result["weight_sum_valid"] is True
    assert result["strategy_count"] == 3
    assert result["weight_sum"] == pytest.approx(0.90)


def test_portfolio_report_fails_on_duplicate_names():
    report = make_report()
    report.loc[1, "strategy"] = "trend"

    result = validate_portfolio_report(
        report
    )

    assert result["passed"] is False
    assert result["strategy_names_valid"] is False


def test_assert_valid_portfolio_passes():
    assert_valid_portfolio(
        make_report()
    )


def test_assert_valid_portfolio_rejects_invalid_report():
    report = make_report()
    report.loc[0, "portfolio_weight"] = 0.80
    report.loc[1, "portfolio_weight"] = 0.40

    with pytest.raises(ValueError):
        assert_valid_portfolio(report)


def test_target_weights_valid():
    assert validate_target_weights(
        {
            "trend": 0.50,
            "momentum": 0.30,
        }
    ) is True


def test_target_weights_full_allocation():
    assert validate_target_weights(
        {
            "trend": 0.60,
            "momentum": 0.40,
        },
        require_full_allocation=True,
    ) is True


def test_target_weights_partial_allocation():
    assert validate_target_weights(
        {
            "trend": 0.50,
            "momentum": 0.30,
        },
        require_full_allocation=True,
    ) is False


def test_target_weights_above_one_rejected():
    assert validate_target_weights(
        {
            "trend": 0.80,
            "momentum": 0.30,
        }
    ) is False


def test_target_negative_weight_rejected():
    assert validate_target_weights(
        {
            "trend": -0.10,
            "momentum": 0.50,
        }
    ) is False


def test_target_invalid_weight_rejected():
    assert validate_target_weights(
        {
            "trend": "invalid",
        }
    ) is False


def test_invalid_min_max_bounds():
    with pytest.raises(ValueError):
        validate_weight_bounds(
            make_report(),
            min_weight=0.70,
            max_weight=0.60,
        )

    with pytest.raises(ValueError):
        validate_weight_bounds(
            make_report(),
            min_weight=-0.10,
        )

    with pytest.raises(ValueError):
        validate_weight_bounds(
            make_report(),
            max_weight=1.10,
        )
