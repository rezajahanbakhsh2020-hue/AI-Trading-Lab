import pandas as pd
import pytest

from src.evaluation.portfolio_rebalancing import (
    build_rebalancing_plan,
    calculate_rebalance_capital,
    calculate_rebalancing,
    calculate_turnover,
    should_rebalance,
)


def make_report():
    return pd.DataFrame(
        {
            "strategy": [
                "trend",
                "momentum",
            ],
            "portfolio_weight": [
                0.60,
                0.30,
            ],
        }
    )


def make_targets():
    return {
        "trend": 0.50,
        "momentum": 0.40,
    }


def test_rebalancing_calculates_weight_changes():
    result = calculate_rebalancing(
        make_report(),
        make_targets(),
    )

    assert result["current_weight"].tolist() == [
        pytest.approx(0.60),
        pytest.approx(0.30),
    ]

    assert result["target_weight"].tolist() == [
        pytest.approx(0.50),
        pytest.approx(0.40),
    ]

    assert result["weight_change"].tolist() == [
        pytest.approx(-0.10),
        pytest.approx(0.10),
    ]


def test_turnover_is_calculated():
    result = calculate_rebalancing(
        make_report(),
        make_targets(),
    )

    assert calculate_turnover(
        result
    ) == pytest.approx(0.10)


def test_new_strategy_is_supported():
    result = calculate_rebalancing(
        make_report(),
        {
            "trend": 0.50,
            "momentum": 0.30,
            "mean_reversion": 0.10,
        },
    )

    row = result[
        result["strategy"] == "mean_reversion"
    ].iloc[0]

    assert row["current_weight"] == pytest.approx(0.0)
    assert row["target_weight"] == pytest.approx(0.10)
    assert row["weight_change"] == pytest.approx(0.10)


def test_removed_strategy_gets_zero_target():
    result = calculate_rebalancing(
        make_report(),
        {
            "trend": 0.50,
        },
    )

    row = result[
        result["strategy"] == "momentum"
    ].iloc[0]

    assert row["target_weight"] == pytest.approx(0.0)
    assert row["weight_change"] == pytest.approx(-0.30)


def test_rebalance_capital():
    result = calculate_rebalancing(
        make_report(),
        make_targets(),
    )

    result = calculate_rebalance_capital(
        result,
        10000.0,
    )

    assert result["capital_change"].tolist() == [
        pytest.approx(-1000.0),
        pytest.approx(1000.0),
    ]


def test_build_rebalancing_plan():
    result = build_rebalancing_plan(
        make_report(),
        make_targets(),
        capital=10000.0,
    )

    assert "weight_change" in result.columns
    assert "absolute_weight_change" in result.columns
    assert "rebalance_required" in result.columns
    assert "turnover" in result.columns
    assert "capital_change" in result.columns

    assert result["turnover"].iloc[0] == pytest.approx(
        0.10
    )


def test_should_rebalance_above_threshold():
    assert should_rebalance(
        make_report(),
        make_targets(),
        threshold=0.05,
    ) is True


def test_should_not_rebalance_below_threshold():
    assert should_rebalance(
        make_report(),
        make_targets(),
        threshold=0.15,
    ) is False


def test_negative_threshold_rejected():
    with pytest.raises(ValueError):
        should_rebalance(
            make_report(),
            make_targets(),
            threshold=-0.01,
        )


def test_negative_capital_rejected():
    result = calculate_rebalancing(
        make_report(),
        make_targets(),
    )

    with pytest.raises(ValueError):
        calculate_rebalance_capital(
            result,
            -100.0,
        )


def test_negative_current_weight_rejected():
    report = make_report()
    report.loc[0, "portfolio_weight"] = -0.10

    with pytest.raises(ValueError):
        calculate_rebalancing(
            report,
            make_targets(),
        )


def test_target_weights_above_one_rejected():
    with pytest.raises(ValueError):
        calculate_rebalancing(
            make_report(),
            {
                "trend": 0.80,
                "momentum": 0.40,
            },
        )


def test_invalid_target_weight_rejected():
    with pytest.raises(ValueError):
        calculate_rebalancing(
            make_report(),
            {
                "trend": "invalid",
                "momentum": 0.40,
            },
        )


def test_invalid_report_weight_rejected():
    report = make_report()

    values = report["portfolio_weight"].astype(object)
    values.iloc[0] = "invalid"
    report["portfolio_weight"] = values

    with pytest.raises(ValueError):
        calculate_rebalancing(
            report,
            make_targets(),
        )


def test_empty_report_with_targets():
    report = pd.DataFrame(
        {
            "strategy": pd.Series(dtype=str),
            "portfolio_weight": pd.Series(dtype=float),
        }
    )

    result = calculate_rebalancing(
        report,
        {
            "trend": 0.60,
            "momentum": 0.30,
        },
    )

    assert len(result) == 2
    assert result["current_weight"].tolist() == [
        pytest.approx(0.0),
        pytest.approx(0.0),
    ]

    assert calculate_turnover(
        result
    ) == pytest.approx(0.45)
