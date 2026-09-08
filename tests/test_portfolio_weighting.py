import pandas as pd
import pytest

from src.evaluation.portfolio_weighting import (
    allocate_portfolio,
    calculate_portfolio_weights,
    validate_portfolio_weights,
)


def make_report():
    return pd.DataFrame(
        {
            "strategy": [
                "trend",
                "momentum",
                "mean_reversion",
            ],
            "portfolio_score": [
                0.80,
                0.60,
                0.40,
            ],
            "stability_score": [
                0.90,
                0.80,
                0.70,
            ],
            "sharpe_ratio": [
                1.60,
                1.30,
                1.10,
            ],
            "max_drawdown": [
                -0.08,
                -0.10,
                -0.12,
            ],
        }
    )


def test_weights_are_created():
    result = calculate_portfolio_weights(
        make_report()
    )

    assert "portfolio_weight" in result.columns
    assert result["portfolio_weight"].notna().all()


def test_weights_sum_to_one():
    result = calculate_portfolio_weights(
        make_report()
    )

    assert result["portfolio_weight"].sum() == pytest.approx(
        1.0
    )


def test_weights_are_non_negative():
    result = calculate_portfolio_weights(
        make_report()
    )

    assert (
        result["portfolio_weight"] >= 0
    ).all()


def test_stronger_strategy_gets_higher_weight():
    result = calculate_portfolio_weights(
        make_report()
    )

    weights = dict(
        zip(
            result["strategy"],
            result["portfolio_weight"],
        )
    )

    assert weights["trend"] > weights["momentum"]
    assert weights["momentum"] > weights["mean_reversion"]


def test_max_weight_is_respected():
    result = calculate_portfolio_weights(
        make_report(),
        max_weight=0.50,
    )

    assert (
        result["portfolio_weight"]
        <= 0.50 + 1e-9
    ).all()

    assert result["portfolio_weight"].sum() == pytest.approx(
        1.0
    )


def test_zero_scores_produce_zero_weight():
    report = make_report()
    report["portfolio_score"] = [
        0.0,
        0.0,
        0.0,
    ]

    result = calculate_portfolio_weights(
        report
    )

    assert (
        result["portfolio_weight"] == 0.0
    ).all()


def test_empty_report():
    report = make_report().iloc[0:0]

    result = calculate_portfolio_weights(
        report
    )

    assert result.empty
    assert "portfolio_weight" in result.columns
    assert validate_portfolio_weights(result)


def test_validation_accepts_valid_weights():
    result = calculate_portfolio_weights(
        make_report()
    )

    assert validate_portfolio_weights(result)


def test_validation_rejects_negative_weights():
    report = make_report()
    report["portfolio_weight"] = [
        0.5,
        0.6,
        -0.1,
    ]

    assert not validate_portfolio_weights(
        report
    )


def test_validation_rejects_wrong_total():
    report = make_report()
    report["portfolio_weight"] = [
        0.2,
        0.2,
        0.2,
    ]

    assert not validate_portfolio_weights(
        report
    )


def test_allocate_portfolio():
    result = allocate_portfolio(
        make_report()
    )

    assert validate_portfolio_weights(
        result
    )


def test_invalid_max_weight():
    with pytest.raises(ValueError):
        calculate_portfolio_weights(
            make_report(),
            max_weight=0.0,
        )


def test_invalid_min_weight():
    with pytest.raises(ValueError):
        calculate_portfolio_weights(
            make_report(),
            min_weight=-0.1,
        )


def test_min_weight_cannot_exceed_max_weight():
    with pytest.raises(ValueError):
        calculate_portfolio_weights(
            make_report(),
            min_weight=0.7,
            max_weight=0.6,
        )


def test_missing_required_column():
    report = make_report().drop(
        columns=["portfolio_score"]
    )

    with pytest.raises(ValueError):
        calculate_portfolio_weights(report)


def test_input_is_not_modified():
    report = make_report()
    original = report.copy(deep=True)

    calculate_portfolio_weights(report)

    pd.testing.assert_frame_equal(
        report,
        original,
    )
