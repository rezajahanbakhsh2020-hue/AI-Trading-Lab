import pandas as pd
import pytest

from src.evaluation.portfolio_allocation import (
    build_capital_allocation,
    calculate_capital_allocation,
    calculate_exposure,
    calculate_unallocated_capital,
    validate_capital_allocation,
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
                0.20,
            ],
        }
    )


def test_capital_allocation_is_created():
    result = calculate_capital_allocation(
        make_report(),
        capital=10000.0,
    )

    assert "allocated_capital" in result.columns
    assert result["allocated_capital"].tolist() == [
        pytest.approx(5000.0),
        pytest.approx(3000.0),
        pytest.approx(2000.0),
    ]


def test_allocated_capital_equals_total_capital():
    result = calculate_capital_allocation(
        make_report(),
        capital=10000.0,
    )

    assert result["allocated_capital"].sum() == pytest.approx(
        10000.0
    )


def test_exposure_matches_weight():
    result = calculate_exposure(
        make_report(),
        capital=10000.0,
    )

    assert result["exposure"].tolist() == [
        pytest.approx(0.50),
        pytest.approx(0.30),
        pytest.approx(0.20),
    ]


def test_unallocated_capital():
    report = pd.DataFrame(
        {
            "strategy": [
                "trend",
                "momentum",
            ],
            "portfolio_weight": [
                0.50,
                0.30,
            ],
        }
    )

    remaining = calculate_unallocated_capital(
        report,
        capital=10000.0,
    )

    assert remaining == pytest.approx(
        2000.0
    )


def test_zero_capital():
    result = calculate_exposure(
        make_report(),
        capital=0.0,
    )

    assert (
        result["allocated_capital"] == 0.0
    ).all()

    assert (
        result["exposure"] == 0.0
    ).all()


def test_empty_report():
    report = make_report().iloc[0:0]

    result = build_capital_allocation(
        report,
        capital=10000.0,
    )

    assert result.empty


def test_validation_accepts_valid_allocation():
    result = calculate_capital_allocation(
        make_report(),
        capital=10000.0,
    )

    assert validate_capital_allocation(
        result,
        capital=10000.0,
    )


def test_negative_capital_rejected():
    with pytest.raises(ValueError):
        calculate_capital_allocation(
            make_report(),
            capital=-100.0,
        )


def test_negative_weight_rejected():
    report = make_report()
    report.loc[0, "portfolio_weight"] = -0.1

    with pytest.raises(ValueError):
        calculate_capital_allocation(
            report,
            capital=10000.0,
        )


def test_weights_must_sum_to_one():
    report = make_report()
    report["portfolio_weight"] = [
        0.40,
        0.30,
        0.20,
    ]

    with pytest.raises(ValueError):
        calculate_capital_allocation(
            report,
            capital=10000.0,
        )


def test_missing_weight_column():
    report = make_report().drop(
        columns=["portfolio_weight"]
    )

    with pytest.raises(ValueError):
        calculate_capital_allocation(
            report,
            capital=10000.0,
        )


def test_input_is_not_modified():
    report = make_report()
    original = report.copy(deep=True)

    calculate_capital_allocation(
        report,
        capital=10000.0,
    )

    pd.testing.assert_frame_equal(
        report,
        original,
    )


def test_final_allocation_contains_exposure():
    result = build_capital_allocation(
        make_report(),
        capital=25000.0,
    )

    assert "allocated_capital" in result.columns
    assert "exposure" in result.columns
    assert result["exposure"].sum() == pytest.approx(
        1.0
    )
