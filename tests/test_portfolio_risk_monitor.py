import pandas as pd
import pytest

from src.evaluation.portfolio_risk_monitor import (
    build_portfolio_risk_report,
    calculate_cash_weight,
    calculate_effective_strategy_count,
    calculate_max_strategy_weight,
    calculate_total_exposure,
    calculate_weight_concentration,
    check_portfolio_risk_limits,
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


def test_total_exposure():
    assert calculate_total_exposure(
        make_report()
    ) == pytest.approx(0.90)


def test_cash_weight():
    assert calculate_cash_weight(
        make_report()
    ) == pytest.approx(0.10)


def test_max_strategy_weight():
    assert calculate_max_strategy_weight(
        make_report()
    ) == pytest.approx(0.50)


def test_weight_concentration():
    assert calculate_weight_concentration(
        make_report()
    ) == pytest.approx(
        0.50**2 + 0.30**2 + 0.10**2
    )


def test_effective_strategy_count():
    concentration = (
        0.50**2
        + 0.30**2
        + 0.10**2
    )

    assert calculate_effective_strategy_count(
        make_report()
    ) == pytest.approx(
        1.0 / concentration
    )


def test_risk_limits_pass():
    result = check_portfolio_risk_limits(
        make_report(),
        max_strategy_weight=0.60,
        max_concentration=0.50,
    )

    assert result["passed"] is True
    assert result["max_weight_ok"] is True
    assert result["concentration_ok"] is True
    assert result["exposure_ok"] is True


def test_max_strategy_weight_limit_fails():
    result = check_portfolio_risk_limits(
        make_report(),
        max_strategy_weight=0.40,
        max_concentration=0.50,
    )

    assert result["passed"] is False
    assert result["max_weight_ok"] is False


def test_concentration_limit_fails():
    result = check_portfolio_risk_limits(
        make_report(),
        max_strategy_weight=0.60,
        max_concentration=0.30,
    )

    assert result["passed"] is False
    assert result["concentration_ok"] is False


def test_build_risk_report():
    result = build_portfolio_risk_report(
        make_report()
    )

    assert "total_exposure" in result.columns
    assert "cash_weight" in result.columns
    assert "concentration" in result.columns
    assert "effective_strategy_count" in result.columns
    assert "max_strategy_weight" in result.columns
    assert "risk_limits_passed" in result.columns

    assert len(result) == 3
    assert result["total_exposure"].iloc[0] == pytest.approx(
        0.90
    )


def test_negative_weight_rejected():
    report = make_report()
    report.loc[0, "portfolio_weight"] = -0.10

    with pytest.raises(ValueError):
        calculate_total_exposure(report)


def test_weights_above_one_rejected():
    report = make_report()
    report.loc[0, "portfolio_weight"] = 0.90

    with pytest.raises(ValueError):
        calculate_total_exposure(report)


def test_invalid_weight_rejected():
    report = make_report()

    values = report["portfolio_weight"].astype(object)
    values.iloc[0] = "invalid"
    report["portfolio_weight"] = values

    with pytest.raises(ValueError):
        calculate_total_exposure(report)


def test_empty_report():
    report = pd.DataFrame(
        columns=[
            "strategy",
            "portfolio_weight",
        ]
    )

    assert calculate_total_exposure(
        report
    ) == pytest.approx(0.0)

    assert calculate_cash_weight(
        report
    ) == pytest.approx(1.0)

    assert calculate_max_strategy_weight(
        report
    ) == pytest.approx(0.0)

    assert calculate_weight_concentration(
        report
    ) == pytest.approx(0.0)

    assert calculate_effective_strategy_count(
        report
    ) == pytest.approx(0.0)


def test_missing_required_column_rejected():
    report = pd.DataFrame(
        {
            "strategy": ["trend"],
        }
    )

    with pytest.raises(ValueError):
        calculate_total_exposure(report)


def test_invalid_risk_limit_rejected():
    with pytest.raises(ValueError):
        check_portfolio_risk_limits(
            make_report(),
            max_strategy_weight=1.5,
        )

    with pytest.raises(ValueError):
        check_portfolio_risk_limits(
            make_report(),
            max_concentration=-0.1,
        )
