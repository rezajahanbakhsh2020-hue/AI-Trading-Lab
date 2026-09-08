import pytest

from src.evaluation.report_status import (
    build_status_summary,
    determine_report_status,
    is_report_acceptable,
)


@pytest.fixture
def passing_report():
    return {
        "sharpe_ratio": 1.5,
        "max_drawdown": -0.10,
        "win_rate": 0.60,
    }


@pytest.fixture
def failing_report():
    return {
        "sharpe_ratio": 0.7,
        "max_drawdown": -0.10,
        "win_rate": 0.60,
    }


def test_determine_report_status_passes_good_report(
    passing_report,
):
    assert determine_report_status(passing_report) == "pass"


def test_determine_report_status_fails_bad_report(
    failing_report,
):
    assert determine_report_status(failing_report) == "fail"


def test_determine_report_status_returns_incomplete_for_empty_report():
    assert determine_report_status({}) == "incomplete"


def test_determine_report_status_supports_partial_report():
    report = {
        "sharpe_ratio": 1.5,
    }

    assert determine_report_status(report) == "pass"


def test_determine_report_status_fails_when_drawdown_is_too_large():
    report = {
        "max_drawdown": -0.30,
    }

    assert determine_report_status(report) == "fail"


def test_determine_report_status_fails_when_win_rate_is_too_low():
    report = {
        "win_rate": 0.40,
    }

    assert determine_report_status(report) == "fail"


def test_determine_report_status_uses_custom_thresholds():
    report = {
        "sharpe_ratio": 1.2,
        "max_drawdown": -0.15,
        "win_rate": 0.55,
    }

    result = determine_report_status(
        report,
        minimum_sharpe=1.1,
        maximum_drawdown=-0.10,
        minimum_win_rate=0.50,
    )

    assert result == "fail"


def test_determine_report_status_accepts_boundary_values():
    report = {
        "sharpe_ratio": 1.0,
        "max_drawdown": -0.20,
        "win_rate": 0.50,
    }

    assert determine_report_status(report) == "pass"


def test_determine_report_status_rejects_non_mapping():
    with pytest.raises(TypeError):
        determine_report_status(["invalid"])


def test_determine_report_status_rejects_non_numeric_threshold():
    with pytest.raises(ValueError):
        determine_report_status(
            {"sharpe_ratio": 1.5},
            minimum_sharpe="1.0",
        )


def test_determine_report_status_rejects_boolean_threshold():
    with pytest.raises(ValueError):
        determine_report_status(
            {"sharpe_ratio": 1.5},
            minimum_sharpe=True,
        )


def test_is_report_acceptable_returns_true(
    passing_report,
):
    assert is_report_acceptable(passing_report) is True


def test_is_report_acceptable_returns_false(
    failing_report,
):
    assert is_report_acceptable(failing_report) is False


def test_build_status_summary_for_passing_report(
    passing_report,
):
    assert build_status_summary(passing_report) == {
        "status": "pass",
        "acceptable": True,
    }


def test_build_status_summary_for_failing_report(
    failing_report,
):
    assert build_status_summary(failing_report) == {
        "status": "fail",
        "acceptable": False,
    }


def test_build_status_summary_for_incomplete_report():
    assert build_status_summary({}) == {
        "status": "incomplete",
        "acceptable": False,
    }
