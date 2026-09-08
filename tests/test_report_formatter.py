import pytest

from src.evaluation.report_formatter import (
    format_report_markdown,
    format_report_summary,
    format_report_text,
)


@pytest.fixture
def sample_report():
    return {
        "total_return": 0.125,
        "max_drawdown": -0.05,
        "sharpe_ratio": 1.23456,
        "strategy": "XAU/USD",
    }


def test_format_report_text_formats_numeric_values(sample_report):
    result = format_report_text(sample_report)

    assert "total_return: 0.1250" in result
    assert "max_drawdown: -0.0500" in result
    assert "sharpe_ratio: 1.2346" in result
    assert "strategy: XAU/USD" in result


def test_format_report_text_supports_custom_precision():
    report = {
        "metric": 1.23456,
    }

    result = format_report_text(
        report,
        precision=2,
    )

    assert result == "metric: 1.23"


def test_format_report_text_supports_integer_values():
    report = {
        "trade_count": 10,
    }

    result = format_report_text(report)

    assert result == "trade_count: 10.0000"


def test_format_report_text_supports_boolean_values():
    report = {
        "acceptable": True,
    }

    result = format_report_text(report)

    assert result == "acceptable: True"


def test_format_report_text_supports_empty_report():
    assert format_report_text({}) == ""


def test_format_report_text_rejects_non_mapping():
    with pytest.raises(TypeError):
        format_report_text(["invalid"])


def test_format_report_text_rejects_negative_precision(
    sample_report,
):
    with pytest.raises(ValueError):
        format_report_text(
            sample_report,
            precision=-1,
        )


def test_format_report_markdown_creates_table(sample_report):
    result = format_report_markdown(sample_report)

    assert result.startswith("# Evaluation Report")
    assert "| Metric | Value |" in result
    assert "| --- | --- |" in result
    assert "| total_return | 0.125 |" in result
    assert "| strategy | XAU/USD |" in result


def test_format_report_markdown_supports_custom_title():
    report = {
        "sharpe_ratio": 1.5,
    }

    result = format_report_markdown(
        report,
        title="XAU/USD Results",
    )

    assert result.startswith("# XAU/USD Results")


def test_format_report_markdown_supports_empty_report():
    result = format_report_markdown({})

    assert result == (
        "# Evaluation Report\n\n"
        "| Metric | Value |\n"
        "| --- | --- |"
    )


def test_format_report_markdown_rejects_non_mapping():
    with pytest.raises(TypeError):
        format_report_markdown(["invalid"])


def test_format_report_markdown_rejects_non_string_title():
    with pytest.raises(TypeError):
        format_report_markdown(
            {"metric": 1.0},
            title=123,
        )


def test_format_report_summary_includes_main_metrics(
    sample_report,
):
    result = format_report_summary(sample_report)

    assert "total_return=0.125" in result
    assert "max_drawdown=-0.05" in result
    assert "sharpe_ratio=1.23456" in result


def test_format_report_summary_ignores_unrelated_fields():
    report = {
        "total_return": 0.10,
        "strategy": "XAU/USD",
        "custom": 123,
    }

    result = format_report_summary(report)

    assert result == "total_return=0.1"


def test_format_report_summary_supports_empty_report():
    assert format_report_summary({}) == ""


def test_format_report_summary_rejects_non_mapping():
    with pytest.raises(TypeError):
        format_report_summary(["invalid"])
