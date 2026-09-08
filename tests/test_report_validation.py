import pytest

from src.evaluation.report_validation import (
validate_numeric_fields,
validate_report,
validate_report_fields,
)

@pytest.fixture
def sample_report():
return {
"total_return": 0.15,
"max_drawdown": -0.06,
"sharpe_ratio": 1.42,
"strategy": "XAU/USD",
}

def test_validate_report_accepts_mapping(sample_report):
assert validate_report(sample_report) is True

def test_validate_report_accepts_required_fields(sample_report):
assert validate_report(
sample_report,
required_fields=[
"total_return",
"max_drawdown",
"sharpe_ratio",
],
) is True

def test_validate_report_rejects_non_mapping():
with pytest.raises(TypeError):
validate_report(["invalid"])

def test_validate_report_rejects_missing_required_field(
sample_report,
):
with pytest.raises(ValueError):
validate_report(
sample_report,
required_fields=[
"total_return",
"missing_metric",
],
)

def test_validate_report_rejects_string_required_fields(
sample_report,
):
with pytest.raises(TypeError):
validate_report(
sample_report,
required_fields="total_return",
)

def test_validate_report_rejects_non_string_required_field(
sample_report,
):
with pytest.raises(TypeError):
validate_report(
sample_report,
required_fields=["total_return", 123],
)

def test_validate_numeric_fields_accepts_finite_values(
sample_report,
):
assert validate_numeric_fields(
sample_report,
[
"total_return",
"max_drawdown",
"sharpe_ratio",
],
) is True

def test_validate_numeric_fields_accepts_integers(
sample_report,
):
report = dict(sample_report)
report["trade_count"] = 25

assert validate_numeric_fields(
    report,
    ["trade_count"],
) is True

def test_validate_numeric_fields_rejects_missing_field(
sample_report,
):
with pytest.raises(ValueError):
validate_numeric_fields(
sample_report,
["missing_metric"],
)

def test_validate_numeric_fields_rejects_non_numeric_value(
sample_report,
):
with pytest.raises(ValueError):
validate_numeric_fields(
sample_report,
["strategy"],
)

def test_validate_numeric_fields_rejects_boolean(
sample_report,
):
report = dict(sample_report)
report["valid"] = True

with pytest.raises(ValueError):
    validate_numeric_fields(
        report,
        ["valid"],
    )

def test_validate_numeric_fields_rejects_nan(
sample_report,
):
report = dict(sample_report)
report["metric"] = float("nan")

with pytest.raises(ValueError):
    validate_numeric_fields(
        report,
        ["metric"],
    )

def test_validate_numeric_fields_rejects_infinity(
sample_report,
):
report = dict(sample_report)
report["metric"] = float("inf")

with pytest.raises(ValueError):
    validate_numeric_fields(
        report,
        ["metric"],
    )

def test_validate_numeric_fields_rejects_string_fields():
with pytest.raises(TypeError):
validate_numeric_fields(
{"metric": 1.0},
"metric",
)

def test_validate_report_fields_combines_validations(
sample_report,
):
assert validate_report_fields(
sample_report,
required_fields=[
"total_return",
"sharpe_ratio",
],
numeric_fields=[
"total_return",
"sharpe_ratio",
],
) is True

def test_validate_report_fields_accepts_only_required_fields(
sample_report,
):
assert validate_report_fields(
sample_report,
required_fields=["strategy"],
) is True

def test_validate_report_fields_accepts_only_numeric_fields(
sample_report,
):
assert validate_report_fields(
sample_report,
numeric_fields=["max_drawdown"],
) is True
