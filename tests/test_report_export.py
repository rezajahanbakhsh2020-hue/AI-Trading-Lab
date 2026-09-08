import json

import pytest

from src.evaluation.report_export import (
    export_report,
    export_report_csv,
    export_report_json,
)


@pytest.fixture
def sample_report():
    return {
        "total_return": 0.1286,
        "max_drawdown": -0.05,
        "sharpe_ratio": 1.25,
        "win_rate": 2 / 3,
        "profit_factor": 3.6,
    }


def test_export_report_json(sample_report, tmp_path):
    output_path = tmp_path / "report.json"

    result = export_report_json(sample_report, output_path)

    assert result == output_path
    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert data == sample_report


def test_export_report_json_creates_parent_directories(
    sample_report,
    tmp_path,
):
    output_path = tmp_path / "nested" / "reports" / "report.json"

    export_report_json(sample_report, output_path)

    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert data["total_return"] == sample_report["total_return"]


def test_export_report_csv(sample_report, tmp_path):
    output_path = tmp_path / "report.csv"

    result = export_report_csv(sample_report, output_path)

    assert result == output_path
    assert output_path.exists()

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert "total_return" in lines[0]
    assert "max_drawdown" in lines[0]
    assert "0.1286" in lines[1]


def test_export_report_creates_both_formats(
    sample_report,
    tmp_path,
):
    json_path = tmp_path / "output" / "report.json"
    csv_path = tmp_path / "output" / "report.csv"

    result = export_report(
        sample_report,
        json_path=json_path,
        csv_path=csv_path,
    )

    assert result == {
        "json": json_path,
        "csv": csv_path,
    }

    assert json_path.exists()
    assert csv_path.exists()


def test_export_report_supports_json_only(
    sample_report,
    tmp_path,
):
    json_path = tmp_path / "report.json"

    result = export_report(
        sample_report,
        json_path=json_path,
    )

    assert result == {"json": json_path}
    assert json_path.exists()


def test_export_report_supports_csv_only(
    sample_report,
    tmp_path,
):
    csv_path = tmp_path / "report.csv"

    result = export_report(
        sample_report,
        csv_path=csv_path,
    )

    assert result == {"csv": csv_path}
    assert csv_path.exists()


def test_export_report_requires_output_path(sample_report):
    with pytest.raises(ValueError):
        export_report(sample_report)


def test_export_report_rejects_invalid_report(tmp_path):
    with pytest.raises(TypeError):
        export_report_json(
            ["not", "a", "mapping"],
            tmp_path / "report.json",
        )


def test_export_report_csv_rejects_invalid_report(tmp_path):
    with pytest.raises(TypeError):
        export_report_csv(
            ["not", "a", "mapping"],
            tmp_path / "report.csv",
        )
