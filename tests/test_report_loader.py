import json

import pytest

from src.evaluation.report_loader import (
    load_latest_report,
    load_report,
    report_exists,
)


@pytest.fixture
def sample_report():
    return {
        "total_return": 0.15,
        "max_drawdown": -0.06,
        "sharpe_ratio": 1.42,
    }


def test_load_report_reads_json(sample_report, tmp_path):
    path = tmp_path / "report.json"

    path.write_text(
        json.dumps(sample_report),
        encoding="utf-8",
    )

    result = load_report(path)

    assert result == sample_report


def test_load_report_supports_path_string(
    sample_report,
    tmp_path,
):
    path = tmp_path / "report.json"

    path.write_text(
        json.dumps(sample_report),
        encoding="utf-8",
    )

    result = load_report(str(path))

    assert result == sample_report


def test_load_report_supports_unicode(
    tmp_path,
):
    report = {
        "strategy": "XAU/USD",
        "description": "گزارش ارزیابی",
    }

    path = tmp_path / "report.json"

    path.write_text(
        json.dumps(report, ensure_ascii=False),
        encoding="utf-8",
    )

    assert load_report(path) == report


def test_load_report_raises_for_missing_file(tmp_path):
    path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        load_report(path)


def test_load_report_rejects_directory(tmp_path):
    directory = tmp_path / "report"

    directory.mkdir()

    with pytest.raises(ValueError):
        load_report(directory)


def test_load_report_rejects_invalid_json(tmp_path):
    path = tmp_path / "invalid.json"

    path.write_text(
        "{invalid json",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_report(path)


def test_load_report_rejects_non_object_json(tmp_path):
    path = tmp_path / "list.json"

    path.write_text(
        json.dumps([1, 2, 3]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_report(path)


def test_report_exists_returns_true_for_file(tmp_path):
    path = tmp_path / "report.json"

    path.write_text("{}", encoding="utf-8")

    assert report_exists(path) is True


def test_report_exists_returns_false_for_missing_file(
    tmp_path,
):
    assert report_exists(tmp_path / "missing.json") is False


def test_report_exists_returns_false_for_directory(tmp_path):
    directory = tmp_path / "reports"

    directory.mkdir()

    assert report_exists(directory) is False


def test_load_latest_report_returns_newest(
    sample_report,
    tmp_path,
):
    first = tmp_path / "evaluation_report_20260908T090000000000Z.json"
    second = tmp_path / "evaluation_report_20260908T100000000000Z.json"

    first.write_text(
        json.dumps({"version": 1}),
        encoding="utf-8",
    )

    second.write_text(
        json.dumps(sample_report),
        encoding="utf-8",
    )

    result = load_latest_report(tmp_path)

    assert result == sample_report


def test_load_latest_report_supports_custom_name(
    sample_report,
    tmp_path,
):
    path = tmp_path / "xauusd_20260908T100000000000Z.json"

    path.write_text(
        json.dumps(sample_report),
        encoding="utf-8",
    )

    result = load_latest_report(
        tmp_path,
        name="xauusd",
    )

    assert result == sample_report


def test_load_latest_report_raises_when_empty(
    tmp_path,
):
    with pytest.raises(FileNotFoundError):
        load_latest_report(tmp_path)
