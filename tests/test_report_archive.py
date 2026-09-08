import json

import pytest

from src.evaluation.report_archive import (
    archive_report,
    latest_archived_report,
    list_archived_reports,
)


@pytest.fixture
def sample_report():
    return {
        "total_return": 0.125,
        "max_drawdown": -0.04,
        "sharpe_ratio": 1.35,
    }


def test_archive_report_creates_timestamped_json(
    sample_report,
    tmp_path,
):
    result = archive_report(
        sample_report,
        tmp_path,
    )

    assert result.exists()
    assert result.suffix == ".json"
    assert result.parent == tmp_path
    assert result.name.startswith("evaluation_report_")

    with result.open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert data == sample_report


def test_archive_report_creates_directory(
    sample_report,
    tmp_path,
):
    archive_dir = tmp_path / "archives" / "reports"

    result = archive_report(
        sample_report,
        archive_dir,
    )

    assert archive_dir.exists()
    assert result.exists()


def test_archive_report_supports_custom_name(
    sample_report,
    tmp_path,
):
    result = archive_report(
        sample_report,
        tmp_path,
        name="xauusd_evaluation",
    )

    assert result.name.startswith("xauusd_evaluation_")


def test_list_archived_reports_returns_matching_files(
    sample_report,
    tmp_path,
):
    first = archive_report(
        sample_report,
        tmp_path,
        name="evaluation",
    )

    second = archive_report(
        sample_report,
        tmp_path,
        name="evaluation",
    )

    unrelated = tmp_path / "other.json"
    unrelated.write_text("{}", encoding="utf-8")

    reports = list_archived_reports(
        tmp_path,
        name="evaluation",
    )

    assert reports == sorted([first, second])
    assert unrelated not in reports


def test_list_archived_reports_returns_empty_for_missing_directory(
    tmp_path,
):
    reports = list_archived_reports(
        tmp_path / "missing",
    )

    assert reports == []


def test_latest_archived_report_returns_newest(
    sample_report,
    tmp_path,
):
    first = archive_report(
        sample_report,
        tmp_path,
        name="evaluation",
    )

    second = archive_report(
        sample_report,
        tmp_path,
        name="evaluation",
    )

    latest = latest_archived_report(
        tmp_path,
        name="evaluation",
    )

    assert latest == sorted([first, second])[-1]


def test_latest_archived_report_returns_none_when_empty(
    tmp_path,
):
    assert latest_archived_report(tmp_path) is None


def test_archive_report_rejects_invalid_report(tmp_path):
    with pytest.raises(TypeError):
        archive_report(
            ["invalid"],
            tmp_path,
        )


def test_archive_report_rejects_empty_name(
    sample_report,
    tmp_path,
):
    with pytest.raises(ValueError):
        archive_report(
            sample_report,
            tmp_path,
            name="",
        )


def test_archive_report_rejects_whitespace_name(
    sample_report,
    tmp_path,
):
    with pytest.raises(ValueError):
        archive_report(
            sample_report,
            tmp_path,
            name="   ",
        )
