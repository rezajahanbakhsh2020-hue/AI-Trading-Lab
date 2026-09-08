from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def archive_report(
    report: Mapping[str, Any],
    directory: str | Path,
    name: str = "evaluation_report",
) -> Path:
    """
    Archive an evaluation report as a timestamped JSON file.

    Parameters
    ----------
    report:
        Evaluation report represented as a mapping.
    directory:
        Directory where the archived report will be stored.
    name:
        Base name used for the archived report.

    Returns
    -------
    Path
        Path of the archived report.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string.")

    archive_dir = Path(directory)
    archive_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    output_path = archive_dir / f"{name}_{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(dict(report), file, indent=2, ensure_ascii=False)

    return output_path


def list_archived_reports(
    directory: str | Path,
    name: str = "evaluation_report",
) -> list[Path]:
    """
    Return archived reports sorted from oldest to newest.

    Parameters
    ----------
    directory:
        Directory containing archived reports.
    name:
        Base name used when reports were archived.

    Returns
    -------
    list[Path]
        Matching archived report paths.
    """
    archive_dir = Path(directory)

    if not archive_dir.exists():
        return []

    pattern = f"{name}_*.json"

    return sorted(
        path
        for path in archive_dir.glob(pattern)
        if path.is_file()
    )


def latest_archived_report(
    directory: str | Path,
    name: str = "evaluation_report",
) -> Path | None:
    """
    Return the newest archived report.

    Parameters
    ----------
    directory:
        Directory containing archived reports.
    name:
        Base name used when reports were archived.

    Returns
    -------
    Path | None
        Newest archived report, or None when no archive exists.
    """
    reports = list_archived_reports(directory, name=name)

    if not reports:
        return None

    return reports[-1]
