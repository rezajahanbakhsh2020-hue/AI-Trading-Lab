from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_report(path: str | Path) -> dict[str, Any]:
    """
    Load an evaluation report from a JSON file.

    Parameters
    ----------
    path:
        Path to the JSON report.

    Returns
    -------
    dict[str, Any]
        Loaded report.

    Raises
    ------
    FileNotFoundError
        If the report file does not exist.
    ValueError
        If the JSON content is invalid or is not an object.
    """
    report_path = Path(path)

    if not report_path.exists():
        raise FileNotFoundError(
            f"Report file does not exist: {report_path}"
        )

    if not report_path.is_file():
        raise ValueError(
            f"Report path is not a file: {report_path}"
        )

    try:
        with report_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON report: {report_path}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            "Report JSON must contain an object at the top level."
        )

    return data


def report_exists(path: str | Path) -> bool:
    """
    Check whether a report path exists and is a regular file.

    Parameters
    ----------
    path:
        Path to the report.

    Returns
    -------
    bool
        True when the path points to an existing file.
    """
    return Path(path).is_file()


def load_latest_report(
    directory: str | Path,
    name: str = "evaluation_report",
) -> dict[str, Any]:
    """
    Load the newest archived report from a directory.

    Parameters
    ----------
    directory:
        Directory containing archived reports.
    name:
        Base name used by the archived reports.

    Returns
    -------
    dict[str, Any]
        Contents of the newest report.

    Raises
    ------
    FileNotFoundError
        If no matching report exists.
    """
    archive_dir = Path(directory)

    reports = sorted(
        path
        for path in archive_dir.glob(f"{name}_*.json")
        if path.is_file()
    )

    if not reports:
        raise FileNotFoundError(
            f"No archived reports found in: {archive_dir}"
        )

    return load_report(reports[-1])
