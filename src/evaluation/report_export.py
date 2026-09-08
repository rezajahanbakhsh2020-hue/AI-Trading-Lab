import csv
import json
from pathlib import Path
from typing import Mapping, Any


def _validate_report(report: Mapping[str, Any]) -> None:
    """Validate that the supplied report is a mapping."""
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")


def export_report_json(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """
    Export an evaluation report to a JSON file.

    Parameters
    ----------
    report:
        Evaluation report represented as a mapping.
    path:
        Destination JSON file path.

    Returns
    -------
    Path
        The path of the created file.
    """
    _validate_report(report)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(dict(report), file, indent=2, ensure_ascii=False)

    return output_path


def export_report_csv(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """
    Export an evaluation report to a CSV file.

    The CSV contains one row with the report metrics as columns.

    Parameters
    ----------
    report:
        Evaluation report represented as a mapping.
    path:
        Destination CSV file path.

    Returns
    -------
    Path
        The path of the created file.
    """
    _validate_report(report)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(report.keys())

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(dict(report))

    return output_path


def export_report(
    report: Mapping[str, Any],
    json_path: str | Path | None = None,
    csv_path: str | Path | None = None,
) -> dict[str, Path]:
    """
    Export an evaluation report to one or both supported formats.

    Parameters
    ----------
    report:
        Evaluation report represented as a mapping.
    json_path:
        Optional destination for JSON output.
    csv_path:
        Optional destination for CSV output.

    Returns
    -------
    dict[str, Path]
        Mapping of exported format names to created file paths.

    Raises
    ------
    ValueError
        If neither output path is supplied.
    """
    _validate_report(report)

    if json_path is None and csv_path is None:
        raise ValueError(
            "At least one of json_path or csv_path must be provided."
        )

    exported: dict[str, Path] = {}

    if json_path is not None:
        exported["json"] = export_report_json(report, json_path)

    if csv_path is not None:
        exported["csv"] = export_report_csv(report, csv_path)

    return exported
