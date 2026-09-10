from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from io import StringIO
from typing import Any

from src.evaluation.live_decision_record import (
    DECISION_RECORD_FIELDS,
    validate_live_decision_record,
)


def _normalize_records(
    history: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(history, (str, bytes)) or not isinstance(
        history, Iterable
    ):
        raise TypeError("history must be iterable")

    records: list[dict[str, Any]] = []

    for item in history:
        if not isinstance(item, Mapping):
            raise TypeError("each history item must be a mapping")

        record = {
            field: item.get(field)
            for field in DECISION_RECORD_FIELDS
        }
        validate_live_decision_record(record)
        records.append(record)

    return records


def export_live_decisions_json(
    history: Iterable[Mapping[str, Any]],
) -> str:
    """
    Export normalized live decision history as JSON.

    Trading values are serialized exactly as recorded and are not
    recalculated.
    """
    records = _normalize_records(history)

    return json.dumps(
        records,
        ensure_ascii=False,
        indent=2,
    )


def export_live_decisions_csv(
    history: Iterable[Mapping[str, Any]],
) -> str:
    """
    Export normalized live decision history as CSV.
    """
    records = _normalize_records(history)

    output = StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=list(DECISION_RECORD_FIELDS),
    )

    writer.writeheader()
    writer.writerows(records)

    return output.getvalue()


def export_live_decisions(
    history: Iterable[Mapping[str, Any]],
    format: str = "json",
) -> str:
    """
    Export live decision history using the requested format.

    Supported formats are JSON and CSV.
    """
    normalized_format = str(format).strip().lower()

    if normalized_format == "json":
        return export_live_decisions_json(history)

    if normalized_format == "csv":
        return export_live_decisions_csv(history)

    raise ValueError(
        "unsupported format; expected 'json' or 'csv'"
    )
