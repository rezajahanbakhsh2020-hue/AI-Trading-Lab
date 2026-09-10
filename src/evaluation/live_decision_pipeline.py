from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.live_decision_evaluation import (
    evaluate_live_decision_history,
)
from src.evaluation.live_decision_record import (
    record_live_decision,
)
from src.evaluation.live_decision_store import (
    append_live_decision_to_store,
    load_live_decision_history,
)


def process_live_decision(
    snapshot: Mapping[str, Any],
    history: list[Mapping[str, Any]] | None = None,
    store_path: str | None = None,
) -> dict[str, Any]:
    """
    Process one live snapshot through the Decision Record pipeline.

    The snapshot is normalized into a decision record, optionally persisted,
    and evaluated together with the supplied history.

    No signal, risk level, stability score, or performance value is
    recalculated.
    """
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    record = record_live_decision(snapshot)

    if history is None:
        records: list[dict[str, Any]] = []
    else:
        if not isinstance(history, list):
            raise TypeError("history must be a list")

        records = []

        for item in history:
            if not isinstance(item, Mapping):
                raise TypeError(
                    "each history item must be a mapping"
                )

            records.append(dict(item))

    if store_path is not None:
        persisted_history = append_live_decision_to_store(
            record,
            store_path,
        )
        records = persisted_history
    else:
        records.append(record)

    evaluation = evaluate_live_decision_history(records)

    return {
        "record": record,
        "history": records,
        "evaluation": evaluation,
        "persisted": store_path is not None,
        "store_path": store_path,
    }


def load_and_evaluate_live_decisions(
    store_path: str,
) -> dict[str, Any]:
    """
    Load persisted decision history and evaluate it.
    """
    history = load_live_decision_history(store_path)

    evaluation = evaluate_live_decision_history(history)

    return {
        "history": history,
        "evaluation": evaluation,
        "persisted": True,
        "store_path": store_path,
    }


def validate_live_decision_pipeline_result(
    result: Mapping[str, Any],
) -> bool:
    """
    Validate the structure of a live decision pipeline result.
    """
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")

    required_fields = (
        "record",
        "history",
        "evaluation",
        "persisted",
        "store_path",
    )

    missing = [
        field
        for field in required_fields
        if field not in result
    ]

    if missing:
        raise ValueError(
            "pipeline result is missing required fields: "
            + ", ".join(missing)
        )

    if not isinstance(result["record"], Mapping):
        raise ValueError("record must be a mapping")

    if not isinstance(result["history"], list):
        raise ValueError("history must be a list")

    if not isinstance(result["evaluation"], Mapping):
        raise ValueError("evaluation must be a mapping")

    if not isinstance(result["persisted"], bool):
        raise ValueError("persisted must be boolean")

    if result["store_path"] is not None and not isinstance(
        result["store_path"],
        str,
    ):
        raise ValueError(
            "store_path must be a string or None"
        )

    return True
