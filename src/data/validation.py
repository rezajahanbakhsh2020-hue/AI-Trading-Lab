from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
]


def validate_market_data(df: pd.DataFrame) -> None:
    """
    Validate the structure and basic integrity of market data.

    Raises:
        ValueError: If the data is invalid.
    """

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Market data is empty.")

    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Market data contains missing values.")

    if (df["open"] <= 0).any():
        raise ValueError("Open prices must be positive.")

    if (df["high"] <= 0).any():
        raise ValueError("High prices must be positive.")

    if (df["low"] <= 0).any():
        raise ValueError("Low prices must be positive.")

    if (df["close"] <= 0).any():
        raise ValueError("Close prices must be positive.")

    if (df["high"] < df["low"]).any():
        raise ValueError("High price cannot be lower than low price.")


def verify_dataset_and_manifest_integrity(
    csv_path: str | Path,
    manifest_path: str | Path,
    commit_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Verify that published CSV and Manifest belong to the exact same generation.

    Fails closed if hashes, row counts, or metadata mismatch.
    """
    csv_p = Path(csv_path)
    man_p = Path(manifest_path)

    if not csv_p.exists():
        raise FileNotFoundError(f"Dataset CSV not found at {csv_p}")

    if not man_p.exists():
        raise FileNotFoundError(f"Dataset Manifest not found at {man_p}")

    csv_bytes = csv_p.read_bytes()
    actual_sha = hashlib.sha256(csv_bytes).hexdigest()

    try:
        manifest_data = json.loads(man_p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Corrupt manifest JSON at {man_p}: {exc}") from exc

    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"Manifest JSON at {man_p} must be an object (dict), got {type(manifest_data).__name__}."
        )

    required_identity_fields = ("dataset_id", "symbol", "timeframe")
    for field in required_identity_fields:
        if field not in manifest_data:
            raise ValueError(f"Manifest at {man_p} is missing required identity field '{field}'.")
        val = manifest_data[field]
        if not isinstance(val, str):
            raise TypeError(f"Manifest identity field '{field}' must be a string, got {type(val).__name__}.")
        if not val.strip():
            raise ValueError(f"Manifest identity field '{field}' cannot be empty or whitespace-only.")

    expected_sha = manifest_data.get("output_sha256")
    if not expected_sha or actual_sha != expected_sha:
        raise RuntimeError(
            f"Mismatched dataset/manifest generation detected! "
            f"Actual CSV SHA256 ({actual_sha[:12]}) != Manifest output SHA256 ({str(expected_sha)[:12]})."
        )

    if commit_path:
        com_p = Path(commit_path)
        if com_p.exists():
            try:
                commit_data = json.loads(com_p.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"Corrupt commit marker at {com_p}: {exc}") from exc

            if not isinstance(commit_data, dict):
                raise ValueError(
                    f"Commit marker JSON at {com_p} must be an object (dict), got {type(commit_data).__name__}."
                )

            if commit_data.get("output_sha256") != actual_sha:
                raise RuntimeError("Commit marker output SHA256 mismatch!")

            for field in required_identity_fields:
                if field not in commit_data:
                    raise ValueError(f"Commit marker at {com_p} is missing required identity field '{field}'.")
                val = commit_data[field]
                if not isinstance(val, str):
                    raise TypeError(f"Commit marker identity field '{field}' must be a string, got {type(val).__name__}.")
                if not val.strip():
                    raise ValueError(f"Commit marker identity field '{field}' cannot be empty or whitespace-only.")

                man_val = manifest_data[field]
                if val.strip() != man_val.strip():
                    raise ValueError(
                        f"Commit marker identity field '{field}' ('{val}') conflicts with manifest '{field}' ('{man_val}')."
                    )

    df = pd.read_csv(io.BytesIO(csv_bytes))
    if len(df) != manifest_data.get("row_count"):
        raise RuntimeError(f"Row count mismatch: CSV ({len(df)}) != Manifest ({manifest_data.get('row_count')}).")

    return df, manifest_data
