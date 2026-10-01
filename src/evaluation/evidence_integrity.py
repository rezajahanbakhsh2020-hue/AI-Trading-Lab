"""Research Evidence Integrity Gate & Canonical Partition Validator for Project 1.

Provides authoritative, deterministic, side-effect-free structural, temporal,
and genealogical integrity validation for ResearchEvidence artifacts prior to
qualification or promotion binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Any

import pandas as pd


def _parse_exact_utc_timestamp(ts_str: str) -> datetime:
    """Strictly parse exact UTC ISO-8601 timestamp string.

    Fails closed on empty, naive (no timezone), or non-UTC offset strings.
    """
    if not ts_str or not isinstance(ts_str, str) or not ts_str.strip():
        raise ValueError("Timestamp string is empty or invalid.")
    cleaned = ts_str.strip()
    iso_str = cleaned.replace("Z", "+00:00")
    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        raise ValueError(f"Timestamp '{ts_str}' lacks required explicit timezone information.")
    if dt.utcoffset() != timedelta(0):
        raise ValueError(
            f"Timestamp '{ts_str}' offset ({dt.utcoffset()}) is not UTC (+00:00). Silent normalization is forbidden."
        )
    return dt

from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    EvidencePartition,
    EvidencePartitionRole,
    ExecutionAssumptions,
    RejectionReason,
    ResearchEvidence,
    ResearchExperimentSpec,
    compute_experiment_fingerprint,
)


@dataclass(frozen=True)
class IntegrityValidationResult:
    """Immutable result of a Research Evidence Integrity Gate validation."""

    valid: bool
    rejection_reasons: tuple[RejectionReason, ...]
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a boolean.")
        for r in self.rejection_reasons:
            if not isinstance(r, RejectionReason):
                raise TypeError(f"Rejection reason '{r}' must be a RejectionReason enum member.")


class ResearchEvidenceIntegrityGate:
    """Canonical, side-effect-free Research Evidence Integrity Gate."""

    @classmethod
    def validate(
        cls,
        evidence: Any,
        *,
        require_exact_timestamps: bool = True,
        require_walk_forward: bool = False,
        require_oos: bool = False,
        min_observations: int = 1,
    ) -> IntegrityValidationResult:
        """Validate structural, temporal, and genealogical integrity of ResearchEvidence.

        Fails closed if the evidence is structurally inconsistent, tampered,
        non-monotonic, overlapping, or genealogically corrupt.
        """
        rejection_reasons: list[RejectionReason] = []
        notes: list[str] = []

        # 1. Type validation
        if not isinstance(evidence, ResearchEvidence):
            return IntegrityValidationResult(
                valid=False,
                rejection_reasons=(RejectionReason.EVIDENCE_INCOMPLETENESS,),
                notes=("Input object is not a valid ResearchEvidence instance.",),
            )

        spec = evidence.spec

        # 2. Spec & Identity integrity
        if not isinstance(spec, ResearchExperimentSpec):
            rejection_reasons.append(RejectionReason.SPECIFICATION_INVALID)
            notes.append("Missing or invalid ResearchExperimentSpec.")
        else:
            # Check required spec fields
            for attr_name in (
                "hypothesis",
                "methodology_version",
                "strategy_name",
                "strategy_version",
                "benchmark_reference",
            ):
                val = getattr(spec, attr_name, None)
                if not isinstance(val, str) or not val.strip():
                    if RejectionReason.SPECIFICATION_INVALID not in rejection_reasons:
                        rejection_reasons.append(RejectionReason.SPECIFICATION_INVALID)
                    notes.append(f"Spec field '{attr_name}' is missing or empty.")

            # Experiment fingerprint integrity check
            if not evidence.experiment_fingerprint or not evidence.experiment_fingerprint.strip():
                rejection_reasons.append(RejectionReason.FAILED_REPRODUCIBILITY)
                notes.append("Evidence experiment_fingerprint is missing or empty.")
            elif evidence.experiment_fingerprint != spec.fingerprint:
                rejection_reasons.append(RejectionReason.FAILED_REPRODUCIBILITY)
                notes.append(
                    f"Evidence experiment_fingerprint '{evidence.experiment_fingerprint}' "
                    f"does not match spec fingerprint '{spec.fingerprint}'."
                )
            else:
                try:
                    recomputed_fp = compute_experiment_fingerprint(
                        hypothesis=spec.hypothesis,
                        methodology_version=spec.methodology_version,
                        strategy_name=spec.strategy_name,
                        strategy_version=spec.strategy_version,
                        dataset_scope=spec.dataset_scope,
                        execution_assumptions=spec.execution_assumptions,
                        code_provenance=spec.code_provenance,
                        benchmark_reference=spec.benchmark_reference,
                        parameters=spec.parameters,
                        random_seed=spec.random_seed,
                        walk_forward_protocol=getattr(spec, "walk_forward_protocol", None),
                    )
                    if (
                        recomputed_fp != spec.fingerprint
                        or recomputed_fp != evidence.experiment_fingerprint
                    ):
                        rejection_reasons.append(RejectionReason.FAILED_REPRODUCIBILITY)
                        notes.append(
                            "SHA-256 fingerprint re-computation mismatch (tampered evidence detected)."
                        )
                except Exception as exc:
                    rejection_reasons.append(RejectionReason.FAILED_REPRODUCIBILITY)
                    notes.append(f"Fingerprint re-computation failed: {exc}")

            # DatasetScope validation
            ds = getattr(spec, "dataset_scope", None)
            if not isinstance(ds, DatasetScope):
                rejection_reasons.append(RejectionReason.INVALID_DATASET_SCOPE)
                notes.append("DatasetScope is missing or invalid.")
            else:
                if (
                    not ds.dataset_id
                    or not ds.dataset_id.strip()
                    or not ds.symbol
                    or not ds.symbol.strip()
                    or not ds.timeframe
                    or not ds.timeframe.strip()
                ):
                    rejection_reasons.append(RejectionReason.INVALID_DATASET_SCOPE)
                    notes.append("DatasetScope fields (dataset_id, symbol, timeframe) must be non-empty.")
                if not ds.start_date or not ds.start_date.strip() or not ds.end_date or not ds.end_date.strip():
                    rejection_reasons.append(RejectionReason.INVALID_DATASET_SCOPE)
                    notes.append("DatasetScope date boundaries must be non-empty.")
                elif ds.start_date > ds.end_date:
                    rejection_reasons.append(RejectionReason.INVALID_DATASET_SCOPE)
                    notes.append(
                        f"DatasetScope start_date '{ds.start_date}' is later than end_date '{ds.end_date}'."
                    )

            # ExecutionAssumptions validation
            ea = getattr(spec, "execution_assumptions", None)
            if not isinstance(ea, ExecutionAssumptions):
                rejection_reasons.append(RejectionReason.EXECUTION_ASSUMPTION_VIOLATION)
                notes.append("ExecutionAssumptions missing or invalid.")
            else:
                for field_name in ("transaction_cost", "slippage", "latency_ms"):
                    val = getattr(ea, field_name, None)
                    if not isinstance(val, (int, float)) or math.isnan(val) or val < 0.0:
                        if RejectionReason.EXECUTION_ASSUMPTION_VIOLATION not in rejection_reasons:
                            rejection_reasons.append(RejectionReason.EXECUTION_ASSUMPTION_VIOLATION)
                        notes.append(f"Execution assumption '{field_name}' is invalid or negative: {val}")

            # CodeProvenance validation
            cp = getattr(spec, "code_provenance", None)
            if not isinstance(cp, CodeProvenance):
                rejection_reasons.append(RejectionReason.EVIDENCE_INCOMPLETENESS)
                notes.append("CodeProvenance missing or invalid.")
            elif not cp.commit_sha or not cp.commit_sha.strip():
                rejection_reasons.append(RejectionReason.EVIDENCE_INCOMPLETENESS)
                notes.append("CodeProvenance commit_sha is missing or empty.")

        # 3. Partition identity and canonical partition validator
        partitions = getattr(evidence, "partitions", ()) or ()
        if not isinstance(partitions, (list, tuple)):
            rejection_reasons.append(RejectionReason.EVIDENCE_INCOMPLETENESS)
            notes.append("Partitions must be a sequence of EvidencePartition objects.")
            partitions = ()

        seen_roles: set[EvidencePartitionRole] = set()
        partition_by_role: dict[EvidencePartitionRole, EvidencePartition] = {}

        for p in partitions:
            if not isinstance(p, EvidencePartition):
                rejection_reasons.append(RejectionReason.EVIDENCE_INCOMPLETENESS)
                notes.append(f"Invalid partition object found in partitions: {p}")
                continue

            if p.role in seen_roles:
                rejection_reasons.append(RejectionReason.EVIDENCE_INCOMPLETENESS)
                notes.append(f"Duplicate partition role detected: {p.role.value}")
            seen_roles.add(p.role)
            partition_by_role[p.role] = p

            # Check individual partition observation sufficiency
            if p.observations < min_observations:
                rejection_reasons.append(RejectionReason.INSUFFICIENT_DATA)
                notes.append(
                    f"Partition '{p.role.value}' has insufficient observations ({p.observations} < {min_observations})."
                )

            # Check individual partition date chronology
            if p.start_date > p.end_date:
                rejection_reasons.append(RejectionReason.FAILED_VALIDATION)
                notes.append(
                    f"Partition '{p.role.value}' has reversed dates: start_date '{p.start_date}' > end_date '{p.end_date}'."
                )

            # Check exact UTC timestamps
            if p.start_timestamp_utc is None or p.end_timestamp_utc is None or not str(p.start_timestamp_utc).strip() or not str(p.end_timestamp_utc).strip():
                if require_exact_timestamps:
                    rejection_reasons.append(RejectionReason.EVIDENCE_INCOMPLETENESS)
                    notes.append(
                        f"Partition '{p.role.value}' is missing required exact UTC timestamps."
                    )
            else:
                try:
                    s_str = str(p.start_timestamp_utc).strip()
                    e_str = str(p.end_timestamp_utc).strip()
                    dt_start = _parse_exact_utc_timestamp(s_str)
                    dt_end = _parse_exact_utc_timestamp(e_str)
                    p_start_dt = pd.to_datetime(dt_start, utc=True)
                    p_end_dt = pd.to_datetime(dt_end, utc=True)
                    if pd.isna(p_start_dt) or pd.isna(p_end_dt):
                        raise ValueError("NaT value parsed.")
                    if p_start_dt > p_end_dt:
                        rejection_reasons.append(RejectionReason.FAILED_VALIDATION)
                        notes.append(
                            f"Partition '{p.role.value}' has reversed timestamps: "
                            f"start '{p.start_timestamp_utc}' > end '{p.end_timestamp_utc}'."
                        )
                    if p_start_dt.strftime("%Y-%m-%d") != p.start_date[:10] or p_end_dt.strftime("%Y-%m-%d") != p.end_date[:10]:
                        rejection_reasons.append(RejectionReason.FAILED_VALIDATION)
                        notes.append(
                            f"Partition '{p.role.value}' date/timestamp mismatch: "
                            f"dates [{p.start_date}, {p.end_date}] vs timestamps [{s_str}, {e_str}]."
                        )
                except Exception as exc:
                    rejection_reasons.append(RejectionReason.EVIDENCE_INCOMPLETENESS)
                    notes.append(
                        f"Partition '{p.role.value}' has invalid UTC timestamps: {exc}"
                    )

        # Check required roles
        if EvidencePartitionRole.IN_SAMPLE not in partition_by_role:
            rejection_reasons.append(RejectionReason.EVIDENCE_INCOMPLETENESS)
            notes.append("Missing required In-Sample partition.")

        if require_oos and EvidencePartitionRole.OUT_OF_SAMPLE not in partition_by_role:
            rejection_reasons.append(RejectionReason.FAILED_OOS)
            notes.append("Missing required Out-of-Sample partition.")

        if require_walk_forward and EvidencePartitionRole.WALK_FORWARD not in partition_by_role:
            rejection_reasons.append(RejectionReason.FAILED_WALK_FORWARD)
            notes.append("Missing required Walk-Forward partition.")

        # Check IS -> Validation -> OOS Chronology & Non-Overlap
        ordered_roles = [
            EvidencePartitionRole.IN_SAMPLE,
            EvidencePartitionRole.VALIDATION,
            EvidencePartitionRole.OUT_OF_SAMPLE,
        ]
        active_ordered_parts = [
            partition_by_role[r] for r in ordered_roles if r in partition_by_role
        ]

        for i in range(len(active_ordered_parts) - 1):
            curr_p = active_ordered_parts[i]
            next_p = active_ordered_parts[i + 1]

            # Compare exact timestamps if both have them
            if (
                curr_p.end_timestamp_utc is not None
                and next_p.start_timestamp_utc is not None
            ):
                curr_end = pd.to_datetime(curr_p.end_timestamp_utc, utc=True)
                next_start = pd.to_datetime(next_p.start_timestamp_utc, utc=True)
                if curr_end >= next_start:
                    rejection_reasons.append(RejectionReason.FAILED_VALIDATION)
                    notes.append(
                        f"Partition timestamp overlap or reversal between '{curr_p.role.value}' "
                        f"(end: {curr_p.end_timestamp_utc}) and '{next_p.role.value}' "
                        f"(start: {next_p.start_timestamp_utc})."
                    )
            else:
                if curr_p.end_date >= next_p.start_date:
                    rejection_reasons.append(RejectionReason.FAILED_VALIDATION)
                    notes.append(
                        f"Partition date overlap or reversal between '{curr_p.role.value}' "
                        f"(end: {curr_p.end_date}) and '{next_p.role.value}' "
                        f"(start: {next_p.start_date})."
                    )

        # Validate partition bounds against DatasetScope if spec and dataset_scope exist
        ds_scope = getattr(spec, "dataset_scope", None)
        if isinstance(ds_scope, DatasetScope):
            ds_start_date = ds_scope.start_date[:10]
            ds_end_date = ds_scope.end_date[:10]
            for p in partition_by_role.values():
                if p.start_date[:10] < ds_start_date or p.end_date[:10] > ds_end_date:
                    if RejectionReason.INVALID_DATASET_SCOPE not in rejection_reasons:
                        rejection_reasons.append(RejectionReason.INVALID_DATASET_SCOPE)
                    notes.append(
                        f"Partition '{p.role.value}' date range [{p.start_date}, {p.end_date}] "
                        f"extends beyond DatasetScope [{ds_start_date}, {ds_end_date}]."
                    )

        dedup_rejections = tuple(dict.fromkeys(rejection_reasons))

        return IntegrityValidationResult(
            valid=len(dedup_rejections) == 0,
            rejection_reasons=dedup_rejections,
            notes=tuple(notes),
        )
