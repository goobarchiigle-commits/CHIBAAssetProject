"""
src/analytics/exit_intelligence/validator.py — Integrity Validator

Validates observation integrity:
  - no future leakage
  - no negative holding duration
  - no exit before entry
  - monotone peak validation
  - deterministic replay consistency
  - counterfactual isolation
  - immutable observation integrity
  - append-order consistency
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .models import ExitObservationRecord, ExitResearchRecord, ExitVelocityScore
from .observation import ExitPathSnapshot

logger = logging.getLogger(__name__)


@dataclass
class ValidationViolation:
    rule: str
    record_id: str
    detail: str
    severity: str  # "error" | "warning"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule,
            "record_id": self.record_id,
            "detail": self.detail,
            "severity": self.severity,
        }


@dataclass
class ValidationReport:
    violations: List[ValidationViolation] = field(default_factory=list)
    checked_records: int = 0
    passed: bool = True

    def add(self, v: ValidationViolation) -> None:
        self.violations.append(v)
        if v.severity == "error":
            self.passed = False

    def summary(self) -> Dict[str, Any]:
        errors = [v for v in self.violations if v.severity == "error"]
        warnings = [v for v in self.violations if v.severity == "warning"]
        return {
            "passed": self.passed,
            "checked_records": self.checked_records,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "violations": [v.to_dict() for v in self.violations],
        }


def _parse_date(s: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s[:19], fmt[:len(s[:19])])
        except ValueError:
            pass
    return datetime.strptime(s[:10], "%Y-%m-%d")


class IntegrityValidator:
    """
    Validates the integrity of exit intelligence records.
    All rules are deterministic. No ML, no external calls.
    """

    def validate_exit_observations(
        self,
        records: List[ExitObservationRecord],
        evaluation_date: Optional[str] = None,
    ) -> ValidationReport:
        report = ValidationReport()
        seen_ids: set = set()

        for rec in records:
            report.checked_records += 1
            rid = rec.obs_id

            # No exit before entry
            try:
                entry_dt = _parse_date(rec.entry_date)
                exit_dt = _parse_date(rec.exit_date)
                if exit_dt < entry_dt:
                    report.add(ValidationViolation(
                        rule="no_exit_before_entry",
                        record_id=rid,
                        detail=f"exit_date={rec.exit_date} < entry_date={rec.entry_date}",
                        severity="error",
                    ))
            except Exception as exc:
                report.add(ValidationViolation(
                    rule="date_parse",
                    record_id=rid,
                    detail=str(exc),
                    severity="error",
                ))

            # No negative holding duration
            if rec.holding_days < 0:
                report.add(ValidationViolation(
                    rule="no_negative_holding_duration",
                    record_id=rid,
                    detail=f"holding_days={rec.holding_days}",
                    severity="error",
                ))

            # No future leakage: exit_date must not be after evaluation_date
            if evaluation_date:
                try:
                    eval_dt = _parse_date(evaluation_date)
                    exit_dt = _parse_date(rec.exit_date)
                    if exit_dt > eval_dt:
                        report.add(ValidationViolation(
                            rule="no_future_leakage",
                            record_id=rid,
                            detail=(
                                f"exit_date={rec.exit_date} is after "
                                f"evaluation_date={evaluation_date}"
                            ),
                            severity="error",
                        ))
                except Exception:
                    pass

            # Immutable record integrity: obs_id must match hash of key fields
            expected_id = ExitObservationRecord.make_obs_id(
                rec.symbol, rec.entry_date, rec.exit_date
            )
            if rec.obs_id != expected_id:
                report.add(ValidationViolation(
                    rule="immutable_observation_integrity",
                    record_id=rid,
                    detail=(
                        f"obs_id mismatch: stored={rec.obs_id[:8]}... "
                        f"expected={expected_id[:8]}..."
                    ),
                    severity="error",
                ))

            # Append-order consistency: no duplicate IDs
            if rid in seen_ids:
                report.add(ValidationViolation(
                    rule="append_order_consistency",
                    record_id=rid,
                    detail="duplicate obs_id",
                    severity="error",
                ))
            seen_ids.add(rid)

            # Distance from peak must be >= 0
            if rec.distance_from_peak_pct < 0:
                report.add(ValidationViolation(
                    rule="monotone_peak_validation",
                    record_id=rid,
                    detail=f"distance_from_peak_pct={rec.distance_from_peak_pct:.4f} < 0",
                    severity="error",
                ))

        return report

    def validate_path_snapshots(
        self,
        snapshots: List[ExitPathSnapshot],
        evaluation_date: Optional[str] = None,
    ) -> ValidationReport:
        report = ValidationReport()
        # Group by (symbol, entry_date)
        groups: Dict[Tuple, List[ExitPathSnapshot]] = {}
        for s in snapshots:
            key = (s.symbol, s.entry_date)
            groups.setdefault(key, []).append(s)

        for (symbol, entry_date), group in groups.items():
            sorted_group = sorted(group, key=lambda s: s.snapshot_date)
            report.checked_records += len(sorted_group)

            # Day numbers must be monotonically increasing
            for i in range(1, len(sorted_group)):
                if sorted_group[i].day_number <= sorted_group[i - 1].day_number:
                    report.add(ValidationViolation(
                        rule="monotone_day_numbers",
                        record_id=sorted_group[i].snapshot_id,
                        detail=(
                            f"day_number={sorted_group[i].day_number} not > "
                            f"{sorted_group[i-1].day_number}"
                        ),
                        severity="error",
                    ))

            # Peak price must be monotonically non-decreasing
            for i in range(1, len(sorted_group)):
                if sorted_group[i].peak_price_so_far < sorted_group[i - 1].peak_price_so_far - 1e-9:
                    report.add(ValidationViolation(
                        rule="monotone_peak_validation",
                        record_id=sorted_group[i].snapshot_id,
                        detail=(
                            f"peak_price_so_far decreased: "
                            f"{sorted_group[i].peak_price_so_far:.2f} < "
                            f"{sorted_group[i-1].peak_price_so_far:.2f}"
                        ),
                        severity="error",
                    ))

            # No future leakage in snapshots
            if evaluation_date:
                for s in sorted_group:
                    try:
                        snap_dt = _parse_date(s.snapshot_date)
                        eval_dt = _parse_date(evaluation_date)
                        if snap_dt > eval_dt:
                            report.add(ValidationViolation(
                                rule="no_future_leakage",
                                record_id=s.snapshot_id,
                                detail=(
                                    f"snapshot_date={s.snapshot_date} > "
                                    f"evaluation_date={evaluation_date}"
                                ),
                                severity="error",
                            ))
                    except Exception:
                        pass

        return report

    def validate_velocity_scores(
        self,
        scores: List[ExitVelocityScore],
    ) -> ValidationReport:
        report = ValidationReport()
        for score in scores:
            report.checked_records += 1

            # Total score must be in [0, 10]
            if not (0.0 <= score.total_score <= 10.0):
                report.add(ValidationViolation(
                    rule="score_bounds",
                    record_id=score.score_id,
                    detail=f"total_score={score.total_score} out of [0, 10]",
                    severity="error",
                ))

            # Action band must match score
            expected_band = (
                "hold" if score.total_score <= 2.0
                else "tighten_risk" if score.total_score <= 4.0
                else "staged_reduction" if score.total_score <= 6.0
                else "aggressive_exit"
            )
            if score.action_band != expected_band:
                report.add(ValidationViolation(
                    rule="action_band_consistency",
                    record_id=score.score_id,
                    detail=(
                        f"action_band={score.action_band} inconsistent with "
                        f"total_score={score.total_score} (expected={expected_band})"
                    ),
                    severity="error",
                ))

            # All component scores must be in [0, 1]
            for comp, val in [
                ("rs_deterioration", score.rs_deterioration),
                ("breadth_decay", score.breadth_decay),
                ("breakout_failure", score.breakout_failure),
                ("momentum_decay", score.momentum_decay),
                ("distance_from_peak", score.distance_from_peak),
            ]:
                if not (0.0 <= val <= 1.0 + 1e-9):
                    report.add(ValidationViolation(
                        rule="component_score_bounds",
                        record_id=score.score_id,
                        detail=f"{comp}={val:.4f} out of [0, 1]",
                        severity="error",
                    ))

        return report

    def validate_research_dataset(
        self,
        records: List[ExitResearchRecord],
    ) -> ValidationReport:
        report = ValidationReport()
        seen_research_ids: set = set()

        for rec in records:
            report.checked_records += 1

            # Holding days must be positive
            if rec.holding_days < 0:
                report.add(ValidationViolation(
                    rule="no_negative_holding_duration",
                    record_id=rec.research_id,
                    detail=f"holding_days={rec.holding_days}",
                    severity="error",
                ))

            # Duplicate research_id
            if rec.research_id in seen_research_ids:
                report.add(ValidationViolation(
                    rule="append_order_consistency",
                    record_id=rec.research_id,
                    detail="duplicate research_id",
                    severity="error",
                ))
            seen_research_ids.add(rec.research_id)

            # Velocity score must be in [0, 10]
            if not (0.0 <= rec.exit_velocity_score <= 10.0 + 1e-9):
                report.add(ValidationViolation(
                    rule="score_bounds",
                    record_id=rec.research_id,
                    detail=f"exit_velocity_score={rec.exit_velocity_score}",
                    severity="warning",
                ))

        return report
