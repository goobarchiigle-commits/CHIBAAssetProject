"""
src/live/future_leader_integrity.py — Research Integrity State for Future Leader Screener

Purpose:
  Record daily research quality for forward expectancy analysis.
  Enables filtering of PARTIAL_FAILURE / MAJOR_FAILURE days from research datasets.

INVARIANTS:
  - observation_only research infrastructure only
  - no execution path connection
  - FAIL-CLOSED on integrity log write (raises IOError)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

# MAJOR_FAILURE requires BOTH ratio >= 25% AND absolute count >= 5
# (prevents over-sensitivity on small-universe days)
_MAJOR_FAILURE_MISSING_RATIO: float = 0.25
_MAJOR_FAILURE_MISSING_COUNT_MIN: int = 5


class IntegrityState(Enum):
    CLEAN           = "CLEAN"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"
    MAJOR_FAILURE   = "MAJOR_FAILURE"


@dataclass
class ResearchIntegrityState:
    """
    Daily research quality snapshot for the Future Leader screener.
    Stored append-only as JSONL for forward-return analysis filtering.
    PARTIAL_FAILURE / MAJOR_FAILURE days should be excluded from expectancy studies.
    """
    observation_date:            str
    integrity_state:             IntegrityState
    total_input_symbols:         int
    successfully_scored_symbols: int
    missing_symbols_count:       int
    rsr_build_failed:            bool
    ohlcv_load_failed:           bool
    partial_compute_failed:          bool
    warning_count:                   int   = 0
    data_quality_weight:             float = 0.0
    partial_failure_symbols_count:   int   = 0

    @classmethod
    def classify(
        cls,
        observation_date:              str,
        total_input_symbols:           int,
        successfully_scored_symbols:   int,
        missing_symbols_count:         int,
        rsr_build_failed:              bool,
        ohlcv_load_failed:             bool,
        partial_compute_failed:        bool,
        warning_count:                 int = 0,
        partial_failure_symbols_count: int = 0,
    ) -> "ResearchIntegrityState":
        """Classify integrity state from failure flags and symbol counts."""
        if total_input_symbols > 0:
            missing_ratio = missing_symbols_count / total_input_symbols
        else:
            missing_ratio = 1.0  # no symbols at all → treat as full missing

        if (
            total_input_symbols == 0
            or rsr_build_failed
            or ohlcv_load_failed
            or (
                missing_ratio >= _MAJOR_FAILURE_MISSING_RATIO
                and missing_symbols_count >= _MAJOR_FAILURE_MISSING_COUNT_MIN
            )
        ):
            state  = IntegrityState.MAJOR_FAILURE
            weight = 0.0
        elif partial_compute_failed or missing_symbols_count > 0:
            state  = IntegrityState.PARTIAL_FAILURE
            weight = 0.5
        else:
            state  = IntegrityState.CLEAN
            weight = 1.0

        return cls(
            observation_date=observation_date,
            integrity_state=state,
            total_input_symbols=total_input_symbols,
            successfully_scored_symbols=successfully_scored_symbols,
            missing_symbols_count=missing_symbols_count,
            rsr_build_failed=rsr_build_failed,
            ohlcv_load_failed=ohlcv_load_failed,
            partial_compute_failed=partial_compute_failed,
            warning_count=warning_count,
            data_quality_weight=weight,
            partial_failure_symbols_count=partial_failure_symbols_count,
        )

    def to_dict(self) -> dict:
        return {
            "observation_date":              self.observation_date,
            "integrity_state":               self.integrity_state.value,
            "data_quality_weight":           self.data_quality_weight,
            "total_input_symbols":           self.total_input_symbols,
            "successfully_scored_symbols":   self.successfully_scored_symbols,
            "missing_symbols_count":         self.missing_symbols_count,
            "partial_failure_symbols_count": self.partial_failure_symbols_count,
            "rsr_build_failed":              self.rsr_build_failed,
            "ohlcv_load_failed":             self.ohlcv_load_failed,
            "partial_compute_failed":        self.partial_compute_failed,
            "warning_count":                 self.warning_count,
        }


def write_integrity_log(
    state:    ResearchIntegrityState,
    log_path: Path,
) -> None:
    """
    Persist integrity state to JSONL. FAIL-CLOSED: raises IOError on write failure.
    Append-only: prior records are never modified.
    """
    record = {
        "observation_ts": datetime.now(JST).isoformat(),
        **state.to_dict(),
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        raise IOError(
            f"[FUTURE_LEADER_INTEGRITY] FAIL-CLOSED: log write failed → {log_path}: {exc}"
        ) from exc


def format_integrity_mail_section(state: ResearchIntegrityState) -> str:
    """Format integrity state for daily mail/report inclusion."""
    lines = [
        "=== RESEARCH INTEGRITY ===",
        f"state: {state.integrity_state.value}",
        f"data_quality_weight: {state.data_quality_weight}",
        f"scored: {state.successfully_scored_symbols}/{state.total_input_symbols}",
        f"missing_symbols: {state.missing_symbols_count}",
        f"partial_failure_symbols: {state.partial_failure_symbols_count}",
        f"rsr_build_failed: {str(state.rsr_build_failed).lower()}",
        f"ohlcv_load_failed: {str(state.ohlcv_load_failed).lower()}",
        f"partial_compute_failed: {str(state.partial_compute_failed).lower()}",
    ]
    if state.warning_count > 0:
        lines.append(f"warnings: {state.warning_count}")
    return "\n".join(lines)
