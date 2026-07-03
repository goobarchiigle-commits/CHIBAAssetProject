"""
stale_visibility.py — Structured allocator staleness warning telemetry

Detects and logs staleness across four allocator state planes:
  1. Exposure state     — rolling correlation, effective-N
  2. Lifecycle state    — alpha persistence map per symbol
  3. Stability window   — rolling rank/concentration/aggression history
  4. Fallback mode      — any Phase 5A fallback activation

All warnings are append-only. Never block execution. Fail-open only.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1

WARN_STALE_EXPOSURE: str = "stale_exposure_state"
WARN_STALE_LIFECYCLE: str = "stale_lifecycle_state"
WARN_STALE_STABILITY: str = "stale_stability_window"
WARN_FALLBACK_ACTIVATED: str = "fallback_mode_activation"

DEFAULT_EXPOSURE_STALE_SEC: float = 3600.0
DEFAULT_LIFECYCLE_STALE_SEC: float = 86400.0
DEFAULT_STABILITY_STALE_SEC: float = 86400.0


@dataclass
class StaleVisibilityWarning:
    """One structured staleness warning record. Append-only."""
    timestamp: str
    warning_type: str
    symbol: str
    detail: str
    state_age_sec: Optional[float]
    threshold_sec: Optional[float]
    fallback_reason: str
    schema_version: int = SCHEMA_VERSION


def append_stale_warning(warning: StaleVisibilityWarning, path: Path) -> None:
    """Fail-safe append. Write failure logs WARNING, never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(warning), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("[STALE_VISIBILITY] write failed (%s) — execution continues", exc)


def load_stale_warnings(path: Path) -> list[StaleVisibilityWarning]:
    """Load all warnings from JSONL. Corrupted lines are skipped with WARNING."""
    if not path.exists():
        return []
    warnings: list[StaleVisibilityWarning] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            warnings.append(StaleVisibilityWarning(
                timestamp=d.get("timestamp", ""),
                warning_type=d.get("warning_type", ""),
                symbol=d.get("symbol", ""),
                detail=d.get("detail", ""),
                state_age_sec=d.get("state_age_sec"),
                threshold_sec=d.get("threshold_sec"),
                fallback_reason=d.get("fallback_reason", ""),
            ))
        except Exception as exc:
            logger.warning("[STALE_VISIBILITY] parse error (%s) — line skipped", exc)
    return warnings


def _compute_age(state_load_time_sec: float, now_sec: float, threshold_sec: float) -> float:
    if state_load_time_sec <= 0.0:
        return threshold_sec + 1.0  # unknown load time → always stale
    return now_sec - state_load_time_sec


def check_exposure_staleness(
    state_load_time_sec: float,
    now_sec: float,
    timestamp: str,
    symbol: str = "",
    threshold_sec: float = DEFAULT_EXPOSURE_STALE_SEC,
) -> Optional[StaleVisibilityWarning]:
    """Returns warning if exposure state is stale, None if fresh."""
    age_sec = _compute_age(state_load_time_sec, now_sec, threshold_sec)
    if age_sec <= threshold_sec:
        return None
    return StaleVisibilityWarning(
        timestamp=timestamp,
        warning_type=WARN_STALE_EXPOSURE,
        symbol=symbol,
        detail=f"exposure_state age={age_sec:.0f}s > threshold={threshold_sec:.0f}s",
        state_age_sec=round(age_sec, 2),
        threshold_sec=threshold_sec,
        fallback_reason=f"stale exposure state: age={age_sec:.0f}s",
    )


def check_lifecycle_staleness(
    state_load_time_sec: float,
    now_sec: float,
    timestamp: str,
    symbol: str = "",
    threshold_sec: float = DEFAULT_LIFECYCLE_STALE_SEC,
) -> Optional[StaleVisibilityWarning]:
    """Returns warning if lifecycle state is stale, None if fresh."""
    age_sec = _compute_age(state_load_time_sec, now_sec, threshold_sec)
    if age_sec <= threshold_sec:
        return None
    return StaleVisibilityWarning(
        timestamp=timestamp,
        warning_type=WARN_STALE_LIFECYCLE,
        symbol=symbol,
        detail=f"lifecycle_state age={age_sec:.0f}s > threshold={threshold_sec:.0f}s",
        state_age_sec=round(age_sec, 2),
        threshold_sec=threshold_sec,
        fallback_reason=f"stale lifecycle state: age={age_sec:.0f}s",
    )


def check_stability_staleness(
    state_load_time_sec: float,
    now_sec: float,
    timestamp: str,
    symbol: str = "",
    threshold_sec: float = DEFAULT_STABILITY_STALE_SEC,
) -> Optional[StaleVisibilityWarning]:
    """Returns warning if stability window is stale, None if fresh."""
    age_sec = _compute_age(state_load_time_sec, now_sec, threshold_sec)
    if age_sec <= threshold_sec:
        return None
    return StaleVisibilityWarning(
        timestamp=timestamp,
        warning_type=WARN_STALE_STABILITY,
        symbol=symbol,
        detail=f"stability_window age={age_sec:.0f}s > threshold={threshold_sec:.0f}s",
        state_age_sec=round(age_sec, 2),
        threshold_sec=threshold_sec,
        fallback_reason=f"stale stability window: age={age_sec:.0f}s",
    )


def check_fallback_activation(
    fallback_mode: bool,
    fallback_reason: str,
    timestamp: str,
    symbol: str = "",
) -> Optional[StaleVisibilityWarning]:
    """Returns warning if fallback mode is active, None otherwise."""
    if not fallback_mode:
        return None
    return StaleVisibilityWarning(
        timestamp=timestamp,
        warning_type=WARN_FALLBACK_ACTIVATED,
        symbol=symbol,
        detail=f"fallback mode activated: {fallback_reason}",
        state_age_sec=None,
        threshold_sec=None,
        fallback_reason=fallback_reason,
    )


def emit_stale_warnings(
    exposure_load_time: float,
    lifecycle_load_time: float,
    stability_load_time: float,
    now_sec: float,
    timestamp: str,
    path: Path,
    symbol: str = "",
    fallback_mode: bool = False,
    fallback_reason: str = "",
) -> list[StaleVisibilityWarning]:
    """
    Convenience: run all four staleness checks and append any warnings to path.
    Returns list of warnings emitted (may be empty).
    Never raises.
    """
    checks = [
        check_exposure_staleness(exposure_load_time, now_sec, timestamp, symbol),
        check_lifecycle_staleness(lifecycle_load_time, now_sec, timestamp, symbol),
        check_stability_staleness(stability_load_time, now_sec, timestamp, symbol),
        check_fallback_activation(fallback_mode, fallback_reason, timestamp, symbol),
    ]
    emitted: list[StaleVisibilityWarning] = []
    for w in checks:
        if w is not None:
            append_stale_warning(w, path)
            emitted.append(w)
    return emitted
