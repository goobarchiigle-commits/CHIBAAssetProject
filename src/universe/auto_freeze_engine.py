"""
src/universe/auto_freeze_engine.py — Outcome-based Auto Freeze Engine

Reads pre-aggregated promotion/rotation/exploration outcome summaries and
computes a weighted degradation score that drives an auto_freeze_mode.

Integrates with GovernanceHealthEngine:
    effective_mode = max(governance_mode, auto_freeze_mode)   # most restrictive wins

Sample protection:
    Triggers with sample_count < MIN_*_SAMPLES → INSUFFICIENT_DATA;
    their weight is NOT added to degradation_score.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Sample protection ──────────────────────────────────────────────────────────
MIN_PROMOTION_SAMPLES:   int = 10
MIN_ROTATION_SAMPLES:    int = 10
MIN_EXPLORATION_SAMPLES: int = 5

# ── Trigger weights ────────────────────────────────────────────────────────────
TRIGGER_WEIGHTS: Dict[str, int] = {
    "promotion_alpha_collapse":    3,
    "rotation_alpha_collapse":     3,
    "promotion_precision_collapse": 2,
    "exploration_failure":         1,
    "turnover_anomaly":            1,
}
MAX_DEGRADATION_SCORE: int = sum(TRIGGER_WEIGHTS.values())   # 10

# ── Trigger thresholds ─────────────────────────────────────────────────────────
PROMOTION_ALPHA_THRESHOLD:     float = 0.0    # alpha_30d < 0 → collapse
ROTATION_ALPHA_THRESHOLD:      float = 0.0    # alpha_30d < 0 → collapse
PRECISION_WIN_RATE_THRESHOLD:  float = 0.33   # win_rate < 33% → collapse
EXPLORATION_SUCCESS_THRESHOLD: float = 0.25   # success_rate < 25% → failure
TURNOVER_ANOMALY_RATIO:        float = 2.0    # recent_count > avg * 2.0 → anomaly

# ── Weighted score → mode ──────────────────────────────────────────────────────
FREEZE_SCORE_THRESHOLD:    int = 8
SAFE_MODE_SCORE_THRESHOLD: int = 6
DEGRADED_SCORE_THRESHOLD:  int = 3
WARNING_SCORE_THRESHOLD:   int = 1

_SCHEMA_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
# Input types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OutcomeSummary:
    """Pre-aggregated outcome metrics for one cohort (promotions / rotations / explorations)."""
    sample_count:  int
    alpha_30d:     Optional[float] = None   # mean 30d forward return
    win_rate_30d:  Optional[float] = None   # fraction with positive 30d return
    success_rate:  Optional[float] = None   # explorations: fraction that became leaders


@dataclass
class TurnoverStats:
    """Universe turnover statistics for anomaly detection."""
    recent_count:  int   = 0      # additions + removals in the most recent window
    expected_avg:  float = 1.0    # expected average for the same window


# ─────────────────────────────────────────────────────────────────────────────
# Output type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AutoFreezeState:
    """Full output of one AutoFreezeEngine run."""
    auto_freeze_mode:  str
    degradation_score: float
    triggered_checks:  List[str]
    trigger_values:    Dict[str, float]   # all measured values (triggered or not)
    computed_at:       str
    schema_version:    int = _SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "AutoFreezeState":
        return AutoFreezeState(
            auto_freeze_mode=d.get("auto_freeze_mode", "NORMAL"),
            degradation_score=float(d.get("degradation_score", 0.0)),
            triggered_checks=list(d.get("triggered_checks", [])),
            trigger_values={k: float(v) for k, v in d.get("trigger_values", {}).items()},
            computed_at=d.get("computed_at", ""),
            schema_version=int(d.get("schema_version", _SCHEMA_VERSION)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class AutoFreezeEngine:
    """
    Outcome-based freeze trigger.

    Checks five collapse/anomaly conditions with weighted scores.
    Triggers with insufficient samples are skipped (INSUFFICIENT_DATA semantics).
    FAIL_OPEN: any error returns NORMAL auto_freeze_mode.
    """

    def __init__(self, state_file: Path) -> None:
        self._state_file = state_file

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        promotion_outcomes:   Optional[OutcomeSummary] = None,
        rotation_outcomes:    Optional[OutcomeSummary] = None,
        exploration_outcomes: Optional[OutcomeSummary] = None,
        turnover_stats:       Optional[TurnoverStats]  = None,
    ) -> AutoFreezeState:
        """Compute degradation score and auto_freeze_mode. FAIL_OPEN."""
        try:
            return self._run_internal(
                promotion_outcomes, rotation_outcomes,
                exploration_outcomes, turnover_stats,
            )
        except Exception as exc:
            logger.warning("[AUTO_FREEZE] run() failed (FAIL_OPEN): %s", exc)
            return AutoFreezeState(
                auto_freeze_mode="NORMAL",
                degradation_score=0.0,
                triggered_checks=[],
                trigger_values={},
                computed_at=_now_utc(),
            )

    def write_state(self, state: AutoFreezeState) -> None:
        """Atomic write of auto_freeze_state.json. FAIL_OPEN."""
        try:
            _write_atomic(
                self._state_file,
                json.dumps(state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            )
        except Exception as exc:
            logger.warning("[AUTO_FREEZE] write_state failed (FAIL_OPEN): %s", exc)

    def load_state(self) -> Optional[AutoFreezeState]:
        """Load auto_freeze_state.json. Returns None on missing/corrupt."""
        if not self._state_file.exists():
            return None
        try:
            return AutoFreezeState.from_dict(
                json.loads(self._state_file.read_text(encoding="utf-8"))
            )
        except Exception as exc:
            logger.warning("[AUTO_FREEZE] load_state failed: %s", exc)
            return None

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run_internal(
        self,
        promotion_outcomes:   Optional[OutcomeSummary],
        rotation_outcomes:    Optional[OutcomeSummary],
        exploration_outcomes: Optional[OutcomeSummary],
        turnover_stats:       Optional[TurnoverStats],
    ) -> AutoFreezeState:
        now      = _now_utc()
        score    = 0
        triggered: List[str]        = []
        values:    Dict[str, float] = {}

        # ── 1. Promotion alpha collapse (weight=3) ────────────────────────────
        if promotion_outcomes is not None:
            if promotion_outcomes.alpha_30d is not None:
                values["promotion_alpha_30d"] = promotion_outcomes.alpha_30d
            if promotion_outcomes.sample_count >= MIN_PROMOTION_SAMPLES:
                if (promotion_outcomes.alpha_30d is not None
                        and promotion_outcomes.alpha_30d < PROMOTION_ALPHA_THRESHOLD):
                    triggered.append("promotion_alpha_collapse")
                    score += TRIGGER_WEIGHTS["promotion_alpha_collapse"]
            else:
                logger.debug(
                    "[AUTO_FREEZE] promotion samples=%d < min=%d → INSUFFICIENT_DATA (skip)",
                    promotion_outcomes.sample_count, MIN_PROMOTION_SAMPLES,
                )

        # ── 2. Rotation alpha collapse (weight=3) ─────────────────────────────
        if rotation_outcomes is not None:
            if rotation_outcomes.alpha_30d is not None:
                values["rotation_alpha_30d"] = rotation_outcomes.alpha_30d
            if rotation_outcomes.sample_count >= MIN_ROTATION_SAMPLES:
                if (rotation_outcomes.alpha_30d is not None
                        and rotation_outcomes.alpha_30d < ROTATION_ALPHA_THRESHOLD):
                    triggered.append("rotation_alpha_collapse")
                    score += TRIGGER_WEIGHTS["rotation_alpha_collapse"]
            else:
                logger.debug(
                    "[AUTO_FREEZE] rotation samples=%d < min=%d → INSUFFICIENT_DATA (skip)",
                    rotation_outcomes.sample_count, MIN_ROTATION_SAMPLES,
                )

        # ── 3. Promotion precision collapse / win-rate (weight=2) ─────────────
        if promotion_outcomes is not None:
            if promotion_outcomes.win_rate_30d is not None:
                values["promotion_win_rate_30d"] = promotion_outcomes.win_rate_30d
            if promotion_outcomes.sample_count >= MIN_PROMOTION_SAMPLES:
                if (promotion_outcomes.win_rate_30d is not None
                        and promotion_outcomes.win_rate_30d < PRECISION_WIN_RATE_THRESHOLD):
                    triggered.append("promotion_precision_collapse")
                    score += TRIGGER_WEIGHTS["promotion_precision_collapse"]

        # ── 4. Exploration failure (weight=1) ─────────────────────────────────
        if exploration_outcomes is not None:
            if exploration_outcomes.success_rate is not None:
                values["exploration_success_rate"] = exploration_outcomes.success_rate
            if exploration_outcomes.sample_count >= MIN_EXPLORATION_SAMPLES:
                if (exploration_outcomes.success_rate is not None
                        and exploration_outcomes.success_rate < EXPLORATION_SUCCESS_THRESHOLD):
                    triggered.append("exploration_failure")
                    score += TRIGGER_WEIGHTS["exploration_failure"]
            else:
                logger.debug(
                    "[AUTO_FREEZE] exploration samples=%d < min=%d → INSUFFICIENT_DATA (skip)",
                    exploration_outcomes.sample_count, MIN_EXPLORATION_SAMPLES,
                )

        # ── 5. Turnover anomaly (weight=1) ────────────────────────────────────
        if turnover_stats is not None:
            values["turnover_recent"]   = float(turnover_stats.recent_count)
            values["turnover_expected"] = float(turnover_stats.expected_avg)
            if (turnover_stats.expected_avg > 0
                    and turnover_stats.recent_count
                    > turnover_stats.expected_avg * TURNOVER_ANOMALY_RATIO):
                triggered.append("turnover_anomaly")
                score += TRIGGER_WEIGHTS["turnover_anomaly"]

        mode = _score_to_freeze_mode(score)

        if triggered:
            logger.warning(
                "[AUTO_FREEZE] mode=%s score=%d triggered=%s values=%s",
                mode, score, triggered, {k: f"{v:.3f}" for k, v in values.items()},
            )
        else:
            logger.info("[AUTO_FREEZE] mode=%s score=%d no_triggers", mode, score)

        return AutoFreezeState(
            auto_freeze_mode=mode,
            degradation_score=float(score),
            triggered_checks=triggered,
            trigger_values=values,
            computed_at=now,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_to_freeze_mode(score: int) -> str:
    if score >= FREEZE_SCORE_THRESHOLD:
        return "FREEZE"
    if score >= SAFE_MODE_SCORE_THRESHOLD:
        return "SAFE_MODE"
    if score >= DEGRADED_SCORE_THRESHOLD:
        return "DEGRADED"
    if score >= WARNING_SCORE_THRESHOLD:
        return "WARNING"
    return "NORMAL"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_atomic(path: Path, content: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
