"""
deployment_ramp.py — Runtime-trust-based capital deployment ramp

Tracks rolling execution quality. Poor quality → freeze effective_capital growth.
Requires N consecutive good executions to recover from FROZEN state.

State machine: RAMPING ↔ FROZEN → RECOVERING → RAMPING
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

QUALITY_WINDOW: int = 20              # rolling window size
QUALITY_FREEZE_THRESHOLD: float = 0.60   # score below this → FROZEN
QUALITY_RAMP_THRESHOLD: float = 0.80    # must sustain above this to recover
FREEZE_RECOVERY_WINDOW: int = 5          # N consecutive good execs to unfreeze


@dataclass
class ExecutionQualityRecord:
    fill_ratio: float           # filled_qty / submitted_qty  (1.0 = perfect)
    partial_fill_ratio: float   # partial fills / total       (0.0 = perfect)
    slippage_bps: float         # average realized slippage bps
    reject_rate: float          # rejected_orders / total     (0.0 = perfect)
    execution_latency_ms: float # average execution latency
    trading_day: str            # ISO date


@dataclass
class DeploymentRampState:
    mode: str = "RAMPING"                              # RAMPING | FROZEN | RECOVERING
    quality_history: list = field(default_factory=list)  # rolling ExecutionQualityRecord dicts
    consecutive_good: int = 0
    schema_version: int = 1


def compute_quality_score(record: ExecutionQualityRecord) -> float:
    """
    Composite execution quality score [0, 1].
    Slippage and reject rate weighted most heavily.
    """
    fill_score    = max(0.0, min(1.0, record.fill_ratio))
    slippage_norm = max(0.0, 1.0 - record.slippage_bps / 100.0)   # 100 bps → 0.0
    reject_score  = max(0.0, 1.0 - record.reject_rate * 3.0)       # 33% reject → 0.0
    latency_norm  = max(0.0, 1.0 - record.execution_latency_ms / 2000.0)  # 2 s → 0.0

    score = (
        0.35 * fill_score
        + 0.30 * slippage_norm
        + 0.25 * reject_score
        + 0.10 * latency_norm
    )
    return round(max(0.0, min(1.0, score)), 4)


def update_deployment_ramp(
    ramp: DeploymentRampState,
    record: ExecutionQualityRecord,
    freeze_threshold: float = QUALITY_FREEZE_THRESHOLD,
    ramp_threshold: float = QUALITY_RAMP_THRESHOLD,
    recovery_window: int = FREEZE_RECOVERY_WINDOW,
    window: int = QUALITY_WINDOW,
) -> tuple[DeploymentRampState, float]:
    """
    Update ramp state with new execution quality record.

    Returns (new_state, growth_limit_modifier):
        1.0 = RAMPING  (full growth allowed)
        0.5 = RECOVERING (half growth)
        0.0 = FROZEN   (growth suspended)
    """
    score = compute_quality_score(record)

    # Append to rolling window
    history = list(ramp.quality_history[-(window - 1):])
    history.append({**asdict(record), "quality_score": score})

    consecutive_good = ramp.consecutive_good
    mode = ramp.mode

    if ramp.mode == "RAMPING":
        if score < freeze_threshold:
            mode = "FROZEN"
            consecutive_good = 0
            logger.warning(
                "DeploymentRamp RAMPING→FROZEN (score=%.3f < threshold=%.3f day=%s)",
                score, freeze_threshold, record.trading_day,
            )
        else:
            mode = "RAMPING"
            consecutive_good = min(consecutive_good + 1, recovery_window * 4)

    elif ramp.mode == "FROZEN":
        if score >= ramp_threshold:
            consecutive_good += 1
            if consecutive_good >= recovery_window:
                mode = "RECOVERING"
                logger.info(
                    "DeploymentRamp FROZEN→RECOVERING (%d/%d good execs)",
                    consecutive_good, recovery_window,
                )
            # else stay FROZEN, accumulating consecutive_good
        else:
            mode = "FROZEN"
            consecutive_good = 0

    elif ramp.mode == "RECOVERING":
        if score >= ramp_threshold:
            mode = "RAMPING"
            consecutive_good = min(consecutive_good + 1, recovery_window * 4)
            logger.info("DeploymentRamp RECOVERING→RAMPING (score=%.3f)", score)
        else:
            mode = "FROZEN"
            consecutive_good = 0
            logger.warning(
                "DeploymentRamp RECOVERING→FROZEN (score=%.3f < threshold=%.3f)",
                score, ramp_threshold,
            )

    else:
        mode = "RAMPING"

    new_state = DeploymentRampState(
        mode=mode,
        quality_history=history,
        consecutive_good=consecutive_good,
    )

    modifier = {"RAMPING": 1.0, "RECOVERING": 0.5, "FROZEN": 0.0}.get(mode, 0.0)
    return new_state, modifier


def load_deployment_ramp(path: Path) -> DeploymentRampState:
    """Load ramp state. Returns default RAMPING state on missing/corrupt file."""
    if not path.exists():
        return DeploymentRampState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return DeploymentRampState(
            mode=data.get("mode", "RAMPING"),
            quality_history=data.get("quality_history", []),
            consecutive_good=int(data.get("consecutive_good", 0)),
        )
    except Exception as exc:
        logger.warning("deployment_ramp load failed (%s) — defaulting to RAMPING", exc)
        return DeploymentRampState()


def save_deployment_ramp(ramp: DeploymentRampState, path: Path) -> None:
    """Atomic write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(asdict(ramp), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)
