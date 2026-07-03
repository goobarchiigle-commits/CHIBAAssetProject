"""
capital_freeze.py — Capital freeze circuit breaker

Any single threshold breach triggers freeze (OR logic, fail-closed).
Freeze suspends effective_capital growth; existing positions unaffected.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FreezeConfig:
    slippage_bps_max: float = 50.0      # >50 bps → freeze
    reject_rate_max: float = 0.30       # >30% reject → freeze
    fill_ratio_min: float = 0.70        # <70% fill → freeze
    liquidity_stress_max: float = 0.70  # stress score >0.70 → freeze
    partial_fill_max: float = 0.50      # >50% partials → freeze


@dataclass
class FreezeState:
    is_frozen: bool = False
    freeze_reason: Optional[str] = None
    trigger_count: int = 0
    schema_version: int = 1


def evaluate_freeze(
    slippage_bps: float,
    reject_rate: float,
    fill_ratio: float,
    liquidity_stress_score: float,
    partial_fill_ratio: float = 0.0,
    cfg: Optional[FreezeConfig] = None,
) -> tuple[bool, Optional[str]]:
    """
    Returns (should_freeze, reason_string).
    OR logic: any single threshold breach → freeze.
    """
    if cfg is None:
        cfg = FreezeConfig()

    checks = [
        (slippage_bps > cfg.slippage_bps_max,
         f"slippage_bps={slippage_bps:.1f} > limit={cfg.slippage_bps_max}"),
        (reject_rate > cfg.reject_rate_max,
         f"reject_rate={reject_rate:.2%} > limit={cfg.reject_rate_max:.2%}"),
        (fill_ratio < cfg.fill_ratio_min,
         f"fill_ratio={fill_ratio:.2%} < limit={cfg.fill_ratio_min:.2%}"),
        (liquidity_stress_score > cfg.liquidity_stress_max,
         f"liquidity_stress={liquidity_stress_score:.3f} > limit={cfg.liquidity_stress_max}"),
        (partial_fill_ratio > cfg.partial_fill_max,
         f"partial_fill_ratio={partial_fill_ratio:.2%} > limit={cfg.partial_fill_max:.2%}"),
    ]

    for triggered, reason in checks:
        if triggered:
            return True, reason
    return False, None


def update_freeze_state(
    state: FreezeState,
    slippage_bps: float,
    reject_rate: float,
    fill_ratio: float,
    liquidity_stress_score: float,
    partial_fill_ratio: float = 0.0,
    cfg: Optional[FreezeConfig] = None,
) -> FreezeState:
    """Evaluate metrics and return updated FreezeState."""
    should_freeze, reason = evaluate_freeze(
        slippage_bps, reject_rate, fill_ratio,
        liquidity_stress_score, partial_fill_ratio, cfg,
    )

    if should_freeze:
        if not state.is_frozen:
            logger.warning("CapitalFreezeBreaker: FROZEN — %s", reason)
        return FreezeState(
            is_frozen=True,
            freeze_reason=reason,
            trigger_count=state.trigger_count + 1,
        )
    else:
        if state.is_frozen:
            logger.info("CapitalFreezeBreaker: UNFROZEN (all metrics healthy)")
        return FreezeState(
            is_frozen=False,
            freeze_reason=None,
            trigger_count=state.trigger_count,
        )


def load_freeze_state(path: Path) -> FreezeState:
    """Load freeze state. Returns unfrozen default on missing/corrupt file."""
    if not path.exists():
        return FreezeState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return FreezeState(
            is_frozen=bool(data.get("is_frozen", False)),
            freeze_reason=data.get("freeze_reason"),
            trigger_count=int(data.get("trigger_count", 0)),
        )
    except Exception as exc:
        logger.warning("freeze_state load failed (%s) — defaulting to unfrozen", exc)
        return FreezeState()


def save_freeze_state(state: FreezeState, path: Path) -> None:
    """Atomic write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(asdict(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)
