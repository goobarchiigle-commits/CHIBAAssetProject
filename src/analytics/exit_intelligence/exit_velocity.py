"""
src/analytics/exit_intelligence/exit_velocity.py — Exit Pressure Layer

Deterministic ExitVelocityScorer.
Produces ExitVelocityScore (0–10) with explainable components.
No adaptive mutation. No ML. Deterministic weighting only.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from .models import (
    ConvexityRetentionMetrics,
    ExitVelocityScore,
    MomentumDecayMetrics,
    PeakDecayVelocityMetrics,
    PersistenceMetrics,
    PortfolioExitPressureSnapshot,
    StructuralFailureMetrics,
)
from .observation import ExitPathSnapshot

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))


def _now_jst() -> str:
    return datetime.now(JST).isoformat()


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _atomic_append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            if path.exists():
                fh.write(path.read_text(encoding="utf-8"))
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# Fixed component weights — deterministic, never auto-tuned
_WEIGHTS: dict = {
    "rs_deterioration":            0.12,
    "breadth_decay":               0.10,
    "breakout_failure":            0.12,
    "vwap_rejection":              0.08,
    "upper_shadow":                0.06,
    "volume_decay":                0.08,
    "volatility_compression":      0.06,  # inverted: low compression = pressure
    "momentum_decay":              0.14,
    "distance_from_peak":          0.12,
    "execution_deterioration":     0.06,
    "portfolio_replacement":       0.06,
}
assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1.0"


def _action_band(total: float) -> str:
    if total <= 2.0:
        return "hold"
    elif total <= 4.0:
        return "tighten_risk"
    elif total <= 6.0:
        return "staged_reduction"
    else:
        return "aggressive_exit"


class ExitVelocityScorer:
    """
    Compute deterministic ExitVelocityScore from deterioration metrics.

    Score = weighted sum of 11 component scores × 10 (scale to 0–10).
    Each component is normalized to [0, 1] where 1 = maximum pressure.
    """

    def score(
        self,
        symbol: str,
        scored_date: str,
        latest_snapshot: ExitPathSnapshot,
        momentum_decay: MomentumDecayMetrics,
        structural_failure: StructuralFailureMetrics,
        peak_decay: PeakDecayVelocityMetrics,
        convexity_retention: ConvexityRetentionMetrics,
        persistence: PersistenceMetrics,
        portfolio_pressure: Optional[PortfolioExitPressureSnapshot] = None,
        avg_exec_slippage_bps: float = 0.0,
    ) -> ExitVelocityScore:
        """
        Compute exit velocity score. All inputs are pre-computed analytics.
        Returns immutable ExitVelocityScore with component breakdown.
        """
        # RS deterioration: higher RS rank = less pressure
        rs_deterioration = _clamp(1.0 - latest_snapshot.rs_rank / 100.0)

        # Breadth decay
        breadth_decay = _clamp(1.0 - latest_snapshot.breadth_score)

        # Breakout failure: direct from structural detector
        breakout_failure = _clamp(structural_failure.structural_failure_score)

        # VWAP rejection: upper shadow expansion + negative momentum
        vwap_rejection = _clamp(
            0.5 * structural_failure.convexity_collapse  # type: ignore[arg-type]
            + 0.5 * momentum_decay.upper_shadow_expansion
        )
        vwap_rejection = _clamp(
            (0.7 if structural_failure.vwap_reclaim_failure else 0.0)
            + 0.3 * momentum_decay.upper_shadow_expansion
        )

        # Upper shadow
        upper_shadow = _clamp(momentum_decay.upper_shadow_expansion)

        # Volume decay
        volume_decay = _clamp(momentum_decay.volume_deterioration)

        # Volatility compression: inverted — constructive compression LOWERS pressure
        vol_comp_raw = _clamp(latest_snapshot.volatility_compression)
        volatility_compression_pressure = _clamp(
            1.0 - vol_comp_raw if convexity_retention.is_constructive_compression else vol_comp_raw
        )

        # Momentum decay: normalize velocity (negative slope = pressure)
        mom_vel = momentum_decay.momentum_velocity
        # Negative velocity (decaying) → pressure; cap at ±0.05/day range
        momentum_decay_score = _clamp((-mom_vel + 0.05) / 0.10)

        # Distance from peak
        distance_from_peak = _clamp(
            latest_snapshot.distance_from_peak_pct / 0.25  # 25% distance = max pressure
        )

        # Execution deterioration: normalize slippage
        execution_deterioration = _clamp(avg_exec_slippage_bps / 50.0)  # 50bps = max

        # Portfolio replacement pressure
        if portfolio_pressure is not None:
            portfolio_replacement = _clamp(portfolio_pressure.replacement_pressure)
        else:
            portfolio_replacement = 0.0

        components = {
            "rs_deterioration": rs_deterioration,
            "breadth_decay": breadth_decay,
            "breakout_failure": breakout_failure,
            "vwap_rejection": vwap_rejection,
            "upper_shadow": upper_shadow,
            "volume_decay": volume_decay,
            "volatility_compression": volatility_compression_pressure,
            "momentum_decay": momentum_decay_score,
            "distance_from_peak": distance_from_peak,
            "execution_deterioration": execution_deterioration,
            "portfolio_replacement": portfolio_replacement,
        }

        # Weighted sum → scale 0–10
        raw_weighted = sum(components[k] * _WEIGHTS[k] for k in _WEIGHTS)
        total_score = _clamp(raw_weighted * 10.0, 0.0, 10.0)

        # Convexity preservation: if persistent leader / HTF, dampen score
        if convexity_retention.is_persistent_leader or convexity_retention.is_high_tight_flag:
            total_score = _clamp(total_score * 0.75, 0.0, 10.0)

        return ExitVelocityScore(
            score_id=ExitVelocityScore.make_score_id(symbol, scored_date),
            symbol=symbol,
            scored_date=scored_date,
            rs_deterioration=rs_deterioration,
            breadth_decay=breadth_decay,
            breakout_failure=breakout_failure,
            vwap_rejection=vwap_rejection,
            upper_shadow=upper_shadow,
            volume_decay=volume_decay,
            volatility_compression=volatility_compression_pressure,
            momentum_decay=momentum_decay_score,
            distance_from_peak=distance_from_peak,
            execution_deterioration=execution_deterioration,
            portfolio_replacement_pressure=portfolio_replacement,
            total_score=round(total_score, 4),
            action_band=_action_band(total_score),
            component_weights=dict(_WEIGHTS),
            generated_at=_now_jst(),
        )


class ExitVelocityStore:
    """Append-only store for ExitVelocityScore."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, score: ExitVelocityScore) -> None:
        _atomic_append(self._path, score.to_dict())
        logger.debug("[EVS] appended score=%s symbol=%s band=%s",
                     score.total_score, score.symbol, score.action_band)

    def load_all(self) -> List[ExitVelocityScore]:
        if not self._path.exists():
            return []
        records = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                records.append(ExitVelocityScore.from_dict(d))
            except Exception as exc:
                logger.warning("[EVS] parse error: %s", exc)
        return records

    def load_latest_by_symbol(self, symbol: str) -> Optional[ExitVelocityScore]:
        candidates = [r for r in self.load_all() if r.symbol == symbol]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.scored_date)
