"""
src/universe/universe_shrink_engine.py — Universe size self-healing

Evaluates non-held live symbols for removal when live_size > target_size.
RSR-only instant removal is FORBIDDEN; at least 2 evidence sources required.

Shrink score formula:
    shrink_score = W_RSR * rsr_weakness_score
                 + W_FL  * fl_weakness_score
                 + W_PE  * pe_weakness_score

Candidate gate (BOTH conditions required):
    1. shrink_score >= SHRINK_SCORE_THRESHOLD
    2. rsr_weak_days >= RSR_MIN_WEAK_DAYS          (consecutive-day guard)

ShrinkPressure is duck-type compatible with DemotionActuationEngine.run():
    pressure_score = shrink_score
    is_candidate   = (gate 1 AND gate 2)

Design:
    - FAIL_OPEN: per-symbol errors → ShrinkPressure(pressure_score=0, is_candidate=False)
    - Only fires when live_size > target_size
    - Held positions are excluded by the caller
    - No lookahead: uses T-1 RSR snapshots
    - Deterministic: same inputs → same outputs
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional

logger = logging.getLogger(__name__)

# ── Evidence weights (must sum to 1.0) ────────────────────────────────────────
W_RSR: float = 0.50   # RSR persistence carries most weight
W_FL:  float = 0.30   # Future Leader growth signal
W_PE:  float = 0.20   # Predictive Expansion momentum

# ── RSR weakness thresholds ───────────────────────────────────────────────────
RSR_SHRINK_THRESHOLD: float = 25.0   # RSR below this = weak signal per day
RSR_MIN_WEAK_DAYS:    int   = 3      # consecutive days required before flagging
RSR_HISTORY_DAYS:     int   = 7      # how many snapshot days to load

# ── Combined score threshold ──────────────────────────────────────────────────
SHRINK_SCORE_THRESHOLD: float = 0.65   # conservative; avoids false positives

# ── Missing data neutral values ───────────────────────────────────────────────
MISSING_FL_WEAKNESS: float = 0.50    # neutral when no FL record exists
MISSING_PE_WEAKNESS: float = 0.50    # neutral when no PE record exists

_SCHEMA_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ShrinkPressure:
    """Per-symbol shrink evaluation result. Duck-type compatible with DemotionPressure."""
    symbol:              str
    shrink_score:        float   # weighted composite [0, 1]
    rsr_weak_days:       int     # consecutive days with RSR < RSR_SHRINK_THRESHOLD
    rsr_weakness_score:  float   # [0, 1]
    fl_weakness_score:   float   # [0, 1]  1 - fl_score/100
    pe_weakness_score:   float   # [0, 1]  1 - predictive_alpha/100
    pressure_score:      float   # = shrink_score (duck-type alias)
    is_candidate:        bool    # gate 1 AND gate 2
    reason:              str

    def to_dict(self) -> dict:
        return {**asdict(self), "schema_version": _SCHEMA_VERSION}


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class UniverseShrinkEngine:
    """
    Compute shrink pressure for non-held live symbols.

    Usage:
        engine = UniverseShrinkEngine(rsr_dir, fl_candidates_file, pe_scores_file)
        pressures = engine.compute(live_universe, held_symbols, target_size)
        # returns {} when live_size <= target_size (no action needed)
    """

    def __init__(
        self,
        rsr_snapshot_dir:    Path,
        fl_candidates_file:  Path,
        pe_scores_file:      Path,
    ) -> None:
        self._rsr_dir    = rsr_snapshot_dir
        self._fl_file    = fl_candidates_file
        self._pe_file    = pe_scores_file

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute(
        self,
        live_universe:  Dict[str, str],
        held_symbols:   FrozenSet[str],
        target_size:    int,
    ) -> Dict[str, ShrinkPressure]:
        """
        Returns {symbol: ShrinkPressure} for non-held symbols when oversized.
        Returns {} when live_size <= target_size (no shrink needed).
        FAIL_OPEN: per-symbol errors yield pressure_score=0, is_candidate=False.
        """
        live_size = len(live_universe)
        if live_size <= target_size:
            return {}

        rsr_history  = self._load_rsr_history()
        fl_scores    = self._load_fl_scores()
        pe_scores    = self._load_pe_scores()

        results: Dict[str, ShrinkPressure] = {}
        for sym in sorted(live_universe):
            if sym in held_symbols:
                continue
            try:
                sp = self._compute_one(sym, rsr_history, fl_scores, pe_scores)
                results[sym] = sp
            except Exception as exc:
                logger.warning("[SHRINK] compute(%s) error (FAIL_OPEN): %s", sym, exc)
                results[sym] = _zero_pressure(sym)

        cands = [s for s, p in results.items() if p.is_candidate]
        if cands:
            logger.info(
                "[SHRINK] live=%d target=%d  candidates=%s",
                live_size, target_size, cands,
            )
        else:
            logger.info(
                "[SHRINK] live=%d target=%d  no shrink candidates (all symbols pass)",
                live_size, target_size,
            )
        return results

    # ── Per-symbol scoring ─────────────────────────────────────────────────────

    def _compute_one(
        self,
        symbol:      str,
        rsr_history: Dict[str, List[float]],
        fl_scores:   Dict[str, float],
        pe_scores:   Dict[str, float],
    ) -> ShrinkPressure:
        # ── RSR weakness ──────────────────────────────────────────────────────
        rsr_series  = rsr_history.get(symbol, [])
        weak_days   = _consecutive_weak_days(rsr_series, RSR_SHRINK_THRESHOLD)
        # Normalize: RSR_MIN_WEAK_DAYS * 1.5 = saturation point
        rsr_weakness = min(1.0, weak_days / (RSR_MIN_WEAK_DAYS * 1.5))

        # ── Future Leader weakness ────────────────────────────────────────────
        if symbol in fl_scores:
            fl_weakness = round(1.0 - fl_scores[symbol] / 100.0, 4)
            fl_weakness = max(0.0, min(1.0, fl_weakness))
        else:
            fl_weakness = MISSING_FL_WEAKNESS

        # ── Predictive Expansion weakness ─────────────────────────────────────
        if symbol in pe_scores:
            pe_weakness = round(1.0 - pe_scores[symbol] / 100.0, 4)
            pe_weakness = max(0.0, min(1.0, pe_weakness))
        else:
            pe_weakness = MISSING_PE_WEAKNESS

        # ── Composite ────────────────────────────────────────────────────────
        shrink_score = round(
            W_RSR * rsr_weakness
            + W_FL  * fl_weakness
            + W_PE  * pe_weakness,
            4,
        )

        # ── Gate ─────────────────────────────────────────────────────────────
        is_candidate = (
            shrink_score >= SHRINK_SCORE_THRESHOLD
            and weak_days >= RSR_MIN_WEAK_DAYS
        )

        reason = (
            f"rsr_weak={weak_days}d score={shrink_score:.3f}"
            if is_candidate
            else f"pass: rsr_weak={weak_days}d score={shrink_score:.3f}"
        )

        return ShrinkPressure(
            symbol             = symbol,
            shrink_score       = shrink_score,
            rsr_weak_days      = weak_days,
            rsr_weakness_score = round(rsr_weakness, 4),
            fl_weakness_score  = round(fl_weakness, 4),
            pe_weakness_score  = round(pe_weakness, 4),
            pressure_score     = shrink_score,   # duck-type alias
            is_candidate       = is_candidate,
            reason             = reason,
        )

    # ── Data loaders ──────────────────────────────────────────────────────────

    def _load_rsr_history(self) -> Dict[str, List[float]]:
        """Load last RSR_HISTORY_DAYS snapshots. Returns {symbol: [rsr, ...]}."""
        history: Dict[str, List[float]] = {}
        if not self._rsr_dir.exists():
            return history
        try:
            for snap_file in sorted(self._rsr_dir.glob("*.json"))[-RSR_HISTORY_DAYS:]:
                snap = json.loads(snap_file.read_text(encoding="utf-8"))
                for sym, rsr in snap.get("scores", {}).items():
                    history.setdefault(sym, []).append(float(rsr))
        except Exception as exc:
            logger.warning("[SHRINK] RSR history load error: %s", exc)
        return history

    def _load_fl_scores(self) -> Dict[str, float]:
        """Load most recent FL score per symbol from candidates.jsonl."""
        if not self._fl_file.exists():
            return {}
        latest: Dict[str, float] = {}
        try:
            for line in self._fl_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sym = rec.get("symbol")
                if sym:
                    # Later records overwrite → most recent wins
                    latest[sym] = float(rec.get("future_leader_score", 0.0))
        except Exception as exc:
            logger.warning("[SHRINK] FL candidates load error: %s", exc)
        return latest

    def _load_pe_scores(self) -> Dict[str, float]:
        """Load most recent predictive_alpha_score from predictive_scores.jsonl."""
        if not self._pe_file.exists():
            return {}
        try:
            lines = [
                l for l in self._pe_file.read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
            if not lines:
                return {}
            latest_rec = json.loads(lines[-1])
            return {
                sym: float(score)
                for sym, score in latest_rec.get("predictive_alpha_score", {}).items()
            }
        except Exception as exc:
            logger.warning("[SHRINK] PE scores load error: %s", exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _consecutive_weak_days(rsr_series: List[float], threshold: float) -> int:
    """Count trailing consecutive days where RSR < threshold."""
    count = 0
    for rsr in reversed(rsr_series):
        if rsr < threshold:
            count += 1
        else:
            break
    return count


def _zero_pressure(symbol: str) -> ShrinkPressure:
    return ShrinkPressure(
        symbol=symbol,
        shrink_score=0.0,
        rsr_weak_days=0,
        rsr_weakness_score=0.0,
        fl_weakness_score=0.0,
        pe_weakness_score=0.0,
        pressure_score=0.0,
        is_candidate=False,
        reason="error_fail_open",
    )
