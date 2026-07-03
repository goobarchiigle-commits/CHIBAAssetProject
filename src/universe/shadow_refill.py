"""
src/universe/shadow_refill.py — Shadow Universe Auto Refill

Replenishes shadow_universe.json when shadow-only symbols fall below MIN_SHADOW_SIZE.

Candidate source:
    Symbols present in RSR snapshots (or PE scores) but absent from live universe.

Ranking:
    predictive_alpha_score (primary) → current RSR (secondary)

Viability filter:
    current RSR >= RSR_VIABILITY_MIN (not already in free-fall)

Design:
    - FAIL_OPEN: any error logs and returns no candidates (no shadow write)
    - Atomic write: shadow_universe.json updated via .tmp → replace
    - Idempotent: same inputs → same refill result
    - No execution side-effects (read + write shadow file only)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_SHADOW_SIZE:    int   = 10     # trigger refill when shadow-only < this
SHADOW_REFILL_BATCH: int  = 5     # add at most this many symbols per refill
RSR_VIABILITY_MIN:  float = 30.0  # candidate must have RSR >= this

_SCHEMA_VERSION = 1


class ShadowRefillEngine:
    """
    Checks whether shadow needs refilling and computes candidates.

    Usage:
        engine = ShadowRefillEngine()
        if engine.should_refill(shadow, live):
            candidates = engine.compute_refill(shadow, live, rsr_scores, pe_scores_file, rsr_dir)
            added = engine.execute_refill(shadow_file, candidates)
    """

    # ── Public API ─────────────────────────────────────────────────────────────

    def should_refill(
        self,
        shadow_universe: Dict[str, str],
        live_universe:   Dict[str, str],
    ) -> bool:
        """True when shadow-only symbols < MIN_SHADOW_SIZE."""
        shadow_only_count = sum(
            1 for s in shadow_universe if s not in live_universe
        )
        return shadow_only_count < MIN_SHADOW_SIZE

    def compute_refill(
        self,
        shadow_universe:  Dict[str, str],
        live_universe:    Dict[str, str],
        rsr_scores:       Optional[Dict[str, float]],
        pe_scores_file:   Optional[Path],
        rsr_snapshot_dir: Optional[Path] = None,
    ) -> List[Tuple[str, str]]:
        """
        Compute (symbol, sector) list to add to shadow.

        Priority: predictive_alpha_score descending, then RSR descending.
        Returns [] on any error (FAIL_OPEN).
        """
        try:
            rsr = dict(rsr_scores or {})
            if not rsr and rsr_snapshot_dir:
                rsr = _load_latest_rsr(rsr_snapshot_dir)

            pe_map = _load_pe_scores(pe_scores_file) if pe_scores_file else {}

            # Candidate pool: symbols with RSR/PE data but NOT in live universe
            live_set   = set(live_universe)
            shadow_set = set(shadow_universe)

            all_known = set(rsr) | set(pe_map)
            pool = {
                sym for sym in all_known
                if sym not in live_set
            }

            if not pool:
                logger.info("[SHADOW_REFILL] no candidates outside live universe")
                return []

            # Viability filter: RSR >= RSR_VIABILITY_MIN (exclude already-failed)
            viable = [
                sym for sym in pool
                if rsr.get(sym, 0.0) >= RSR_VIABILITY_MIN
            ]
            if not viable:
                logger.info("[SHADOW_REFILL] no viable candidates (RSR >= %.0f)", RSR_VIABILITY_MIN)
                return []

            # Rank by PE score desc, then RSR desc
            ranked = sorted(
                viable,
                key=lambda s: (-pe_map.get(s, 0.0), -rsr.get(s, 0.0)),
            )

            # Determine how many slots to fill
            shadow_only_count = sum(1 for s in shadow_set if s not in live_set)
            slots_needed = max(0, MIN_SHADOW_SIZE - shadow_only_count)
            to_add = ranked[:min(slots_needed, SHADOW_REFILL_BATCH)]

            # Retrieve sector info: PE file sometimes has no sector, use empty fallback
            sector_map = _load_sector_map(pe_scores_file) if pe_scores_file else {}
            result = [(sym, sector_map.get(sym, "")) for sym in to_add]

            if result:
                logger.info(
                    "[SHADOW_REFILL] computed %d refill candidates: %s",
                    len(result), [s for s, _ in result],
                )
            return result

        except Exception as exc:
            logger.warning("[SHADOW_REFILL] compute_refill failed (FAIL_OPEN): %s", exc)
            return []

    def execute_refill(
        self,
        shadow_file: Path,
        candidates:  List[Tuple[str, str]],
    ) -> List[Tuple[str, str]]:
        """
        Atomically write candidates into shadow_universe.json.
        Returns list of (symbol, sector) actually added.
        FAIL_OPEN: returns [] on any error.
        """
        if not candidates:
            return []
        try:
            existing: dict = {}
            if shadow_file.exists():
                existing = json.loads(shadow_file.read_text(encoding="utf-8"))

            symbols: Dict[str, str] = dict(existing.get("symbols", {}))
            added: List[Tuple[str, str]] = []
            for sym, sec in candidates:
                if sym not in symbols:
                    symbols[sym] = sec
                    added.append((sym, sec))

            if not added:
                return []

            updated = {
                **existing,
                "symbols": {k: v for k, v in sorted(symbols.items())},
                "last_refill_at": datetime.now(timezone.utc).isoformat(),
                "last_refill_added": sorted(s for s, _ in added),
                "schema_version": _SCHEMA_VERSION,
            }
            tmp = shadow_file.with_suffix(".tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(
                json.dumps(updated, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(shadow_file)
            logger.info(
                "[SHADOW_REFILL] shadow updated: added=%s  total=%d",
                [s for s, _ in added], len(symbols),
            )
            return added

        except Exception as exc:
            logger.warning("[SHADOW_REFILL] execute_refill failed (FAIL_OPEN): %s", exc)
            return []


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_latest_rsr(rsr_dir: Path) -> Dict[str, float]:
    try:
        snaps = sorted(rsr_dir.glob("*.json"))
        if not snaps:
            return {}
        data = json.loads(snaps[-1].read_text(encoding="utf-8"))
        return {s: float(v) for s, v in data.get("scores", {}).items()}
    except Exception as exc:
        logger.warning("[SHADOW_REFILL] RSR load error: %s", exc)
        return {}


def _load_pe_scores(pe_scores_file: Path) -> Dict[str, float]:
    if not pe_scores_file or not pe_scores_file.exists():
        return {}
    try:
        lines = [l for l in pe_scores_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            return {}
        rec = json.loads(lines[-1])
        return {s: float(v) for s, v in rec.get("predictive_alpha_score", {}).items()}
    except Exception as exc:
        logger.warning("[SHADOW_REFILL] PE scores load error: %s", exc)
        return {}


def _load_sector_map(pe_scores_file: Path) -> Dict[str, str]:
    """Attempt to extract sector info from PE scores file metadata (best-effort)."""
    if not pe_scores_file or not pe_scores_file.exists():
        return {}
    try:
        lines = [l for l in pe_scores_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            return {}
        rec = json.loads(lines[-1])
        return dict(rec.get("sector_map", {}))
    except Exception:
        return {}
