"""
src/universe/outcome_tracker.py — Phase 2.5 Outcome Tracking Layer

Records promotion, rotation, and exploration decisions at governance time.
Forward returns are null at record time; materialized later by outcome_materializer.py.

Design:
  - observation_only: reads governance output only, never touches execution path
  - append-only JSONL: no in-place mutation, no record overwrite
  - atomic write: each record flushed immediately (no buffering)
  - FAIL_OPEN: all errors emit WARNING; never raises to caller
  - deterministic: UTC timestamps, sort_keys=True, stable record_id via sha256
  - replayable: record_id deduplication prevents double-writing on replay

Schema version: 1
Files:
  logs/universe/promotion_outcome.jsonl
  logs/universe/rotation_outcome.jsonl
  logs/universe/exploration_outcome.jsonl
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_JST = timezone(timedelta(hours=9))

# Round-trip rotation cost as a fraction of trade value.
# commission (0.055% × 2 sides) + slippage (0.1% × 2 sides) = 0.31%
ROTATION_COST_FRACTION: float = 0.0031


# ── Utilities ─────────────────────────────────────────────────────────────────

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_jst() -> str:
    return datetime.now(_JST).strftime("%Y-%m-%d")


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _record_id(fields: dict) -> str:
    """Deterministic 16-char record identifier for deduplication."""
    canonical = json.dumps(fields, sort_keys=True, ensure_ascii=False)
    return _sha256_hex(canonical)[:16]


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _load_existing_record_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rid = rec.get("record_id")
                if rid:
                    ids.add(rid)
            except Exception:
                pass
    except Exception as exc:
        logger.warning("[OUTCOME_TRACKER] existing record_id load error: %s", exc)
    return ids


# ── Public entry point ────────────────────────────────────────────────────────

def record_governance_outcomes(
    gov_result: Any,                                # GovernanceResult (type-erased to avoid circular import)
    shadow_ev_scores: Dict[str, Any],               # {symbol: EVScore}; empty dict if EV engine failed
    live_ev_scores: Dict[str, Any],                 # {symbol: EVScore}; empty dict if EV engine failed
    live_ev_floor: Optional[float],
    outcome_dir: Path,
    run_id: str,
) -> None:
    """
    Top-level FAIL_OPEN entry point.
    Call after governance completes; wraps all sub-writers in try/except.

    Args:
        gov_result:       GovernanceResult from run_universe_governance()
        shadow_ev_scores: {symbol: EVScore} from UniverseEVEngine (may be empty)
        live_ev_scores:   {symbol: EVScore} from UniverseEVEngine (may be empty)
        live_ev_floor:    minimum EV composite across live universe (may be None)
        outcome_dir:      logs/universe/ directory
        run_id:           execution run ID
    """
    try:
        _record_all(gov_result, shadow_ev_scores, live_ev_scores, live_ev_floor,
                    outcome_dir, run_id)
    except Exception as exc:
        logger.warning("[OUTCOME_TRACKER] record_governance_outcomes failed (FAIL_OPEN): %s", exc)


# ── Internal orchestrator ─────────────────────────────────────────────────────

def _record_all(
    gov_result: Any,
    shadow_ev: Dict[str, Any],
    live_ev: Dict[str, Any],
    live_ev_floor: Optional[float],
    outcome_dir: Path,
    run_id: str,
) -> None:
    promo_file  = outcome_dir / "promotion_outcome.jsonl"
    rot_file    = outcome_dir / "rotation_outcome.jsonl"
    explor_file = outcome_dir / "exploration_outcome.jsonl"

    # Determine weakest live EV symbol (used as rotation partner).
    weakest_sym: Optional[str] = None
    weakest_ev: Optional[float] = None
    if live_ev:
        try:
            weakest_sym = min(live_ev, key=lambda s: live_ev[s].composite)
            weakest_ev  = float(live_ev[weakest_sym].composite)
        except Exception:
            pass

    # Pre-load existing record_ids to prevent duplicates on replay.
    existing_promo  = _load_existing_record_ids(promo_file)
    existing_rot    = _load_existing_record_ids(rot_file)
    existing_explor = _load_existing_record_ids(explor_file)

    promoted_count    = 0
    rotation_count    = 0
    exploration_count = 0

    # Iterate over PROMOTE decisions only.
    for decision in gov_result.decisions:
        if decision.action != "PROMOTE":
            continue

        sym    = decision.symbol
        reason = decision.reason or ""

        # EV data for this candidate (may be absent if EV engine failed).
        ev_score = shadow_ev.get(sym)  # EVScore or None

        candidate_ev     = float(ev_score.composite)        if ev_score else None
        expectancy_comp  = float(ev_score.expectancy_score) if ev_score else None
        survivability    = float(ev_score.survivability)    if ev_score else None
        rsr_accel_comp   = float(ev_score.rsr_acceleration) if ev_score else None
        liquidity_comp   = float(ev_score.liquidity_quality)if ev_score else None
        exec_comp        = float(ev_score.exec_stability)   if ev_score else None
        sample_n         = int(ev_score.sample_n)           if ev_score else None

        # Compute replacement_delta if we have the data.
        replacement_delta: Optional[float] = None
        if candidate_ev is not None and live_ev_floor is not None:
            from src.universe.universe_ev_engine import ROTATION_COST_EV
            replacement_delta = round(candidate_ev - live_ev_floor - ROTATION_COST_EV, 4)

        via_exploration = "exploration" in reason

        # ── Write promotion_outcome record ─────────────────────────────────────
        _write_promotion(
            out_path         = promo_file,
            existing_ids     = existing_promo,
            run_id           = run_id,
            symbol           = sym,
            sector           = decision.sector,
            reason           = reason,
            candidate_ev     = candidate_ev,
            live_ev_floor    = live_ev_floor,
            replacement_delta= replacement_delta,
            expectancy_comp  = expectancy_comp,
            survivability    = survivability,
            rsr_accel_comp   = rsr_accel_comp,
            liquidity_comp   = liquidity_comp,
            exec_comp        = exec_comp,
            deployability    = float(decision.deployability_score),
            rsr_score        = float(decision.rsr_score),
            via_exploration  = via_exploration,
            sample_n         = sample_n,
        )
        promoted_count += 1

        # ── Write rotation_outcome record ──────────────────────────────────────
        _write_rotation(
            out_path         = rot_file,
            existing_ids     = existing_rot,
            run_id           = run_id,
            promoted_symbol  = sym,
            demoted_symbol   = weakest_sym,
            promoted_ev      = candidate_ev,
            demoted_ev       = weakest_ev,
            replacement_delta= replacement_delta,
        )
        rotation_count += 1

        # ── Write exploration_outcome record ───────────────────────────────────
        if via_exploration:
            _write_exploration(
                out_path      = explor_file,
                existing_ids  = existing_explor,
                run_id        = run_id,
                symbol        = sym,
                sector        = decision.sector,
                sample_n      = sample_n,
                candidate_ev  = candidate_ev,
                live_ev_floor = live_ev_floor,
            )
            exploration_count += 1

    logger.info(
        "[OUTCOME_TRACKER] written: promotions=%d rotations=%d explorations=%d  dir=%s",
        promoted_count, rotation_count, exploration_count, outcome_dir.name,
    )


# ── Individual record writers ─────────────────────────────────────────────────

def _write_promotion(
    out_path: Path,
    existing_ids: set[str],
    run_id: str,
    symbol: str,
    sector: str,
    reason: str,
    candidate_ev: Optional[float],
    live_ev_floor: Optional[float],
    replacement_delta: Optional[float],
    expectancy_comp: Optional[float],
    survivability: Optional[float],
    rsr_accel_comp: Optional[float],
    liquidity_comp: Optional[float],
    exec_comp: Optional[float],
    deployability: float,
    rsr_score: float,
    via_exploration: bool,
    sample_n: Optional[int],
) -> None:
    today_jst = _today_jst()
    id_seed = {"run_id": run_id, "symbol": symbol, "type": "promotion", "date": today_jst}
    record_id = _record_id(id_seed)

    if record_id in existing_ids:
        logger.debug("[OUTCOME_TRACKER] dup skipped: promotion %s %s", symbol, today_jst)
        return

    record: dict = {
        "record_id":              record_id,
        "schema_version":         SCHEMA_VERSION,
        "record_type":            "promotion",

        "promoted_at":            today_jst,
        "promoted_at_utc":        _now_utc(),
        "run_id":                 run_id,
        "symbol":                 symbol,
        "sector":                 sector,

        # EV scores at promotion time (null if EV engine unavailable)
        "candidate_ev":           round(candidate_ev, 4) if candidate_ev is not None else None,
        "live_ev_floor":          round(live_ev_floor, 4) if live_ev_floor is not None else None,
        "replacement_delta":      replacement_delta,

        # EV component breakdown
        "expectancy_component":   round(expectancy_comp, 4) if expectancy_comp is not None else None,
        "survivability_component":round(survivability, 4)   if survivability   is not None else None,
        "rsr_accel_component":    round(rsr_accel_comp, 4)  if rsr_accel_comp  is not None else None,
        "liquidity_component":    round(liquidity_comp, 4)  if liquidity_comp  is not None else None,
        "exec_component":         round(exec_comp, 4)       if exec_comp       is not None else None,

        # Deployability gate scores
        "deployability_score":    round(deployability, 4),
        "rsr_score":              round(rsr_score, 2),

        "promotion_reason":       reason,
        "via_exploration":        via_exploration,
        "sample_n":               sample_n,

        # Forward returns — null at record time, filled by outcome_materializer.py
        "forward_return_5d":      None,
        "forward_return_10d":     None,
        "forward_return_20d":     None,
        "forward_return_30d":     None,
        "benchmark_return_5d":    None,
        "benchmark_return_20d":   None,
        "realized_alpha_5d":      None,
        "realized_alpha_20d":     None,

        "materialized":           False,
        "materialized_at":        None,
    }
    try:
        _append_jsonl(out_path, record)
        existing_ids.add(record_id)
    except Exception as exc:
        logger.warning("[OUTCOME_TRACKER] promotion write failed %s: %s", symbol, exc)


def _write_rotation(
    out_path: Path,
    existing_ids: set[str],
    run_id: str,
    promoted_symbol: str,
    demoted_symbol: Optional[str],
    promoted_ev: Optional[float],
    demoted_ev: Optional[float],
    replacement_delta: Optional[float],
) -> None:
    today_jst = _today_jst()
    id_seed = {
        "run_id": run_id,
        "promoted_symbol": promoted_symbol,
        "type": "rotation",
        "date": today_jst,
    }
    record_id = _record_id(id_seed)

    if record_id in existing_ids:
        return

    record: dict = {
        "record_id":             record_id,
        "schema_version":        SCHEMA_VERSION,
        "record_type":           "rotation",

        "rotation_at":           today_jst,
        "rotation_at_utc":       _now_utc(),
        "run_id":                run_id,

        "promoted_symbol":       promoted_symbol,
        "demoted_symbol":        demoted_symbol,           # weakest-EV live symbol at decision time

        "promoted_ev":           round(promoted_ev, 4) if promoted_ev is not None else None,
        "demoted_ev":            round(demoted_ev, 4)  if demoted_ev  is not None else None,
        "replacement_delta":     replacement_delta,

        # Round-trip cost estimate (commission + slippage, 2 sides)
        "rotation_cost_fraction": ROTATION_COST_FRACTION,

        # Forward returns — null at record time
        "promoted_forward_20d":  None,
        "demoted_forward_20d":   None,
        "net_rotation_alpha":    None,     # promoted_forward_20d - demoted_forward_20d

        "materialized":          False,
    }
    try:
        _append_jsonl(out_path, record)
        existing_ids.add(record_id)
    except Exception as exc:
        logger.warning("[OUTCOME_TRACKER] rotation write failed %s: %s", promoted_symbol, exc)


def _write_exploration(
    out_path: Path,
    existing_ids: set[str],
    run_id: str,
    symbol: str,
    sector: str,
    sample_n: Optional[int],
    candidate_ev: Optional[float],
    live_ev_floor: Optional[float],
) -> None:
    today_jst = _today_jst()
    id_seed = {"run_id": run_id, "symbol": symbol, "type": "exploration", "date": today_jst}
    record_id = _record_id(id_seed)

    if record_id in existing_ids:
        return

    # Net advantage before exploration bypass was applied.
    net_adv_pre_bypass: Optional[float] = None
    if candidate_ev is not None and live_ev_floor is not None:
        from src.universe.universe_ev_engine import ROTATION_COST_EV, PROMOTION_EV_THRESHOLD
        net_adv_pre_bypass = round(
            candidate_ev - live_ev_floor - ROTATION_COST_EV - PROMOTION_EV_THRESHOLD, 4
        )

    record: dict = {
        "record_id":                    record_id,
        "schema_version":               SCHEMA_VERSION,
        "record_type":                  "exploration",

        "explored_at":                  today_jst,
        "explored_at_utc":              _now_utc(),
        "run_id":                       run_id,
        "symbol":                       symbol,
        "sector":                       sector,

        "sample_n":                     sample_n,
        "candidate_ev":                 round(candidate_ev, 4) if candidate_ev is not None else None,
        "live_ev_floor":                round(live_ev_floor, 4) if live_ev_floor is not None else None,
        "net_advantage_pre_bypass":     net_adv_pre_bypass,    # negative = would have been rejected

        "selected_via_exploration":     True,

        # Forward returns — null at record time
        "forward_return_20d":           None,
        "survived_20d":                 None,     # True if still in live universe after 20d
        "promoted_to_mainline":         None,     # True if later gets mainline promotion

        "materialized":                 False,
    }
    try:
        _append_jsonl(out_path, record)
        existing_ids.add(record_id)
    except Exception as exc:
        logger.warning("[OUTCOME_TRACKER] exploration write failed %s: %s", symbol, exc)
