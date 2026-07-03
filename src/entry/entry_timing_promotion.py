"""
entry_timing_promotion.py — Entry Timing Promotion Engine

Elevates Entry Timing from observation-only to a self-validating, self-promoting
production signal by managing boost_weight through a 4-tier evidence ladder.

Tiers:
  Tier 0 → observation only          boost_weight = 0.06
  Tier 1 → evidence validated        boost_weight = 0.10
  Tier 2 → production alpha          boost_weight = 0.15
  Tier 3 → full operating alpha      boost_weight = 0.20

Promotion conditions:
  0 → 1  monotonicity ≥ 0.60  AND  top_decile_win_rate > bottom + 5%
          AND sample_count ≥ 100  (30d window)
  1 → 2  Tier1 ≥ 30 days  AND  expectancy_uplift ≥ 10%  (60d window)
  2 → 3  Tier2 ≥ 60 days  AND  expectancy_uplift ≥ 10%  AND  forward_return_uplift
          (90d window)

Safety (immediate demotion by one tier):
  - monotonicity_score < 0.45
  - expectancy_uplift ≤ 0
  - sample_count < 30 (insufficient data for current tier evaluation)

Outputs:
  ENTRY_TIMING_PROMOTION_FILE  — point-in-time JSON snapshot
  ENTRY_TIMING_HISTORY_FILE    — append-only CSV tier history

Integration:
  - signal_bridge.py reads effective_boost_weight when auto_apply_boost: true
  - Weekly Executive Review: format_et_promotion_section()
  - run_live_signal.py: DRY + LIVE hooks after weekly review

Fail policy: FAIL_OPEN throughout. Evaluation failure preserves current tier.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_JST = timezone(timedelta(hours=9))
SCHEMA_VERSION = "v1"

# ── Tier constants ─────────────────────────────────────────────────────────────
TIER_BOOST_WEIGHTS: Dict[int, float] = {
    0: 0.06,
    1: 0.10,
    2: 0.15,
    3: 0.20,
}
MIN_TIER = 0
MAX_TIER = 3

# ── Promotion thresholds ───────────────────────────────────────────────────────
TIER1_MONOTONICITY_MIN    = 0.60
TIER1_WIN_RATE_GAP_MIN    = 0.05     # top_decile - bottom_decile ≥ 5%
TIER1_SAMPLE_MIN          = 100
TIER1_WINDOW_DAYS         = 30

TIER2_TIER1_DAYS_MIN      = 30       # hold Tier1 ≥ 30 days
TIER2_EXPECTANCY_UPLIFT   = 0.10     # 10% relative uplift
TIER2_MONOTONICITY_MIN    = 0.60
TIER2_WINDOW_DAYS         = 60

TIER3_TIER2_DAYS_MIN      = 60       # hold Tier2 ≥ 60 days
TIER3_EXPECTANCY_UPLIFT   = 0.10
TIER3_MONOTONICITY_MIN    = 0.60
TIER3_WINDOW_DAYS         = 90

# ── Safety demotion thresholds ─────────────────────────────────────────────────
SAFETY_MONOTONICITY_FLOOR = 0.45    # < 0.45 in primary window → demote
SAFETY_EXPECTANCY_FLOOR   = 0.0     # uplift ≤ 0 → demote
SAFETY_SAMPLE_FLOOR       = 30      # fewer records → insufficient data

# ── Evaluation windows ─────────────────────────────────────────────────────────
WINDOW_DAYS: List[int] = [30, 60, 90]

# ── Recommendation labels ──────────────────────────────────────────────────────
REC_PROMOTE            = "PROMOTE"
REC_HOLD               = "HOLD"
REC_DEMOTE             = "DEMOTE"
REC_INSUFFICIENT_DATA  = "INSUFFICIENT_DATA"
REC_AT_MAX             = "AT_MAX_TIER"


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WindowKPIs:
    window_days:             int
    sample_count:            int         # records with materialized returns
    monotonicity_score:      Optional[float]   # 0-1; None if insufficient
    top_decile_win_rate:     Optional[float]   # deciles 9+10
    bottom_decile_win_rate:  Optional[float]   # deciles 1+2
    top_decile_expectancy:   Optional[float]   # avg 3d return, deciles 9+10
    bottom_decile_expectancy: Optional[float]  # avg 3d return, deciles 1+2
    all_expectancy:          Optional[float]   # avg 3d return across all records
    expectancy_uplift:       Optional[float]   # relative: (top-bottom)/|all|
    win_rate_gap:            Optional[float]   # top_win_rate - bottom_win_rate
    sufficient:              bool              # sample_count >= SAFETY_SAMPLE_FLOOR


@dataclass
class PromotionState:
    current_tier:          int
    current_boost_weight:  float
    tier_start_date:       str          # ISO date when current tier started
    last_evaluated:        str          # ISO date of last evaluation
    windows:               Dict[str, Any]   # "30d", "60d", "90d" → WindowKPIs dict
    promotion_recommendation: str
    safety_triggered:      bool
    safety_reasons:        List[str]
    events:                List[str]    # recent tier transitions
    schema_version:        str = SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# KPI computation from telemetry
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _avg(vals: List[float]) -> Optional[float]:
    return (sum(vals) / len(vals)) if vals else None


def _compute_monotonicity(decile_win_rates: Dict[int, Optional[float]]) -> Optional[float]:
    """
    Fraction of adjacent decile pairs where win_rate is non-decreasing.
    Requires ≥ 3 deciles with data.
    """
    pairs_checked = 0
    pairs_rising  = 0
    prev: Optional[float] = None
    for d in range(1, 11):
        wr = decile_win_rates.get(d)
        if wr is None:
            prev = None
            continue
        if prev is not None:
            pairs_checked += 1
            if wr >= prev:
                pairs_rising += 1
        prev = wr
    if pairs_checked < 2:
        return None
    return round(pairs_rising / pairs_checked, 4)


def compute_window_kpis(
    records: List[dict],
    window_days: int,
    today: date,
) -> WindowKPIs:
    """
    Compute KPIs from telemetry records filtered to the last `window_days` days.
    Only records with materialized_3d=True contribute.
    FAIL_OPEN: returns WindowKPIs(sample_count=0) on exception.
    """
    try:
        cutoff = today - timedelta(days=window_days)
        filtered = []
        for r in records:
            try:
                rec_date = datetime.strptime(r["date"], "%Y-%m-%d").date()
                if rec_date >= cutoff and r.get("materialized_3d") is True:
                    filtered.append(r)
            except Exception:
                continue

        sample_count = len(filtered)
        if sample_count == 0:
            return WindowKPIs(
                window_days=window_days, sample_count=0,
                monotonicity_score=None, top_decile_win_rate=None,
                bottom_decile_win_rate=None, top_decile_expectancy=None,
                bottom_decile_expectancy=None, all_expectancy=None,
                expectancy_uplift=None, win_rate_gap=None,
                sufficient=False,
            )

        # Group by score_decile
        by_decile: Dict[int, List[dict]] = {d: [] for d in range(1, 11)}
        for r in filtered:
            d = int(r.get("score_decile", 5))
            if 1 <= d <= 10:
                by_decile[d].append(r)

        # Per-decile win_rate and expectancy
        decile_win_rates: Dict[int, Optional[float]] = {}
        decile_expectations: Dict[int, Optional[float]] = {}
        for d in range(1, 11):
            recs  = by_decile[d]
            rets  = [_safe_float(r.get("subsequent_3d_return")) for r in recs]
            rets  = [v for v in rets if v is not None]
            if not rets:
                decile_win_rates[d]    = None
                decile_expectations[d] = None
                continue
            decile_win_rates[d]    = round(sum(1 for v in rets if v > 0) / len(rets), 4)
            decile_expectations[d] = round(_avg(rets), 6)  # type: ignore[arg-type]

        # Top/bottom deciles (9+10 vs 1+2)
        top_rets    = [v for d in (9, 10) for r in by_decile[d] if (v := _safe_float(r.get("subsequent_3d_return"))) is not None]
        bottom_rets = [v for d in (1, 2)  for r in by_decile[d] if (v := _safe_float(r.get("subsequent_3d_return"))) is not None]
        all_rets    = [v for r in filtered if (v := _safe_float(r.get("subsequent_3d_return"))) is not None]

        top_wr    = round(sum(1 for v in top_rets    if v > 0) / len(top_rets),    4) if top_rets    else None
        bottom_wr = round(sum(1 for v in bottom_rets if v > 0) / len(bottom_rets), 4) if bottom_rets else None
        top_exp   = round(_avg(top_rets),    6) if top_rets    else None  # type: ignore[arg-type]
        bot_exp   = round(_avg(bottom_rets), 6) if bottom_rets else None  # type: ignore[arg-type]
        all_exp   = round(_avg(all_rets),    6) if all_rets    else None  # type: ignore[arg-type]

        # Expectancy uplift: (top - bottom) / max(|all_avg|, 0.001)
        uplift: Optional[float] = None
        if top_exp is not None and bot_exp is not None:
            denom = max(abs(all_exp) if all_exp is not None else 0.0, 0.001)
            uplift = round((top_exp - bot_exp) / denom, 4)

        win_rate_gap: Optional[float] = None
        if top_wr is not None and bottom_wr is not None:
            win_rate_gap = round(top_wr - bottom_wr, 4)

        mono = _compute_monotonicity(decile_win_rates)

        return WindowKPIs(
            window_days              = window_days,
            sample_count             = sample_count,
            monotonicity_score       = mono,
            top_decile_win_rate      = top_wr,
            bottom_decile_win_rate   = bottom_wr,
            top_decile_expectancy    = top_exp,
            bottom_decile_expectancy = bot_exp,
            all_expectancy           = all_exp,
            expectancy_uplift        = uplift,
            win_rate_gap             = win_rate_gap,
            sufficient               = sample_count >= SAFETY_SAMPLE_FLOOR,
        )

    except Exception as exc:
        logger.warning("[ETP] compute_window_kpis(%dd) failed: %s", window_days, exc)
        return WindowKPIs(
            window_days=window_days, sample_count=0,
            monotonicity_score=None, top_decile_win_rate=None,
            bottom_decile_win_rate=None, top_decile_expectancy=None,
            bottom_decile_expectancy=None, all_expectancy=None,
            expectancy_uplift=None, win_rate_gap=None,
            sufficient=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Safety and promotion logic
# ─────────────────────────────────────────────────────────────────────────────

def _days_since(date_str: str, today: date) -> int:
    try:
        start = datetime.strptime(date_str, "%Y-%m-%d").date()
        return max(0, (today - start).days)
    except Exception:
        return 0


def check_safety_triggers(
    w30: WindowKPIs,
) -> Tuple[bool, List[str]]:
    """
    Return (triggered, reasons). Uses 30d window as the primary safety check.
    Triggered if ANY of:
      - monotonicity_score < SAFETY_MONOTONICITY_FLOOR  (when data available)
      - expectancy_uplift <= SAFETY_EXPECTANCY_FLOOR     (when data available)
      - sample_count < SAFETY_SAMPLE_FLOOR               (insufficient data)
    """
    reasons: List[str] = []

    if not w30.sufficient:
        reasons.append(f"sample_count={w30.sample_count} < {SAFETY_SAMPLE_FLOOR}")

    if w30.monotonicity_score is not None and w30.monotonicity_score < SAFETY_MONOTONICITY_FLOOR:
        reasons.append(
            f"monotonicity={w30.monotonicity_score:.3f} < {SAFETY_MONOTONICITY_FLOOR}"
        )

    if w30.expectancy_uplift is not None and w30.expectancy_uplift <= SAFETY_EXPECTANCY_FLOOR:
        reasons.append(
            f"expectancy_uplift={w30.expectancy_uplift:.3f} <= {SAFETY_EXPECTANCY_FLOOR}"
        )

    return bool(reasons), reasons


def check_tier1_promotion(w30: WindowKPIs) -> Tuple[bool, str]:
    """Tier 0 → 1: 30d window checks."""
    if not w30.sufficient or w30.sample_count < TIER1_SAMPLE_MIN:
        return False, f"sample_count={w30.sample_count} < {TIER1_SAMPLE_MIN}"
    if w30.monotonicity_score is None or w30.monotonicity_score < TIER1_MONOTONICITY_MIN:
        return False, f"monotonicity={w30.monotonicity_score} < {TIER1_MONOTONICITY_MIN}"
    if w30.win_rate_gap is None or w30.win_rate_gap < TIER1_WIN_RATE_GAP_MIN:
        return False, f"win_rate_gap={w30.win_rate_gap} < {TIER1_WIN_RATE_GAP_MIN}"
    return True, "all Tier1 conditions met"


def check_tier2_promotion(
    w60: WindowKPIs,
    days_in_tier: int,
) -> Tuple[bool, str]:
    """Tier 1 → 2: 60d window + duration check."""
    if days_in_tier < TIER2_TIER1_DAYS_MIN:
        return False, f"tier1_days={days_in_tier} < {TIER2_TIER1_DAYS_MIN}"
    if not w60.sufficient:
        return False, f"60d sample_count={w60.sample_count} insufficient"
    if w60.monotonicity_score is None or w60.monotonicity_score < TIER2_MONOTONICITY_MIN:
        return False, f"60d monotonicity={w60.monotonicity_score} < {TIER2_MONOTONICITY_MIN}"
    if w60.expectancy_uplift is None or w60.expectancy_uplift < TIER2_EXPECTANCY_UPLIFT:
        return False, f"60d expectancy_uplift={w60.expectancy_uplift} < {TIER2_EXPECTANCY_UPLIFT}"
    return True, "all Tier2 conditions met"


def check_tier3_promotion(
    w90: WindowKPIs,
    days_in_tier: int,
) -> Tuple[bool, str]:
    """Tier 2 → 3: 90d window + duration check."""
    if days_in_tier < TIER3_TIER2_DAYS_MIN:
        return False, f"tier2_days={days_in_tier} < {TIER3_TIER2_DAYS_MIN}"
    if not w90.sufficient:
        return False, f"90d sample_count={w90.sample_count} insufficient"
    if w90.monotonicity_score is None or w90.monotonicity_score < TIER3_MONOTONICITY_MIN:
        return False, f"90d monotonicity={w90.monotonicity_score} < {TIER3_MONOTONICITY_MIN}"
    if w90.expectancy_uplift is None or w90.expectancy_uplift < TIER3_EXPECTANCY_UPLIFT:
        return False, f"90d expectancy_uplift={w90.expectancy_uplift} < {TIER3_EXPECTANCY_UPLIFT}"
    # forward_return_uplift: top_decile_expectancy > 0 in 90d window
    if w90.top_decile_expectancy is None or w90.top_decile_expectancy <= 0:
        return False, f"90d top_decile_expectancy={w90.top_decile_expectancy} <= 0"
    return True, "all Tier3 conditions met"


def evaluate_tier_transition(
    current_tier:    int,
    tier_start_date: str,
    w30: WindowKPIs,
    w60: WindowKPIs,
    w90: WindowKPIs,
    today: date,
) -> Tuple[int, str, List[str]]:
    """
    Core promotion/demotion logic.
    Returns (new_tier, recommendation, events).
    Never raises.
    """
    try:
        days_in_tier = _days_since(tier_start_date, today)
        events: List[str] = []

        # 1. Safety check (uses 30d window)
        triggered, reasons = check_safety_triggers(w30)
        if triggered and current_tier > MIN_TIER:
            new_tier = current_tier - 1
            events.append(
                f"DEMOTED {current_tier}→{new_tier} on {today}: {'; '.join(reasons)}"
            )
            return new_tier, REC_DEMOTE, events

        # 2. Promotion check
        if current_tier == 0:
            ok, msg = check_tier1_promotion(w30)
            if ok:
                events.append(f"PROMOTED 0→1 on {today}: {msg}")
                return 1, REC_PROMOTE, events
            return 0, REC_INSUFFICIENT_DATA if not w30.sufficient else REC_HOLD, events

        elif current_tier == 1:
            ok, msg = check_tier2_promotion(w60, days_in_tier)
            if ok:
                events.append(f"PROMOTED 1→2 on {today}: {msg}")
                return 2, REC_PROMOTE, events
            return 1, REC_HOLD, events

        elif current_tier == 2:
            ok, msg = check_tier3_promotion(w90, days_in_tier)
            if ok:
                events.append(f"PROMOTED 2→3 on {today}: {msg}")
                return 3, REC_PROMOTE, events
            return 2, REC_HOLD, events

        elif current_tier == 3:
            return 3, REC_AT_MAX, events

        return current_tier, REC_HOLD, events

    except Exception as exc:
        logger.warning("[ETP] tier transition evaluation failed: %s", exc)
        return current_tier, REC_HOLD, []


# ─────────────────────────────────────────────────────────────────────────────
# State persistence
# ─────────────────────────────────────────────────────────────────────────────

def _load_promotion_state(promotion_file: Path) -> Optional[PromotionState]:
    """Load existing promotion state. Returns None if file missing or corrupt."""
    try:
        if not promotion_file.exists():
            return None
        data = json.loads(promotion_file.read_text(encoding="utf-8"))
        return PromotionState(
            current_tier           = int(data.get("current_tier", 0)),
            current_boost_weight   = float(data.get("current_boost_weight", 0.06)),
            tier_start_date        = str(data.get("tier_start_date", "")),
            last_evaluated         = str(data.get("last_evaluated", "")),
            windows                = data.get("windows", {}),
            promotion_recommendation = str(data.get("promotion_recommendation", REC_INSUFFICIENT_DATA)),
            safety_triggered       = bool(data.get("safety_triggered", False)),
            safety_reasons         = list(data.get("safety_reasons", [])),
            events                 = list(data.get("events", [])),
            schema_version         = str(data.get("schema_version", SCHEMA_VERSION)),
        )
    except Exception as exc:
        logger.warning("[ETP] load_promotion_state failed: %s", exc)
        return None


def _default_state(today: str) -> PromotionState:
    return PromotionState(
        current_tier           = 0,
        current_boost_weight   = TIER_BOOST_WEIGHTS[0],
        tier_start_date        = today,
        last_evaluated         = today,
        windows                = {},
        promotion_recommendation = REC_INSUFFICIENT_DATA,
        safety_triggered       = False,
        safety_reasons         = [],
        events                 = [],
    )


def _save_promotion_state(state: PromotionState, promotion_file: Path) -> None:
    """Atomic JSON write. FAIL_OPEN."""
    try:
        promotion_file.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(state)
        content = json.dumps(data, ensure_ascii=False, indent=2, default=str)
        fd, tmp = tempfile.mkstemp(dir=str(promotion_file.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tf:
                tf.write(content)
                tf.flush()
                os.fsync(tf.fileno())
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise
        Path(tmp).replace(promotion_file)
    except Exception as exc:
        logger.warning("[ETP] save_promotion_state failed: %s", exc)


def _append_history_csv(
    state:        PromotionState,
    history_file: Path,
    w30:          WindowKPIs,
    event:        str,
) -> None:
    """Append one row to the CSV history. FAIL_OPEN."""
    try:
        history_file.parent.mkdir(parents=True, exist_ok=True)
        write_header = not history_file.exists()
        with history_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "date", "tier", "boost_weight",
                    "monotonicity_30d", "win_rate_gap_30d", "expectancy_uplift_30d",
                    "sample_count_30d", "recommendation", "safety_triggered", "event",
                ])
            writer.writerow([
                state.last_evaluated,
                state.current_tier,
                state.current_boost_weight,
                w30.monotonicity_score,
                w30.win_rate_gap,
                w30.expectancy_uplift,
                w30.sample_count,
                state.promotion_recommendation,
                state.safety_triggered,
                event,
            ])
    except Exception as exc:
        logger.warning("[ETP] append_history_csv failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Public main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_entry_timing_promotion(
    telemetry_file:  Path,
    promotion_file:  Path,
    history_file:    Path,
    today_str:       Optional[str] = None,
) -> PromotionState:
    """
    Main evaluation entry point.

    1. Load telemetry JSONL
    2. Compute 30d / 60d / 90d window KPIs
    3. Load existing promotion state (or initialize at Tier 0)
    4. Evaluate safety triggers and tier transitions
    5. Write promotion JSON + append history CSV
    6. Return updated PromotionState

    FAIL_OPEN: returns existing (or default) state on any critical failure.
    """
    today = _parse_today(today_str)
    today_iso = today.strftime("%Y-%m-%d")

    existing = _load_promotion_state(promotion_file) or _default_state(today_iso)

    # Load telemetry
    records: List[dict] = []
    try:
        if telemetry_file.exists():
            for line in telemetry_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
    except Exception as exc:
        logger.warning("[ETP] telemetry load failed: %s", exc)

    # Compute window KPIs
    try:
        w30 = compute_window_kpis(records, 30, today)
        w60 = compute_window_kpis(records, 60, today)
        w90 = compute_window_kpis(records, 90, today)
    except Exception as exc:
        logger.warning("[ETP] window KPI computation failed: %s", exc)
        return existing

    # Evaluate tier transition
    safety_triggered, safety_reasons = check_safety_triggers(w30)
    new_tier, recommendation, events = evaluate_tier_transition(
        current_tier    = existing.current_tier,
        tier_start_date = existing.tier_start_date,
        w30             = w30,
        w60             = w60,
        w90             = w90,
        today           = today,
    )

    # Update tier_start_date if tier changed
    tier_start_date = existing.tier_start_date
    if new_tier != existing.current_tier:
        tier_start_date = today_iso

    # Keep last 20 events
    all_events = (existing.events or []) + events
    all_events = all_events[-20:]

    # Build updated state
    state = PromotionState(
        current_tier             = new_tier,
        current_boost_weight     = TIER_BOOST_WEIGHTS.get(new_tier, 0.06),
        tier_start_date          = tier_start_date,
        last_evaluated           = today_iso,
        windows                  = {
            "30d": asdict(w30),
            "60d": asdict(w60),
            "90d": asdict(w90),
        },
        promotion_recommendation = recommendation,
        safety_triggered         = safety_triggered,
        safety_reasons           = safety_reasons,
        events                   = all_events,
    )

    _save_promotion_state(state, promotion_file)

    event_label = events[0] if events else "EVALUATED"
    _append_history_csv(state, history_file, w30, event_label)

    logger.info(
        "[ETP] tier=%d→%d boost=%.2f rec=%s mono30=%.3f n30=%d",
        existing.current_tier, new_tier,
        state.current_boost_weight,
        recommendation,
        w30.monotonicity_score if w30.monotonicity_score is not None else -1.0,
        w30.sample_count,
    )
    return state


def _parse_today(today_str: Optional[str]) -> date:
    if today_str:
        try:
            return datetime.strptime(today_str, "%Y-%m-%d").date()
        except Exception:
            pass
    return datetime.now(_JST).date()


# ─────────────────────────────────────────────────────────────────────────────
# Effective boost_weight lookup (for signal_bridge.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_effective_boost_weight(
    promotion_file:      Path,
    fallback_weight:     float = 0.06,
    auto_apply_enabled:  bool  = False,
) -> float:
    """
    Return effective boost_weight for signal_bridge.py composite scoring.

    If auto_apply_enabled=False (default):  always return fallback_weight
    If auto_apply_enabled=True AND promotion file exists: return current_boost_weight
    FAIL_OPEN: returns fallback_weight on any error.
    """
    if not auto_apply_enabled:
        return fallback_weight
    try:
        state = _load_promotion_state(promotion_file)
        if state is not None:
            return state.current_boost_weight
    except Exception as exc:
        logger.warning("[ETP] get_effective_boost_weight failed: %s", exc)
    return fallback_weight


# ─────────────────────────────────────────────────────────────────────────────
# Report section for Weekly Executive Review
# ─────────────────────────────────────────────────────────────────────────────

def format_et_promotion_section(
    promotion_file: Path,
    today_str:      Optional[str] = None,
) -> str:
    """
    Format the [ENTRY TIMING STATUS] section for Weekly Executive Review.
    FAIL_OPEN: returns empty string on error.
    """
    try:
        state = _load_promotion_state(promotion_file)
        if state is None:
            return ""

        today = _parse_today(today_str)
        days_in_tier = _days_since(state.tier_start_date, today)

        tier_label = {
            0: "Tier 0 (観測専用)",
            1: "Tier 1 (証拠検証済み)",
            2: "Tier 2 (本番アルファ)",
            3: "Tier 3 (フル運用アルファ)",
        }.get(state.current_tier, f"Tier {state.current_tier}")

        rec_emoji = {
            REC_PROMOTE:           "⬆ PROMOTE",
            REC_HOLD:              "▶ HOLD",
            REC_DEMOTE:            "⬇ DEMOTE",
            REC_INSUFFICIENT_DATA: "⏳ INSUFFICIENT_DATA",
            REC_AT_MAX:            "✅ AT_MAX_TIER",
        }.get(state.promotion_recommendation, state.promotion_recommendation)

        lines = [
            "",
            "━━ [ENTRY TIMING STATUS — 自己昇格ダッシュボード] ━━━━━━━━━━",
            f"  Tier               : {tier_label}",
            f"  Boost Weight       : {state.current_boost_weight:.2f}  (Tier {state.current_tier})",
            f"  Tier継続日数        : {days_in_tier}日",
            f"  推奨アクション      : {rec_emoji}",
        ]

        if state.safety_triggered:
            lines.append(f"  ⚠ 安全装置         : 発動中 — {'; '.join(state.safety_reasons)}")

        # 30d window KPIs
        w30 = state.windows.get("30d", {})
        if w30:
            mono  = w30.get("monotonicity_score")
            gap   = w30.get("win_rate_gap")
            uplift = w30.get("expectancy_uplift")
            n30   = w30.get("sample_count", 0)
            lines.append(f"  ── 30日ウィンドウ KPI ────────────────────────────────")
            lines.append(f"  サンプル数          : {n30}")
            if mono is not None:
                mono_tag = "✅" if mono >= 0.60 else ("⚠" if mono >= 0.45 else "❌")
                lines.append(f"  単調性              : {mono:.3f}  {mono_tag}  (閾値: ≥0.60)")
            if gap is not None:
                gap_tag = "✅" if gap >= 0.05 else "⚠"
                lines.append(f"  Win率差(Top-Bottom) : {gap:+.1%}  {gap_tag}  (閾値: ≥5%)")
            if uplift is not None:
                up_tag = "✅" if uplift >= 0.10 else ("⚠" if uplift > 0 else "❌")
                lines.append(f"  期待値Uplift        : {uplift:+.2f}x  {up_tag}  (閾値: ≥10%)")

        # 60d / 90d summary if available
        for wkey, wlabel in [("60d", "60日"), ("90d", "90日")]:
            wdata = state.windows.get(wkey, {})
            if wdata and wdata.get("sample_count", 0) >= 10:
                m  = wdata.get("monotonicity_score")
                up = wdata.get("expectancy_uplift")
                n  = wdata.get("sample_count", 0)
                parts = [f"n={n}"]
                if m  is not None: parts.append(f"mono={m:.3f}")
                if up is not None: parts.append(f"uplift={up:+.2f}x")
                lines.append(f"  {wlabel}ウィンドウ       : {', '.join(parts)}")

        # Tier progression status
        lines.append(f"  ── 昇格ステータス ──────────────────────────────────────")
        next_tier = state.current_tier + 1
        if next_tier <= MAX_TIER:
            _add_next_tier_progress(lines, state, w30, today, days_in_tier)
        else:
            lines.append("  最高ティア到達済み。維持条件を継続監視中。")

        # Recent events
        if state.events:
            lines.append(f"  ── 直近イベント ────────────────────────────────────")
            for ev in state.events[-3:]:
                lines.append(f"  {ev}")

        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)

    except Exception as exc:
        logger.warning("[ETP] format_et_promotion_section failed: %s", exc)
        return ""


def _add_next_tier_progress(
    lines: List[str],
    state: PromotionState,
    w30_raw: dict,
    today: date,
    days_in_tier: int,
) -> None:
    """Append next-tier gap analysis to lines list. FAIL_OPEN."""
    try:
        t = state.current_tier
        if t == 0:
            n30 = w30_raw.get("sample_count", 0)
            mono = w30_raw.get("monotonicity_score")
            gap  = w30_raw.get("win_rate_gap")
            remaining = max(0, TIER1_SAMPLE_MIN - n30)
            lines.append(f"  Tier1昇格残り       :")
            if remaining > 0:
                lines.append(f"    サンプル不足: あと{remaining}件")
            if mono is not None:
                lines.append(f"    単調性: {mono:.3f} / 必要 ≥ {TIER1_MONOTONICITY_MIN}")
            if gap is not None:
                lines.append(f"    Win率差: {gap:+.1%} / 必要 ≥ {TIER1_WIN_RATE_GAP_MIN:.0%}")
        elif t == 1:
            days_rem = max(0, TIER2_TIER1_DAYS_MIN - days_in_tier)
            lines.append(f"  Tier2昇格残り       : {days_rem}日 + 60d expectancy_uplift≥10%")
        elif t == 2:
            days_rem = max(0, TIER3_TIER2_DAYS_MIN - days_in_tier)
            lines.append(f"  Tier3昇格残り       : {days_rem}日 + 90d expectancy_uplift≥10%")
    except Exception:
        pass
