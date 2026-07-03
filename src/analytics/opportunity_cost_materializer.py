"""
opportunity_cost_materializer.py — Phase 5D OC Materialization & Regime Attribution

Enriches OC events after the forward-return window completes.
Attaches regime context at event time.
Runs multi-dimensional analytics on the enriched records.

Observation-only. Append-only. Fail-open. Deterministic. Replay-safe.

Input  (read-only): OPPORTUNITY_COST_EVENTS_FILE / oc_events.jsonl
Output (append):    OC_5D_ENRICHED_FILE (companion; originals never mutated)

Rollout phases (all analytics available from day 1):
  Phase 1 — materialize_oc_events_5d + aggregate_5d_kpis (Tasks 1+2+6)
  Phase 2 — staleness + slot attribution  (Tasks 4+5)
  Phase 3 — persistence analysis          (Task 3) — available immediately
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.analytics.opportunity_cost import (
    SEVERITY_LOW,
    SEVERITY_MODERATE,
    SEVERITY_SEVERE,
    MAT_BUFFER_DAYS,
    _add_business_days,
    _append_jsonl,
    _compute_return,
    _due_for_materialization,
    _load_jsonl,
    _safe,
)

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))
SCHEMA_VERSION_5D = "5d_v1"
FORWARD_BUSINESS_DAYS = 5

# ── Regime labels ──────────────────────────────────────────────────────────────
REGIME_BULL    = "bull"
REGIME_BEAR    = "bear"
REGIME_NORMAL  = "normal"
REGIME_CRASH   = "crash"
REGIME_UNKNOWN = "unknown"

# ── Signal environment labels ──────────────────────────────────────────────────
SIGNAL_TREND_PERSISTENT = "trend_persistent"
SIGNAL_FRESH_BREAKOUT   = "fresh_breakout"
SIGNAL_COMPRESSION      = "compression"
SIGNAL_RANGE_BOUND      = "range_bound"
SIGNAL_UNKNOWN          = "unknown"

# ── Age bucket labels ──────────────────────────────────────────────────────────
AGE_BUCKET_LT_10  = "lt_10d"
AGE_BUCKET_10_20  = "10_20d"
AGE_BUCKET_GT_20  = "gt_20d"

# Callable type alias: (date_str) → Optional[regime dict]
RegimeProviderFn = Callable[[str], Optional[dict]]


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegimeSnapshot:
    """Regime context snapshot at OC event date."""
    regime: str               # bull|bear|normal|crash|unknown
    regime_confidence: float  # 0-1
    breadth_50: float         # fraction of universe with RSR ≥ 50
    breadth_75: float         # fraction of universe with RSR ≥ 75
    market_drawdown: float    # negative = drawdown from peak (e.g. -0.041)
    signal_environment: str   # trend_persistent|fresh_breakout|compression|range_bound|unknown


@dataclass
class OC5DEnrichment:
    """
    Companion materialization record for one OC event.

    event_id = "{date}/{rejected_symbol}" — dedup key.
    Never modifies original event rows in the source file.
    """
    event_id: str
    date: str
    rejected_symbol: str
    blocked_by_symbol: str
    # Forward returns (None = data unavailable)
    rejected_candidate_5d_return: Optional[float]
    blocked_position_5d_return: Optional[float]
    rotation_counterfactual_delta: Optional[float]   # rejected - blocked
    materialized_at: str
    materialization_complete: bool
    # Regime attribution
    regime: str
    regime_confidence: float
    breadth_50: float
    breadth_75: float
    market_drawdown: float
    signal_environment: str
    # Copied from original event for analytics (avoids join at query time)
    priority_gap: float
    opportunity_cost_severity: str
    blocked_position_age_days: int
    max_positions: int
    active_positions: int
    schema_version: str = SCHEMA_VERSION_5D


_EMPTY_REGIME = RegimeSnapshot(
    regime=REGIME_UNKNOWN,
    regime_confidence=0.0,
    breadth_50=0.0,
    breadth_75=0.0,
    market_drawdown=0.0,
    signal_environment=SIGNAL_UNKNOWN,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pure computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_event_id(date: str, rejected_symbol: str) -> str:
    return f"{date}/{rejected_symbol}"


def _avg_list(lst: List[float]) -> Optional[float]:
    return round(sum(lst) / len(lst), 6) if lst else None


def classify_signal_environment(
    regime: str,
    breadth_50: float,
    breadth_75: float,
) -> str:
    """
    Deterministic O(1) signal_environment from regime + breadth metrics.

    Used to identify where OC severity matters most:
      trend_persistent → strong continuation, high breadth
      fresh_breakout   → bull regime, moderate breadth
      compression      → breadth collapse across both thresholds
      range_bound      → normal/bear with moderate breadth
      unknown          → insufficient data
    """
    if regime == REGIME_BULL and breadth_50 >= 0.50:
        return SIGNAL_TREND_PERSISTENT
    if regime == REGIME_BULL and breadth_50 >= 0.30:
        return SIGNAL_FRESH_BREAKOUT
    if breadth_50 < 0.30 and breadth_75 < 0.15:
        return SIGNAL_COMPRESSION
    if regime in (REGIME_NORMAL, REGIME_BEAR) and 0.30 <= breadth_50 < 0.50:
        return SIGNAL_RANGE_BOUND
    return SIGNAL_UNKNOWN


def _build_regime_snapshot(
    date: str,
    regime_provider: Optional[RegimeProviderFn],
) -> RegimeSnapshot:
    """
    Retrieve regime context for a date via the injected provider.
    Fail-open: returns empty snapshot on any error or missing provider.
    """
    if regime_provider is None:
        return _EMPTY_REGIME
    try:
        raw = regime_provider(date)
        if not raw:
            return _EMPTY_REGIME
        regime      = str(raw.get("regime", REGIME_UNKNOWN))
        confidence  = _safe(raw.get("regime_confidence", 0.0))
        b50         = _safe(raw.get("breadth_50", 0.0))
        b75         = _safe(raw.get("breadth_75", 0.0))
        drawdown    = _safe(raw.get("market_drawdown", 0.0))
        signal_env  = str(raw.get("signal_environment", ""))
        if not signal_env:
            signal_env = classify_signal_environment(regime, b50, b75)
        return RegimeSnapshot(
            regime=regime,
            regime_confidence=confidence,
            breadth_50=b50,
            breadth_75=b75,
            market_drawdown=drawdown,
            signal_environment=signal_env,
        )
    except Exception as exc:
        logger.debug("_build_regime_snapshot failed for %s: %s", date, exc)
        return _EMPTY_REGIME


def _age_bucket(age_days: int) -> str:
    """Classify blocked_position_age_days into lt_10d / 10_20d / gt_20d."""
    if age_days < 10:
        return AGE_BUCKET_LT_10
    if age_days < 20:
        return AGE_BUCKET_10_20
    return AGE_BUCKET_GT_20


# ─────────────────────────────────────────────────────────────────────────────
# Task 1+2 — Materialization + Regime Attribution
# ─────────────────────────────────────────────────────────────────────────────

def materialize_oc_events_5d(
    events_file: Path,
    enriched_file: Path,
    price_fetcher: Callable[[str, str], Optional[float]],
    today: str,
    regime_provider: Optional[RegimeProviderFn] = None,
    forward_business_days: int = FORWARD_BUSINESS_DAYS,
    buffer_days: int = MAT_BUFFER_DAYS,
) -> int:
    """
    Tasks 1+2: Materialize eligible OC events and attach regime attribution.

    For each un-enriched OC event that is ≥ buffer_days calendar days old:
      - Fetches 5-business-day forward returns for rejected + blocked symbols
      - Attaches regime snapshot from regime_provider (or empty if None)
      - Writes OC5DEnrichment companion record to enriched_file

    Idempotent: skips already-enriched events (by event_id).
    Never mutates original events.
    Fail-open: per-event errors are logged + skipped.

    Args:
        events_file:          source OC events JSONL (read-only)
        enriched_file:        companion output JSONL (append-only)
        price_fetcher:        (symbol, "YYYY-MM-DD") → Optional[float]
        today:                "YYYY-MM-DD"
        regime_provider:      ("YYYY-MM-DD") → Optional[regime dict]
        forward_business_days: look-ahead window (default 5)
        buffer_days:          calendar-day safety buffer (default 8)

    Returns:
        number of events newly enriched
    """
    try:
        events = _load_jsonl(events_file)
    except Exception as exc:
        logger.warning("materialize_oc_events_5d: failed loading events: %s", exc)
        return 0

    # Build dedup set from already-enriched records
    done_ids: set = set()
    for r in _load_jsonl(enriched_file):
        eid = r.get("event_id", "")
        if eid:
            done_ids.add(eid)
        else:
            done_ids.add(_make_event_id(r.get("date", ""), r.get("rejected_symbol", "")))

    count = 0
    for ev in events:
        try:
            event_date  = ev.get("date", "")
            rej_sym     = ev.get("rejected_symbol", "")
            blocked_sym = ev.get("blocked_by_symbol", "")

            if not event_date or not rej_sym:
                continue

            event_id = _make_event_id(event_date, rej_sym)
            if event_id in done_ids:
                continue

            if not _due_for_materialization(event_date, today, buffer_days):
                continue

            target_date = _add_business_days(event_date, forward_business_days)

            rej_ret = _compute_return(rej_sym,    event_date, target_date, price_fetcher)
            blk_ret = _compute_return(blocked_sym, event_date, target_date, price_fetcher)

            delta: Optional[float] = None
            if rej_ret is not None and blk_ret is not None:
                delta = round(rej_ret - blk_ret, 6)

            snap = _build_regime_snapshot(event_date, regime_provider)

            ts = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
            enrichment = OC5DEnrichment(
                event_id=event_id,
                date=event_date,
                rejected_symbol=rej_sym,
                blocked_by_symbol=blocked_sym,
                rejected_candidate_5d_return=rej_ret,
                blocked_position_5d_return=blk_ret,
                rotation_counterfactual_delta=delta,
                materialized_at=ts,
                materialization_complete=(rej_ret is not None and blk_ret is not None),
                regime=snap.regime,
                regime_confidence=snap.regime_confidence,
                breadth_50=snap.breadth_50,
                breadth_75=snap.breadth_75,
                market_drawdown=snap.market_drawdown,
                signal_environment=snap.signal_environment,
                priority_gap=_safe(ev.get("priority_gap", 0.0)),
                opportunity_cost_severity=str(ev.get("opportunity_cost_severity", "")),
                blocked_position_age_days=int(ev.get("blocked_position_age_days", 0)),
                max_positions=int(ev.get("max_positions", 3)),
                active_positions=int(ev.get("active_positions", 0)),
            )
            row = asdict(enrichment)
            _append_jsonl(row, enriched_file)
            done_ids.add(event_id)
            count += 1

        except Exception as exc:
            logger.debug(
                "materialize_oc_events_5d: skipped event date=%s sym=%s: %s",
                ev.get("date"), ev.get("rejected_symbol"), exc,
            )

    return count


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Severity Persistence Analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_severity_persistence(enriched_file: Path) -> dict:
    """
    Task 3: avg rotation_counterfactual_delta per severity bucket.

    Validates: does priority_gap actually predict future alpha differential?

    Returns dict with avg_delta and sample_count for low/moderate/severe.
    Fail-open: returns all-None dict on error.
    """
    try:
        records = _load_jsonl(enriched_file)
        completed = [
            r for r in records
            if r.get("rotation_counterfactual_delta") is not None
        ]

        def _bucket(sev: str) -> List[float]:
            return [
                _safe(r["rotation_counterfactual_delta"])
                for r in completed
                if r.get("opportunity_cost_severity") == sev
            ]

        severe   = _bucket(SEVERITY_SEVERE)
        moderate = _bucket(SEVERITY_MODERATE)
        low      = _bucket(SEVERITY_LOW)

        return {
            "avg_delta_severe":       _avg_list(severe),
            "avg_delta_moderate":     _avg_list(moderate),
            "avg_delta_low":          _avg_list(low),
            "sample_count_severe":    len(severe),
            "sample_count_moderate":  len(moderate),
            "sample_count_low":       len(low),
        }
    except Exception as exc:
        logger.debug("compute_severity_persistence failed: %s", exc)
        return {
            "avg_delta_severe": None, "avg_delta_moderate": None, "avg_delta_low": None,
            "sample_count_severe": 0, "sample_count_moderate": 0, "sample_count_low": 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — Holding Staleness Attribution
# ─────────────────────────────────────────────────────────────────────────────

def compute_holding_staleness(enriched_file: Path) -> dict:
    """
    Task 4: relationship between blocked_position_age_days and rotation delta.

    Hypothesis: older weak holdings → greater alpha drag.

    Age buckets:
      lt_10d  : blocked position held < 10 days
      10_20d  : 10 ≤ age < 20 days
      gt_20d  : age ≥ 20 days

    Returns avg_delta + count per bucket. Fail-open.
    """
    try:
        records = _load_jsonl(enriched_file)
        completed = [
            r for r in records
            if r.get("rotation_counterfactual_delta") is not None
        ]

        buckets: Dict[str, List[float]] = {
            AGE_BUCKET_LT_10: [],
            AGE_BUCKET_10_20: [],
            AGE_BUCKET_GT_20: [],
        }
        for r in completed:
            age = int(r.get("blocked_position_age_days", 0))
            buckets[_age_bucket(age)].append(_safe(r["rotation_counterfactual_delta"]))

        return {
            "avg_delta_age_lt_10d":  _avg_list(buckets[AGE_BUCKET_LT_10]),
            "avg_delta_age_10_20d":  _avg_list(buckets[AGE_BUCKET_10_20]),
            "avg_delta_age_gt_20d":  _avg_list(buckets[AGE_BUCKET_GT_20]),
            "count_age_lt_10d":      len(buckets[AGE_BUCKET_LT_10]),
            "count_age_10_20d":      len(buckets[AGE_BUCKET_10_20]),
            "count_age_gt_20d":      len(buckets[AGE_BUCKET_GT_20]),
        }
    except Exception as exc:
        logger.debug("compute_holding_staleness failed: %s", exc)
        return {
            "avg_delta_age_lt_10d": None, "avg_delta_age_10_20d": None, "avg_delta_age_gt_20d": None,
            "count_age_lt_10d": 0, "count_age_10_20d": 0, "count_age_gt_20d": 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Task 5 — Slot Efficiency Attribution
# ─────────────────────────────────────────────────────────────────────────────

def compute_slot_efficiency(enriched_file: Path) -> dict:
    """
    Task 5: rotation delta conditioned on max_positions slot count.

    Distinguishes: is lost alpha due to poor rotation or insufficient slot count?
    Critical before any future rotation governance research.

    Returns avg_delta + count for slots_3, slots_4, slots_5. Fail-open.
    """
    try:
        records = _load_jsonl(enriched_file)
        completed = [
            r for r in records
            if r.get("rotation_counterfactual_delta") is not None
        ]

        by_slot: Dict[int, List[float]] = {}
        for r in completed:
            mp = int(r.get("max_positions", 3))
            by_slot.setdefault(mp, []).append(_safe(r["rotation_counterfactual_delta"]))

        return {
            "avg_delta_slots_3":  _avg_list(by_slot.get(3, [])),
            "avg_delta_slots_4":  _avg_list(by_slot.get(4, [])),
            "avg_delta_slots_5":  _avg_list(by_slot.get(5, [])),
            "count_slots_3":      len(by_slot.get(3, [])),
            "count_slots_4":      len(by_slot.get(4, [])),
            "count_slots_5":      len(by_slot.get(5, [])),
        }
    except Exception as exc:
        logger.debug("compute_slot_efficiency failed: %s", exc)
        return {
            "avg_delta_slots_3": None, "avg_delta_slots_4": None, "avg_delta_slots_5": None,
            "count_slots_3": 0, "count_slots_4": 0, "count_slots_5": 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Task 6 — KPI Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_5d_kpis(
    enriched_file: Path,
    events_file: Path,
) -> dict:
    """
    Task 6: Full Phase 5D KPI aggregation.

    Most important KPI: positive_delta_rate
        = fraction of enriched events where rotation_counterfactual_delta > 0
        = "how often would rotation have actually helped?"

    Returns all required KPIs including regime/severity/age/slot breakdowns.
    Fail-open: returns _empty_5d_kpis() on any error.
    """
    try:
        events    = _load_jsonl(events_file)
        enriched  = _load_jsonl(enriched_file)
        completed = [r for r in enriched if r.get("rotation_counterfactual_delta") is not None]

        event_count        = len(events)
        materialized_count = len(enriched)

        gaps     = [_safe(r.get("priority_gap", 0.0)) for r in enriched]
        rej_rets = [
            _safe(r["rejected_candidate_5d_return"])
            for r in completed if r.get("rejected_candidate_5d_return") is not None
        ]
        blk_rets = [
            _safe(r["blocked_position_5d_return"])
            for r in completed if r.get("blocked_position_5d_return") is not None
        ]
        deltas = [_safe(r["rotation_counterfactual_delta"]) for r in completed]

        positive_delta_rate = (
            round(sum(1 for d in deltas if d > 0) / len(deltas), 4)
            if deltas else 0.0
        )
        severe_event_rate = (
            round(
                sum(1 for r in enriched if r.get("opportunity_cost_severity") == SEVERITY_SEVERE)
                / len(enriched),
                4,
            )
            if enriched else 0.0
        )

        # By-regime breakdown
        by_regime: Dict[str, List[float]] = {}
        for r in completed:
            reg = r.get("regime", REGIME_UNKNOWN)
            by_regime.setdefault(reg, []).append(_safe(r["rotation_counterfactual_delta"]))
        avg_delta_by_regime = {k: _avg_list(v) for k, v in by_regime.items()}

        sev_stats  = compute_severity_persistence(enriched_file)
        age_stats  = compute_holding_staleness(enriched_file)
        slot_stats = compute_slot_efficiency(enriched_file)

        return {
            "opportunity_cost_event_count":       event_count,
            "materialized_event_count":           materialized_count,
            "avg_priority_gap":                   _avg_list(gaps),
            "avg_rejected_candidate_5d_return":   _avg_list(rej_rets),
            "avg_blocked_position_5d_return":     _avg_list(blk_rets),
            "avg_rotation_counterfactual_delta":  _avg_list(deltas),
            "positive_delta_rate":                positive_delta_rate,
            "severe_event_rate":                  severe_event_rate,
            "avg_delta_by_regime":                avg_delta_by_regime,
            "avg_delta_by_severity": {
                "severe":   sev_stats["avg_delta_severe"],
                "moderate": sev_stats["avg_delta_moderate"],
                "low":      sev_stats["avg_delta_low"],
            },
            "avg_delta_by_holding_age": {
                "lt_10d":  age_stats["avg_delta_age_lt_10d"],
                "10_20d":  age_stats["avg_delta_age_10_20d"],
                "gt_20d":  age_stats["avg_delta_age_gt_20d"],
            },
            "avg_delta_by_slot_count": {
                "slots_3": slot_stats["avg_delta_slots_3"],
                "slots_4": slot_stats["avg_delta_slots_4"],
                "slots_5": slot_stats["avg_delta_slots_5"],
            },
            "schema_version": SCHEMA_VERSION_5D,
        }

    except Exception as exc:
        logger.warning("aggregate_5d_kpis failed: %s", exc)
        return _empty_5d_kpis()


def _empty_5d_kpis() -> dict:
    return {
        "opportunity_cost_event_count":       0,
        "materialized_event_count":           0,
        "avg_priority_gap":                   None,
        "avg_rejected_candidate_5d_return":   None,
        "avg_blocked_position_5d_return":     None,
        "avg_rotation_counterfactual_delta":  None,
        "positive_delta_rate":                0.0,
        "severe_event_rate":                  0.0,
        "avg_delta_by_regime":                {},
        "avg_delta_by_severity":              {"severe": None, "moderate": None, "low": None},
        "avg_delta_by_holding_age":           {"lt_10d": None, "10_20d": None, "gt_20d": None},
        "avg_delta_by_slot_count":            {"slots_3": None, "slots_4": None, "slots_5": None},
        "schema_version":                     SCHEMA_VERSION_5D,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task 7 — DRY/LIVE Report
# ─────────────────────────────────────────────────────────────────────────────

def format_oc_materialization_report(
    enriched_file: Path,
    events_file: Optional[Path] = None,
    date: Optional[str] = None,
    max_events: int = 5,
) -> str:
    """
    Task 7: Format the OPPORTUNITY COST MATERIALIZATION section.

    Shows per-event detail for recent completed materializations
    plus an aggregate summary. Fails open (returns "" on any error).
    """
    try:
        if date is None:
            date = datetime.now(JST).strftime("%Y-%m-%d")

        all_enriched = _load_jsonl(enriched_file)
        completed = [
            r for r in all_enriched
            if r.get("rotation_counterfactual_delta") is not None
        ][-max_events:]

        if not completed:
            return ""

        sep   = "─" * 58
        lines = [sep, "  OPPORTUNITY COST MATERIALIZATION", sep]

        for r in completed:
            rej     = r.get("rejected_symbol", "?")
            blk     = r.get("blocked_by_symbol", "?")
            rej_r   = r.get("rejected_candidate_5d_return")
            blk_r   = r.get("blocked_position_5d_return")
            delta   = _safe(r.get("rotation_counterfactual_delta", 0.0))
            sev     = str(r.get("opportunity_cost_severity", ""))
            regime  = r.get("regime", "unknown")
            sig_env = r.get("signal_environment", "")
            age     = int(r.get("blocked_position_age_days", 0))

            sign    = "+" if delta >= 0 else ""
            rej_str = f"{rej_r:+.1%}" if rej_r is not None else "pending"
            blk_str = f"{blk_r:+.1%}" if blk_r is not None else "pending"
            reg_str = f"{regime}/{sig_env}" if sig_env and sig_env != SIGNAL_UNKNOWN else regime

            lines.append(f"  Rejected: {rej}  Blocked: {blk}")
            lines.append(f"    Rejected 5d return  : {rej_str}")
            lines.append(f"    Blocked  5d return  : {blk_str}")
            lines.append(f"    Counterfactual delta: {sign}{delta:.1%}")
            lines.append(
                f"    Severity: {sev}  "
                f"Regime: {reg_str}  "
                f"Holding age: {age}d"
            )
            lines.append("")

        if events_file is not None:
            kpis       = aggregate_5d_kpis(enriched_file, events_file)
            pos_rate   = kpis["positive_delta_rate"]
            avg_sev    = kpis["avg_delta_by_severity"].get("severe")
            by_regime  = kpis.get("avg_delta_by_regime", {})

            worst_regime: Optional[str] = None
            worst_delta:  Optional[float] = None
            for reg, avg in by_regime.items():
                if avg is not None and (worst_delta is None or avg > worst_delta):
                    worst_delta = avg
                    worst_regime = reg

            lines.append(f"  Positive delta rate          : {pos_rate:.0%}")
            if avg_sev is not None:
                lines.append(f"  Avg severe-gap delta         : {avg_sev:+.1%}")
            if worst_regime:
                lines.append(f"  Worst regime for slot starv. : {worst_regime}")

        lines.append(sep)
        return "\n".join(lines)

    except Exception as exc:
        logger.debug("format_oc_materialization_report failed: %s", exc)
        return ""
