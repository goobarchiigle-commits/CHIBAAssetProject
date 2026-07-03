"""
opportunity_cost.py — Portfolio Rotation Opportunity Cost Intelligence

Observation-only telemetry for portfolio-full rejection events.

Records when high-quality new candidates are blocked because
max_positions is occupied by lower-priority holdings.

Key metric:
  priority_gap = candidate_priority_estimate - lowest_holding_priority
  >= 40  → severe_opportunity_cost
  >= 20  → moderate_opportunity_cost
  <  20  → low_opportunity_cost

No execution changes. No auto-rotation. Append-only JSONL. Fail-open.
O(n) where n = max_positions (typically 3).

Telemetry: runtime/analytics/opportunity_cost/
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import date as _date_type, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))
SCHEMA_VERSION = "v1"

# ── Gap severity thresholds ───────────────────────────────────────────────────
PRIORITY_GAP_SEVERE    = 40.0
PRIORITY_GAP_MODERATE  = 20.0

SEVERITY_SEVERE   = "severe_opportunity_cost"
SEVERITY_MODERATE = "moderate_opportunity_cost"
SEVERITY_LOW      = "low_opportunity_cost"

# ── Candidate priority estimation weights ─────────────────────────────────────
# Applied to new (non-held) candidates to generate a comparable 0-100 score.
W_COMPRESSION  = 0.35
W_BQ           = 0.30
W_RSR          = 0.25
W_SURVIVABILITY = 0.10

RSR_FLOOR = 60.0   # RSR below this → 0 pts
RSR_CEIL  = 100.0  # RSR at or above this → 100 pts

# ── Materialization: calendar days to wait before 5-business-day return ───────
MAT_BUFFER_DAYS = 8   # 5 business days ≈ 7 calendar days; 8 = safe margin


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RejectedCandidate:
    """Enriched info for a BUY candidate blocked due to portfolio being full."""
    symbol: str
    continuation_score: float      # 0-100 from strategy signal / CDP
    breakout_quality: float        # 0-100; 0 if unknown
    compression_score: float       # 0-100; 50 default
    survivability: float           # 0-1; 0.70 default
    rsr: float                     # RSR percentile (strategy-space, e.g. 75-100)


@dataclass
class HeldPositionSummary:
    """Priority snapshot of one currently-held position."""
    symbol: str
    priority_score: float    # 0-100 from ContinuationPriorityResult
    priority_tier: str       # core_continuation|high_quality|neutral|exit_candidate
    phase: str               # current_phase from ContinuationPriorityInput
    hold_days: int = 0       # calendar days held
    unrealized_pnl_pct: float = 0.0   # e.g. 0.12 = +12%
    rank: int = 1            # priority rank among held positions (1=highest)


@dataclass
class OpportunityCostRecord:
    """Single portfolio-full rejection event."""
    date: str
    rejected_symbol: str
    candidate_priority_estimate: float
    candidate_breakout_quality: float
    candidate_compression_score: float
    blocked_by_symbol: str
    blocked_position_priority: float
    blocked_position_tier: str
    priority_gap: float
    blocked_phase: str
    portfolio_full: bool
    opportunity_cost_severity: str
    schema_version: str = SCHEMA_VERSION
    # Phase 5C enrichment fields
    run_id: str = ""
    candidate_virtual_rank: int = 0
    blocked_position_age_days: int = 0
    blocked_unrealized_pnl_pct: float = 0.0
    blocked_rank: int = 1
    active_positions: int = 0
    max_positions: int = 3


@dataclass
class MaterializedRecord:
    """OpportunityCostRecord with 5-business-day forward returns filled in."""
    date: str
    rejected_symbol: str
    candidate_priority_estimate: float
    candidate_breakout_quality: float
    candidate_compression_score: float
    blocked_by_symbol: str
    blocked_position_priority: float
    blocked_position_tier: str
    priority_gap: float
    blocked_phase: str
    portfolio_full: bool
    opportunity_cost_severity: str
    schema_version: str
    rejected_candidate_5d_return: Optional[float]   # None = data unavailable
    blocked_position_5d_return: Optional[float]
    rotation_counterfactual_delta: Optional[float]  # rejected - blocked
    materialized_date: str
    materialization_complete: bool
    # Phase 5C enrichment fields (mirrored from OpportunityCostRecord)
    run_id: str = ""
    candidate_virtual_rank: int = 0
    blocked_position_age_days: int = 0
    blocked_unrealized_pnl_pct: float = 0.0
    blocked_rank: int = 1
    active_positions: int = 0
    max_positions: int = 3


# ─────────────────────────────────────────────────────────────────────────────
# Pure computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _rsr_strength_score(rsr: float) -> float:
    """Map RSR percentile → [0-100]. Matches continuation_priority formula."""
    v = _safe(rsr)
    span = RSR_CEIL - RSR_FLOOR
    if span <= 0:
        return 0.0
    return round(max(0.0, min(100.0, (v - RSR_FLOOR) / span * 100.0)), 2)


def estimate_candidate_priority(candidate: RejectedCandidate) -> float:
    """
    Estimate 0-100 priority score for a new (non-held) BUY candidate.

    Uses a comparable formula to ContinuationPriority but reflects
    entry-quality rather than hold-quality metrics.
    """
    comp  = max(0.0, min(100.0, _safe(candidate.compression_score, 50.0)))
    bq    = max(0.0, min(100.0, _safe(candidate.breakout_quality, 50.0)))
    rsr_s = _rsr_strength_score(_safe(candidate.rsr, 75.0))
    surv  = max(0.0, min(1.0, _safe(candidate.survivability, 0.70))) * 100.0

    raw = (
        comp  * W_COMPRESSION
        + bq  * W_BQ
        + rsr_s * W_RSR
        + surv  * W_SURVIVABILITY
    )
    return round(min(100.0, max(0.0, raw)), 2)


def classify_priority_gap(gap: float) -> str:
    """
    Severity classification for a priority_gap value.
      gap >= 40 → severe_opportunity_cost
      gap >= 20 → moderate_opportunity_cost
      else      → low_opportunity_cost
    """
    if gap >= PRIORITY_GAP_SEVERE:
        return SEVERITY_SEVERE
    if gap >= PRIORITY_GAP_MODERATE:
        return SEVERITY_MODERATE
    return SEVERITY_LOW


def find_lowest_priority_holding(
    positions: List[HeldPositionSummary],
) -> Optional[HeldPositionSummary]:
    """
    Return the held position with the lowest priority_score.
    None if positions is empty.
    """
    if not positions:
        return None
    return min(positions, key=lambda p: p.priority_score)


def build_opportunity_cost_record(
    candidate: RejectedCandidate,
    blocked_by: HeldPositionSummary,
    date: str,
    portfolio_full: bool,
    run_id: str = "",
    candidate_virtual_rank: int = 0,
    active_positions: int = 0,
    max_positions: int = 3,
    override_priority: Optional[float] = None,
) -> OpportunityCostRecord:
    """
    Build one OpportunityCostRecord. Pure function.
    Does NOT filter on gap > 0; caller decides whether to persist.

    override_priority: if set, use directly instead of estimate_candidate_priority()
    """
    candidate_priority = (
        round(override_priority, 2)
        if override_priority is not None
        else estimate_candidate_priority(candidate)
    )
    gap = round(candidate_priority - blocked_by.priority_score, 2)
    severity = classify_priority_gap(gap)

    return OpportunityCostRecord(
        date=date,
        rejected_symbol=candidate.symbol,
        candidate_priority_estimate=candidate_priority,
        candidate_breakout_quality=round(_safe(candidate.breakout_quality, 50.0), 2),
        candidate_compression_score=round(_safe(candidate.compression_score, 50.0), 2),
        blocked_by_symbol=blocked_by.symbol,
        blocked_position_priority=round(blocked_by.priority_score, 2),
        blocked_position_tier=blocked_by.priority_tier,
        priority_gap=gap,
        blocked_phase=blocked_by.phase,
        portfolio_full=portfolio_full,
        opportunity_cost_severity=severity,
        run_id=run_id,
        candidate_virtual_rank=candidate_virtual_rank,
        blocked_position_age_days=blocked_by.hold_days,
        blocked_unrealized_pnl_pct=blocked_by.unrealized_pnl_pct,
        blocked_rank=blocked_by.rank,
        active_positions=active_positions,
        max_positions=max_positions,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers (fail-open)
# ─────────────────────────────────────────────────────────────────────────────

def _append_jsonl(record: dict, path: Path) -> None:
    """Append one record to a JSONL file. Fail-open."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("opportunity_cost JSONL append failed (%s): %s", path, exc)


def _load_jsonl(path: Path) -> List[dict]:
    """Load all records from a JSONL file. Returns [] on any error."""
    if not path.exists():
        return []
    records: List[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    except Exception as exc:
        logger.warning("opportunity_cost JSONL load failed (%s): %s", path, exc)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Main event recording
# ─────────────────────────────────────────────────────────────────────────────

def record_opportunity_cost_events(
    rejected_candidates: List[RejectedCandidate],
    held_positions: List[HeldPositionSummary],
    portfolio_full: bool,
    date: str,
    events_file: Path,
    min_gap_to_record: float = 0.0,
) -> List[OpportunityCostRecord]:
    """
    Main entry point: record opportunity cost events for a morning run.

    For each rejected candidate, finds the lowest-priority holding
    and records an event if priority_gap > min_gap_to_record.

    Fail-open: JSONL write errors are logged, not raised.

    Args:
        rejected_candidates: BUY candidates that were blocked (portfolio full)
        held_positions:       continuation priority data for current holdings
        portfolio_full:       True if max_positions was reached
        date:                 "YYYY-MM-DD"
        events_file:          JSONL output path
        min_gap_to_record:    only persist events with gap > this (default: 0)

    Returns:
        list of recorded OpportunityCostRecord (empty if none qualified)
    """
    if not portfolio_full or not rejected_candidates or not held_positions:
        return []

    lowest = find_lowest_priority_holding(held_positions)
    if lowest is None:
        return []

    recorded: List[OpportunityCostRecord] = []
    for cand in rejected_candidates:
        rec = build_opportunity_cost_record(cand, lowest, date, portfolio_full)
        if rec.priority_gap > min_gap_to_record:
            ts = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
            row = asdict(rec)
            row["ts"] = ts
            _append_jsonl(row, events_file)
            recorded.append(rec)

    return recorded


# ─────────────────────────────────────────────────────────────────────────────
# Signal dict adapter (run_live_signal.py integration)
# ─────────────────────────────────────────────────────────────────────────────

def make_rejected_candidate_from_signal(signal: dict) -> RejectedCandidate:
    """
    Build RejectedCandidate from a run_live_signal result.signals dict.
    Uses safe defaults for any absent fields.
    """
    return RejectedCandidate(
        symbol=str(signal.get("symbol", "")),
        continuation_score=_safe(signal.get("continuation_score", 0.0)),
        breakout_quality=_safe(
            signal.get("breakout_quality_score", signal.get("bq_score", 50.0)), 50.0
        ),
        compression_score=_safe(signal.get("compression_score", 50.0), 50.0),
        survivability=_safe(signal.get("survivability", 0.70), 0.70),
        rsr=_safe(signal.get("rsr", 75.0), 75.0),
    )


def make_held_summary_from_cp_result(cp_tuple: tuple) -> Optional[HeldPositionSummary]:
    """
    Build HeldPositionSummary from a (_cp_sig, ContinuationPriorityResult) tuple
    as returned by the run_live_signal continuation_priority hook.
    Returns None on error.
    """
    try:
        sig, cp_res = cp_tuple
        return HeldPositionSummary(
            symbol=str(cp_res.symbol),
            priority_score=float(cp_res.priority_score),
            priority_tier=str(cp_res.priority_tier),
            phase=str(sig.get("compression_phase", sig.get("current_phase", "neutral"))),
        )
    except Exception as exc:
        logger.debug("make_held_summary_from_cp_result failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Materialization
# ─────────────────────────────────────────────────────────────────────────────

def _add_business_days(start_date_str: str, n: int) -> str:
    """
    Add n business days (Mon-Fri) to start_date_str ("YYYY-MM-DD").
    Ignores Japanese public holidays (acceptable for 5-day observational telemetry).
    Returns ISO date string.
    """
    try:
        d = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    except ValueError:
        return start_date_str
    added = 0
    while added < n:
        d += timedelta(days=1)
        if d.weekday() < 5:   # Mon=0 … Fri=4
            added += 1
    return d.isoformat()


def _due_for_materialization(event_date: str, today: str, buffer_days: int = MAT_BUFFER_DAYS) -> bool:
    """True when enough calendar days have passed to expect 5-business-day data."""
    try:
        e = datetime.strptime(event_date, "%Y-%m-%d").date()
        t = datetime.strptime(today, "%Y-%m-%d").date()
        return (t - e).days >= buffer_days
    except ValueError:
        return False


def _compute_return(
    symbol: str,
    event_date: str,
    target_date: str,
    price_fetcher: Callable[[str, str], Optional[float]],
) -> Optional[float]:
    """
    Compute (price_at_target - price_at_event) / price_at_event.
    Returns None if either price is unavailable.
    """
    p0 = price_fetcher(symbol, event_date)
    p1 = price_fetcher(symbol, target_date)
    if p0 is None or p1 is None or p0 <= 0:
        return None
    return round((p1 - p0) / p0, 6)


def materialize_outcomes(
    events_file: Path,
    materialized_file: Path,
    price_fetcher: Callable[[str, str], Optional[float]],
    today: str,
    forward_business_days: int = 5,
    buffer_days: int = MAT_BUFFER_DAYS,
) -> int:
    """
    Scan pending events and materialize 5-business-day returns.

    Only processes events that:
      - Are not already materialized
      - Have waited at least buffer_days calendar days

    Appends MaterializedRecord objects to materialized_file.
    Fail-open: never raises.

    Returns:
        number of events newly materialized
    """
    events   = _load_jsonl(events_file)
    done_keys = {
        (r.get("date", ""), r.get("rejected_symbol", ""))
        for r in _load_jsonl(materialized_file)
    }

    count = 0
    for ev in events:
        event_date    = ev.get("date", "")
        rej_sym       = ev.get("rejected_symbol", "")
        blocked_sym   = ev.get("blocked_by_symbol", "")

        key = (event_date, rej_sym)
        if key in done_keys:
            continue
        if not _due_for_materialization(event_date, today, buffer_days):
            continue

        target_date = _add_business_days(event_date, forward_business_days)

        rej_ret  = _compute_return(rej_sym,     event_date, target_date, price_fetcher)
        blk_ret  = _compute_return(blocked_sym, event_date, target_date, price_fetcher)

        delta: Optional[float] = None
        if rej_ret is not None and blk_ret is not None:
            delta = round(rej_ret - blk_ret, 6)

        mat = MaterializedRecord(
            date=event_date,
            rejected_symbol=rej_sym,
            candidate_priority_estimate=_safe(ev.get("candidate_priority_estimate")),
            candidate_breakout_quality=_safe(ev.get("candidate_breakout_quality", 50.0)),
            candidate_compression_score=_safe(ev.get("candidate_compression_score", 50.0)),
            blocked_by_symbol=blocked_sym,
            blocked_position_priority=_safe(ev.get("blocked_position_priority")),
            blocked_position_tier=str(ev.get("blocked_position_tier", "")),
            priority_gap=_safe(ev.get("priority_gap")),
            blocked_phase=str(ev.get("blocked_phase", "")),
            portfolio_full=bool(ev.get("portfolio_full", True)),
            opportunity_cost_severity=str(ev.get("opportunity_cost_severity", "")),
            schema_version=str(ev.get("schema_version", SCHEMA_VERSION)),
            rejected_candidate_5d_return=rej_ret,
            blocked_position_5d_return=blk_ret,
            rotation_counterfactual_delta=delta,
            materialized_date=today,
            materialization_complete=(rej_ret is not None and blk_ret is not None),
            run_id=str(ev.get("run_id", "")),
            candidate_virtual_rank=int(ev.get("candidate_virtual_rank", 0)),
            blocked_position_age_days=int(ev.get("blocked_position_age_days", 0)),
            blocked_unrealized_pnl_pct=_safe(ev.get("blocked_unrealized_pnl_pct", 0.0)),
            blocked_rank=int(ev.get("blocked_rank", 1)),
            active_positions=int(ev.get("active_positions", 0)),
            max_positions=int(ev.get("max_positions", 3)),
        )
        row = asdict(mat)
        row["ts"] = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
        _append_jsonl(row, materialized_file)
        done_keys.add(key)
        count += 1

    return count


# ─────────────────────────────────────────────────────────────────────────────
# KPI aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_kpis(materialized_file: Path) -> dict:
    """
    Compute opportunity cost KPIs from the materialized records file.

    Returns dict with keys:
      opportunity_cost_event_count
      avg_priority_gap
      avg_rejected_candidate_5d_return
      avg_blocked_position_5d_return
      avg_rotation_counterfactual_delta
      severe_event_count
      moderate_event_count
      low_event_count
      complete_materialization_count
    """
    records = _load_jsonl(materialized_file)
    if not records:
        return _empty_kpis()

    gaps:       List[float] = []
    rej_rets:   List[float] = []
    blk_rets:   List[float] = []
    deltas:     List[float] = []
    severe = moderate = low = complete = 0

    for r in records:
        gaps.append(_safe(r.get("priority_gap", 0.0)))

        sev = r.get("opportunity_cost_severity", "")
        if sev == SEVERITY_SEVERE:
            severe += 1
        elif sev == SEVERITY_MODERATE:
            moderate += 1
        else:
            low += 1

        rej_ret = r.get("rejected_candidate_5d_return")
        blk_ret = r.get("blocked_position_5d_return")
        delta   = r.get("rotation_counterfactual_delta")

        if rej_ret is not None:
            rej_rets.append(_safe(rej_ret))
        if blk_ret is not None:
            blk_rets.append(_safe(blk_ret))
        if delta is not None:
            deltas.append(_safe(delta))
        if r.get("materialization_complete"):
            complete += 1

    def _avg(lst: List[float]) -> Optional[float]:
        return round(sum(lst) / len(lst), 6) if lst else None

    total = len(records)
    severe_event_rate = round(severe / total, 4) if total > 0 else 0.0

    total_slots = sum(int(r.get("max_positions", 3)) for r in records)
    opportunity_cost_per_slot = round(total / total_slots, 4) if total_slots > 0 else 0.0

    return {
        "opportunity_cost_event_count":        total,
        "avg_priority_gap":                    _avg(gaps),
        "avg_rejected_candidate_5d_return":    _avg(rej_rets),
        "avg_blocked_position_5d_return":      _avg(blk_rets),
        "avg_rotation_counterfactual_delta":   _avg(deltas),
        "severe_event_count":                  severe,
        "moderate_event_count":                moderate,
        "low_event_count":                     low,
        "complete_materialization_count":      complete,
        "severe_event_rate":                   severe_event_rate,
        "opportunity_cost_per_slot":           opportunity_cost_per_slot,
        "schema_version":                      SCHEMA_VERSION,
    }


def _empty_kpis() -> dict:
    return {
        "opportunity_cost_event_count":      0,
        "avg_priority_gap":                  None,
        "avg_rejected_candidate_5d_return":  None,
        "avg_blocked_position_5d_return":    None,
        "avg_rotation_counterfactual_delta": None,
        "severe_event_count":                0,
        "moderate_event_count":              0,
        "low_event_count":                   0,
        "complete_materialization_count":    0,
        "severe_event_rate":                 0.0,
        "opportunity_cost_per_slot":         0.0,
        "schema_version":                    SCHEMA_VERSION,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5C: CapitalDeploymentOS integration helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tier_from_continuation_score(score: float) -> str:
    """Map continuation_score (0-100) → priority tier label."""
    s = _safe(score)
    if s >= 80.0:
        return "core_continuation"
    if s >= 65.0:
        return "high_quality"
    if s >= 45.0:
        return "neutral"
    return "exit_candidate"


def emit_opportunity_cost_from_deployment(
    candidate: Any,
    context: Any,
    held_priorities: Dict[str, float],
    candidate_virtual_rank: int,
    max_positions: int,
    events_file: Path,
    run_id: str = "",
) -> None:
    """
    Fail-open OC telemetry hook called from CapitalDeploymentOS.

    Call ONLY when: portfolio_full=True AND governance_passed=True.
    Uses continuation_score of held candidates as priority proxy.
    Never raises; all errors are logged and swallowed.

    Args:
        candidate:             CandidateInfo (capital_deployment_os)
        context:               DeploymentContext (capital_deployment_os)
        held_priorities:       {symbol: continuation_score} for held positions
        candidate_virtual_rank: rank of this candidate in deployment queue
        max_positions:         dynamic_max_positions value for this run
        events_file:           JSONL append target
        run_id:                optional run identifier for dedup / tracing
    """
    try:
        if not held_priorities:
            return

        # Build HeldPositionSummary list from priorities + position state
        held_list: List[HeldPositionSummary] = []
        for sym, score in held_priorities.items():
            hp = context.current_positions.get(sym)
            held_list.append(HeldPositionSummary(
                symbol=sym,
                priority_score=_safe(score),
                priority_tier=_tier_from_continuation_score(_safe(score)),
                phase="",
                hold_days=hp.hold_days if hp is not None else 0,
                unrealized_pnl_pct=hp.unrealized_pnl_pct if hp is not None else 0.0,
                rank=0,  # assigned below
            ))

        # Rank: 1 = highest priority, len = lowest (most likely to be replaced)
        held_list.sort(key=lambda h: h.priority_score, reverse=True)
        for idx, h in enumerate(held_list):
            h.rank = idx + 1

        lowest = find_lowest_priority_holding(held_list)
        if lowest is None:
            return

        date = getattr(context, "date", "") or datetime.now(JST).strftime("%Y-%m-%d")
        active_pos = len(context.current_positions)

        # Use continuation_score directly as priority proxy (no new scoring)
        candidate_priority = round(_safe(getattr(candidate, "continuation_score", 0.0)), 2)
        gap = round(candidate_priority - lowest.priority_score, 2)
        severity = classify_priority_gap(gap)

        rec = OpportunityCostRecord(
            date=date,
            rejected_symbol=str(getattr(candidate, "symbol", "")),
            candidate_priority_estimate=candidate_priority,
            candidate_breakout_quality=50.0,
            candidate_compression_score=round(
                _safe(getattr(candidate, "predictive_score", 50.0), 50.0), 2
            ),
            blocked_by_symbol=lowest.symbol,
            blocked_position_priority=round(lowest.priority_score, 2),
            blocked_position_tier=lowest.priority_tier,
            priority_gap=gap,
            blocked_phase=lowest.phase,
            portfolio_full=True,
            opportunity_cost_severity=severity,
            run_id=run_id,
            candidate_virtual_rank=candidate_virtual_rank,
            blocked_position_age_days=lowest.hold_days,
            blocked_unrealized_pnl_pct=lowest.unrealized_pnl_pct,
            blocked_rank=lowest.rank,
            active_positions=active_pos,
            max_positions=max_positions,
        )

        ts = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
        row = asdict(rec)
        row["ts"] = ts
        _append_jsonl(row, events_file)

    except Exception as exc:
        logger.warning("emit_opportunity_cost_from_deployment failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run report formatter (Task 3)
# ─────────────────────────────────────────────────────────────────────────────

def format_opportunity_cost_report(
    events_file: Path,
    materialized_file: Optional[Path] = None,
    date: Optional[str] = None,
    max_events: int = 10,
) -> str:
    """
    Format a human-readable opportunity cost section for dry/live reports.

    Example output:
    ──────────────────────────────────────────
    OPPORTUNITY COST EVENTS (2026-05-27)
    ──────────────────────────────────────────
    Rejected: 7011.T  priority=91.4  [compression=79  BQ=84]
    Blocked : 5803.T  priority=41.3  tier=exit_candidate  phase=weak_breakout
    Gap     : +50.1  → SEVERE_OPPORTUNITY_COST
    ──────────────────────────────────────────

    Fails open (returns empty string on any error).
    """
    try:
        if date is None:
            date = datetime.now(JST).strftime("%Y-%m-%d")

        all_events = _load_jsonl(events_file)
        today_events = [
            e for e in all_events
            if e.get("date", "") == date
        ][-max_events:]

        if not today_events:
            return ""

        lines: List[str] = []
        sep = "─" * 58
        lines.append(sep)
        lines.append(f"  OPPORTUNITY COST EVENTS  ({date})")
        lines.append(sep)

        for ev in today_events:
            rej  = ev.get("rejected_symbol", "?")
            blk  = ev.get("blocked_by_symbol", "?")
            gap  = _safe(ev.get("priority_gap", 0.0))
            sev  = ev.get("opportunity_cost_severity", "")
            cpri = _safe(ev.get("candidate_priority_estimate", 0.0))
            bpri = _safe(ev.get("blocked_position_priority", 0.0))
            tier = ev.get("blocked_position_tier", "?")
            phase = ev.get("blocked_phase", "?")
            bq   = _safe(ev.get("candidate_breakout_quality", 0.0))
            comp = _safe(ev.get("candidate_compression_score", 0.0))

            sev_tag = sev.upper().replace("_OPPORTUNITY_COST", "").replace("_", " ")
            sign    = "+" if gap >= 0 else ""

            lines.append(
                f"  Rejected: {rej:<8s}  priority={cpri:.1f}"
                f"  [compression={comp:.0f}  BQ={bq:.0f}]"
            )
            age_days = ev.get("blocked_position_age_days", 0)
            blk_pnl  = _safe(ev.get("blocked_unrealized_pnl_pct", 0.0))
            age_str  = f"  age={age_days}d" if age_days else ""
            pnl_str  = f"  pnl={blk_pnl:+.1%}" if age_days or blk_pnl != 0.0 else ""
            lines.append(
                f"  Blocked : {blk:<8s}  priority={bpri:.1f}"
                f"  tier={tier}  phase={phase}{age_str}{pnl_str}"
            )
            lines.append(
                f"  Gap     : {sign}{gap:.1f}  → {sev_tag}"
            )
            lines.append("")

        # Materialized summary if available
        if materialized_file is not None:
            kpis = aggregate_kpis(materialized_file)
            n    = kpis["opportunity_cost_event_count"]
            avg_delta = kpis["avg_rotation_counterfactual_delta"]
            if n > 0:
                delta_str = (
                    f"{avg_delta:+.2%}" if avg_delta is not None else "pending"
                )
                lines.append(
                    f"  KPI (all-time):  events={n}"
                    f"  avg_gap={kpis['avg_priority_gap']:.1f}"
                    f"  avg_rotation_delta={delta_str}"
                    f"  severe={kpis['severe_event_count']}"
                )

        lines.append(sep)
        return "\n".join(lines)

    except Exception as exc:
        logger.debug("format_opportunity_cost_report failed: %s", exc)
        return ""
