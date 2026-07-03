"""
src/analytics/future_leader_transition.py — Future Leader Failure Transition Analytics

Purpose:
  Observe which deterioration sequence leads to persistence collapse.
  Records consecutive failure-state transitions for the same symbol within
  a 10-day window.  Research telemetry only — never connected to execution.

INVARIANTS (変更禁止):
  - observation_only=True: no execution path connection
  - pair telemetry only: single (from_state → to_state) step per record
  - multi-step chain: forbidden
  - Markov model / graph analysis: forbidden
  - transition prediction: forbidden
  - lookahead-safe: uses only already-materialized JSONL records (no OHLCV reload)
  - append-only JSONL: (symbol, from_obs_date, to_obs_date) never reprocessed
  - CLEAN-only aggregation: data_quality_weight >= 1.0 AND NOT insufficient_forward_data
  - PARTIAL_FAILURE: persisted to JSONL, excluded from aggregation

State priority (per observation):
  1. primary_failure_reason  — if failure_detected=True and reason is not None/"none"
  2. failure_archetype       — if archetype is not "unresolved"
  Observations with no resolvable state are skipped.

禁止:
  - allocator / broker / signal_bridge / deployable_universe / order / execution / routing imports
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

_QUALITY_CLEAN_MIN: float = 1.0
_DELTA_MAX_DAYS:    int   = 10
_MIN_SAMPLE:        int   = 3

# States excluded from transition endpoints
_SKIP_STATES: Set[str] = {"none", "unresolved", ""}


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransitionRecord:
    """
    Single deterioration-state transition for one symbol.
    observation_only=True: invariant, never connected to execution path.
    """
    symbol:               str
    observation_date:     str            # from_observation_date (earlier)
    to_observation_date:  str            # to_observation_date (later)
    transition_pair:      str            # "{from_state}->{to_state}"
    from_state:           str
    to_state:             str
    transition_days:      int
    regime:               str
    survival_days:        Optional[int]
    half_life_day:        Optional[int]   # None — not available per-record
    data_quality_weight:  float
    integrity_state:      str
    insufficient_forward_data: bool
    observation_only:     bool = True    # invariant: always True

    def to_dict(self) -> dict:
        return {
            "symbol":                  self.symbol,
            "observation_date":        self.observation_date,
            "to_observation_date":     self.to_observation_date,
            "transition_pair":         self.transition_pair,
            "from_state":              self.from_state,
            "to_state":                self.to_state,
            "transition_days":         self.transition_days,
            "regime":                  self.regime,
            "survival_days":           self.survival_days,
            "half_life_day":           self.half_life_day,
            "data_quality_weight":     self.data_quality_weight,
            "integrity_state":         self.integrity_state,
            "insufficient_forward_data": self.insufficient_forward_data,
            "observation_only":        self.observation_only,
        }


@dataclass
class TransitionSummary:
    """Aggregated metrics per transition_pair. CLEAN-only."""
    transition_pair:         str
    count:                   int
    median_transition_days:  Optional[float]
    median_survival_days:    Optional[float]
    median_half_life_day:    Optional[float]   # None when no data


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl(log_dir: Path, pattern: str) -> List[dict]:
    records: List[dict] = []
    if not log_dir.exists():
        return records
    for p in sorted(log_dir.glob(pattern)):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        except Exception:
            pass
    return records


def _load_failure_records(failure_log_dir: Path) -> List[dict]:
    return _load_jsonl(failure_log_dir, "future_leader_failure_*.jsonl")


def _load_archetype_records(archetype_log_dir: Path) -> List[dict]:
    return _load_jsonl(archetype_log_dir, "future_leader_archetype_*.jsonl")


def _load_survivability_records(survivability_log_dir: Path) -> List[dict]:
    return _load_jsonl(survivability_log_dir, "future_leader_survivability_*.jsonl")


def _load_regime_lookup(regime_log_dir: Path) -> Dict[str, str]:
    """Load {observation_date: regime_tag}. Last occurrence per date wins."""
    result: Dict[str, str] = {}
    for rec in _load_jsonl(regime_log_dir, "future_leader_regime_*.jsonl"):
        dt  = rec.get("observation_date", "")
        tag = rec.get("regime_tag", "")
        if dt and tag:
            result[dt] = tag
    return result


def _load_transition_keys(transition_log_dir: Path) -> Set[Tuple[str, str, str]]:
    """Load already-classified (symbol, from_obs_date, to_obs_date) triples."""
    keys: Set[Tuple[str, str, str]] = set()
    for rec in _load_jsonl(transition_log_dir, "future_leader_transition_*.jsonl"):
        sym    = rec.get("symbol", "")
        from_d = rec.get("observation_date", "")
        to_d   = rec.get("to_observation_date", "")
        if sym and from_d and to_d:
            keys.add((sym, from_d, to_d))
    return keys


def _write_transition_log(
    records:            List[TransitionRecord],
    transition_log_dir: Path,
    today:              Optional[date] = None,
) -> None:
    """Append new transition records to today's JSONL. Fail-open."""
    if not records:
        return
    if today is None:
        today = datetime.now(JST).date()

    today_str = today.strftime("%Y%m%d")
    log_path  = transition_log_dir / f"future_leader_transition_{today_str}.jsonl"

    try:
        transition_log_dir.mkdir(parents=True, exist_ok=True)
        ts    = datetime.now(JST).isoformat()
        lines = []
        for r in records:
            d = {"materialization_ts": ts, **r.to_dict()}
            lines.append(json.dumps(d, ensure_ascii=False, default=str))
        with log_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("[FL_TRANS] wrote %d records → %s", len(records), log_path)
    except Exception as e:
        logger.warning("[FL_TRANS] log write failed (fail-open): %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# State resolution — pure
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_state(merged: dict) -> Optional[str]:
    """
    Resolve state name from a merged observation record.
    Priority 1: primary_failure_reason (when failure_detected=True and not None/"none").
    Priority 2: failure_archetype (when not "unresolved").
    Returns None when no resolvable state — observation skipped for transitions.
    Pure function: no I/O.
    """
    if merged.get("failure_detected", False):
        reason = merged.get("primary_failure_reason")
        if reason and reason not in _SKIP_STATES:
            return reason
    arch = merged.get("failure_archetype")
    if arch and arch not in _SKIP_STATES:
        return arch
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Merge lookup — pure
# ─────────────────────────────────────────────────────────────────────────────

def _build_merged_lookup(
    failure_records:       List[dict],
    archetype_records:     List[dict],
    survivability_records: List[dict],
) -> Dict[Tuple[str, str], dict]:
    """
    Merge failure, archetype, and survivability records into a unified lookup.
    Key: (symbol, observation_date).  First occurrence per key wins for each source.
    Pure function: no I/O.
    """
    # Base from failure records
    merged: Dict[Tuple[str, str], dict] = {}
    for rec in failure_records:
        sym = rec.get("symbol", "")
        dt  = rec.get("observation_date", "")
        if not (sym and dt):
            continue
        key = (sym, dt)
        if key not in merged:
            merged[key] = {
                "symbol":                  sym,
                "observation_date":        dt,
                "failure_detected":        rec.get("failure_detected", False),
                "primary_failure_reason":  rec.get("primary_failure_reason") or rec.get("failure_reason"),
                "failure_archetype":       None,
                "failure_day_offset":      rec.get("failure_day_offset"),
                "data_quality_weight":     float(rec.get("data_quality_weight", 1.0)),
                "integrity_state":         str(rec.get("integrity_state", "CLEAN")),
                "insufficient_forward_data": False,
            }

    # Overlay archetype
    for rec in archetype_records:
        sym = rec.get("symbol", "")
        dt  = rec.get("observation_date", "")
        if not (sym and dt):
            continue
        key = (sym, dt)
        if key in merged:
            if merged[key]["failure_archetype"] is None:
                merged[key]["failure_archetype"] = rec.get("failure_archetype")
        else:
            # Archetype-only record (no failure record for this date)
            merged[key] = {
                "symbol":                  sym,
                "observation_date":        dt,
                "failure_detected":        False,
                "primary_failure_reason":  None,
                "failure_archetype":       rec.get("failure_archetype"),
                "failure_day_offset":      None,
                "data_quality_weight":     float(rec.get("data_quality_weight", 1.0)),
                "integrity_state":         str(rec.get("integrity_state", "CLEAN")),
                "insufficient_forward_data": False,
            }

    # Overlay survivability (insufficient_forward_data flag)
    for rec in survivability_records:
        sym = rec.get("symbol", "")
        dt  = rec.get("observation_date", "")
        if not (sym and dt):
            continue
        key = (sym, dt)
        if key in merged:
            if rec.get("insufficient_forward_data", False):
                merged[key]["insufficient_forward_data"] = True

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Transition detection — pure
# ─────────────────────────────────────────────────────────────────────────────

def detect_transitions(
    merged_lookup:   Dict[Tuple[str, str], dict],
    existing_keys:   Set[Tuple[str, str, str]],
    regime_lookup:   Dict[str, str],
) -> List[TransitionRecord]:
    """
    Detect new deterioration-state transitions from merged observation records.

    For each symbol: sort observation_dates, iterate consecutive pairs (i, i+1).
    Record a transition when:
      - 0 < delta_days <= _DELTA_MAX_DAYS
      - both from_state and to_state are resolvable (not None/unresolved/none)
      - (symbol, from_date, to_date) not already in existing_keys

    Pure except for date arithmetic. No I/O. No OHLCV. No lookahead.
    """
    # Group dates by symbol
    symbol_dates: Dict[str, List[str]] = {}
    for (sym, obs_date) in merged_lookup:
        symbol_dates.setdefault(sym, []).append(obs_date)

    new_transitions: List[TransitionRecord] = []

    for sym, dates in sorted(symbol_dates.items()):
        dates_sorted = sorted(set(dates))

        for i in range(len(dates_sorted) - 1):
            d1_str = dates_sorted[i]
            d2_str = dates_sorted[i + 1]

            try:
                d1 = date.fromisoformat(d1_str)
                d2 = date.fromisoformat(d2_str)
                delta_days = (d2 - d1).days
            except ValueError:
                continue

            if delta_days <= 0 or delta_days > _DELTA_MAX_DAYS:
                continue

            triple = (sym, d1_str, d2_str)
            if triple in existing_keys:
                continue

            rec1 = merged_lookup.get((sym, d1_str))
            rec2 = merged_lookup.get((sym, d2_str))
            if rec1 is None or rec2 is None:
                continue

            from_state = _resolve_state(rec1)
            to_state   = _resolve_state(rec2)

            if from_state is None or to_state is None:
                continue

            regime        = regime_lookup.get(d1_str, "unclassified")
            survival_days = rec1.get("failure_day_offset")
            survival_days = int(survival_days) if survival_days is not None else None
            dq_weight     = float(rec1.get("data_quality_weight", 1.0))
            integrity     = str(rec1.get("integrity_state", "CLEAN"))
            insuf_fwd     = bool(rec1.get("insufficient_forward_data", False))

            new_transitions.append(TransitionRecord(
                symbol=sym,
                observation_date=d1_str,
                to_observation_date=d2_str,
                transition_pair=f"{from_state}->{to_state}",
                from_state=from_state,
                to_state=to_state,
                transition_days=delta_days,
                regime=regime,
                survival_days=survival_days,
                half_life_day=None,
                data_quality_weight=dq_weight,
                integrity_state=integrity,
                insufficient_forward_data=insuf_fwd,
                observation_only=True,
            ))

    return new_transitions


# ─────────────────────────────────────────────────────────────────────────────
# Main materialization pipeline
# ─────────────────────────────────────────────────────────────────────────────

def materialize_transitions(
    failure_log_dir:       Path,
    archetype_log_dir:     Path,
    survivability_log_dir: Path,
    regime_log_dir:        Path,
    transition_log_dir:    Path,
    today:                 Optional[date] = None,
) -> int:
    """
    Detect and persist new failure-state transitions.

    observation_only=True: research telemetry only.
    Input: already-materialized JSONL (no OHLCV reload, no live market access).
    append-only: (symbol, from_obs_date, to_obs_date) never reprocessed.
    PARTIAL_FAILURE records persisted; excluded from aggregation.

    Returns count of newly persisted transitions.
    """
    if today is None:
        today = datetime.now(JST).date()

    fail_recs  = _load_failure_records(failure_log_dir)
    arch_recs  = _load_archetype_records(archetype_log_dir)
    surv_recs  = _load_survivability_records(survivability_log_dir)

    if not fail_recs and not arch_recs:
        logger.info("[FL_TRANS] no failure/archetype records — nothing to process")
        return 0

    merged_lookup = _build_merged_lookup(fail_recs, arch_recs, surv_recs)
    if not merged_lookup:
        logger.info("[FL_TRANS] empty merged lookup — nothing to process")
        return 0

    existing_keys = _load_transition_keys(transition_log_dir)
    regime_lookup = _load_regime_lookup(regime_log_dir)

    new_records = detect_transitions(merged_lookup, existing_keys, regime_lookup)

    if new_records:
        _write_transition_log(new_records, transition_log_dir, today=today)

    logger.info("[FL_TRANS] detected=%d existing=%d", len(new_records), len(existing_keys))
    return len(new_records)


# ─────────────────────────────────────────────────────────────────────────────
# Analytics — pure functions
# ─────────────────────────────────────────────────────────────────────────────

def _filter_clean(records: List[dict]) -> List[dict]:
    """
    CLEAN-only: data_quality_weight >= 1.0 AND NOT insufficient_forward_data.
    PARTIAL_FAILURE excluded.
    """
    return [
        r for r in records
        if float(r.get("data_quality_weight", 0.0)) >= _QUALITY_CLEAN_MIN
        and not r.get("insufficient_forward_data", False)
    ]


def _opt_median(vals: List[float]) -> Optional[float]:
    if len(vals) < _MIN_SAMPLE:
        return None
    return round(float(np.median(vals)), 4)


def compute_transition_summary(
    transition_log_dir: Path,
) -> List[TransitionSummary]:
    """
    Compute per-transition-pair analytics from CLEAN transition records.

    CLEAN filter: data_quality_weight >= 1.0 AND NOT insufficient_forward_data.
    PARTIAL_FAILURE excluded from aggregation.
    Returns summaries sorted by count descending (frequency ranking).
    Pure except for loading JSONL.
    """
    all_records = _load_jsonl(transition_log_dir, "future_leader_transition_*.jsonl")
    clean       = _filter_clean(all_records)

    groups: Dict[str, List[dict]] = {}
    for r in clean:
        pair = r.get("transition_pair", "")
        if pair:
            groups.setdefault(pair, []).append(r)

    summaries: List[TransitionSummary] = []
    for pair, recs in groups.items():
        trans_days  = [float(r["transition_days"])  for r in recs if r.get("transition_days")  is not None]
        surv_days   = [float(r["survival_days"])    for r in recs if r.get("survival_days")    is not None]
        hl_days     = [float(r["half_life_day"])    for r in recs if r.get("half_life_day")    is not None]

        summaries.append(TransitionSummary(
            transition_pair=pair,
            count=len(recs),
            median_transition_days=_opt_median(trans_days),
            median_survival_days=_opt_median(surv_days),
            median_half_life_day=_opt_median(hl_days) if hl_days else None,
        ))

    # Frequency ranking: sort by count descending
    summaries.sort(key=lambda s: -s.count)
    return summaries


# ─────────────────────────────────────────────────────────────────────────────
# Report formatter — pure function
# ─────────────────────────────────────────────────────────────────────────────

def format_transition_section(summaries: List[TransitionSummary]) -> str:
    """
    Render FAILURE TRANSITIONS section. Pure function: no I/O.

    Example:
      === FAILURE TRANSITIONS ===

      volume_faded->rsr_reversal:
        count: 18
        median transition: 4d
        median survival after transition: 6d
    """
    lines = ["=== FAILURE TRANSITIONS ==="]

    if not summaries:
        lines.append("(no transition records)")
        return "\n".join(lines)

    for s in summaries:
        lines.append("")
        lines.append(f"{s.transition_pair}:")
        lines.append(f"  count: {s.count}")
        if s.median_transition_days is not None:
            lines.append(f"  median transition: {s.median_transition_days:.1f}d")
        if s.median_survival_days is not None:
            lines.append(f"  median survival after transition: {s.median_survival_days:.1f}d")
        if s.median_half_life_day is not None:
            lines.append(f"  median half-life: {s.median_half_life_day:.1f}d")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# run_live_signal.py integration hook
# ─────────────────────────────────────────────────────────────────────────────

def run_future_leader_transition_hook(
    failure_log_dir:       Path,
    archetype_log_dir:     Path,
    survivability_log_dir: Path,
    regime_log_dir:        Path,
    transition_log_dir:    Path,
    reports_dir:           Path,
    today:                 Optional[date] = None,
) -> None:
    """
    Integration hook for run_live_signal.py. Fail-open.

    Positioned after future leader archetype hook.
    1. Materializes transitions for unmaterialized consecutive observation pairs.
    2. Computes transition summary (CLEAN-only, frequency-ranked).
    3. Appends FAILURE TRANSITIONS section to today's future_leader_report.

    observation_only=True: no execution path connection.
    Input: already-materialized JSONL only (no OHLCV reload).
    Failure: warn and return — execution path untouched.
    """
    if today is None:
        today = datetime.now(JST).date()

    today_str   = today.strftime("%Y%m%d")
    report_path = reports_dir / f"future_leader_report_{today_str}.md"

    try:
        count = materialize_transitions(
            failure_log_dir=failure_log_dir,
            archetype_log_dir=archetype_log_dir,
            survivability_log_dir=survivability_log_dir,
            regime_log_dir=regime_log_dir,
            transition_log_dir=transition_log_dir,
            today=today,
        )
        logger.info("[FL_TRANS] hook: materialized %d new transitions", count)

        summaries = compute_transition_summary(transition_log_dir)
        section   = format_transition_section(summaries)

        reports_dir.mkdir(parents=True, exist_ok=True)
        mode = "a" if report_path.exists() else "w"
        with report_path.open(mode, encoding="utf-8") as f:
            f.write("\n\n" + section + "\n")

        logger.info("[FL_TRANS] section written → %s", report_path)

    except Exception as e:
        logger.warning("[FL_TRANS] hook failed (fail-open): %s", e)
