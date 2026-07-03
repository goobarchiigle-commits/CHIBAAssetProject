"""
shadow_recommendation.py — Shadow Recommendation Engine

Reads the latest Evidence Promotion record and maps each candidate to a
Shadow Validation recommendation. This is the Evidence → Shadow promotion layer.

Input:
  evidence_promotion.jsonl (latest record only; read-only)

Status mapping:
  PROMOTABLE       → SHADOW_READY
  OBSERVE          → CONTINUE_OBSERVE
  INSUFFICIENT_DATA → WAIT_DATA

shadow_priority_score (0-100):
  SHADOW_READY:      evidence_score         (unchanged)
  CONTINUE_OBSERVE:  evidence_score × 0.5   (half; still observable)
  WAIT_DATA:         0

shadow_duration_days (recommended shadow observation window):
  priority_score >= 80 → 20d
  priority_score >= 60 → 30d
  else                 → 45d

Weekly governance summary (Task 7):
  Reads SR JSONL (all records); filters to last lookback_days.
  Returns: shadow_ready_count, top_shadow_candidate, avg_shadow_priority

Design:
  observation_only — no signals, no orders, no execution mutation
  fail-open        — every stage wrapped; never blocks live execution
  O(n)             — n = lines in evidence_promotion.jsonl (reads latest only)
  append-only JSONL — 1 record per run; never rewritten
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_JST = timezone(timedelta(hours=9))
SCHEMA_VERSION = "v1"

# ── Evidence status constants (from EP output) ────────────────────────────────
EP_PROMOTABLE   = "PROMOTABLE"
EP_OBSERVE      = "OBSERVE"
EP_INSUFFICIENT = "INSUFFICIENT_DATA"

# ── Shadow status constants ───────────────────────────────────────────────────
SHADOW_READY        = "SHADOW_READY"
CONTINUE_OBSERVE    = "CONTINUE_OBSERVE"
WAIT_DATA           = "WAIT_DATA"

# ── shadow_priority_score multipliers ────────────────────────────────────────
OBSERVE_SCORE_FACTOR:  float = 0.5    # CONTINUE_OBSERVE priority = evidence × 0.5
WAIT_DATA_SCORE:       int   = 0

# ── shadow_duration_days thresholds ──────────────────────────────────────────
DURATION_FAST:        int = 20    # priority_score >= SCORE_FAST_THRESHOLD
DURATION_MEDIUM:      int = 30    # priority_score >= SCORE_MEDIUM_THRESHOLD
DURATION_SLOW:        int = 45    # below SCORE_MEDIUM_THRESHOLD

SCORE_FAST_THRESHOLD:   int = 80
SCORE_MEDIUM_THRESHOLD: int = 60

# ── Candidate ordering (from EP) ─────────────────────────────────────────────
_CANDIDATE_ORDER = [
    "EXIT_EXTENSION",
    "CAPITAL_CONCENTRATION",
    "PRIORITY",
    "SIGNAL_LEADER",
    "CONTINUATION_SIGNAL",
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ShadowRecommendation:
    candidate_type:       str    # e.g. "EXIT_EXTENSION"
    source_module:        str    # "HDC", "CCS", "PCI", "SA", "FCO"
    evidence_status:      str    # from EP: "PROMOTABLE" | "OBSERVE" | "INSUFFICIENT_DATA"
    shadow_status:        str    # "SHADOW_READY" | "CONTINUE_OBSERVE" | "WAIT_DATA"
    evidence_score:       int    # from EP (0-100)
    shadow_priority_score: int   # 0-100 derived from evidence_score + status
    shadow_duration_days: int    # recommended shadow observation window (20 / 30 / 45)


@dataclass
class ShadowRecommendationOutput:
    date:                 str
    n_shadow_ready:       int
    n_continue_observe:   int
    n_wait_data:          int
    top_shadow_candidate: str    # highest shadow_priority_score among SHADOW_READY; "" if none
    recommendations:      List[ShadowRecommendation]
    schema_version:       str = SCHEMA_VERSION


@dataclass
class WeeklyGovernanceSummary:
    lookback_days:        int
    shadow_ready_count:   int    # 最重要KPI 1: total SHADOW_READY recs in window
    top_shadow_candidate: str    # 最重要KPI 2: most common SHADOW_READY type; "" if none
    avg_shadow_priority:  float  # 最重要KPI 3: mean shadow_priority_score for SHADOW_READY


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _safe_int(v: Any, fallback: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return fallback


def _safe_str(v: Any, fallback: str = "") -> str:
    if v is None:
        return fallback
    return str(v)


def _load_latest_record(path: Path) -> dict:
    """Return last valid JSON line from a JSONL file. Returns {} on any error."""
    try:
        if not path.exists():
            return {}
        last: dict = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except Exception:
                pass
        return last
    except Exception as exc:
        logger.debug("[SR] load_latest_record %s failed: %s", path, exc)
        return {}


def _load_all_jsonl(path: Path) -> List[dict]:
    """Load all valid JSON lines from a JSONL file. FAIL_OPEN → empty list."""
    records: List[dict] = []
    try:
        if not path.exists():
            return records
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    except Exception as exc:
        logger.debug("[SR] load_all_jsonl %s failed: %s", path, exc)
    return records


def _append_jsonl(record: dict, path: Path) -> None:
    """Atomic append-only JSONL write. FAIL_OPEN."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tf:
                existing = path.read_text(encoding="utf-8") if path.exists() else ""
                tf.write(existing)
                tf.write(line + "\n")
                tf.flush()
                os.fsync(tf.fileno())
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise
        Path(tmp).replace(path)
    except Exception as exc:
        logger.warning("[SR] append_jsonl failed: %s", exc)


# ── Core logic ────────────────────────────────────────────────────────────────

def _map_shadow_status(evidence_status: str) -> str:
    """Map EP status to shadow status."""
    if evidence_status == EP_PROMOTABLE:
        return SHADOW_READY
    if evidence_status == EP_OBSERVE:
        return CONTINUE_OBSERVE
    return WAIT_DATA


def _compute_shadow_priority(evidence_score: int, shadow_status: str) -> int:
    """Derive shadow_priority_score from evidence_score + status."""
    if shadow_status == SHADOW_READY:
        return max(0, min(100, evidence_score))
    if shadow_status == CONTINUE_OBSERVE:
        return max(0, min(100, int(evidence_score * OBSERVE_SCORE_FACTOR)))
    return WAIT_DATA_SCORE


def _compute_shadow_duration(shadow_priority_score: int) -> int:
    """Map shadow_priority_score to recommended shadow observation window."""
    if shadow_priority_score >= SCORE_FAST_THRESHOLD:
        return DURATION_FAST
    if shadow_priority_score >= SCORE_MEDIUM_THRESHOLD:
        return DURATION_MEDIUM
    return DURATION_SLOW


def _build_recommendation(candidate: dict) -> ShadowRecommendation:
    """Build a ShadowRecommendation from one EP candidate dict."""
    ct             = _safe_str(candidate.get("candidate_type"))
    src            = _safe_str(candidate.get("source_module"))
    ev_status      = _safe_str(candidate.get("status"), EP_INSUFFICIENT)
    ev_score       = _safe_int(candidate.get("evidence_score"), 0)
    shadow_status  = _map_shadow_status(ev_status)
    shadow_priority = _compute_shadow_priority(ev_score, shadow_status)
    shadow_duration = _compute_shadow_duration(shadow_priority)
    return ShadowRecommendation(
        candidate_type=ct,
        source_module=src,
        evidence_status=ev_status,
        shadow_status=shadow_status,
        evidence_score=ev_score,
        shadow_priority_score=shadow_priority,
        shadow_duration_days=shadow_duration,
    )


# ── Main entry point (Task 1-5) ───────────────────────────────────────────────

def run_shadow_recommendation(
    ep_file:     Path,
    output_file: Path,
    today:       str,
) -> ShadowRecommendationOutput:
    """
    Read latest EP record, build shadow recommendations, append to JSONL.
    Never raises. FAIL_OPEN on all errors.
    """
    try:
        ep_record = _load_latest_record(ep_file)
        candidates = ep_record.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []

        recommendations: List[ShadowRecommendation] = [
            _build_recommendation(c) for c in candidates
        ]

        n_shadow_ready      = sum(1 for r in recommendations if r.shadow_status == SHADOW_READY)
        n_continue_observe  = sum(1 for r in recommendations if r.shadow_status == CONTINUE_OBSERVE)
        n_wait_data         = sum(1 for r in recommendations if r.shadow_status == WAIT_DATA)

        ready = [r for r in recommendations if r.shadow_status == SHADOW_READY]
        top_shadow_candidate = (
            max(ready, key=lambda r: r.shadow_priority_score).candidate_type
            if ready else ""
        )

        output = ShadowRecommendationOutput(
            date=today,
            n_shadow_ready=n_shadow_ready,
            n_continue_observe=n_continue_observe,
            n_wait_data=n_wait_data,
            top_shadow_candidate=top_shadow_candidate,
            recommendations=recommendations,
        )

        append_recommendation_record(output, output_file)
        return output

    except Exception as exc:
        logger.warning("[SR] run_shadow_recommendation failed: %s", exc)
        return ShadowRecommendationOutput(
            date=today,
            n_shadow_ready=0,
            n_continue_observe=0,
            n_wait_data=0,
            top_shadow_candidate="",
            recommendations=[],
        )


def append_recommendation_record(output: ShadowRecommendationOutput, path: Path) -> None:
    """Serialize to JSONL (append-only). FAIL_OPEN."""
    try:
        record = {
            "date": output.date,
            "n_shadow_ready": output.n_shadow_ready,
            "n_continue_observe": output.n_continue_observe,
            "n_wait_data": output.n_wait_data,
            "top_shadow_candidate": output.top_shadow_candidate,
            "recommendations": [
                {
                    "candidate_type":       r.candidate_type,
                    "source_module":        r.source_module,
                    "evidence_status":      r.evidence_status,
                    "shadow_status":        r.shadow_status,
                    "evidence_score":       r.evidence_score,
                    "shadow_priority_score": r.shadow_priority_score,
                    "shadow_duration_days": r.shadow_duration_days,
                }
                for r in output.recommendations
            ],
            "schema_version": output.schema_version,
        }
        _append_jsonl(record, path)
    except Exception as exc:
        logger.warning("[SR] append_recommendation_record failed: %s", exc)


# ── Weekly governance summary (Task 7) ────────────────────────────────────────

def compute_weekly_governance_summary(
    sr_file:      Path,
    today:        str,
    lookback_days: int = 7,
) -> WeeklyGovernanceSummary:
    """
    Aggregate SR JSONL records from the last lookback_days.

    Returns:
      shadow_ready_count  — total SHADOW_READY recommendations in the window
      top_shadow_candidate — most common SHADOW_READY candidate_type; "" if none
      avg_shadow_priority  — mean shadow_priority_score for SHADOW_READY recs

    FAIL_OPEN: returns zero-state summary on any error.
    """
    try:
        cutoff = (
            datetime.strptime(today, "%Y-%m-%d") - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")
        records = _load_all_jsonl(sr_file)

        ready_scores: List[int] = []
        candidate_counter: Counter = Counter()

        for rec in records:
            rec_date = _safe_str(rec.get("date", ""))
            if rec_date < cutoff:
                continue
            for r in rec.get("recommendations", []):
                if _safe_str(r.get("shadow_status")) == SHADOW_READY:
                    score = _safe_int(r.get("shadow_priority_score"), 0)
                    ct    = _safe_str(r.get("candidate_type"))
                    ready_scores.append(score)
                    candidate_counter[ct] += 1

        shadow_ready_count   = len(ready_scores)
        top_shadow_candidate = candidate_counter.most_common(1)[0][0] if candidate_counter else ""
        avg_shadow_priority  = (
            sum(ready_scores) / len(ready_scores) if ready_scores else 0.0
        )

        return WeeklyGovernanceSummary(
            lookback_days=lookback_days,
            shadow_ready_count=shadow_ready_count,
            top_shadow_candidate=top_shadow_candidate,
            avg_shadow_priority=round(avg_shadow_priority, 2),
        )

    except Exception as exc:
        logger.warning("[SR] compute_weekly_governance_summary failed: %s", exc)
        return WeeklyGovernanceSummary(
            lookback_days=lookback_days,
            shadow_ready_count=0,
            top_shadow_candidate="",
            avg_shadow_priority=0.0,
        )


# ── Report formatter (Task 6) ──────────────────────────────────────────────────

_STATUS_TAG = {
    SHADOW_READY:     "[SHADOW_READY]    ",
    CONTINUE_OBSERVE: "[CONTINUE_OBSERVE]",
    WAIT_DATA:        "[WAIT_DATA]       ",
}


def format_recommendation_report(output: ShadowRecommendationOutput) -> str:
    """Return SHADOW RECOMMENDATIONS report section string."""
    if not output.recommendations:
        return ""
    lines: list[str] = []
    sep = "═" * 60
    lines.append(sep)
    lines.append("SHADOW RECOMMENDATIONS")
    lines.append("─" * 60)
    lines.append(
        f"Shadow Ready: {output.n_shadow_ready}"
        f"  │  Continue Observe: {output.n_continue_observe}"
        f"  │  Wait Data: {output.n_wait_data}"
    )
    if output.top_shadow_candidate:
        lines.append(f"Top candidate: {output.top_shadow_candidate}")
    lines.append("")

    by_type = {r.candidate_type: r for r in output.recommendations}
    for ct in _CANDIDATE_ORDER:
        r = by_type.get(ct)
        if r is None:
            continue
        tag      = _STATUS_TAG.get(r.shadow_status, "[?]")
        label    = r.candidate_type.ljust(22)
        priority = f"priority={r.shadow_priority_score:3d}"
        duration = f"shadow={r.shadow_duration_days}d"
        src      = f"({r.source_module})"
        ev_score = f"ev={r.evidence_score}"
        lines.append(f"  {tag}  {label}  {priority}  {duration}  {src}  {ev_score}")
    lines.append(sep)
    return "\n".join(lines)
