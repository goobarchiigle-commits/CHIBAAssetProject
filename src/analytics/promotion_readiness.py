"""
promotion_readiness.py — Promotion Readiness Engine (Phase 6B-Lite)

Evaluates whether each shadow candidate is ready for real promotion,
based on accumulated outcome evidence.

Inputs (read-only):
  EVIDENCE_PROMOTION_FILE    — evidence_promotion.jsonl
  SHADOW_RECOMMENDATION_FILE — shadow_recommendation.jsonl
  SHADOW_OUTCOME_FILE        — shadow_outcome.jsonl

Output:
  PROMOTION_READINESS_FILE   — promotion_readiness.jsonl
    append-only JSONL; 1 record per (date, candidate) pair

Readiness Score (0-100):
  40% × success_rate_pct     (success_rate × 100, range 0-100)
  30% × evidence_score_norm  (evidence_score / 100, range 0-1)
  30% × sample_size_norm     (sample_size_score / 100, range 0-1)

Status gates:
  PROMOTION_READY: score >= 80 AND n_completed >= 3 AND success_rate >= 60%
  SHADOW_CONTINUE: score >= 60
  MORE_DATA:       otherwise

Design:
  observation_only — no signals, no orders, no execution mutation
  fail-open        — never blocks live execution
  idempotent       — (date, candidate) pairs never re-evaluated
  O(n)             — linear in JSONL lines
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_JST = timezone(timedelta(hours=9))
SCHEMA_VERSION = "v1"

# ── Status constants ──────────────────────────────────────────────────────────
STATUS_PROMOTION_READY = "PROMOTION_READY"
STATUS_SHADOW_CONTINUE = "SHADOW_CONTINUE"
STATUS_MORE_DATA       = "MORE_DATA"

# ── Score thresholds ──────────────────────────────────────────────────────────
SCORE_PROMOTION_THRESHOLD:     float = 80.0
SCORE_SHADOW_CONTINUE_THRESHOLD: float = 60.0
PROMOTION_N_COMPLETED_MIN:     int   = 3
PROMOTION_SUCCESS_RATE_MIN:    float = 0.60   # fraction (not %)

# ── Weight constants ──────────────────────────────────────────────────────────
W_SUCCESS_RATE:   float = 40.0   # weight applied to success_rate_pct (0-100)
W_EVIDENCE_SCORE: float = 30.0   # weight applied to evidence_score (0-100)
W_SAMPLE_SIZE:    float = 30.0   # weight applied to sample_size_score (0-100)

# ── Sample-size score breakpoints (n → score) ─────────────────────────────────
_SAMPLE_BREAKPOINTS: List[Tuple[int, float]] = [
    (0,   0.0),
    (1,  20.0),
    (2,  40.0),
    (3,  60.0),
    (5,  80.0),
    (10, 100.0),
]

# ── Candidate ordering ────────────────────────────────────────────────────────
_CANDIDATE_ORDER: List[str] = [
    "EXIT_EXTENSION",
    "CAPITAL_CONCENTRATION",
    "PRIORITY",
    "SIGNAL_LEADER",
    "CONTINUATION_SIGNAL",
]

# ── Shadow status constant (from SR module) ───────────────────────────────────
_SHADOW_READY = "SHADOW_READY"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class OutcomeStats:
    n_completed:  int
    success_rate: float   # fraction 0.0-1.0
    avg_delta:    float


@dataclass
class PromotionReadinessRecord:
    date:               str
    candidate:          str
    readiness_score:    float   # 0.0-100.0
    success_rate:       float   # percentage 0.0-100.0 (for output)
    n_completed:        int
    avg_delta:          float
    evidence_score:     int     # 0-100 from EP
    sample_size_score:  float   # 0-100 derived
    shadow_status:      str     # latest shadow_status from SR
    status:             str     # PROMOTION_READY | SHADOW_CONTINUE | MORE_DATA
    schema_version:     str = SCHEMA_VERSION


@dataclass
class PromotionReadinessSummary:
    promotion_ready_count: int
    top_ready_candidate:   str    # highest score among PROMOTION_READY; "" if none
    avg_readiness_score:   float


@dataclass
class PromotionReadinessRunResult:
    date:        str
    new_records: List[PromotionReadinessRecord]   # written this run
    snapshot:    List[PromotionReadinessRecord]   # latest per candidate from output
    summary:     PromotionReadinessSummary


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _safe_int(v: Any, fallback: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return fallback


def _safe_float(v: Any, fallback: float = 0.0) -> float:
    if v is None:
        return fallback
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


def _load_all_jsonl(path: Path) -> List[dict]:
    """Load all valid JSON lines. FAIL_OPEN → empty list."""
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
        logger.debug("[PR] load_all_jsonl %s failed: %s", path, exc)
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
        logger.warning("[PR] append_jsonl failed: %s", exc)


# ── Core computations ─────────────────────────────────────────────────────────

def _sample_size_score(n: int) -> float:
    """Map n_completed to sample_size_score (0-100) via linear interpolation."""
    if n <= 0:
        return 0.0
    if n >= _SAMPLE_BREAKPOINTS[-1][0]:
        return _SAMPLE_BREAKPOINTS[-1][1]
    for i in range(len(_SAMPLE_BREAKPOINTS) - 1):
        n0, s0 = _SAMPLE_BREAKPOINTS[i]
        n1, s1 = _SAMPLE_BREAKPOINTS[i + 1]
        if n0 <= n <= n1:
            if n1 == n0:
                return float(s0)
            t = (n - n0) / (n1 - n0)
            return s0 + t * (s1 - s0)
    return 100.0


def _compute_readiness_score(
    success_rate: float,   # fraction 0.0-1.0
    evidence_score: int,   # 0-100
    n_completed: int,
) -> float:
    """Compute readiness_score in range 0.0-100.0, rounded to 1dp."""
    ss = _sample_size_score(n_completed)
    score = (
        W_SUCCESS_RATE   * success_rate               # 0-40
        + W_EVIDENCE_SCORE * (evidence_score / 100.0) # 0-30
        + W_SAMPLE_SIZE    * (ss / 100.0)             # 0-30
    )
    return round(max(0.0, min(100.0, score)), 1)


def _compute_status(
    readiness_score: float,
    n_completed: int,
    success_rate: float,   # fraction 0.0-1.0
) -> str:
    """Derive promotion status from score and gate conditions."""
    if (readiness_score >= SCORE_PROMOTION_THRESHOLD
            and n_completed >= PROMOTION_N_COMPLETED_MIN
            and success_rate >= PROMOTION_SUCCESS_RATE_MIN):
        return STATUS_PROMOTION_READY
    if readiness_score >= SCORE_SHADOW_CONTINUE_THRESHOLD:
        return STATUS_SHADOW_CONTINUE
    return STATUS_MORE_DATA


def _compute_outcome_stats(outcome_records: List[dict], candidate: str) -> OutcomeStats:
    """Compute n_completed / success_rate / avg_delta for one candidate."""
    recs = [r for r in outcome_records if r.get("candidate") == candidate]
    n = len(recs)
    if n == 0:
        return OutcomeStats(n_completed=0, success_rate=0.0, avg_delta=0.0)
    successes = sum(1 for r in recs if r.get("success", False))
    deltas = [_safe_float(r.get("delta"), 0.0) for r in recs]
    return OutcomeStats(
        n_completed=n,
        success_rate=successes / n,
        avg_delta=sum(deltas) / len(deltas) if deltas else 0.0,
    )


def _latest_evidence_score(ep_record: dict, candidate: str) -> int:
    """Extract evidence_score for candidate from latest EP record."""
    for c in ep_record.get("candidates", []):
        if c.get("candidate_type") == candidate:
            return _safe_int(c.get("evidence_score"), 0)
    return 0


def _latest_shadow_status(sr_record: dict, candidate: str) -> str:
    """Extract the latest shadow_status for candidate from latest SR record."""
    for r in sr_record.get("recommendations", []):
        if r.get("candidate_type") == candidate:
            return str(r.get("shadow_status", ""))
    return ""


def _load_evaluated_keys(output_file: Path) -> Set[Tuple[str, str]]:
    """Return set of (date, candidate) already written to output JSONL."""
    keys: Set[Tuple[str, str]] = set()
    for r in _load_all_jsonl(output_file):
        date = str(r.get("date", ""))
        ct   = str(r.get("candidate", ""))
        if date and ct:
            keys.add((date, ct))
    return keys


def _latest_per_candidate(all_records: List[dict]) -> List[PromotionReadinessRecord]:
    """Return the most-recent PromotionReadinessRecord per candidate, in _CANDIDATE_ORDER."""
    latest: Dict[str, dict] = {}
    for r in all_records:
        ct   = str(r.get("candidate", ""))
        date = str(r.get("date", ""))
        if ct and date:
            if ct not in latest or date > latest[ct].get("date", ""):
                latest[ct] = r

    result: List[PromotionReadinessRecord] = []
    for ct in _CANDIDATE_ORDER:
        r = latest.get(ct)
        if r is None:
            continue
        result.append(PromotionReadinessRecord(
            date=str(r.get("date", "")),
            candidate=ct,
            readiness_score=_safe_float(r.get("readiness_score"), 0.0),
            success_rate=_safe_float(r.get("success_rate"), 0.0),
            n_completed=_safe_int(r.get("n_completed"), 0),
            avg_delta=_safe_float(r.get("avg_delta"), 0.0),
            evidence_score=_safe_int(r.get("evidence_score"), 0),
            sample_size_score=_safe_float(r.get("sample_size_score"), 0.0),
            shadow_status=str(r.get("shadow_status", "")),
            status=str(r.get("status", STATUS_MORE_DATA)),
        ))
    return result


def _compute_summary(snapshot: List[PromotionReadinessRecord]) -> PromotionReadinessSummary:
    """Aggregate snapshot into summary KPIs."""
    if not snapshot:
        return PromotionReadinessSummary(
            promotion_ready_count=0,
            top_ready_candidate="",
            avg_readiness_score=0.0,
        )
    ready = [r for r in snapshot if r.status == STATUS_PROMOTION_READY]
    top = max(ready, key=lambda r: r.readiness_score).candidate if ready else ""
    avg = sum(r.readiness_score for r in snapshot) / len(snapshot)
    return PromotionReadinessSummary(
        promotion_ready_count=len(ready),
        top_ready_candidate=top,
        avg_readiness_score=round(avg, 1),
    )


# ── Serialization ─────────────────────────────────────────────────────────────

def append_readiness_record(record: PromotionReadinessRecord, path: Path) -> None:
    """Serialize one record to JSONL (append-only). FAIL_OPEN."""
    try:
        data = {
            "date":              record.date,
            "candidate":         record.candidate,
            "readiness_score":   record.readiness_score,
            "success_rate":      record.success_rate,
            "n_completed":       record.n_completed,
            "avg_delta":         record.avg_delta,
            "evidence_score":    record.evidence_score,
            "sample_size_score": record.sample_size_score,
            "shadow_status":     record.shadow_status,
            "status":            record.status,
            "schema_version":    record.schema_version,
        }
        _append_jsonl(data, path)
    except Exception as exc:
        logger.warning("[PR] append_readiness_record failed: %s", exc)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_promotion_readiness(
    ep_file:      Path,
    sr_file:      Path,
    outcome_file: Path,
    output_file:  Path,
    today:        str,
) -> PromotionReadinessRunResult:
    """
    Evaluate promotion readiness for all 5 candidates.

    For each candidate in _CANDIDATE_ORDER:
      - Skip if (today, candidate) already in output JSONL (idempotent)
      - Compute outcome stats from SHADOW_OUTCOME_FILE
      - Extract evidence_score from latest EP record
      - Compute readiness_score and status
      - Append record to output JSONL

    Never raises. FAIL_OPEN on all errors.
    """
    try:
        ep_records      = _load_all_jsonl(ep_file)
        sr_records      = _load_all_jsonl(sr_file)
        outcome_records = _load_all_jsonl(outcome_file)
        already         = _load_evaluated_keys(output_file)

        latest_ep = ep_records[-1] if ep_records else {}
        latest_sr = sr_records[-1] if sr_records else {}

        new_records: List[PromotionReadinessRecord] = []

        for ct in _CANDIDATE_ORDER:
            if (today, ct) in already:
                continue

            stats       = _compute_outcome_stats(outcome_records, ct)
            ev_score    = _latest_evidence_score(latest_ep, ct)
            sh_status   = _latest_shadow_status(latest_sr, ct)
            ss          = _sample_size_score(stats.n_completed)
            readiness   = _compute_readiness_score(stats.success_rate, ev_score, stats.n_completed)
            status      = _compute_status(readiness, stats.n_completed, stats.success_rate)

            record = PromotionReadinessRecord(
                date=today,
                candidate=ct,
                readiness_score=readiness,
                success_rate=round(stats.success_rate * 100.0, 1),
                n_completed=stats.n_completed,
                avg_delta=round(stats.avg_delta, 6),
                evidence_score=ev_score,
                sample_size_score=round(ss, 1),
                shadow_status=sh_status,
                status=status,
            )
            append_readiness_record(record, output_file)
            new_records.append(record)
            already.add((today, ct))

        all_out  = _load_all_jsonl(output_file)
        snapshot = _latest_per_candidate(all_out)
        summary  = _compute_summary(snapshot)

        return PromotionReadinessRunResult(
            date=today,
            new_records=new_records,
            snapshot=snapshot,
            summary=summary,
        )

    except Exception as exc:
        logger.warning("[PR] run_promotion_readiness failed: %s", exc)
        return PromotionReadinessRunResult(
            date=today,
            new_records=[],
            snapshot=[],
            summary=PromotionReadinessSummary(
                promotion_ready_count=0,
                top_ready_candidate="",
                avg_readiness_score=0.0,
            ),
        )


# ── Report formatter ──────────────────────────────────────────────────────────

_STATUS_TAG = {
    STATUS_PROMOTION_READY: "PROMOTION_READY",
    STATUS_SHADOW_CONTINUE: "SHADOW_CONTINUE",
    STATUS_MORE_DATA:       "MORE_DATA      ",
}


def format_readiness_report(snapshot: List[PromotionReadinessRecord]) -> str:
    """Return PROMOTION READINESS report section string."""
    if not snapshot:
        return ""
    n_ready    = sum(1 for r in snapshot if r.status == STATUS_PROMOTION_READY)
    n_continue = sum(1 for r in snapshot if r.status == STATUS_SHADOW_CONTINUE)
    n_more     = sum(1 for r in snapshot if r.status == STATUS_MORE_DATA)
    lines: List[str] = []
    sep = "═" * 60
    lines.append(sep)
    lines.append("PROMOTION READINESS")
    lines.append("─" * 60)
    lines.append(
        f"Ready: {n_ready}"
        f"  │  Continue: {n_continue}"
        f"  │  More Data: {n_more}"
    )
    lines.append("")
    by_ct = {r.candidate: r for r in snapshot}
    for ct in _CANDIDATE_ORDER:
        r = by_ct.get(ct)
        if r is None:
            continue
        tag   = _STATUS_TAG.get(r.status, r.status)
        label = ct.ljust(24)
        score = f"score={r.readiness_score:.0f}"
        lines.append(f"  {label}  {score}  {tag}")
    lines.append(sep)
    return "\n".join(lines)
