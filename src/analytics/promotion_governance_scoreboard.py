"""
promotion_governance_scoreboard.py — Promotion Governance Scoreboard (Phase 6G-Lite)

Ranks all promotion candidates using evidence, shadow outcomes, readiness, and
deployment impact. Read-only analytics layer.

Inputs (all read-only):
  PROMOTION_READINESS_FILE          — readiness_score per candidate
  PROMOTION_CANDIDATE_REGISTRY_FILE — ACTIVE status filter
  PROMOTION_IMPACT_TRACKER_FILE     — impact_delta per candidate
  SHADOW_OUTCOME_FILE               — success per candidate (aggregated)
  EVIDENCE_PROMOTION_FILE           — evidence_score per candidate

Output:
  PROMOTION_GOVERNANCE_SCOREBOARD_FILE — append-only JSONL
    one record per (date, candidate) per run

Governance score formula (all components → 0-100 range):
  governance_score =
    0.35 * readiness_score          (0-100)
  + 0.25 * evidence_score           (0-100)
  + 0.20 * (success_rate * 100)     (success_rate = fraction 0-1 from shadow outcomes)
  + 0.20 * normalized_impact_score  (clamp(impact_delta*100, 0, 100))

Design:
  observation_only — no signals, no orders, no parameter changes
  fail-open        — never blocks live execution
  idempotent       — (date, candidate) pair never re-added
  O(n)             — linear in JSONL lines
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_JST = timezone(timedelta(hours=9))
SCHEMA_VERSION = "v1"

CLASSIFICATION_PROMOTE_FIRST = "PROMOTE_FIRST"
CLASSIFICATION_PROMOTE_LATER = "PROMOTE_LATER"
CLASSIFICATION_OBSERVE       = "OBSERVE"

SCORE_PROMOTE_FIRST_THRESHOLD = 80.0
SCORE_PROMOTE_LATER_THRESHOLD = 60.0

_W_READINESS   = 0.35
_W_EVIDENCE    = 0.25
_W_SUCCESS     = 0.20
_W_IMPACT      = 0.20

_REGISTRY_STATUS_ACTIVE = "ACTIVE"

_CANDIDATE_ORDER: List[str] = [
    "EXIT_EXTENSION",
    "CAPITAL_CONCENTRATION",
    "PRIORITY",
    "SIGNAL_LEADER",
    "CONTINUATION_SIGNAL",
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ScoreboardRecord:
    date:               str
    candidate:          str
    readiness_score:    float
    evidence_score:     float
    success_rate:       float   # fraction 0-1 (from shadow outcomes)
    impact_delta:       float
    governance_score:   float
    rank:               int
    classification:     str
    schema_version:     str = SCHEMA_VERSION


@dataclass
class ScoreboardSummary:
    top_candidate:          str
    top_governance_score:   float
    n_promote_first:        int
    n_promote_later:        int
    n_observe:              int
    promotion_recommendation: str   # "PROMOTE <top_candidate>" or "" if none qualify


@dataclass
class ScoreboardRunResult:
    date:              str
    new_records:       List[ScoreboardRecord]
    ranked_records:    List[ScoreboardRecord]   # latest per candidate, sorted by rank
    summary:           ScoreboardSummary


# ── I/O helpers ───────────────────────────────────────────────────────────────

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
        logger.debug("[PGS] load_all_jsonl %s failed: %s", path, exc)
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
        logger.warning("[PGS] append_jsonl failed: %s", exc)


# ── Metric loaders ────────────────────────────────────────────────────────────

def _latest_per_candidate(records: List[dict], field: str) -> Dict[str, float]:
    """Return {candidate: latest field value} keyed by _CANDIDATE_ORDER members."""
    latest_date: Dict[str, str]  = {}
    latest_val:  Dict[str, float] = {}
    for r in records:
        ct   = str(r.get("candidate", ""))
        date = str(r.get("date", ""))
        if ct not in _CANDIDATE_ORDER or not date or field not in r:
            continue
        if ct not in latest_date or date > latest_date[ct]:
            latest_date[ct] = date
            latest_val[ct]  = _safe_float(r.get(field), 0.0)
    return latest_val


def _active_candidates(registry_file: Path) -> Set[str]:
    """Return set of candidate names with ACTIVE registry status."""
    active: Set[str] = set()
    for r in _load_all_jsonl(registry_file):
        if str(r.get("registry_status", "")) == _REGISTRY_STATUS_ACTIVE:
            ct = str(r.get("candidate", ""))
            if ct in _CANDIDATE_ORDER:
                active.add(ct)
    return active


def _success_rate_per_candidate(shadow_outcome_file: Path) -> Dict[str, float]:
    """
    Compute success_rate (fraction 0-1) per candidate from shadow outcome records.
    Each record has success: bool.
    """
    counts: Dict[str, int] = {}
    hits:   Dict[str, int] = {}
    for r in _load_all_jsonl(shadow_outcome_file):
        ct = str(r.get("candidate", ""))
        if ct not in _CANDIDATE_ORDER:
            continue
        counts[ct] = counts.get(ct, 0) + 1
        if bool(r.get("success", False)):
            hits[ct] = hits.get(ct, 0) + 1
    return {
        ct: hits.get(ct, 0) / counts[ct]
        for ct in _CANDIDATE_ORDER
        if ct in counts and counts[ct] > 0
    }


# ── Score computation ─────────────────────────────────────────────────────────

def _normalize_impact(impact_delta: float) -> float:
    """clamp(impact_delta * 100, 0, 100)."""
    raw = impact_delta * 100.0
    return max(0.0, min(100.0, raw))


def compute_governance_score(
    readiness_score: float,
    evidence_score:  float,
    success_rate:    float,   # fraction 0-1
    impact_delta:    float,
) -> float:
    """Return governance score in [0, 100]."""
    norm_impact = _normalize_impact(impact_delta)
    score = (
        _W_READINESS * readiness_score
        + _W_EVIDENCE * evidence_score
        + _W_SUCCESS  * (success_rate * 100.0)
        + _W_IMPACT   * norm_impact
    )
    return round(min(100.0, max(0.0, score)), 2)


def _classify(governance_score: float) -> str:
    if governance_score >= SCORE_PROMOTE_FIRST_THRESHOLD:
        return CLASSIFICATION_PROMOTE_FIRST
    if governance_score >= SCORE_PROMOTE_LATER_THRESHOLD:
        return CLASSIFICATION_PROMOTE_LATER
    return CLASSIFICATION_OBSERVE


# ── Idempotency ───────────────────────────────────────────────────────────────

def _load_evaluated_keys(output_file: Path) -> Set[Tuple[str, str]]:
    keys: Set[Tuple[str, str]] = set()
    for r in _load_all_jsonl(output_file):
        date = str(r.get("date", ""))
        ct   = str(r.get("candidate", ""))
        if date and ct:
            keys.add((date, ct))
    return keys


def _load_latest_scoreboard(output_file: Path) -> List[ScoreboardRecord]:
    """Return latest ScoreboardRecord per candidate, sorted by rank ascending."""
    latest: Dict[str, dict] = {}
    for r in _load_all_jsonl(output_file):
        ct   = str(r.get("candidate", ""))
        date = str(r.get("date", ""))
        if ct not in _CANDIDATE_ORDER or not date:
            continue
        if ct not in latest or date > latest[ct].get("date", ""):
            latest[ct] = r

    records = []
    for ct, r in latest.items():
        records.append(ScoreboardRecord(
            date=str(r.get("date", "")),
            candidate=ct,
            readiness_score=_safe_float(r.get("readiness_score"), 0.0),
            evidence_score=_safe_float(r.get("evidence_score"), 0.0),
            success_rate=_safe_float(r.get("success_rate"), 0.0),
            impact_delta=_safe_float(r.get("impact_delta"), 0.0),
            governance_score=_safe_float(r.get("governance_score"), 0.0),
            rank=int(r.get("rank", 0)),
            classification=str(r.get("classification", CLASSIFICATION_OBSERVE)),
        ))
    # sort by rank; unranked (0) go last
    records.sort(key=lambda r: r.rank if r.rank > 0 else 999)
    return records


# ── Summary ───────────────────────────────────────────────────────────────────

def _compute_summary(ranked_records: List[ScoreboardRecord]) -> ScoreboardSummary:
    if not ranked_records:
        return ScoreboardSummary(
            top_candidate="",
            top_governance_score=0.0,
            n_promote_first=0,
            n_promote_later=0,
            n_observe=0,
            promotion_recommendation="",
        )
    top = ranked_records[0]
    n_pf = sum(1 for r in ranked_records if r.classification == CLASSIFICATION_PROMOTE_FIRST)
    n_pl = sum(1 for r in ranked_records if r.classification == CLASSIFICATION_PROMOTE_LATER)
    n_ob = sum(1 for r in ranked_records if r.classification == CLASSIFICATION_OBSERVE)
    rec_str = f"PROMOTE {top.candidate}" if top.classification == CLASSIFICATION_PROMOTE_FIRST else ""
    return ScoreboardSummary(
        top_candidate=top.candidate,
        top_governance_score=round(top.governance_score, 2),
        n_promote_first=n_pf,
        n_promote_later=n_pl,
        n_observe=n_ob,
        promotion_recommendation=rec_str,
    )


# ── Serialization ─────────────────────────────────────────────────────────────

def append_scoreboard_record(record: ScoreboardRecord, path: Path) -> None:
    """Serialize one ScoreboardRecord to JSONL (append-only). FAIL_OPEN."""
    try:
        data = {
            "date":             record.date,
            "candidate":        record.candidate,
            "readiness_score":  record.readiness_score,
            "evidence_score":   record.evidence_score,
            "success_rate":     record.success_rate,
            "impact_delta":     record.impact_delta,
            "governance_score": record.governance_score,
            "rank":             record.rank,
            "classification":   record.classification,
            "schema_version":   record.schema_version,
        }
        _append_jsonl(data, path)
    except Exception as exc:
        logger.warning("[PGS] append_scoreboard_record failed: %s", exc)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_promotion_governance_scoreboard(
    readiness_file:  Path,
    registry_file:   Path,
    impact_file:     Path,
    shadow_file:     Path,
    evidence_file:   Path,
    output_file:     Path,
    today:           str,
) -> ScoreboardRunResult:
    """
    For each candidate in _CANDIDATE_ORDER, compute governance score and rank.

    Idempotent: (date, candidate) pair never re-written.
    Never raises. FAIL_OPEN on all errors.
    """
    try:
        evaluated = _load_evaluated_keys(output_file)

        # load per-candidate metrics
        readiness_scores = _latest_per_candidate(
            _load_all_jsonl(readiness_file), "readiness_score"
        )
        evidence_scores = _latest_per_candidate(
            _load_all_jsonl(evidence_file), "evidence_score"
        )
        impact_deltas = _latest_per_candidate(
            _load_all_jsonl(impact_file), "impact_delta"
        )
        success_rates = _success_rate_per_candidate(shadow_file)
        active_set    = _active_candidates(registry_file)

        # build unranked candidates
        unranked: List[Tuple[str, float, float, float, float, float]] = []
        # (candidate, readiness, evidence, success_rate, impact_delta, governance_score)
        for ct in _CANDIDATE_ORDER:
            if (today, ct) in evaluated:
                continue
            readiness = readiness_scores.get(ct, 0.0)
            evidence  = evidence_scores.get(ct, 0.0)
            sr        = success_rates.get(ct, 0.0)
            impact    = impact_deltas.get(ct, 0.0)
            gov_score = compute_governance_score(readiness, evidence, sr, impact)
            unranked.append((ct, readiness, evidence, sr, impact, gov_score))

        # sort by governance_score descending for ranking
        unranked.sort(key=lambda x: x[5], reverse=True)

        new_records: List[ScoreboardRecord] = []
        for rank_i, (ct, readiness, evidence, sr, impact, gov_score) in enumerate(unranked, start=1):
            classification = _classify(gov_score)
            record = ScoreboardRecord(
                date=today,
                candidate=ct,
                readiness_score=readiness,
                evidence_score=evidence,
                success_rate=sr,
                impact_delta=impact,
                governance_score=gov_score,
                rank=rank_i,
                classification=classification,
            )
            append_scoreboard_record(record, output_file)
            new_records.append(record)
            evaluated.add((today, ct))

        ranked_records = _load_latest_scoreboard(output_file)
        summary        = _compute_summary(ranked_records)

        return ScoreboardRunResult(
            date=today,
            new_records=new_records,
            ranked_records=ranked_records,
            summary=summary,
        )

    except Exception as exc:
        logger.warning("[PGS] run_promotion_governance_scoreboard failed: %s", exc)
        return ScoreboardRunResult(
            date=today,
            new_records=[],
            ranked_records=[],
            summary=ScoreboardSummary(
                top_candidate="",
                top_governance_score=0.0,
                n_promote_first=0,
                n_promote_later=0,
                n_observe=0,
                promotion_recommendation="",
            ),
        )


# ── Report formatter ──────────────────────────────────────────────────────────

def format_scoreboard_report(
    ranked_records: List[ScoreboardRecord],
    summary:        ScoreboardSummary,
) -> str:
    """Return PROMOTION GOVERNANCE SCOREBOARD report section string."""
    if not ranked_records:
        return ""
    lines: List[str] = []
    sep = "═" * 60
    lines.append(sep)
    lines.append("PROMOTION GOVERNANCE SCOREBOARD")
    lines.append("─" * 60)
    lines.append(f"{'Rank':<5} {'Candidate':<22} {'Score':>6}  Classification")
    lines.append("─" * 60)
    for r in ranked_records:
        lines.append(
            f"  #{r.rank:<4} {r.candidate:<22} {r.governance_score:>6.1f}  {r.classification}"
        )
    lines.append("─" * 60)
    lines.append(f"Top Candidate: {summary.top_candidate}"
                 f"  (score={summary.top_governance_score:.1f})")
    if summary.promotion_recommendation:
        lines.append(f"Promotion Recommendation: {summary.promotion_recommendation}")
    lines.append(f"PROMOTE_FIRST={summary.n_promote_first}"
                 f"  PROMOTE_LATER={summary.n_promote_later}"
                 f"  OBSERVE={summary.n_observe}")
    lines.append(sep)
    return "\n".join(lines)
