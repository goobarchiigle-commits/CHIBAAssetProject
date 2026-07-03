"""
evidence_promotion.py — Evidence Promotion Engine

Reads the most recent KPI records from five observation modules and evaluates
whether each has accumulated sufficient evidence to become a promotion candidate.

Source modules (read-only):
  HDC — Hold Duration Calibration Intelligence
  CCS — Capital Concentration Shadow Intelligence
  PCI — Priority Calibration Intelligence
  SA  — Signal Attribution Intelligence
  FCO — Forward Continuation Outcome Intelligence

Candidate types:
  EXIT_EXTENSION       (HDC)  — suppression adds value to hold duration
  CAPITAL_CONCENTRATION (CCS) — priority-proportional weighting beats equal-weight
  PRIORITY              (PCI) — priority_score monotonically predicts future return
  SIGNAL_LEADER         (SA)  — a single signal consistently leads future return
  CONTINUATION_SIGNAL   (FCO) — a continuation signal leads forward 5d return

candidate_status:
  INSUFFICIENT_DATA — not enough records to evaluate
  OBSERVE           — data present; KPIs exist but thresholds not met
  PROMOTABLE        — all promotion rules satisfied

evidence_score (0-100): weighted sum of KPI contributions vs target thresholds.

Design:
  observation_only — no signals, no orders, no execution mutation
  fail-open        — every stage wrapped; never blocks live execution
  O(n)             — n = number of lines in each source JSONL (reads only latest)
  append-only JSONL — 1 record per run; never rewritten
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_JST = timezone(timedelta(hours=9))
SCHEMA_VERSION = "v1"

# ── Candidate type constants ──────────────────────────────────────────────────
CT_EXIT_EXTENSION        = "EXIT_EXTENSION"
CT_CAPITAL_CONCENTRATION = "CAPITAL_CONCENTRATION"
CT_PRIORITY              = "PRIORITY"
CT_SIGNAL_LEADER         = "SIGNAL_LEADER"
CT_CONTINUATION_SIGNAL   = "CONTINUATION_SIGNAL"

# ── Status constants ──────────────────────────────────────────────────────────
STATUS_PROMOTABLE        = "PROMOTABLE"
STATUS_OBSERVE           = "OBSERVE"
STATUS_INSUFFICIENT      = "INSUFFICIENT_DATA"

# ── Promotion thresholds (Task 2) ─────────────────────────────────────────────
# EXIT_EXTENSION (HDC)
HDC_DELTA_MIN:            float = 0.010   # suppression_return_delta > 1%
HDC_SAMPLE_MIN:           int   = 20      # n_materialized >= 20
HDC_INSUFFICIENT_BELOW:   int   = 5       # n_materialized < 5 → no data

# CAPITAL_CONCENTRATION (CCS)
CCS_ALPHA_MIN:            float = 0.020   # mean_concentration_alpha > 2%
CCS_DATES_MIN:            int   = 30      # n_dates_analyzed >= 30
CCS_INSUFFICIENT_BELOW:   int   = 10      # n_dates_analyzed < 10 → no data

# PRIORITY (PCI)
PCI_MONO_MIN:             float = 0.50    # priority_monotonicity > 0.50
PCI_CALIB_MAX:            float = 0.25    # priority_calibration_error < 0.25
PCI_INSUFFICIENT_BELOW:   int   = 10      # n_materialized < 10 → no data

# SIGNAL_LEADER (SA)
SA_IR_MIN:                float = 0.010   # abs(top_signal_ir) > 1%
SA_SAMPLE_MIN:            int   = 20      # n_materialized >= 20
SA_INSUFFICIENT_BELOW:    int   = 5       # n_materialized < 5 → no data

# CONTINUATION_SIGNAL (FCO)
FCO_IR_MIN:               float = 0.010   # top_signal_ir > 1%
FCO_SAMPLE_MIN:           int   = 20      # n_materialized >= 20
FCO_INSUFFICIENT_BELOW:   int   = 5       # n_materialized < 5 → no data

# ── Evidence score full-credit targets ───────────────────────────────────────
HDC_DELTA_FULL:           float = 0.020   # 2% delta = full delta contribution
HDC_SPREAD_FULL:          float = 0.020   # 2% ev_spread = full spread contribution
HDC_SAMPLE_FULL:          int   = 40
CCS_ALPHA_FULL:           float = 0.040   # 4% alpha = full alpha contribution
CCS_DATES_FULL:           int   = 60
SA_IR_FULL:               float = 0.020
SA_SAMPLE_FULL:           int   = 40
FCO_IR_FULL:              float = 0.020
FCO_SAMPLE_FULL:          int   = 40
PCI_BUCKET_FULL:          int   = 10


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class EvidenceCandidate:
    candidate_type:  str          # e.g. "EXIT_EXTENSION"
    source_module:   str          # "HDC", "CCS", "PCI", "SA", "FCO"
    status:          str          # "PROMOTABLE" | "OBSERVE" | "INSUFFICIENT_DATA"
    evidence_score:  int          # 0-100
    supporting_kpis: Dict[str, Any]
    rules:           List[str]    # human-readable rule descriptions
    rules_met:       List[bool]   # parallel to rules[]


@dataclass
class EvidencePromotionOutput:
    date:           str
    candidates:     List[EvidenceCandidate]
    n_promotable:   int
    n_observe:      int
    n_insufficient: int
    top_candidate:  str           # candidate_type with highest score among PROMOTABLE, "" if none
    schema_version: str = SCHEMA_VERSION


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _safe(v: Any, fallback: float = 0.0) -> float:
    if v is None:
        return fallback
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


def _safe_int(v: Any, fallback: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return fallback


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _load_latest_record(path: Path) -> dict:
    """Return the last valid JSON line from a JSONL file. Returns {} on any error."""
    try:
        if not path.exists():
            return {}
        text = path.read_text(encoding="utf-8")
        last: dict = {}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except Exception:
                pass
        return last
    except Exception as exc:
        logger.debug("[EP] load_latest_record %s failed: %s", path, exc)
        return {}


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
        logger.warning("[EP] append_jsonl failed: %s", exc)


# ── Candidate evaluators ──────────────────────────────────────────────────────

def _evaluate_exit_extension(rec: dict) -> EvidenceCandidate:
    """HDC → EXIT_EXTENSION candidate."""
    n_mat    = _safe_int(rec.get("n_materialized"), 0)
    delta    = _safe(rec.get("suppression_return_delta"), 0.0)
    spread   = _safe(rec.get("duration_ev_spread"), 0.0)
    best_bkt = str(rec.get("best_duration_bucket", ""))

    # INSUFFICIENT check
    if n_mat < HDC_INSUFFICIENT_BELOW:
        return EvidenceCandidate(
            candidate_type=CT_EXIT_EXTENSION,
            source_module="HDC",
            status=STATUS_INSUFFICIENT,
            evidence_score=0,
            supporting_kpis={
                "suppression_return_delta": round(delta, 6),
                "n_materialized": n_mat,
                "duration_ev_spread": round(spread, 6),
                "best_duration_bucket": best_bkt,
            },
            rules=[
                f"suppression_return_delta > {HDC_DELTA_MIN:.1%}",
                f"n_materialized >= {HDC_SAMPLE_MIN}",
            ],
            rules_met=[False, False],
        )

    # Rule evaluation
    r1 = delta > HDC_DELTA_MIN
    r2 = n_mat >= HDC_SAMPLE_MIN

    # Evidence score: delta 50pts + sample 30pts + spread 20pts
    score_delta  = _clamp(delta / HDC_DELTA_FULL) * 50.0
    score_sample = _clamp(n_mat / HDC_SAMPLE_FULL) * 30.0
    score_spread = _clamp(spread / HDC_SPREAD_FULL) * 20.0
    ev_score = int(round(score_delta + score_sample + score_spread))

    status = STATUS_PROMOTABLE if (r1 and r2) else STATUS_OBSERVE
    return EvidenceCandidate(
        candidate_type=CT_EXIT_EXTENSION,
        source_module="HDC",
        status=status,
        evidence_score=ev_score,
        supporting_kpis={
            "suppression_return_delta": round(delta, 6),
            "n_materialized": n_mat,
            "duration_ev_spread": round(spread, 6),
            "best_duration_bucket": best_bkt,
        },
        rules=[
            f"suppression_return_delta > {HDC_DELTA_MIN:.1%}",
            f"n_materialized >= {HDC_SAMPLE_MIN}",
        ],
        rules_met=[r1, r2],
    )


def _evaluate_capital_concentration(rec: dict) -> EvidenceCandidate:
    """CCS → CAPITAL_CONCENTRATION candidate."""
    n_dates  = _safe_int(rec.get("n_dates_analyzed"), 0)
    n_mat    = _safe_int(rec.get("n_materialized"), 0)
    alpha    = _safe(rec.get("mean_concentration_alpha"), 0.0)
    pos_rate = _safe(rec.get("positive_alpha_rate"), 0.0)

    if n_dates < CCS_INSUFFICIENT_BELOW:
        return EvidenceCandidate(
            candidate_type=CT_CAPITAL_CONCENTRATION,
            source_module="CCS",
            status=STATUS_INSUFFICIENT,
            evidence_score=0,
            supporting_kpis={
                "mean_concentration_alpha": round(alpha, 6),
                "positive_alpha_rate": round(pos_rate, 4),
                "n_dates_analyzed": n_dates,
                "n_materialized": n_mat,
            },
            rules=[
                f"mean_concentration_alpha > {CCS_ALPHA_MIN:.1%}",
                f"n_dates_analyzed >= {CCS_DATES_MIN}",
            ],
            rules_met=[False, False],
        )

    r1 = alpha > CCS_ALPHA_MIN
    r2 = n_dates >= CCS_DATES_MIN

    score_alpha  = _clamp(alpha / CCS_ALPHA_FULL) * 50.0
    score_rate   = pos_rate * 30.0
    score_dates  = _clamp(n_dates / CCS_DATES_FULL) * 20.0
    ev_score = int(round(score_alpha + score_rate + score_dates))

    status = STATUS_PROMOTABLE if (r1 and r2) else STATUS_OBSERVE
    return EvidenceCandidate(
        candidate_type=CT_CAPITAL_CONCENTRATION,
        source_module="CCS",
        status=status,
        evidence_score=ev_score,
        supporting_kpis={
            "mean_concentration_alpha": round(alpha, 6),
            "positive_alpha_rate": round(pos_rate, 4),
            "n_dates_analyzed": n_dates,
            "n_materialized": n_mat,
        },
        rules=[
            f"mean_concentration_alpha > {CCS_ALPHA_MIN:.1%}",
            f"n_dates_analyzed >= {CCS_DATES_MIN}",
        ],
        rules_met=[r1, r2],
    )


def _evaluate_priority(rec: dict) -> EvidenceCandidate:
    """PCI → PRIORITY candidate."""
    n_mat        = _safe_int(rec.get("n_materialized"), 0)
    mono         = _safe(rec.get("priority_monotonicity"), 0.0)
    calib_err    = _safe(rec.get("priority_calibration_error"), 1.0)
    n_buckets    = _safe_int(rec.get("n_buckets_populated"), 0)

    if n_mat < PCI_INSUFFICIENT_BELOW:
        return EvidenceCandidate(
            candidate_type=CT_PRIORITY,
            source_module="PCI",
            status=STATUS_INSUFFICIENT,
            evidence_score=0,
            supporting_kpis={
                "priority_monotonicity": round(mono, 4),
                "priority_calibration_error": round(calib_err, 4),
                "n_buckets_populated": n_buckets,
                "n_materialized": n_mat,
            },
            rules=[
                f"priority_monotonicity > {PCI_MONO_MIN}",
                f"priority_calibration_error < {PCI_CALIB_MAX}",
            ],
            rules_met=[False, False],
        )

    r1 = mono > PCI_MONO_MIN
    r2 = calib_err < PCI_CALIB_MAX

    score_mono   = _clamp(mono, 0.0, 1.0) * 50.0
    score_calib  = _clamp(1.0 - calib_err, 0.0, 1.0) * 30.0
    score_bucket = _clamp(n_buckets / PCI_BUCKET_FULL) * 20.0
    ev_score = int(round(score_mono + score_calib + score_bucket))

    status = STATUS_PROMOTABLE if (r1 and r2) else STATUS_OBSERVE
    return EvidenceCandidate(
        candidate_type=CT_PRIORITY,
        source_module="PCI",
        status=status,
        evidence_score=ev_score,
        supporting_kpis={
            "priority_monotonicity": round(mono, 4),
            "priority_calibration_error": round(calib_err, 4),
            "n_buckets_populated": n_buckets,
            "n_materialized": n_mat,
        },
        rules=[
            f"priority_monotonicity > {PCI_MONO_MIN}",
            f"priority_calibration_error < {PCI_CALIB_MAX}",
        ],
        rules_met=[r1, r2],
    )


def _evaluate_signal_leader(rec: dict) -> EvidenceCandidate:
    """SA → SIGNAL_LEADER candidate."""
    n_mat      = _safe_int(rec.get("n_materialized"), 0)
    ir         = _safe(rec.get("top_signal_ir"), 0.0)
    top_signal = str(rec.get("top_signal", ""))

    if n_mat < SA_INSUFFICIENT_BELOW:
        return EvidenceCandidate(
            candidate_type=CT_SIGNAL_LEADER,
            source_module="SA",
            status=STATUS_INSUFFICIENT,
            evidence_score=0,
            supporting_kpis={
                "top_signal": top_signal,
                "top_signal_ir": round(ir, 6),
                "n_materialized": n_mat,
            },
            rules=[
                f"abs(top_signal_ir) > {SA_IR_MIN:.1%}",
                f"n_materialized >= {SA_SAMPLE_MIN}",
            ],
            rules_met=[False, False],
        )

    r1 = abs(ir) > SA_IR_MIN
    r2 = n_mat >= SA_SAMPLE_MIN

    score_ir     = _clamp(abs(ir) / SA_IR_FULL) * 60.0
    score_sample = _clamp(n_mat / SA_SAMPLE_FULL) * 40.0
    ev_score = int(round(score_ir + score_sample))

    status = STATUS_PROMOTABLE if (r1 and r2) else STATUS_OBSERVE
    return EvidenceCandidate(
        candidate_type=CT_SIGNAL_LEADER,
        source_module="SA",
        status=status,
        evidence_score=ev_score,
        supporting_kpis={
            "top_signal": top_signal,
            "top_signal_ir": round(ir, 6),
            "n_materialized": n_mat,
        },
        rules=[
            f"abs(top_signal_ir) > {SA_IR_MIN:.1%}",
            f"n_materialized >= {SA_SAMPLE_MIN}",
        ],
        rules_met=[r1, r2],
    )


def _evaluate_continuation_signal(rec: dict) -> EvidenceCandidate:
    """FCO → CONTINUATION_SIGNAL candidate."""
    n_mat      = _safe_int(rec.get("n_materialized"), 0)
    ir         = _safe(rec.get("top_signal_ir"), 0.0)
    top_signal = str(rec.get("top_signal", ""))

    if n_mat < FCO_INSUFFICIENT_BELOW:
        return EvidenceCandidate(
            candidate_type=CT_CONTINUATION_SIGNAL,
            source_module="FCO",
            status=STATUS_INSUFFICIENT,
            evidence_score=0,
            supporting_kpis={
                "top_signal": top_signal,
                "top_signal_ir": round(ir, 6),
                "n_materialized": n_mat,
            },
            rules=[
                f"top_signal_ir > {FCO_IR_MIN:.1%}",
                f"n_materialized >= {FCO_SAMPLE_MIN}",
            ],
            rules_met=[False, False],
        )

    r1 = ir > FCO_IR_MIN
    r2 = n_mat >= FCO_SAMPLE_MIN

    score_ir     = _clamp(ir / FCO_IR_FULL, 0.0, 1.0) * 60.0
    score_sample = _clamp(n_mat / FCO_SAMPLE_FULL) * 40.0
    ev_score = int(round(score_ir + score_sample))

    status = STATUS_PROMOTABLE if (r1 and r2) else STATUS_OBSERVE
    return EvidenceCandidate(
        candidate_type=CT_CONTINUATION_SIGNAL,
        source_module="FCO",
        status=status,
        evidence_score=ev_score,
        supporting_kpis={
            "top_signal": top_signal,
            "top_signal_ir": round(ir, 6),
            "n_materialized": n_mat,
        },
        rules=[
            f"top_signal_ir > {FCO_IR_MIN:.1%}",
            f"n_materialized >= {FCO_SAMPLE_MIN}",
        ],
        rules_met=[r1, r2],
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def run_evidence_promotion(
    hdc_file: Path,
    ccs_file: Path,
    pci_file: Path,
    sa_file:  Path,
    fco_file: Path,
    output_file: Path,
    today: str,
) -> EvidencePromotionOutput:
    """
    Read latest KPI record from each source module and evaluate candidates.
    Never raises. FAIL_OPEN on all errors.
    """
    try:
        hdc_rec = _load_latest_record(hdc_file)
        ccs_rec = _load_latest_record(ccs_file)
        pci_rec = _load_latest_record(pci_file)
        sa_rec  = _load_latest_record(sa_file)
        fco_rec = _load_latest_record(fco_file)

        candidates: List[EvidenceCandidate] = [
            _evaluate_exit_extension(hdc_rec),
            _evaluate_capital_concentration(ccs_rec),
            _evaluate_priority(pci_rec),
            _evaluate_signal_leader(sa_rec),
            _evaluate_continuation_signal(fco_rec),
        ]

        n_promotable   = sum(1 for c in candidates if c.status == STATUS_PROMOTABLE)
        n_observe      = sum(1 for c in candidates if c.status == STATUS_OBSERVE)
        n_insufficient = sum(1 for c in candidates if c.status == STATUS_INSUFFICIENT)

        # top_candidate: highest evidence_score among PROMOTABLE
        promotable = [c for c in candidates if c.status == STATUS_PROMOTABLE]
        top_candidate = (
            max(promotable, key=lambda c: c.evidence_score).candidate_type
            if promotable else ""
        )

        output = EvidencePromotionOutput(
            date=today,
            candidates=candidates,
            n_promotable=n_promotable,
            n_observe=n_observe,
            n_insufficient=n_insufficient,
            top_candidate=top_candidate,
        )

        append_promotion_record(output, output_file)
        return output

    except Exception as exc:
        logger.warning("[EP] run_evidence_promotion failed: %s", exc)
        return EvidencePromotionOutput(
            date=today,
            candidates=[],
            n_promotable=0,
            n_observe=0,
            n_insufficient=0,
            top_candidate="",
        )


def append_promotion_record(output: EvidencePromotionOutput, path: Path) -> None:
    """Serialize output to JSONL (append-only). FAIL_OPEN."""
    try:
        record = {
            "date": output.date,
            "n_promotable": output.n_promotable,
            "n_observe": output.n_observe,
            "n_insufficient": output.n_insufficient,
            "top_candidate": output.top_candidate,
            "candidates": [
                {
                    "candidate_type": c.candidate_type,
                    "source_module": c.source_module,
                    "status": c.status,
                    "evidence_score": c.evidence_score,
                    "supporting_kpis": c.supporting_kpis,
                    "rules": c.rules,
                    "rules_met": c.rules_met,
                }
                for c in output.candidates
            ],
            "schema_version": output.schema_version,
        }
        _append_jsonl(record, path)
    except Exception as exc:
        logger.warning("[EP] append_promotion_record failed: %s", exc)


# ── Report formatter ──────────────────────────────────────────────────────────

_STATUS_TAG = {
    STATUS_PROMOTABLE: "[PROMOTABLE]  ",
    STATUS_OBSERVE:    "[OBSERVE]     ",
    STATUS_INSUFFICIENT: "[INSUFFICIENT]",
}

_CANDIDATE_ORDER = [
    CT_EXIT_EXTENSION,
    CT_CAPITAL_CONCENTRATION,
    CT_PRIORITY,
    CT_SIGNAL_LEADER,
    CT_CONTINUATION_SIGNAL,
]


def _kpi_summary(c: EvidenceCandidate) -> str:
    kpis = c.supporting_kpis
    if c.candidate_type == CT_EXIT_EXTENSION:
        delta = kpis.get("suppression_return_delta", 0.0)
        spread = kpis.get("duration_ev_spread", 0.0)
        n = kpis.get("n_materialized", 0)
        bkt = kpis.get("best_duration_bucket", "?")
        return f"delta={delta:+.2%}  spread={spread:.2%}  best={bkt}  n={n}"
    if c.candidate_type == CT_CAPITAL_CONCENTRATION:
        alpha = kpis.get("mean_concentration_alpha", 0.0)
        rate  = kpis.get("positive_alpha_rate", 0.0)
        nd    = kpis.get("n_dates_analyzed", 0)
        return f"alpha={alpha:+.2%}  pos_rate={rate:.0%}  n_dates={nd}"
    if c.candidate_type == CT_PRIORITY:
        mono = kpis.get("priority_monotonicity", 0.0)
        cerr = kpis.get("priority_calibration_error", 0.0)
        nb   = kpis.get("n_buckets_populated", 0)
        return f"mono={mono:+.3f}  calib_err={cerr:.3f}  buckets={nb}"
    if c.candidate_type == CT_SIGNAL_LEADER:
        sig = kpis.get("top_signal", "?")
        ir  = kpis.get("top_signal_ir", 0.0)
        n   = kpis.get("n_materialized", 0)
        return f"top={sig}  IR={ir:+.4f}  n={n}"
    if c.candidate_type == CT_CONTINUATION_SIGNAL:
        sig = kpis.get("top_signal", "?")
        ir  = kpis.get("top_signal_ir", 0.0)
        n   = kpis.get("n_materialized", 0)
        return f"top={sig}  IR={ir:+.4f}  n={n}"
    return ""


def format_promotion_report(output: EvidencePromotionOutput) -> str:
    """Return EVIDENCE PROMOTION report section string."""
    if not output.candidates:
        return ""
    lines: list[str] = []
    sep = "═" * 60
    lines.append(sep)
    lines.append("EVIDENCE PROMOTION")
    lines.append("─" * 60)
    lines.append(
        f"Candidates: {len(output.candidates)}"
        f"  │  Promotable: {output.n_promotable}"
        f"  │  Observe: {output.n_observe}"
        f"  │  Insufficient: {output.n_insufficient}"
    )
    if output.top_candidate:
        lines.append(f"Top candidate: {output.top_candidate}")
    lines.append("")

    by_type = {c.candidate_type: c for c in output.candidates}
    for ct in _CANDIDATE_ORDER:
        c = by_type.get(ct)
        if c is None:
            continue
        tag   = _STATUS_TAG.get(c.status, "[?]          ")
        label = c.candidate_type.ljust(22)
        score = f"score={c.evidence_score:3d}"
        src   = f"({c.source_module})"
        kpis  = _kpi_summary(c)
        lines.append(f"  {tag}  {label}  {score}  {src}  {kpis}")
    lines.append(sep)
    return "\n".join(lines)
