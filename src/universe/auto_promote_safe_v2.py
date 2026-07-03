"""
auto_promote_safe_v2.py — Probation-based automatic promotion (AUTO_PROMOTE_SAFE_V2)

Replaces manual shadow promotion approval with low-risk probation deployment while
preserving fail-closed governance and minimizing runtime overhead.

Runtime constraints:
  - No SQLite, no sklearn, no extra yfinance/OHLCV loads
  - O(n) universe complexity
  - Append-only JSONL telemetry only
  - All score inputs reuse existing cached metrics (T-1 snapshots)

Probation execution restrictions (enforced in run_live_signal.py):
  allocation_multiplier = 0.25
  max_entries_per_symbol = 1
  addon_disabled = True
  continuation_pyramid_disabled = True

Fail policy:
  - run_probation_gate: FAIL_OPEN (errors return empty sets, never block execution)
  - JSONL writes: FAIL_OPEN (log + continue)
  - State reads: FAIL_OPEN (missing files → empty state)
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────
SCHEMA_VERSION = 1

# ── Probation runtime constants ───────────────────────────────────────────────
DEFAULT_PROBATION_DAYS    = 5
PROBATION_COOLDOWN_DAYS   = 10
ALLOCATION_MULTIPLIER     = 0.25   # 25% of normal allocation
MAX_ACTIVE_PROBATION      = 3      # max concurrent probation slots

# ── Promotion gate thresholds ─────────────────────────────────────────────────
GATE_MIN_RSR_PASS         = 8.0    # RSR percentile lower bound (very loose; other gates are tight)
GATE_MAX_PREDICTIVE_RANK  = 3      # must be top-3 by predictive_alpha_score
GATE_MIN_SECTOR_IGNITION  = 90.0   # sector_ignition_score (0-100)
_HIGH_RSR_BYPASS_RSR      = 90.0   # RSR>=90: skip Gate2 predictive_rank check

# ── P2-A gate: UNCLASSIFIED + partial sector ignition ─────────────────────────
P2A_MIN_SECTOR_IGNITION: float = 50.0          # SI lower bound (WF-validated [50,90))
P2A_EXCLUDED_SYMBOLS: FrozenSet[str] = frozenset({"5706.T"})  # OOS-negative (PF=0.22)

# ── Scoring weights (must sum to 1.0) ─────────────────────────────────────────
WEIGHT_RSR_PASS          = 0.30
WEIGHT_PREDICTIVE        = 0.25
WEIGHT_SECTOR_IGNITION   = 0.20
WEIGHT_COMPRESSION       = 0.15
WEIGHT_DRIFT             = 0.10

# ── Demotion triggers ─────────────────────────────────────────────────────────
RSR_RANK_DROP_THRESHOLD        = 15.0   # RSR drop > 15 points since promotion
BREAKOUT_FAIL_RSR_MIN          = 65.0   # RSR < 65 = failed breakout
VOLUME_COLLAPSE_PROXY_RSR_DROP = 20.0   # RSR drop > 20 = volume collapse proxy

# ── Graduation thresholds ─────────────────────────────────────────────────────
GRADUATION_MIN_RSR_RATIO      = 0.90   # current_rsr >= promotion_rsr * 0.90
GRADUATION_MIN_CONTINUATION   = 0.40   # continuation_days / total_days

# ── Status ────────────────────────────────────────────────────────────────────
STATUS_ACTIVE    = "active"
STATUS_GRADUATED = "graduated"
STATUS_DEMOTED   = "demoted"

# ── CB states that block new probation promotions ─────────────────────────────
CB_SEVERE_STATES = {"CB_ACTIVE", "RECOVERY"}

# ── Candidate taxonomy ────────────────────────────────────────────────────────
CANDIDATE_HIGH_RSR        = "HIGH_RSR"
CANDIDATE_EARLY_IGNITION  = "EARLY_IGNITION"
CANDIDATE_CONTINUATION    = "CONTINUATION"
CANDIDATE_MATURE_LEADER   = "MATURE_LEADER"
CANDIDATE_MEAN_REVERSION  = "MEAN_REVERSION"
CANDIDATE_UNCLASSIFIED    = "UNCLASSIFIED"

ALLOWED_CANDIDATE_TYPES: FrozenSet[str] = frozenset({
    CANDIDATE_HIGH_RSR,
    CANDIDATE_EARLY_IGNITION,
    CANDIDATE_CONTINUATION,
})

# Taxonomy thresholds (derived from existing metric ranges; no ML)
MATURE_LEADER_MIN_RSR          = 87.0   # RSR near ceiling
MATURE_LEADER_MAX_COMPRESSION  = 35.0   # low compression = no setup remaining → late entry
MEAN_REVERSION_MAX_RSR         = 55.0   # weak RSR = no trend
MEAN_REVERSION_MAX_IGNITION    = 55.0   # low sector ignition = bounce, not ignition
IGNITION_MIN_SECTOR_IGNITION   = 75.0   # strong ignition required
IGNITION_MAX_PREDICTIVE_RANK   = 5      # top-5 predictive rank
IGNITION_MAX_RSR               = 85.0   # not yet mature
IGNITION_MIN_DRIFT             = 35.0   # drift presence
CONTINUATION_MIN_RSR           = 70.0   # high RSR
CONTINUATION_MIN_COMPRESSION   = 28.0   # some compression setup
CONTINUATION_MAX_RSR           = 88.0   # not maxed out (below MATURE_LEADER_MIN_RSR)


# ─────────────────────────────────────────────────────────────────────────────
# Internal utilities
# ─────────────────────────────────────────────────────────────────────────────

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha8(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:8]


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _normalize(value: float, scale: float = 100.0) -> float:
    if scale <= 0:
        return 0.0
    return max(0.0, min(1.0, value / scale))


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception as exc:
            logger.warning("[V2] JSONL parse error %s: %s", path.name, exc)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProbationRecord:
    """Append-only event record in probation_promotions.jsonl."""
    symbol:                str
    promoted_at:           str           # ISO8601 (UTC)
    promotion_score:       float         # 0-1
    promotion_reason:      List[str]
    probation_days:        int
    status:                str           # active | graduated | demoted
    batch_id:              str
    rsr_at_promotion:      float         # RSR percentile at promotion time
    sector_ignition_score: float         # sector_ignition component (0-100)
    compression_score:     float         # compression_breakout component (0-100)
    drift_score:           float         # persistent_drift component (0-100)
    candidate_type:        str           = CANDIDATE_UNCLASSIFIED
    schema_version:        int           = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProbationOutcome:
    """Append-only daily observation record in probation_outcomes.jsonl."""
    symbol:                  str
    promoted_at:             str
    observation_date:        str         # YYYY-MM-DD
    days_held:               int
    forward_return_3d:       Optional[float]   # null at record time; materialized later
    forward_return_5d:       Optional[float]
    forward_return_10d:      Optional[float]
    max_favorable_excursion: Optional[float]
    max_adverse_excursion:   Optional[float]
    rsr_delta:               float       # current_rsr - rsr_at_promotion
    continuation_days:       int         # count of days symbol remained active
    status:                  str
    schema_version:          int         = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RejectionRecord:
    """Append-only rejection event record in promotion_rejections.jsonl."""
    symbol:            str
    timestamp:         str           # ISO8601 (UTC)
    candidate_type:    str
    promotion_score:   float         # 0-1; 0.0 when gate failed before scoring
    failed_conditions: List[str]
    metrics:           dict          # rsr_pass, predictive_rank, sector_ignition, cb_state, etc.
    batch_id:          str
    schema_version:    int           = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Score loaders (reuse existing cached metrics — no extra OHLCV)
# ─────────────────────────────────────────────────────────────────────────────

def load_latest_predictive_scores(scores_path: Path) -> Dict:
    """
    Load last record from predictive_scores.jsonl.

    Injects "top_candidates" [(sym, score), ...] computed from
    predictive_alpha_score for use by check_promotion_gate().
    Returns empty dict if file missing or unreadable.
    """
    records = _load_jsonl(scores_path)
    if not records:
        return {}
    rec = dict(records[-1])
    # Reconstruct top_candidates from persisted predictive_alpha_score
    pa: Dict[str, float] = rec.get("predictive_alpha_score") or {}
    top_cands = sorted(pa.items(), key=lambda x: (-x[1], x[0]))
    rec["top_candidates"] = top_cands
    return rec


def load_latest_fl_candidates(candidates_file: Path) -> Dict[str, Dict]:
    """
    Load most recent future leader records keyed by symbol.

    Returns {symbol: record_dict} for the latest run_id batch.
    Returns empty dict if file missing or empty.
    """
    records = _load_jsonl(candidates_file)
    if not records:
        return {}
    latest_run_id = records[-1].get("run_id", "")
    out: Dict[str, Dict] = {}
    for r in reversed(records):
        rid = r.get("run_id", "")
        # Collect records from the latest run; stop when run_id changes
        if rid != latest_run_id and out:
            break
        sym = r.get("symbol", "")
        if sym:
            out[sym] = r
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Active state helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_active_probation(path: Path) -> List[ProbationRecord]:
    """
    Load all currently active probation records.

    Reads append-only JSONL; last record per symbol defines current status.
    """
    records = _load_jsonl(path)
    by_sym: Dict[str, dict] = {}
    for r in records:
        sym = r.get("symbol", "")
        if sym:
            by_sym[sym] = r

    active: List[ProbationRecord] = []
    for sym, r in by_sym.items():
        if r.get("status") != STATUS_ACTIVE:
            continue
        try:
            active.append(ProbationRecord(
                symbol=sym,
                promoted_at=r["promoted_at"],
                promotion_score=_safe_float(r.get("promotion_score", 0.0)),
                promotion_reason=r.get("promotion_reason", []),
                probation_days=int(r.get("probation_days", DEFAULT_PROBATION_DAYS)),
                status=STATUS_ACTIVE,
                batch_id=r.get("batch_id", ""),
                rsr_at_promotion=_safe_float(r.get("rsr_at_promotion", 0.0)),
                sector_ignition_score=_safe_float(r.get("sector_ignition_score", 0.0)),
                compression_score=_safe_float(r.get("compression_score", 0.0)),
                drift_score=_safe_float(r.get("drift_score", 0.0)),
            ))
        except Exception as exc:
            logger.warning("[V2] parse active record %s failed: %s", sym, exc)
    return active


def _get_cooldown_symbols(path: Path) -> Set[str]:
    """
    Return symbols in demotion cooldown (demoted within PROBATION_COOLDOWN_DAYS).

    Uses "updated_at" field if present (actual demotion timestamp), else "promoted_at".
    """
    records = _load_jsonl(path)
    by_sym: Dict[str, dict] = {}
    for r in records:
        sym = r.get("symbol", "")
        if sym:
            by_sym[sym] = r

    cutoff = datetime.now(timezone.utc) - timedelta(days=PROBATION_COOLDOWN_DAYS)
    cooldown: Set[str] = set()
    for sym, r in by_sym.items():
        if r.get("status") != STATUS_DEMOTED:
            continue
        ts_str = r.get("updated_at") or r.get("promoted_at", "")
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if dt >= cutoff:
                cooldown.add(sym)
        except Exception:
            pass
    return cooldown


# ─────────────────────────────────────────────────────────────────────────────
# Candidate taxonomy
# ─────────────────────────────────────────────────────────────────────────────

def _compute_late_entry_risk(rsr: float, compression: float) -> bool:
    """RSR at ceiling AND no compression setup = late-entry proxy. O(1), pure."""
    return rsr >= MATURE_LEADER_MIN_RSR and compression < MATURE_LEADER_MAX_COMPRESSION


def classify_candidate_type(
    symbol: str,
    rsr_scores: Dict[str, float],
    predictive_scores: Dict,
    fl_candidates: Dict[str, Dict],
) -> Tuple[str, bool]:
    """
    Deterministic O(1) candidate classification.

    Returns (candidate_type, late_entry_risk).
    Uses only already-computed in-memory metrics — no extra OHLCV or ML.

    Priority order (first match wins):
      HIGH_RSR → MATURE_LEADER → MEAN_REVERSION → EARLY_IGNITION → CONTINUATION → UNCLASSIFIED
    """
    rsr     = _safe_float(rsr_scores.get(symbol, 0.0))
    si_dict = predictive_scores.get("sector_ignition_score") or {}
    si      = _safe_float(si_dict.get(symbol, 0.0))
    cb_dict = predictive_scores.get("compression_breakout_score") or {}
    comp    = _safe_float(cb_dict.get(symbol, 0.0))
    fl_rec  = fl_candidates.get(symbol) or {}
    drift   = _safe_float(fl_rec.get("persistent_drift_score", 0.0))
    top_cands = predictive_scores.get("top_candidates") or []
    rank = next((i + 1 for i, (s, _) in enumerate(top_cands) if s == symbol), None)

    late_entry_risk = _compute_late_entry_risk(rsr, comp)

    # Priority 1 — MATURE_LEADER: RSR at ceiling + no compression setup
    if late_entry_risk:
        return CANDIDATE_MATURE_LEADER, True

    # Priority 2 — MEAN_REVERSION: weak RSR + low ignition = bounce, no trend
    if rsr < MEAN_REVERSION_MAX_RSR and si < MEAN_REVERSION_MAX_IGNITION:
        return CANDIDATE_MEAN_REVERSION, False

    # Priority 3 — EARLY_IGNITION: strong ignition + top rank + pre-maturity + drift
    if (
        si >= IGNITION_MIN_SECTOR_IGNITION
        and rank is not None
        and rank <= IGNITION_MAX_PREDICTIVE_RANK
        and rsr < IGNITION_MAX_RSR
        and drift >= IGNITION_MIN_DRIFT
    ):
        return CANDIDATE_EARLY_IGNITION, False

    # Priority 4 — CONTINUATION: high RSR + some compression + not maxed out
    if (
        rsr >= CONTINUATION_MIN_RSR
        and comp >= CONTINUATION_MIN_COMPRESSION
        and rsr < CONTINUATION_MAX_RSR
    ):
        return CANDIDATE_CONTINUATION, False

    # Priority 5 — HIGH_RSR: RSR >= 90 + confirmed sector ignition
    # Placed after MATURE_LEADER to preserve late-entry-risk semantics.
    if rsr >= _HIGH_RSR_BYPASS_RSR and si >= GATE_MIN_SECTOR_IGNITION:
        return CANDIDATE_HIGH_RSR, False

    return CANDIDATE_UNCLASSIFIED, False


# ─────────────────────────────────────────────────────────────────────────────
# Promotion scoring
# ─────────────────────────────────────────────────────────────────────────────

def compute_promotion_score(
    symbol: str,
    rsr_scores: Dict[str, float],
    predictive_scores: Dict,
    fl_candidates: Dict[str, Dict],
) -> Tuple[float, List[str]]:
    """
    Compute promotion_score ∈ [0, 1] from existing cached metrics.

    Formula (weights sum to 1.0):
      score = rsr_pass_score * 0.30
            + predictive_score * 0.25
            + sector_ignition_score * 0.20
            + compression_score * 0.15
            + drift_score * 0.10

    All component inputs are in [0, 100]; normalized to [0, 1] before weighting.
    """
    reasons: List[str] = []

    # RSR pass score (percentile 0-100)
    rsr_raw = _safe_float(rsr_scores.get(symbol, 0.0))
    rsr_c   = _normalize(rsr_raw, 100.0)
    if rsr_raw >= 75.0:
        reasons.append("rsr_strong")
    elif rsr_raw >= 50.0:
        reasons.append("rsr_moderate")

    # Predictive alpha score (0-100, from persisted predictive_alpha_score dict)
    pa_scores: Dict = predictive_scores.get("predictive_alpha_score") or {}
    pred_raw = _safe_float(pa_scores.get(symbol, 0.0))
    # Fall back to compression_breakout_score if alpha score absent
    if pred_raw == 0.0:
        cb: Dict = predictive_scores.get("compression_breakout_score") or {}
        pred_raw = _safe_float(cb.get(symbol, 0.0))
    pred_c = _normalize(pred_raw, 100.0)
    if pred_raw >= 70.0:
        reasons.append("predictive_strong")

    # Sector ignition score (0-100)
    si_dict: Dict = predictive_scores.get("sector_ignition_score") or {}
    si_raw  = _safe_float(si_dict.get(symbol, 0.0))
    si_c    = _normalize(si_raw, 100.0)
    if si_raw >= GATE_MIN_SECTOR_IGNITION:
        reasons.append("sector_ignition_ok")

    # Compression breakout score (0-100)
    cb_dict: Dict = predictive_scores.get("compression_breakout_score") or {}
    comp_raw = _safe_float(cb_dict.get(symbol, 0.0))
    comp_c   = _normalize(comp_raw, 100.0)
    if comp_raw >= 60.0:
        reasons.append("compression_ok")

    # Persistent drift score (0-100) from future leader screener
    fl_rec   = fl_candidates.get(symbol) or {}
    drift_raw = _safe_float(fl_rec.get("persistent_drift_score", 0.0))
    drift_c   = _normalize(drift_raw, 100.0)
    if drift_raw >= 50.0:
        reasons.append("drift_ok")

    score = (
        WEIGHT_RSR_PASS        * rsr_c
        + WEIGHT_PREDICTIVE    * pred_c
        + WEIGHT_SECTOR_IGNITION * si_c
        + WEIGHT_COMPRESSION   * comp_c
        + WEIGHT_DRIFT         * drift_c
    )
    return round(max(0.0, min(1.0, score)), 4), reasons


def check_promotion_gate(
    symbol: str,
    rsr_scores: Dict[str, float],
    predictive_scores: Dict,
    fl_candidates: Dict[str, Dict],
) -> Tuple[bool, List[str]]:
    """
    Check hard gate conditions for probation promotion.

    Returns (passed: bool, fail_reasons: list[str]).
    All three conditions must pass simultaneously.
    """
    fail: List[str] = []

    # Gate 1: RSR percentile >= GATE_MIN_RSR_PASS (very loose; ignition gate is strict)
    rsr = _safe_float(rsr_scores.get(symbol, 0.0))
    if rsr < GATE_MIN_RSR_PASS:
        fail.append(f"rsr_too_low:{rsr:.1f}<{GATE_MIN_RSR_PASS}")

    # Gate 2: symbol in top-N by predictive_alpha_score
    # HIGH_RSR fast-track: RSR >= 90 bypasses predictive_rank requirement
    if rsr >= _HIGH_RSR_BYPASS_RSR:
        pass
    else:
        top_cands = predictive_scores.get("top_candidates") or []
        if top_cands:
            rank = next((i + 1 for i, (s, _) in enumerate(top_cands) if s == symbol), None)
            if rank is None or rank > GATE_MAX_PREDICTIVE_RANK:
                actual = rank if rank is not None else "unranked"
                fail.append(f"predictive_rank:{actual}>top{GATE_MAX_PREDICTIVE_RANK}")
        else:
            fail.append("predictive_scores_unavailable")

    # Gate 3: sector_ignition_score >= 90
    si_dict: Dict = predictive_scores.get("sector_ignition_score") or {}
    si = _safe_float(si_dict.get(symbol, 0.0))
    if si < GATE_MIN_SECTOR_IGNITION:
        fail.append(f"sector_ignition:{si:.1f}<{GATE_MIN_SECTOR_IGNITION}")

    return (len(fail) == 0), fail


def check_p2a_unclassified_gate(
    symbol: str,
    rsr_scores: Dict[str, float],
    predictive_scores: Dict,
) -> Tuple[bool, List[str]]:
    """P2-A gate for UNCLASSIFIED candidates (Gate 2 predictive_rank bypassed).

    Conditions:
      1. symbol not in P2A_EXCLUDED_SYMBOLS
      2. RSR >= GATE_MIN_RSR_PASS (8.0)
      3. P2A_MIN_SECTOR_IGNITION (50) <= SI < GATE_MIN_SECTOR_IGNITION (90)

    Gate 2 bypass is intentional; validated in P2-A WF study (4-fold 2021-2024).
    """
    fail: List[str] = []

    if symbol in P2A_EXCLUDED_SYMBOLS:
        fail.append(f"p2a_excluded:{symbol}")

    rsr = _safe_float(rsr_scores.get(symbol, 0.0))
    if rsr < GATE_MIN_RSR_PASS:
        fail.append(f"rsr_too_low:{rsr:.1f}<{GATE_MIN_RSR_PASS}")

    si_dict: Dict = predictive_scores.get("sector_ignition_score") or {}
    si = _safe_float(si_dict.get(symbol, 0.0))
    if si < P2A_MIN_SECTOR_IGNITION:
        fail.append(f"p2a_si_too_low:{si:.1f}<{P2A_MIN_SECTOR_IGNITION}")
    elif si >= GATE_MIN_SECTOR_IGNITION:
        fail.append(f"p2a_si_too_high:{si:.1f}>={GATE_MIN_SECTOR_IGNITION}")

    return (len(fail) == 0), fail


# ─────────────────────────────────────────────────────────────────────────────
# Demotion / graduation logic
# ─────────────────────────────────────────────────────────────────────────────

def check_demotion(
    record: ProbationRecord,
    current_rsr: float,
) -> Tuple[bool, str]:
    """
    Check whether a probation symbol should be demoted.

    Triggers (any one triggers demotion):
      - rsr_rank_drop > RSR_RANK_DROP_THRESHOLD
      - breakout_fail (current_rsr < BREAKOUT_FAIL_RSR_MIN)
      - relative_volume_collapse proxy (rsr_drop > VOLUME_COLLAPSE_PROXY_RSR_DROP)
    """
    rsr_drop = record.rsr_at_promotion - current_rsr

    if rsr_drop > RSR_RANK_DROP_THRESHOLD:
        return True, f"rsr_rank_drop:{rsr_drop:.1f}>{RSR_RANK_DROP_THRESHOLD}"

    if current_rsr < BREAKOUT_FAIL_RSR_MIN:
        return True, f"breakout_fail:rsr={current_rsr:.1f}<{BREAKOUT_FAIL_RSR_MIN}"

    if rsr_drop > VOLUME_COLLAPSE_PROXY_RSR_DROP:
        return True, f"relative_volume_collapse_proxy:drop={rsr_drop:.1f}"

    return False, ""


def check_graduation(
    record: ProbationRecord,
    current_rsr: float,
    outcomes: List[dict],
) -> Tuple[bool, str]:
    """
    After probation_days elapsed, check if symbol should graduate.

    Conditions (all must pass):
      1. probation_days elapsed
      2. Average forward_return_3d > 0 (positive expectancy)
      3. current_rsr >= rsr_at_promotion * GRADUATION_MIN_RSR_RATIO
      4. continuation_days / total_days >= GRADUATION_MIN_CONTINUATION
    """
    try:
        promoted_dt = datetime.fromisoformat(record.promoted_at.replace("Z", "+00:00"))
        elapsed = (datetime.now(timezone.utc) - promoted_dt).days
    except Exception:
        return False, "promoted_at_parse_error"

    if elapsed < record.probation_days:
        return False, f"probation_not_elapsed:{elapsed}/{record.probation_days}d"

    sym_outcomes = [o for o in outcomes if o.get("symbol") == record.symbol]

    # Condition 2: positive expectancy
    # Primary: avg(forward_return_3d) > 0
    # Fallback: avg(rsr_delta) > 0  ← used when fwd returns not yet materialized
    fwd_rets = [
        _safe_float(o["forward_return_3d"])
        for o in sym_outcomes
        if o.get("forward_return_3d") is not None
    ]
    if fwd_rets:
        avg_ret = sum(fwd_rets) / len(fwd_rets)
        graduation_method = "forward_return"
        if avg_ret <= 0:
            return False, f"expectancy_negative:{avg_ret:.4f}"
    else:
        # forward_return_3d not yet materialized — fall back to RSR trend as proxy
        rsr_deltas = [_safe_float(o.get("rsr_delta", 0.0)) for o in sym_outcomes]
        if not rsr_deltas:
            return False, "no_forward_returns"
        avg_ret = sum(rsr_deltas) / len(rsr_deltas)
        graduation_method = "rsr_delta_fallback"
        logger.info(
            "[V2] graduation fallback: %s avg_rsr_delta=%.3f (n=%d)",
            record.symbol, avg_ret, len(rsr_deltas),
        )
        if avg_ret <= 0:
            return False, f"rsr_delta_negative:{avg_ret:.4f}"

    # Condition 3: RSR stable
    rsr_floor = record.rsr_at_promotion * GRADUATION_MIN_RSR_RATIO
    if current_rsr < rsr_floor:
        return False, f"rsr_declining:{current_rsr:.1f}<{rsr_floor:.1f}"

    # Condition 4: continuation persistence
    total = len(sym_outcomes)
    cont  = sum(1 for o in sym_outcomes if _safe_float(o.get("continuation_days", 0)) > 0)
    rate  = cont / total if total > 0 else 0.0
    if rate < GRADUATION_MIN_CONTINUATION:
        return False, f"continuation_low:{rate:.2f}<{GRADUATION_MIN_CONTINUATION}"

    return True, (
        f"graduated:avg_ret={avg_ret:.4f}"
        f" graduation_method:{graduation_method}"
        f" rsr={current_rsr:.1f}"
        f" cont={rate:.2f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rejection telemetry helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_rejection_record(
    symbol: str,
    candidate_type: str,
    promotion_score: float,
    failed_conditions: List[str],
    rsr_scores: Dict[str, float],
    predictive_scores: Dict,
    fl_candidates: Dict[str, Dict],
    cb_state: str,
    late_entry_risk: bool,
    batch_id: str,
    timestamp: str,
) -> dict:
    """Build a RejectionRecord dict from already-computed in-memory metrics."""
    rsr = _safe_float(rsr_scores.get(symbol, 0.0))
    top_cands = predictive_scores.get("top_candidates") or []
    rank = next((i + 1 for i, (s, _) in enumerate(top_cands) if s == symbol), None)
    si_dict = predictive_scores.get("sector_ignition_score") or {}
    si = _safe_float(si_dict.get(symbol, 0.0))
    fl_rec = fl_candidates.get(symbol) or {}
    return RejectionRecord(
        symbol=symbol,
        timestamp=timestamp,
        candidate_type=candidate_type,
        promotion_score=round(promotion_score, 4),
        failed_conditions=list(failed_conditions),
        metrics={
            "rsr_pass": int(rsr),
            "predictive_rank": rank,
            "sector_ignition": round(si, 1),
            "cb_state": cb_state,
            "future_leader_score": fl_rec.get("future_leader_score"),
            "rsr_rank": fl_rec.get("rsr_rank"),
            "late_entry_risk": late_entry_risk,
        },
        batch_id=batch_id,
    ).to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Main probation gate runner
# ─────────────────────────────────────────────────────────────────────────────

def run_probation_gate(
    shadow_universe: Dict[str, str],
    live_universe: Dict[str, str],
    cb_state: str,
    rsr_scores: Dict[str, float],
    predictive_scores: Dict,
    fl_candidates: Dict[str, Dict],
    probation_path: Path,
    outcomes_path: Path,
    run_id: str,
    rejection_path: Optional[Path] = None,
) -> Tuple[Set[str], Set[str]]:
    """
    Run probation gate.

    Step 1: Check active probation symbols for demotion / graduation.
    Step 2: Check shadow candidates for new probation promotion.
    Step 3: Return (active_probation_symbols, newly_promoted_symbols).

    FAIL_OPEN: any error returns (set(), set()) to preserve execution.
    """
    newly_promoted: Set[str] = set()
    active_symbols: Set[str] = set()

    active_records = load_active_probation(probation_path)
    cooldown_syms  = _get_cooldown_symbols(probation_path)
    outcomes       = _load_jsonl(outcomes_path)
    now_utc        = _now_utc()
    batch_id       = _sha8(f"{run_id}:v2_probation")

    # ── Step 1: lifecycle checks on active probation symbols ──────────────────
    surviving: List[ProbationRecord] = []
    _just_graduated: Set[str] = set()   # P1-B: prevent same-run re-promotion
    for rec in active_records:
        sym         = rec.symbol
        current_rsr = _safe_float(rsr_scores.get(sym, 0.0))

        demoted, dem_reason = check_demotion(rec, current_rsr)
        if demoted:
            updated = {
                **rec.to_dict(),
                "status":          STATUS_DEMOTED,
                "demotion_reason": dem_reason,
                "updated_at":      now_utc,
            }
            try:
                _append_jsonl(probation_path, updated)
            except Exception as _w:
                logger.warning("[V2] demotion write failed: %s", _w)
            cooldown_syms.add(sym)
            logger.info("[V2] DEMOTED %s: %s", sym, dem_reason)
            print(f"  [AUTO_PROMOTE_V2] 降格: {sym} ({dem_reason})")
            continue

        graduated, grad_reason = check_graduation(rec, current_rsr, outcomes)
        if graduated:
            updated = {
                **rec.to_dict(),
                "status":            STATUS_GRADUATED,
                "graduation_reason": grad_reason,
                "updated_at":        now_utc,
            }
            try:
                _append_jsonl(probation_path, updated)
            except Exception as _w:
                logger.warning("[V2] graduation write failed: %s", _w)
            _just_graduated.add(sym)   # P1-B: block same-run re-promotion
            logger.info("[V2] GRADUATED %s: %s", sym, grad_reason)
            print(f"  [AUTO_PROMOTE_V2] 卒業候補: {sym} → 本ユニバース昇格推奨")
            continue

        surviving.append(rec)
        active_symbols.add(sym)

    # ── Step 2: CB check — block new promotions in severe CB state ────────────
    if cb_state in CB_SEVERE_STATES:
        logger.info("[V2] CB=%s: skip new probation promotions", cb_state)
        return active_symbols, newly_promoted

    # ── Step 3: budget check ──────────────────────────────────────────────────
    budget = MAX_ACTIVE_PROBATION - len(surviving)
    if budget <= 0:
        logger.info("[V2] budget exhausted (%d/%d active)", len(surviving), MAX_ACTIVE_PROBATION)
        return active_symbols, newly_promoted

    # ── Step 4: score shadow candidates ──────────────────────────────────────
    # P1-B: _just_graduated excluded to prevent same-run re-promotion overwriting
    # STATUS_GRADUATED with STATUS_ACTIVE (which breaks get_graduated_symbols lookup)
    already_seen = active_symbols | cooldown_syms | _just_graduated | set(live_universe.keys())
    candidates: List[Tuple[str, float, List[str], str]] = []
    _rejections: List[dict] = []
    shadow_evaluated = 0

    for sym in sorted(shadow_universe.keys()):   # deterministic input order
        if sym in already_seen:
            continue
        shadow_evaluated += 1

        cand_type, late_risk = classify_candidate_type(
            sym, rsr_scores, predictive_scores, fl_candidates
        )

        # ── P2-A: UNCLASSIFIED with SI=[50,90) — bypasses predictive_rank gate ──
        if cand_type == CANDIDATE_UNCLASSIFIED:
            p2a_ok, p2a_fail = check_p2a_unclassified_gate(
                sym, rsr_scores, predictive_scores
            )
            if not p2a_ok:
                logger.debug("[V2][P2A] %s rejected: %s", sym, p2a_fail)
                _rejections.append(_build_rejection_record(
                    sym, cand_type, 0.0, p2a_fail,
                    rsr_scores, predictive_scores, fl_candidates,
                    cb_state, False, batch_id, now_utc,
                ))
                continue
            score, reasons = compute_promotion_score(
                sym, rsr_scores, predictive_scores, fl_candidates
            )
            candidates.append((sym, score, list(reasons) + ["p2a_unclassified"], cand_type))
            continue

        gate_ok, gate_fail = check_promotion_gate(
            sym, rsr_scores, predictive_scores, fl_candidates
        )

        if not gate_ok:
            logger.debug("[V2] %s gate_fail: %s type=%s", sym, gate_fail, cand_type)
            _rejections.append(_build_rejection_record(
                sym, cand_type, 0.0, gate_fail,
                rsr_scores, predictive_scores, fl_candidates,
                cb_state, late_risk, batch_id, now_utc,
            ))
            continue

        if cand_type not in ALLOWED_CANDIDATE_TYPES:
            tax_fail = [f"taxonomy_blocked:{cand_type}"]
            score, _ = compute_promotion_score(
                sym, rsr_scores, predictive_scores, fl_candidates
            )
            logger.debug("[V2] %s taxonomy_blocked: type=%s score=%.3f", sym, cand_type, score)
            _rejections.append(_build_rejection_record(
                sym, cand_type, score, tax_fail,
                rsr_scores, predictive_scores, fl_candidates,
                cb_state, late_risk, batch_id, now_utc,
            ))
            continue

        score, reasons = compute_promotion_score(
            sym, rsr_scores, predictive_scores, fl_candidates
        )
        candidates.append((sym, score, reasons, cand_type))

    # Deterministic sort: desc score, asc symbol
    candidates.sort(key=lambda x: (-x[1], x[0]))

    # ── Step 5: promote top candidates within budget ──────────────────────────
    si_dict: Dict = predictive_scores.get("sector_ignition_score") or {}
    cb_dict: Dict = predictive_scores.get("compression_breakout_score") or {}

    for sym, score, reasons, cand_type in candidates[:budget]:
        fl_rec = fl_candidates.get(sym) or {}
        rec = ProbationRecord(
            symbol=sym,
            promoted_at=now_utc,
            promotion_score=score,
            promotion_reason=reasons,
            probation_days=DEFAULT_PROBATION_DAYS,
            status=STATUS_ACTIVE,
            batch_id=batch_id,
            rsr_at_promotion=_safe_float(rsr_scores.get(sym, 0.0)),
            sector_ignition_score=_safe_float(si_dict.get(sym, 0.0)),
            compression_score=_safe_float(cb_dict.get(sym, 0.0)),
            drift_score=_safe_float(fl_rec.get("persistent_drift_score", 0.0)),
            candidate_type=cand_type,
        )
        try:
            _append_jsonl(probation_path, rec.to_dict())
        except Exception as _w:
            logger.warning("[V2] promotion write failed: %s", _w)
            continue

        active_symbols.add(sym)
        newly_promoted.add(sym)
        logger.info("[V2] PROBATION %s: score=%.3f type=%s reasons=%s", sym, score, cand_type, reasons)
        print(f"  [AUTO_PROMOTE_V2] 試用昇格: {sym}  score={score:.3f}  type={cand_type}  {reasons}")

    # ── Rejection telemetry ───────────────────────────────────────────────────
    if rejection_path and _rejections:
        for _rej in _rejections:
            try:
                _append_jsonl(rejection_path, _rej)
            except Exception as _rw:
                logger.warning("[V2] rejection write failed: %s", _rw)

    # ── Debug warning: shadow candidates detected but no promotions ───────────
    if shadow_evaluated > 0 and not newly_promoted:
        _warn = "\n".join([
            "",
            "[AUTO_PROMOTE_V2]",
            f"  shadow_candidates_detected={shadow_evaluated}",
            "  probation_promoted=0",
            "  see rejection report for details",
        ])
        print(_warn)
        logger.info("[V2] %d shadow candidate(s) evaluated; probation_promoted=0", shadow_evaluated)

    return active_symbols, newly_promoted


# ─────────────────────────────────────────────────────────────────────────────
# Graduation query helper
# ─────────────────────────────────────────────────────────────────────────────

def get_graduated_symbols(
    probation_path: Path,
    live_universe: Dict[str, str],
) -> Set[str]:
    """
    Return symbols that completed probation (STATUS_GRADUATED) and are not yet
    in live_universe.  Uses last-record-wins per symbol from the append-only JSONL.
    FAIL_OPEN: returns empty set on any error.
    """
    try:
        records = _load_jsonl(probation_path)
        by_sym: Dict[str, dict] = {}
        for r in records:
            sym = r.get("symbol", "")
            if sym:
                by_sym[sym] = r
        return {
            sym
            for sym, r in by_sym.items()
            if r.get("status") == STATUS_GRADUATED
            and sym not in live_universe
        }
    except Exception as exc:
        logger.warning("[V2] get_graduated_symbols failed: %s", exc)
        return set()


# ─────────────────────────────────────────────────────────────────────────────
# Outcome observation hook
# ─────────────────────────────────────────────────────────────────────────────

def run_probation_outcome_observation(
    probation_path: Path,
    outcomes_path: Path,
    rsr_scores: Dict[str, float],
    run_id: str,
) -> None:
    """
    Record daily observation for active probation symbols.

    Forward returns are null at record time (no additional OHLCV loads).
    Reuses T-1 RSR snapshot for rsr_delta computation.
    FAIL_OPEN: errors logged and swallowed.
    """
    active = load_active_probation(probation_path)
    if not active:
        return

    today = date.today().isoformat()

    # Count continuation days per symbol from existing outcomes
    existing = _load_jsonl(outcomes_path)
    cont_count: Dict[str, int] = {}
    for rec in existing:
        sym = rec.get("symbol", "")
        if sym and rec.get("status") == STATUS_ACTIVE:
            cont_count[sym] = cont_count.get(sym, 0) + 1

    for prec in active:
        sym = prec.symbol
        try:
            promoted_dt = datetime.fromisoformat(prec.promoted_at.replace("Z", "+00:00"))
            days_held = max(1, (datetime.now(timezone.utc) - promoted_dt).days + 1)
        except Exception:
            days_held = 1

        rsr_now   = _safe_float(rsr_scores.get(sym, 0.0))
        rsr_delta = rsr_now - prec.rsr_at_promotion

        outcome = ProbationOutcome(
            symbol=sym,
            promoted_at=prec.promoted_at,
            observation_date=today,
            days_held=days_held,
            forward_return_3d=None,
            forward_return_5d=None,
            forward_return_10d=None,
            max_favorable_excursion=None,
            max_adverse_excursion=None,
            rsr_delta=round(rsr_delta, 2),
            continuation_days=cont_count.get(sym, 0) + 1,
            status=STATUS_ACTIVE,
        )
        try:
            _append_jsonl(outcomes_path, outcome.to_dict())
        except Exception as _w:
            logger.warning("[V2] outcome write failed %s: %s", sym, _w)
        logger.debug("[V2] outcome obs: %s days=%d rsr_delta=%.1f", sym, days_held, rsr_delta)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def format_probation_report(
    probation_path: Path,
    outcomes_path: Path,
) -> str:
    """
    Format [AUTO_PROMOTE_SAFE_V2] section for morning report output.
    FAIL_OPEN: returns empty string on any error.
    """
    try:
        records  = _load_jsonl(probation_path)
        outcomes = _load_jsonl(outcomes_path)

        by_sym: Dict[str, dict] = {}
        for r in records:
            sym = r.get("symbol", "")
            if sym:
                by_sym[sym] = r

        active_list    = [r for r in by_sym.values() if r.get("status") == STATUS_ACTIVE]
        graduated_list = [r for r in by_sym.values() if r.get("status") == STATUS_GRADUATED]
        demoted_list   = [r for r in by_sym.values() if r.get("status") == STATUS_DEMOTED]

        fwd_rets = [
            _safe_float(o["forward_return_3d"])
            for o in outcomes
            if o.get("forward_return_3d") is not None
        ]
        avg_exp = sum(fwd_rets) / len(fwd_rets) if fwd_rets else None

        resolved = len(graduated_list) + len(demoted_list)
        hit_rate = len(graduated_list) / resolved if resolved > 0 else None

        lines = [
            "",
            "── [AUTO_PROMOTE_SAFE_V2] ────────────────────────────────",
            f"  試用中     : {[r['symbol'] for r in active_list]}"
            f"  ({len(active_list)}/{MAX_ACTIVE_PROBATION}枠)",
            f"  卒業済み   : {[r['symbol'] for r in graduated_list]}"
            f"  ({len(graduated_list)}銘柄)",
            f"  降格済み   : {[r['symbol'] for r in demoted_list]}"
            f"  ({len(demoted_list)}銘柄)",
        ]
        if avg_exp is not None:
            lines.append(f"  平均期待値 : {avg_exp:+.4f}  (3日先行リターン平均)")
        if hit_rate is not None:
            lines.append(f"  ヒット率   : {hit_rate:.1%}  (卒業 / {resolved}解決済)")
        lines.append("──────────────────────────────────────────────────────────")
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("[V2] format_probation_report failed: %s", exc)
        return ""


def format_explainability_report(
    rejection_path: Path,
    probation_path: Path,
    today: Optional[str] = None,
    max_rejects: int = 10,
) -> str:
    """
    Format [AUTO_PROMOTE_EXPLAINABILITY] section.

    Shows accepted probation symbols and per-symbol rejection reasons for today's run.
    FAIL_OPEN: returns "" on any error.
    """
    try:
        if today is None:
            today = date.today().isoformat()

        all_rejs = _load_jsonl(rejection_path)
        todays_rejs = [r for r in all_rejs if r.get("timestamp", "").startswith(today)]

        # Accepted = promoted today (status active, promoted_at = today)
        precs = _load_jsonl(probation_path)
        accepted = [
            r.get("symbol", "")
            for r in precs
            if r.get("promoted_at", "").startswith(today)
            and r.get("status") == STATUS_ACTIVE
        ]

        lines = [
            "",
            "── [AUTO_PROMOTE_EXPLAINABILITY] ─────────────────────────────",
        ]
        lines.append(f"  試用昇格受理: {accepted if accepted else 'なし'}")

        if not todays_rejs:
            lines.append("  拒否候補: なし")
            lines.append("──────────────────────────────────────────────────────────")
            return "\n".join(lines)

        lines.append("")
        for rej in todays_rejs[:max_rejects]:
            sym = rej.get("symbol", "?")
            ct  = rej.get("candidate_type", CANDIDATE_UNCLASSIFIED)
            metrics = rej.get("metrics") or {}

            lines.append(f"  {sym} rejected:")
            lines.append(f"    * candidate_type={ct}")

            if metrics.get("late_entry_risk"):
                lines.append("    * late_entry_risk=True")

            pred_rank = metrics.get("predictive_rank")
            if pred_rank is not None:
                lines.append(f"    * predictive_rank={pred_rank} (>{GATE_MAX_PREDICTIVE_RANK})")
            else:
                lines.append("    * predictive_rank=unranked")

            si = metrics.get("sector_ignition")
            if si is not None and si < GATE_MIN_SECTOR_IGNITION:
                lines.append(f"    * no_sector_ignition ({si:.1f}<{GATE_MIN_SECTOR_IGNITION})")

            # Remaining failed conditions not already shown above
            skip_prefixes = ("predictive_rank", "sector_ignition", "taxonomy_blocked")
            for cond in rej.get("failed_conditions", []):
                if not any(cond.startswith(p) for p in skip_prefixes):
                    lines.append(f"    * {cond}")

            lines.append("")

        lines.append("──────────────────────────────────────────────────────────")
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("[V2] format_explainability_report failed: %s", exc)
        return ""


def format_rejection_stats(
    rejection_path: Path,
    today: Optional[str] = None,
) -> str:
    """
    Format [AUTO_PROMOTE_REJECTION_STATS] section using Counter.

    Aggregates failed_conditions across today's rejections.
    FAIL_OPEN: returns "" on any error.
    """
    try:
        if today is None:
            today = date.today().isoformat()

        all_rejs = _load_jsonl(rejection_path)
        todays_rejs = [r for r in all_rejs if r.get("timestamp", "").startswith(today)]

        if not todays_rejs:
            return ""

        counter: Counter = Counter()
        for rej in todays_rejs:
            for cond in rej.get("failed_conditions", []):
                key = cond.split(":")[0]   # strip colon-suffixed detail values
                counter[key] += 1

        if not counter:
            return ""

        lines = [
            "",
            "── [AUTO_PROMOTE_REJECTION_STATS] ────────────────────────────",
        ]
        for key, count in counter.most_common():
            unit = "reject" if count == 1 else "rejects"
            lines.append(f"  {key:<34}: {count} {unit}")
        lines.append("──────────────────────────────────────────────────────────")
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("[V2] format_rejection_stats failed: %s", exc)
        return ""
