"""
src/analytics/future_leader_promotion_readiness.py
Promotion Readiness + Lifecycle Governance Engine

Purpose:
  Evaluate Future Leader research telemetry for safe promotion through
  the Research → Shadow → Paper → Limited → Production lifecycle.
  FAIL_CLOSED: promotion is denied unless all quantitative conditions
  are met AND all AUTO_PROMOTION_BLOCKERS are clear.

Architecture (state machine):
  RESEARCH_ONLY  (telemetry accumulation only)
  ↓ score>=0.55, confidence>=0.50, no blockers
  SHADOW_READY   (shadow allocator simulation allowed — future hook)
  ↓ score>=0.65, confidence>=0.60, regime_coverage>=2
  PAPER_READY    (synthetic execution allowed — future hook)
  ↓ score>=0.72, confidence>=0.70
  LIMITED_EXECUTION_READY  (small capital routing — future hook)
  ↓ score>=0.80, confidence>=0.80
  PRODUCTION_READY  (full allocator integration — future hook)

  BLOCKED: any AUTO_PROMOTION_BLOCKER triggers this; auto-promotion prohibited

INVARIANTS (変更禁止):
  - observation_only=True: reads research telemetry, writes promotion audit only
  - FAIL_CLOSED: insufficient data or any blocker → BLOCKED + promotion_ready=False
  - deterministic: identical inputs → identical hash and outputs
  - append-only JSONL: promotion audit is immutable
  - no execution path connection: no live orders, no allocator, no signal mutation

禁止:
  - live order generation
  - predictive_entry_enabled mutation
  - allocator mutation
  - deployable universe mutation
  - shadow allocator simulation (future integration point — see hook comments)
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

# Matches future_leader_half_life.HALF_LIFE_WINDOW (survivability horizon, business days)
_HALF_LIFE_WINDOW: int = 20


# ─────────────────────────────────────────────────────────────────────────────
# FAIL_CLOSED blocker registry — 変更禁止
# ─────────────────────────────────────────────────────────────────────────────

AUTO_PROMOTION_BLOCKERS: Set[str] = {
    "insufficient_clean_sample",
    "negative_recent_expectancy",
    "risk_off_instability",
    "high_instant_rejection",
    "temporal_degradation",
    "survivability_collapse",
    "regime_instability",
}


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────────────────────

MIN_CLEAN_SAMPLE:         int   = 15    # absolute minimum for any state advance
RECENT_WINDOW:            int   = 20    # last N records for "recent" window
CONFIDENCE_SAMPLE_TARGET: int   = 50    # target sample for full confidence
MIN_REGIME_SAMPLE:        int   = 3     # minimum per-regime for regime analysis
_CLEAN_QUALITY_MIN:       float = 1.0   # data_quality_weight threshold

# Blocker thresholds
_BLK_RECENT_EXPCT_MIN: float = -0.010  # recent mean_10d < this → blocked
_BLK_ROFF_SAMPLE_MIN:  int   = 3       # min risk_off records to activate check
_BLK_ROFF_MEAN_MIN:    float = -0.040  # risk_off mean_10d < this → blocked
_BLK_INSTANT_REJ_MAX:  float = 0.45    # instant rejection rate > this → blocked
_BLK_TEMP_DEGRADE:     float = 0.40    # degradation ratio threshold
_BLK_SURV_RATIO:       float = 0.40    # recent_surv < ratio * overall → blocked
_BLK_REGIME_STD:       float = 0.10    # cross-regime expectancy std threshold

# State transition thresholds (score, confidence)
_THRESH_SCORE: Dict[str, float] = {
    "shadow_ready":             0.55,
    "paper_ready":              0.65,
    "limited_execution_ready":  0.72,
    "production_ready":         0.80,
}
_THRESH_CONF: Dict[str, float] = {
    "shadow_ready":             0.50,
    "paper_ready":              0.60,
    "limited_execution_ready":  0.70,
    "production_ready":         0.80,
}
# Extra conditions per state
_THRESH_REGIME_COV: Dict[str, int] = {
    "paper_ready": 2,  # >= 2 covered regimes required for paper_ready
}

# Component scoring weights — must sum to 1.0
_COMP_WEIGHTS: Dict[str, float] = {
    "statistical_edge":   0.30,
    "survivability":      0.25,
    "regime_robustness":  0.20,
    "failure_quality":    0.15,
    "temporal_stability": 0.10,
}
assert abs(sum(_COMP_WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1.0"

# Confidence sub-weights — must sum to 1.0
_CONF_WEIGHTS: Dict[str, float] = {
    "sample":   0.25,
    "regime":   0.20,
    "temporal": 0.15,
    "variance": 0.20,
    "recent":   0.20,
}
assert abs(sum(_CONF_WEIGHTS.values()) - 1.0) < 1e-9, "conf weights must sum to 1.0"

_REGIMES = ["risk_on", "risk_off", "range", "volatile"]


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle State Machine
# ─────────────────────────────────────────────────────────────────────────────

class PromotionState(Enum):
    RESEARCH_ONLY           = "research_only"            # telemetry only; execution disabled
    SHADOW_READY            = "shadow_ready"             # shadow simulation allowed (future)
    PAPER_READY             = "paper_ready"              # synthetic execution allowed (future)
    LIMITED_EXECUTION_READY = "limited_execution_ready"  # small capital routing (future)
    PRODUCTION_READY        = "production_ready"         # full allocator integration (future)
    BLOCKED                 = "blocked"                  # FAIL_CLOSED; auto-promotion prohibited


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PromotionReadinessInputs:
    """
    Aggregated research telemetry for promotion evaluation.
    All record lists are raw dicts loaded from existing analytics JSONL.
    """
    survivability_records: List[dict]     # future_leader_survivability_*.jsonl
    failure_records:       List[dict]     # future_leader_failure_*.jsonl
    regime_tags:           Dict[str, str] # {observation_date: regime_tag}
    evaluation_date:       str            # ISO date string YYYY-MM-DD


@dataclass
class ReadinessComponentScore:
    """Single component in the promotion readiness evaluation."""
    name:       str
    score:      float              # 0.0–1.0
    weight:     float              # contribution weight in final score
    sub_scores: Dict[str, float]   # diagnostic detail
    notes:      List[str]          # human-readable diagnostics


@dataclass
class PromotionReadinessReport:
    """
    Promotion readiness governance output.
    observation_only=True: no execution path connection.

    JSONL schema (to_dict):
      timestamp, promotion_state, promotion_ready, readiness_score,
      promotion_confidence, component_scores, blocking_reasons, warnings,
      sample_size, recent_sample_size, regime_coverage, deterministic_hash
    """
    timestamp:            str

    promotion_state:      PromotionState
    promotion_ready:      bool

    readiness_score:      float   # 0.0–1.0 component-weighted
    promotion_confidence: float   # 0.0–1.0 independent evidence strength

    component_scores:     Dict[str, float]

    blocking_reasons:     List[str]
    warnings:             List[str]

    sample_size:          int
    recent_sample_size:   int

    regime_coverage:      Dict[str, int]  # {regime: count of CLEAN records}

    deterministic_hash:   str

    def to_dict(self) -> dict:
        return {
            "timestamp":            self.timestamp,
            "promotion_state":      self.promotion_state.value,
            "promotion_ready":      self.promotion_ready,
            "readiness_score":      round(self.readiness_score, 4),
            "promotion_confidence": round(self.promotion_confidence, 4),
            "component_scores":     {k: round(v, 4) for k, v in self.component_scores.items()},
            "blocking_reasons":     self.blocking_reasons,
            "warnings":             self.warnings,
            "sample_size":          self.sample_size,
            "recent_sample_size":   self.recent_sample_size,
            "regime_coverage":      self.regime_coverage,
            "deterministic_hash":   self.deterministic_hash,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Filter / sort helpers (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _filter_clean_survivability(records: List[dict]) -> List[dict]:
    """
    CLEAN complete: dq >= 1.0, insufficient_forward_data=False.
    Deduplicates (symbol, observation_date) — first occurrence wins.
    """
    seen: Set[Tuple[str, str]] = set()
    out: List[dict] = []
    for r in records:
        key = (r.get("symbol", ""), r.get("observation_date", ""))
        if not (key[0] and key[1]) or key in seen:
            continue
        seen.add(key)
        if (float(r.get("data_quality_weight", 0.0)) >= _CLEAN_QUALITY_MIN
                and not r.get("insufficient_forward_data", True)):
            out.append(r)
    return out


def _filter_clean_failure(records: List[dict]) -> List[dict]:
    """CLEAN failure: dq >= 1.0. Deduplicates (symbol, observation_date)."""
    seen: Set[Tuple[str, str]] = set()
    out: List[dict] = []
    for r in records:
        key = (r.get("symbol", ""), r.get("observation_date", ""))
        if not (key[0] and key[1]) or key in seen:
            continue
        seen.add(key)
        if float(r.get("data_quality_weight", 0.0)) >= _CLEAN_QUALITY_MIN:
            out.append(r)
    return out


def _sort_by_obs_date(records: List[dict]) -> List[dict]:
    return sorted(records, key=lambda r: r.get("observation_date", ""))


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _fwd10_vals(records: List[dict]) -> List[float]:
    return [float(r["forward_return_10d"]) for r in records if r.get("forward_return_10d") is not None]


def _mfe_vals(records: List[dict]) -> List[float]:
    return [float(r["mfe_10d"]) for r in records if r.get("mfe_10d") is not None]


def _mae_vals(records: List[dict]) -> List[float]:
    return [float(r["mae_10d"]) for r in records if r.get("mae_10d") is not None]


# ─────────────────────────────────────────────────────────────────────────────
# Component scoring functions (pure — no I/O, no side effects)
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_statistical_edge(clean_surv: List[dict]) -> ReadinessComponentScore:
    """
    Statistical Edge: expectancy quality from CLEAN complete survivability records.

    mean_norm:   mean_10d in [-5%,+5%] normalized to [0,1]
    wr_norm:     positive rate in [30%,70%] normalized to [0,1]
    mfm_norm:    MFE/|MAE| ratio normalized (2.0 → 1.0)
    score = 0.50*mean_norm + 0.30*wr_norm + 0.20*mfm_norm
    """
    sub: Dict[str, float] = {}
    notes: List[str] = []

    if not clean_surv:
        return ReadinessComponentScore(
            name="statistical_edge", score=0.0,
            weight=_COMP_WEIGHTS["statistical_edge"],
            sub_scores={}, notes=["no CLEAN records"],
        )

    fwd10 = _fwd10_vals(clean_surv)
    if not fwd10:
        return ReadinessComponentScore(
            name="statistical_edge", score=0.0,
            weight=_COMP_WEIGHTS["statistical_edge"],
            sub_scores={}, notes=["no materialized 10d returns"],
        )

    mean_10d = float(np.mean(fwd10))
    pos_rate = float(np.mean([v > 0 for v in fwd10]))
    std_10d  = float(np.std(fwd10)) if len(fwd10) >= 2 else 0.0

    sub["mean_return_10d"] = round(mean_10d, 6)
    sub["positive_rate"]   = round(pos_rate, 4)
    sub["std_10d"]         = round(std_10d, 6)

    mean_norm = _clamp((mean_10d + 0.05) / 0.10)   # [-5%,+5%] → [0,1]
    wr_norm   = _clamp((pos_rate - 0.30) / 0.40)   # [30%,70%] → [0,1]

    mfe_v = _mfe_vals(clean_surv)
    mae_v = _mae_vals(clean_surv)
    if mfe_v and mae_v:
        med_mfe  = float(np.median(mfe_v))
        med_mae  = abs(float(np.median(mae_v))) + 1e-9
        ratio    = med_mfe / med_mae
        mfm_norm = _clamp(ratio / 2.0)
        sub["mfe_mae_ratio"] = round(ratio, 4)
    else:
        mfm_norm = 0.5
        notes.append("MFE/MAE incomplete — neutral sub-score applied")

    score = _clamp(mean_norm * 0.50 + wr_norm * 0.30 + mfm_norm * 0.20)

    sub["mean_norm"] = round(mean_norm, 4)
    sub["wr_norm"]   = round(wr_norm, 4)
    sub["mfm_norm"]  = round(mfm_norm, 4)

    if mean_10d > 0.02:
        notes.append(f"positive edge: mean_10d={mean_10d*100:+.1f}%")
    elif mean_10d < -0.01:
        notes.append(f"negative edge: mean_10d={mean_10d*100:+.1f}%")

    return ReadinessComponentScore(
        name="statistical_edge", score=round(score, 4),
        weight=_COMP_WEIGHTS["statistical_edge"],
        sub_scores=sub, notes=notes,
    )


def _evaluate_survivability(
    clean_surv: List[dict],
    clean_fail: List[dict],
) -> ReadinessComponentScore:
    """
    Survivability: alpha persistence and stability.

    persistence_score: median survival days normalized to _HALF_LIFE_WINDOW
    stability_score:   recent vs. historical survival comparison (decay detection)
    score = 0.60 * persistence_score + 0.40 * stability_score
    """
    sub:   Dict[str, float] = {}
    notes: List[str]        = []

    if not clean_fail:
        return ReadinessComponentScore(
            name="survivability", score=0.30,
            weight=_COMP_WEIGHTS["survivability"],
            sub_scores={}, notes=["no failure records — minimal score"],
        )

    sorted_fail = _sort_by_obs_date(clean_fail)

    def _days(recs: List[dict]) -> List[int]:
        out = []
        for r in recs:
            if r.get("failure_detected", False):
                offset = r.get("failure_day_offset")
                if offset is not None:
                    out.append(int(offset))
            else:
                out.append(_HALF_LIFE_WINDOW)
        return out

    all_days = _days(sorted_fail)
    if not all_days:
        return ReadinessComponentScore(
            name="survivability", score=0.30,
            weight=_COMP_WEIGHTS["survivability"],
            sub_scores={}, notes=["no resolvable survival data"],
        )

    med_overall = float(np.median(all_days))
    sub["median_survival_days"] = round(med_overall, 2)
    persistence_score = _clamp(med_overall / _HALF_LIFE_WINDOW)

    recent_fail = sorted_fail[-RECENT_WINDOW:] if len(sorted_fail) > RECENT_WINDOW else sorted_fail
    recent_days = _days(recent_fail)

    if len(recent_days) >= 3:
        med_recent = float(np.median(recent_days))
        sub["median_recent_survival_days"] = round(med_recent, 2)
        decay_ratio = (med_overall - med_recent) / max(med_overall, 1.0)
        sub["decay_ratio"]  = round(decay_ratio, 4)
        stability_score     = _clamp(1.0 - max(0.0, decay_ratio))
    else:
        stability_score = 0.50
        notes.append("insufficient recent failure records for stability check")

    score = _clamp(0.60 * persistence_score + 0.40 * stability_score)
    sub["persistence_score"] = round(persistence_score, 4)
    sub["stability_score"]   = round(stability_score, 4)

    if med_overall >= 10:
        notes.append(f"strong persistence: median {med_overall:.1f}d")
    elif med_overall <= 5:
        notes.append(f"weak persistence: median {med_overall:.1f}d")

    return ReadinessComponentScore(
        name="survivability", score=round(score, 4),
        weight=_COMP_WEIGHTS["survivability"],
        sub_scores=sub, notes=notes,
    )


def _evaluate_regime_robustness(
    clean_surv:  List[dict],
    regime_tags: Dict[str, str],
) -> ReadinessComponentScore:
    """
    Regime Robustness: alpha breadth across market environments.

    coverage_score:    covered_regimes / 4 (regimes with >= MIN_REGIME_SAMPLE)
    consistency_score: 1 - clamp(cross_regime_std / 0.08)
                       penalized if risk_off mean < -3%
    score = 0.50 * coverage_score + 0.50 * consistency_score
    """
    sub:   Dict[str, float] = {}
    notes: List[str]        = []

    groups: Dict[str, List[float]] = {r: [] for r in _REGIMES}
    for rec in clean_surv:
        obs_date = rec.get("observation_date", "")
        regime   = regime_tags.get(obs_date, "range")
        fwd10    = rec.get("forward_return_10d")
        if fwd10 is not None:
            groups[regime].append(float(fwd10))

    for r in _REGIMES:
        sub[f"regime_{r}_count"] = float(len(groups[r]))

    covered = [r for r in _REGIMES if len(groups[r]) >= MIN_REGIME_SAMPLE]
    sub["covered_regimes"] = float(len(covered))
    coverage_score = len(covered) / 4.0

    per_means: Dict[str, float] = {}
    for r in covered:
        per_means[r] = float(np.mean(groups[r]))
        sub[f"regime_{r}_mean_10d"] = round(per_means[r], 6)

    if len(per_means) >= 2:
        means_arr = list(per_means.values())
        reg_std   = float(np.std(means_arr))
        sub["cross_regime_std"] = round(reg_std, 6)
        consistency_score = _clamp(1.0 - reg_std / 0.08)
        if "risk_off" in per_means and per_means["risk_off"] < -0.03:
            consistency_score *= 0.70
            notes.append(
                f"risk_off mean_10d={per_means['risk_off']*100:+.1f}% penalty"
            )
    else:
        consistency_score = 0.50
        notes.append(f"only {len(per_means)} covered regime(s) — consistency undetermined")

    score = _clamp(0.50 * coverage_score + 0.50 * consistency_score)
    sub["coverage_score"]    = round(coverage_score, 4)
    sub["consistency_score"] = round(consistency_score, 4)

    if len(covered) == 4:
        notes.append("all 4 regimes covered")
    elif len(covered) == 0:
        notes.append("no regime with sufficient sample")

    return ReadinessComponentScore(
        name="regime_robustness", score=round(score, 4),
        weight=_COMP_WEIGHTS["regime_robustness"],
        sub_scores=sub, notes=notes,
    )


def _evaluate_failure_quality(clean_fail: List[dict]) -> ReadinessComponentScore:
    """
    Failure Quality: how positions fail — instant vs. explainable vs. survival.

    no_failure_rate:        fraction with failure_detected=False (survived window)
    instant_rejection_rate: fraction failing within <= 3 days
    market_selloff_rate:    fraction explained by market environment
    score = 0.40*no_fail_rate + 0.40*(1-instant_rate) + 0.20*(1-selloff_rate)
    """
    sub:   Dict[str, float] = {}
    notes: List[str]        = []

    if not clean_fail:
        return ReadinessComponentScore(
            name="failure_quality", score=0.50,
            weight=_COMP_WEIGHTS["failure_quality"],
            sub_scores={}, notes=["no failure records — neutral score"],
        )

    n = len(clean_fail)
    no_fail_count = sum(1 for r in clean_fail if not r.get("failure_detected", True))
    instant_count = sum(
        1 for r in clean_fail
        if r.get("failure_detected", False)
        and r.get("failure_day_offset") is not None
        and int(r["failure_day_offset"]) <= 3
    )
    selloff_count = sum(
        1 for r in clean_fail
        if r.get("primary_failure_reason") == "market_selloff"
    )

    no_fail_rate  = no_fail_count / n
    instant_rate  = instant_count / n
    selloff_rate  = selloff_count / n

    sub["n_clean_failure"]        = float(n)
    sub["no_failure_rate"]        = round(no_fail_rate, 4)
    sub["instant_rejection_rate"] = round(instant_rate, 4)
    sub["market_selloff_rate"]    = round(selloff_rate, 4)

    score = _clamp(
        no_fail_rate * 0.40
        + (1.0 - instant_rate) * 0.40
        + (1.0 - selloff_rate) * 0.20
    )

    if no_fail_rate >= 0.50:
        notes.append(f"high survival rate: {no_fail_rate*100:.0f}%")
    if instant_rate > 0.30:
        notes.append(f"elevated instant rejection: {instant_rate*100:.0f}%")
    if selloff_rate > 0.40:
        notes.append(f"high market selloff fraction: {selloff_rate*100:.0f}%")

    return ReadinessComponentScore(
        name="failure_quality", score=round(score, 4),
        weight=_COMP_WEIGHTS["failure_quality"],
        sub_scores=sub, notes=notes,
    )


def _evaluate_temporal_stability(clean_surv: List[dict]) -> ReadinessComponentScore:
    """
    Temporal Stability: chronological consistency of returns.

    Splits CLEAN records into earlier half / recent half by observation_date.
    degradation = (early_mean - recent_mean) / max(|early_mean| + eps, eps)
    score = 1 - clamp(degradation)   when degradation > 0
            1.0                       when improving or stable
    """
    sub:   Dict[str, float] = {}
    notes: List[str]        = []

    if not clean_surv:
        return ReadinessComponentScore(
            name="temporal_stability", score=0.50,
            weight=_COMP_WEIGHTS["temporal_stability"],
            sub_scores={}, notes=["no CLEAN records"],
        )

    sorted_surv = _sort_by_obs_date(clean_surv)
    if len(sorted_surv) < 6:
        return ReadinessComponentScore(
            name="temporal_stability", score=0.50,
            weight=_COMP_WEIGHTS["temporal_stability"],
            sub_scores={"n": float(len(sorted_surv))},
            notes=["insufficient records for temporal split"],
        )

    mid         = len(sorted_surv) // 2
    early_vals  = _fwd10_vals(sorted_surv[:mid])
    recent_vals = _fwd10_vals(sorted_surv[mid:])

    if not early_vals or not recent_vals:
        return ReadinessComponentScore(
            name="temporal_stability", score=0.50,
            weight=_COMP_WEIGHTS["temporal_stability"],
            sub_scores={}, notes=["split missing fwd10 data"],
        )

    early_mean  = float(np.mean(early_vals))
    recent_mean = float(np.mean(recent_vals))

    sub["early_mean_10d"]  = round(early_mean, 6)
    sub["recent_mean_10d"] = round(recent_mean, 6)
    sub["n_early"]         = float(len(early_vals))
    sub["n_recent"]        = float(len(recent_vals))

    eps         = 0.001
    degradation = (early_mean - recent_mean) / max(abs(early_mean) + eps, eps)
    sub["degradation_ratio"] = round(degradation, 4)

    score = _clamp(1.0 - _clamp(degradation)) if degradation > 0 else 1.0

    if degradation > 0.30:
        notes.append(f"performance decay: ratio={degradation:.2f}")
    if recent_mean < 0.0 and early_mean > 0.0:
        notes.append("sign flip: early positive → recent negative")

    return ReadinessComponentScore(
        name="temporal_stability", score=round(_clamp(score), 4),
        weight=_COMP_WEIGHTS["temporal_stability"],
        sub_scores=sub, notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Promotion Confidence (pure — independent of readiness_score)
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_promotion_confidence(
    clean_surv:  List[dict],
    regime_tags: Dict[str, str],
) -> Tuple[float, Dict[str, float]]:
    """
    Promotion confidence: how much to trust the readiness_score.
    Measures evidence strength, not alpha quality.

    sample_factor:   min(1, n / CONFIDENCE_SAMPLE_TARGET)
    regime_factor:   covered_regimes / 4
    temporal_factor: months_of_data / 6, clamped
    variance_factor: 1 - std_10d / 0.15, clamped
    recent_factor:   positive rate in RECENT_WINDOW, normalized [30%,70%]→[0,1]

    Returns (confidence: float, sub_factors: Dict[str, float])
    """
    sub: Dict[str, float] = {}

    if not clean_surv:
        empty = {k: 0.0 for k in _CONF_WEIGHTS}
        return 0.0, empty

    n = len(clean_surv)
    sub["n_clean"] = float(n)

    sample_factor = _clamp(n / CONFIDENCE_SAMPLE_TARGET)
    sub["sample_factor"] = round(sample_factor, 4)

    regime_counts: Dict[str, int] = {r: 0 for r in _REGIMES}
    for rec in clean_surv:
        obs_date = rec.get("observation_date", "")
        regime   = regime_tags.get(obs_date, "range")
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    covered_n   = sum(1 for cnt in regime_counts.values() if cnt >= MIN_REGIME_SAMPLE)
    regime_factor = covered_n / 4.0
    sub["regime_factor"] = round(regime_factor, 4)

    obs_dates = sorted(r.get("observation_date", "") for r in clean_surv if r.get("observation_date"))
    temporal_factor = 0.0
    if len(obs_dates) >= 2:
        try:
            oldest = date.fromisoformat(obs_dates[0])
            newest = date.fromisoformat(obs_dates[-1])
            months = (newest - oldest).days / 30.0
            temporal_factor = _clamp(months / 6.0)
        except Exception:
            temporal_factor = 0.30
    sub["temporal_factor"] = round(temporal_factor, 4)

    fwd10 = _fwd10_vals(clean_surv)
    if len(fwd10) >= 3:
        std_10d          = float(np.std(fwd10))
        variance_factor  = _clamp(1.0 - std_10d / 0.15)
    else:
        variance_factor  = 0.0
    sub["variance_factor"] = round(variance_factor, 4)

    sorted_surv   = _sort_by_obs_date(clean_surv)
    recent_recs   = sorted_surv[-RECENT_WINDOW:] if len(sorted_surv) > RECENT_WINDOW else sorted_surv
    recent_fwd10  = _fwd10_vals(recent_recs)
    if recent_fwd10:
        recent_pos_rate = float(np.mean([v > 0 for v in recent_fwd10]))
        recent_factor   = _clamp((recent_pos_rate - 0.30) / 0.40)
    else:
        recent_factor   = 0.0
    sub["recent_factor"] = round(recent_factor, 4)

    confidence = (
        _CONF_WEIGHTS["sample"]   * sample_factor
        + _CONF_WEIGHTS["regime"]   * regime_factor
        + _CONF_WEIGHTS["temporal"] * temporal_factor
        + _CONF_WEIGHTS["variance"] * variance_factor
        + _CONF_WEIGHTS["recent"]   * recent_factor
    )
    return round(_clamp(confidence), 4), sub


# ─────────────────────────────────────────────────────────────────────────────
# FAIL_CLOSED blocker check (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _check_blockers(
    clean_surv:  List[dict],
    clean_fail:  List[dict],
    regime_tags: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    """
    Evaluate all AUTO_PROMOTION_BLOCKERS.
    Returns (blockers: List[str], warnings: List[str]).

    FAIL_CLOSED:
      - insufficient_clean_sample: immediate short-circuit
      - all other blockers checked independently
    """
    blockers: List[str] = []
    warnings: List[str] = []

    n_clean = len(clean_surv)

    # 1. Insufficient clean sample — short-circuit
    if n_clean < MIN_CLEAN_SAMPLE:
        blockers.append("insufficient_clean_sample")
        warnings.append(
            f"only {n_clean} CLEAN records (min={MIN_CLEAN_SAMPLE})"
        )
        return blockers, warnings

    sorted_surv  = _sort_by_obs_date(clean_surv)
    recent_recs  = sorted_surv[-RECENT_WINDOW:] if len(sorted_surv) > RECENT_WINDOW else sorted_surv
    recent_fwd10 = _fwd10_vals(recent_recs)

    # 2. Negative recent expectancy
    if recent_fwd10:
        recent_mean = float(np.mean(recent_fwd10))
        if recent_mean < _BLK_RECENT_EXPCT_MIN:
            blockers.append("negative_recent_expectancy")
            warnings.append(
                f"recent_mean_10d={recent_mean*100:+.1f}% (threshold={_BLK_RECENT_EXPCT_MIN*100:+.1f}%)"
            )
    else:
        warnings.append("no recent fwd10 values — recent expectancy unverifiable")

    # 3. Risk-off instability
    roff_vals = [
        float(r["forward_return_10d"])
        for r in clean_surv
        if regime_tags.get(r.get("observation_date", ""), "range") == "risk_off"
        and r.get("forward_return_10d") is not None
    ]
    if len(roff_vals) >= _BLK_ROFF_SAMPLE_MIN:
        roff_mean = float(np.mean(roff_vals))
        if roff_mean < _BLK_ROFF_MEAN_MIN:
            blockers.append("risk_off_instability")
            warnings.append(
                f"risk_off mean_10d={roff_mean*100:+.1f}% (threshold={_BLK_ROFF_MEAN_MIN*100:+.1f}%)"
            )

    # 4. High instant rejection
    if clean_fail:
        n_fail        = len(clean_fail)
        instant_count = sum(
            1 for r in clean_fail
            if r.get("failure_detected", False)
            and r.get("failure_day_offset") is not None
            and int(r["failure_day_offset"]) <= 3
        )
        instant_rate = instant_count / n_fail
        if instant_rate > _BLK_INSTANT_REJ_MAX:
            blockers.append("high_instant_rejection")
            warnings.append(
                f"instant_rejection_rate={instant_rate*100:.0f}% (max={_BLK_INSTANT_REJ_MAX*100:.0f}%)"
            )

    # 5. Temporal degradation
    all_fwd10 = _fwd10_vals(sorted_surv)
    if len(all_fwd10) >= 10:
        mid         = len(all_fwd10) // 2
        early_mean  = float(np.mean(all_fwd10[:mid]))
        recent_mean_all = float(np.mean(all_fwd10[mid:]))
        eps         = 0.001
        degradation = (early_mean - recent_mean_all) / max(abs(early_mean) + eps, eps)
        if (early_mean > 0.005
                and recent_mean_all < -0.010
                and degradation > _BLK_TEMP_DEGRADE):
            blockers.append("temporal_degradation")
            warnings.append(
                f"early={early_mean*100:+.1f}% → recent={recent_mean_all*100:+.1f}%"
                f" (degradation={degradation:.2f})"
            )

    # 6. Survivability collapse
    if clean_fail and len(clean_fail) >= 10:
        sorted_fail  = _sort_by_obs_date(clean_fail)

        def _surv_days(recs: List[dict]) -> List[int]:
            out = []
            for r in recs:
                if r.get("failure_detected", False):
                    offset = r.get("failure_day_offset")
                    if offset is not None:
                        out.append(int(offset))
                else:
                    out.append(_HALF_LIFE_WINDOW)
            return out

        all_sdays    = _surv_days(sorted_fail)
        recent_fail  = sorted_fail[-RECENT_WINDOW:] if len(sorted_fail) > RECENT_WINDOW else sorted_fail
        recent_sdays = _surv_days(recent_fail)

        if all_sdays and len(recent_sdays) >= 3:
            med_all    = float(np.median(all_sdays))
            med_recent = float(np.median(recent_sdays))
            if med_all > 0 and med_recent < _BLK_SURV_RATIO * med_all:
                blockers.append("survivability_collapse")
                warnings.append(
                    f"survival: overall={med_all:.1f}d → recent={med_recent:.1f}d"
                    f" (ratio={med_recent/med_all:.2f} < {_BLK_SURV_RATIO})"
                )

    # 7. Regime instability
    regime_means: Dict[str, float] = {}
    for rgm in _REGIMES:
        rv = [
            float(r["forward_return_10d"])
            for r in clean_surv
            if regime_tags.get(r.get("observation_date", ""), "range") == rgm
            and r.get("forward_return_10d") is not None
        ]
        if len(rv) >= MIN_REGIME_SAMPLE:
            regime_means[rgm] = float(np.mean(rv))

    if len(regime_means) >= 2:
        means_arr = list(regime_means.values())
        reg_std   = float(np.std(means_arr))
        if reg_std > _BLK_REGIME_STD and any(v < -0.05 for v in means_arr):
            blockers.append("regime_instability")
            warnings.append(
                f"regime std={reg_std*100:.1f}% with extreme negative regime"
            )

    return blockers, warnings


# ─────────────────────────────────────────────────────────────────────────────
# State machine (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _determine_promotion_state(
    readiness_score:  float,
    confidence:       float,
    regime_coverage:  Dict[str, int],
    blockers:         List[str],
) -> PromotionState:
    """
    Determine lifecycle state.
    FAIL_CLOSED: any blocker → BLOCKED.
    Returns highest satisfied state given score / confidence / coverage conditions.

    Future integration points (labeled — do NOT activate here):
      SHADOW_READY   → ShadowAllocatorSimulation.run(report)
      PAPER_READY    → PaperExecutionEngine.simulate(report)
      LIMITED_EXECUTION_READY → CapitalRouter.route_limited(report) + FreezeBreaker guard
      PRODUCTION_READY → DeployableUniverseGovernor.evaluate(report) + Allocator hook
    """
    if blockers:
        return PromotionState.BLOCKED

    covered = sum(1 for cnt in regime_coverage.values() if cnt >= MIN_REGIME_SAMPLE)

    if (readiness_score >= _THRESH_SCORE["production_ready"]
            and confidence >= _THRESH_CONF["production_ready"]):
        return PromotionState.PRODUCTION_READY

    if (readiness_score >= _THRESH_SCORE["limited_execution_ready"]
            and confidence >= _THRESH_CONF["limited_execution_ready"]):
        return PromotionState.LIMITED_EXECUTION_READY

    if (readiness_score >= _THRESH_SCORE["paper_ready"]
            and confidence >= _THRESH_CONF["paper_ready"]
            and covered >= _THRESH_REGIME_COV.get("paper_ready", 0)):
        return PromotionState.PAPER_READY

    if (readiness_score >= _THRESH_SCORE["shadow_ready"]
            and confidence >= _THRESH_CONF["shadow_ready"]):
        return PromotionState.SHADOW_READY

    return PromotionState.RESEARCH_ONLY


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic hash (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_deterministic_hash(inputs: PromotionReadinessInputs) -> str:
    """
    SHA256 of canonical input representation.
    Sorted by (symbol, observation_date) → reproducible across runs.
    """
    def _sort_recs(recs: List[dict]) -> List[dict]:
        return sorted(recs, key=lambda r: (r.get("symbol", ""), r.get("observation_date", "")))

    canonical = {
        "survivability":  _sort_recs(inputs.survivability_records),
        "failure":        _sort_recs(inputs.failure_records),
        "regime_tags":    dict(sorted(inputs.regime_tags.items())),
        "evaluation_date": inputs.evaluation_date,
    }
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation entry point (pure — no I/O)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_promotion_readiness(
    inputs: PromotionReadinessInputs,
) -> PromotionReadinessReport:
    """
    Evaluate promotion readiness for Future Leader research telemetry.

    FAIL_CLOSED: any AUTO_PROMOTION_BLOCKER → BLOCKED, promotion_ready=False.
    Deterministic: same inputs → same output + same deterministic_hash.
    observation_only=True: no execution path connection.

    Component scores are always computed for diagnostic transparency, even when BLOCKED.
    Returns PromotionReadinessReport with full component and blocker breakdown.
    """
    ts = datetime.now(JST).isoformat()

    det_hash   = _compute_deterministic_hash(inputs)
    clean_surv = _filter_clean_survivability(inputs.survivability_records)
    clean_fail = _filter_clean_failure(inputs.failure_records)

    # Regime coverage count from CLEAN survivability records
    regime_coverage: Dict[str, int] = {r: 0 for r in _REGIMES}
    for rec in clean_surv:
        obs_date = rec.get("observation_date", "")
        regime   = inputs.regime_tags.get(obs_date, "range")
        regime_coverage[regime] = regime_coverage.get(regime, 0) + 1

    # Recent sample count
    sorted_surv  = _sort_by_obs_date(clean_surv)
    recent_recs  = sorted_surv[-RECENT_WINDOW:] if len(sorted_surv) > RECENT_WINDOW else sorted_surv
    recent_count = len(recent_recs)

    # FAIL_CLOSED blocker evaluation
    blockers, warnings = _check_blockers(clean_surv, clean_fail, inputs.regime_tags)

    # Component scores (always computed — diagnostic transparency)
    comp_edge = _evaluate_statistical_edge(clean_surv)
    comp_surv = _evaluate_survivability(clean_surv, clean_fail)
    comp_reg  = _evaluate_regime_robustness(clean_surv, inputs.regime_tags)
    comp_fail = _evaluate_failure_quality(clean_fail)
    comp_temp = _evaluate_temporal_stability(clean_surv)

    components = [comp_edge, comp_surv, comp_reg, comp_fail, comp_temp]

    readiness_score = round(
        _clamp(sum(c.score * c.weight for c in components)), 4
    )
    component_scores = {c.name: c.score for c in components}

    confidence, _conf_sub = _evaluate_promotion_confidence(clean_surv, inputs.regime_tags)

    state = _determine_promotion_state(
        readiness_score, confidence, regime_coverage, blockers
    )
    promotion_ready = state not in (PromotionState.BLOCKED, PromotionState.RESEARCH_ONLY)

    # Collect diagnostic warnings from components (cap to avoid log explosion)
    for comp in components:
        for note in comp.notes:
            w = f"[{comp.name}] {note}"
            if w not in warnings:
                warnings.append(w)

    return PromotionReadinessReport(
        timestamp=ts,
        promotion_state=state,
        promotion_ready=promotion_ready,
        readiness_score=readiness_score,
        promotion_confidence=confidence,
        component_scores=component_scores,
        blocking_reasons=sorted(blockers),
        warnings=warnings,
        sample_size=len(clean_surv),
        recent_sample_size=recent_count,
        regime_coverage=regime_coverage,
        deterministic_hash=det_hash,
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSONL persistence (append-only)
# ─────────────────────────────────────────────────────────────────────────────

def append_promotion_report(
    report:  PromotionReadinessReport,
    log_dir: Path,
    today:   Optional[date] = None,
) -> None:
    """
    Append one PromotionReadinessReport to today's JSONL.
    append-only: never overwrites existing records.
    Fail-open: I/O errors are logged but not re-raised.
    """
    if today is None:
        today = datetime.now(JST).date()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"future_leader_promotion_readiness_{today.strftime('%Y%m%d')}.jsonl"
        line     = json.dumps(report.to_dict(), ensure_ascii=False)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        logger.info(
            "[FL_PROMO] appended → %s state=%s hash=%.8s…",
            log_path.name, report.promotion_state.value, report.deterministic_hash,
        )
    except Exception as e:
        logger.warning("[FL_PROMO] append failed (fail-open): %s", e)


def load_promotion_reports(log_dir: Path) -> List[dict]:
    """Load all promotion readiness JSONL records in chronological order. Fail-open."""
    records: List[dict] = []
    if not log_dir.exists():
        return records
    for p in sorted(log_dir.glob("future_leader_promotion_readiness_*.jsonl")):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        except Exception:
            pass
    return records


# ─────────────────────────────────────────────────────────────────────────────
# I/O loaders (separated from computation)
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl_dir(log_dir: Path, pattern: str) -> List[dict]:
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


def _load_regime_tags_from_dir(regime_log_dir: Path) -> Dict[str, str]:
    """Load {observation_date: regime_tag} from regime JSONL. Last write wins."""
    result: Dict[str, str] = {}
    for rec in _load_jsonl_dir(regime_log_dir, "future_leader_regime_*.jsonl"):
        dt  = rec.get("observation_date", "")
        tag = rec.get("regime_tag", "")
        if dt and tag:
            result[dt] = tag
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Report formatter (pure — no I/O)
# ─────────────────────────────────────────────────────────────────────────────

def format_promotion_readiness_section(report: PromotionReadinessReport) -> str:
    """Render PROMOTION READINESS section for daily report. Pure function."""
    lines = [
        "=== PROMOTION READINESS ===",
        f"state:           {report.promotion_state.value}",
        f"promotion_ready: {report.promotion_ready}",
        f"readiness_score: {report.readiness_score:.3f}",
        f"confidence:      {report.promotion_confidence:.3f}",
        f"sample:          {report.sample_size} (recent={report.recent_sample_size})",
        "",
        "component_scores:",
    ]
    for name, score in sorted(report.component_scores.items()):
        lines.append(f"  {name:26s}: {score:.3f}")

    cov   = report.regime_coverage
    total = sum(cov.values())
    lines += [
        "",
        f"regime_coverage (n={total}):",
        f"  risk_on={cov.get('risk_on',0)}  risk_off={cov.get('risk_off',0)}"
        f"  range={cov.get('range',0)}  volatile={cov.get('volatile',0)}",
    ]

    if report.blocking_reasons:
        lines += ["", "BLOCKERS (FAIL_CLOSED):"]
        for b in report.blocking_reasons:
            lines.append(f"  [BLOCKED] {b}")

    if report.warnings:
        lines += ["", "warnings:"]
        for w in report.warnings[:6]:
            lines.append(f"  ! {w}")

    lines += ["", f"hash: {report.deterministic_hash[:16]}…"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# run_live_signal.py integration hook (fail-open)
# ─────────────────────────────────────────────────────────────────────────────

def run_future_leader_promotion_readiness_hook(
    survivability_log_dir: Path,
    failure_log_dir:       Path,
    regime_log_dir:        Path,
    promotion_log_dir:     Path,
    reports_dir:           Path,
    today:                 Optional[date] = None,
    persist:               bool = True,
) -> Optional[PromotionReadinessReport]:
    """
    Integration hook for run_live_signal.py. Fail-open.

    Reads existing analytics JSONL dirs (no OHLCV reload needed).
    Evaluates promotion readiness and writes audit JSONL.
    Appends PROMOTION READINESS section to today's future_leader_report.

    observation_only=True: no execution mutation.
    FAIL_CLOSED enforced internally — this hook does NOT advance any execution state.

    Future integration points (activate ONLY after each layer is ready):
      SHADOW_READY   → call ShadowAllocatorSimulation.run(report)
      PAPER_READY    → call PaperExecutionEngine.simulate(report)
      LIMITED_EXECUTION_READY → call CapitalRouter.route_limited() + FreezeBreaker guard
      PRODUCTION_READY → call DeployableUniverseGovernor.evaluate() + Allocator hook
                         + Auto Rollback guard

    Positioned after run_future_leader_regime_hook in run_live_signal.py.
    """
    if today is None:
        today = datetime.now(JST).date()

    try:
        surv_recs = _load_jsonl_dir(survivability_log_dir, "future_leader_survivability_*.jsonl")
        fail_recs = _load_jsonl_dir(failure_log_dir, "future_leader_failure_*.jsonl")
        reg_tags  = _load_regime_tags_from_dir(regime_log_dir)

        if not surv_recs:
            logger.info("[FL_PROMO] no survivability records — hook skipped")
            return None

        inputs = PromotionReadinessInputs(
            survivability_records=surv_recs,
            failure_records=fail_recs,
            regime_tags=reg_tags,
            evaluation_date=today.isoformat(),
        )
        report = evaluate_promotion_readiness(inputs)

        if persist:
            append_promotion_report(report, promotion_log_dir, today=today)

        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"future_leader_report_{today.strftime('%Y%m%d')}.md"
        section     = format_promotion_readiness_section(report)
        mode = "a" if report_path.exists() else "w"
        with report_path.open(mode, encoding="utf-8") as f:
            f.write("\n\n" + section + "\n")

        logger.info(
            "[FL_PROMO] hook done: state=%s score=%.3f conf=%.3f blockers=%s",
            report.promotion_state.value,
            report.readiness_score,
            report.promotion_confidence,
            report.blocking_reasons,
        )
        return report

    except Exception as e:
        logger.warning("[FL_PROMO] hook failed (fail-open): %s", e)
        return None
