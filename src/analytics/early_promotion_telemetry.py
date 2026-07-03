"""
src/analytics/early_promotion_telemetry.py
Early Promotion Telemetry — detect "5301.T型 mature後検出" before it happens.

Purpose:
  Detect future leader candidates that show early promotion signals
  BEFORE they reach full RSR maturity. Telemetry only — no execution.

  Addresses the root cause of "matured symbols detected too late" problem:
  by the time RSR is strong enough for normal promotion gate, much of
  the upside is already captured by institutional holders.

INVARIANTS:
  - observation_only: True always
  - no LIVE_UNIVERSE auto-addition
  - no execution mutation
  - no predictive_entry_enabled change
  - append-only JSONL

Early promotion condition (ALL must hold):
  accum_score      >= 0.20   — institutional accumulation building
  drift_5d         >= 0.20   — 5-day persistent drift above threshold
  consecutive_days >= 3      — signal persists at least 3 consecutive days
  rsr_zone         == "weak" — RSR still weak (not yet mature — early signal)
  future_leader_score >= 70  — composite score above threshold

Output: EarlyPromotionSignal (telemetry only, no auto-promotion)
JSONL: logs/early_promotion_telemetry/early_promotion_YYYYMMDD.jsonl
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

# ── Early promotion thresholds (変更前ユーザー確認必須) ────────────────────────
_EARLY_ACCUM_MIN:   float = 0.20
_EARLY_DRIFT_MIN:   float = 0.20
_EARLY_CONSEC_MIN:  int   = 3
_EARLY_ZONE:        str   = "weak"    # must be exactly this zone
_EARLY_SCORE_MIN:   float = 70.0


@dataclass(frozen=True)
class EarlyPromotionSignal:
    """
    One early promotion candidate detection.
    observation_only is always True — governance read-only.
    """
    observation_only:    bool   # must always be True
    signal_date:         str
    symbol:              str
    future_leader_score: float
    rsr_zone:            str
    accum_score:         float
    drift_5d:            float
    consecutive_days:    int
    condition_met:       bool
    conditions_detail:   Dict[str, Any]
    timestamp:           str = field(default_factory=lambda: datetime.now(JST).isoformat())

    def __post_init__(self) -> None:
        if not self.observation_only:
            raise ValueError("EarlyPromotionSignal.observation_only must always be True")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_only":    self.observation_only,
            "signal_date":         self.signal_date,
            "symbol":              self.symbol,
            "future_leader_score": self.future_leader_score,
            "rsr_zone":            self.rsr_zone,
            "accum_score":         self.accum_score,
            "drift_5d":            self.drift_5d,
            "consecutive_days":    self.consecutive_days,
            "condition_met":       self.condition_met,
            "conditions_detail":   self.conditions_detail,
            "timestamp":           self.timestamp,
        }


def evaluate_early_promotion(
    symbol:              str,
    signal_date:         str,
    future_leader_score: float,
    rsr_zone:            str,
    accum_score:         float,
    drift_5d:            float,
    consecutive_days:    int,
) -> EarlyPromotionSignal:
    """
    Pure function: evaluate early promotion conditions for one symbol.
    Returns EarlyPromotionSignal with observation_only=True always.
    No side effects.
    """
    cond_accum  = accum_score    >= _EARLY_ACCUM_MIN
    cond_drift  = drift_5d       >= _EARLY_DRIFT_MIN
    cond_consec = consecutive_days >= _EARLY_CONSEC_MIN
    cond_zone   = rsr_zone       == _EARLY_ZONE
    cond_score  = future_leader_score >= _EARLY_SCORE_MIN

    condition_met = cond_accum and cond_drift and cond_consec and cond_zone and cond_score

    conditions_detail = {
        "accum_score":      {"value": round(accum_score, 4), "threshold": _EARLY_ACCUM_MIN,  "met": cond_accum},
        "drift_5d":         {"value": round(drift_5d, 4),    "threshold": _EARLY_DRIFT_MIN,  "met": cond_drift},
        "consecutive_days": {"value": consecutive_days,      "threshold": _EARLY_CONSEC_MIN, "met": cond_consec},
        "rsr_zone":         {"value": rsr_zone,              "required":  _EARLY_ZONE,       "met": cond_zone},
        "future_leader_score": {"value": round(future_leader_score, 2), "threshold": _EARLY_SCORE_MIN, "met": cond_score},
    }

    return EarlyPromotionSignal(
        observation_only=True,
        signal_date=signal_date,
        symbol=symbol,
        future_leader_score=future_leader_score,
        rsr_zone=rsr_zone,
        accum_score=accum_score,
        drift_5d=drift_5d,
        consecutive_days=consecutive_days,
        condition_met=condition_met,
        conditions_detail=conditions_detail,
    )


def append_early_promotion(signal: EarlyPromotionSignal, log_dir: Path) -> None:
    """
    Append EarlyPromotionSignal to dated JSONL.
    Raises IOError on failure (telemetry integrity).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = signal.signal_date.replace("-", "")
    out_path = log_dir / f"early_promotion_{date_str}.jsonl"
    line = json.dumps(signal.to_dict(), ensure_ascii=False) + "\n"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(line)


def load_early_promotion(log_dir: Path, date_str: Optional[str] = None) -> List[EarlyPromotionSignal]:
    """
    Load EarlyPromotionSignal records from JSONL.
    date_str: YYYYMMDD; None loads all.
    """
    if not log_dir.exists():
        return []
    signals: List[EarlyPromotionSignal] = []
    if date_str:
        paths = [log_dir / f"early_promotion_{date_str}.jsonl"]
    else:
        paths = sorted(log_dir.glob("early_promotion_*.jsonl"))
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                signals.append(EarlyPromotionSignal(
                    observation_only=True,   # always True — enforce invariant on load
                    signal_date=str(d.get("signal_date", "")),
                    symbol=str(d.get("symbol", "")),
                    future_leader_score=float(d.get("future_leader_score", 0.0)),
                    rsr_zone=str(d.get("rsr_zone", "")),
                    accum_score=float(d.get("accum_score", 0.0)),
                    drift_5d=float(d.get("drift_5d", 0.0)),
                    consecutive_days=int(d.get("consecutive_days", 0)),
                    condition_met=bool(d.get("condition_met", False)),
                    conditions_detail=dict(d.get("conditions_detail", {})),
                    timestamp=str(d.get("timestamp", "")),
                ))
            except Exception as exc:
                logger.warning("[EARLY_PROM] parse failed: %s", exc)
    return signals


def run_early_promotion_telemetry_hook(
    future_leader_candidates: List[Dict[str, Any]],
    log_dir:                  Path,
    signal_date:              str,
) -> List[EarlyPromotionSignal]:
    """
    Convenience hook for run_live_signal.py integration.
    Evaluates early promotion conditions for each candidate.
    Logs those with condition_met=True (and all if log_all=True).
    Returns list of signals with condition_met=True.

    future_leader_candidates: list of dicts with keys:
      symbol, future_leader_score, rsr_zone, accum_score, drift_5d, consecutive_days
    """
    fired: List[EarlyPromotionSignal] = []

    for cand in future_leader_candidates:
        sym = str(cand.get("symbol", ""))
        if not sym:
            continue
        sig = evaluate_early_promotion(
            symbol=sym,
            signal_date=signal_date,
            future_leader_score=float(cand.get("future_leader_score", 0.0)),
            rsr_zone=str(cand.get("rsr_zone", "")),
            accum_score=float(cand.get("accum_score", 0.0)),
            drift_5d=float(cand.get("drift_5d", 0.0)),
            consecutive_days=int(cand.get("consecutive_days", 0)),
        )
        if sig.condition_met:
            try:
                append_early_promotion(sig, log_dir)
            except Exception as exc:
                logger.warning("[EARLY_PROM] append failed %s: %s", sym, exc)
            fired.append(sig)
            logger.info(
                "[EARLY_PROM] candidate detected: symbol=%s score=%.1f zone=%s consec=%d",
                sym, sig.future_leader_score, sig.rsr_zone, sig.consecutive_days,
            )

    if fired:
        logger.info(
            "[EARLY_PROM] %d early promotion candidate(s) detected on %s — telemetry only",
            len(fired), signal_date,
        )
    return fired
