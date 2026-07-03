"""
continuation_priority.py — Continuation Priority Intelligence

Ranks held positions by capital efficiency (0-100).
Answers: "among max_positions=3 slots, which deserves the scarce capital most?"

Observation-only. No execution changes, no sizing changes, no auto-exit.
Purely ranking visibility for DRY/LIVE reports and forward-return validation.

Score formula:
  base = compression_score * 0.25
       + breakout_quality_score * 0.25
       + rsr_strength_score * 0.20   (normalised, not raw percentile)
       + mfe_quality_score * 0.15    (proxy from unrealized_pnl_pct when peak unknown)
       + hold_quality_score * 0.15   (peak zone 5-40d; penalises overstay)

  bonuses:   healthy_breakout +10, suppression_eligible +5,
             high-MFE/low-giveback +8, rsr>80 +5, active breakout +5
  penalties: distribution -15, failed_breakout -10,
             high giveback -8, negative RSR momentum -5, overstay>50d -5

Priority tiers:
  core_continuation  ≥ 80
  high_quality       65-79
  neutral            45-64
  exit_candidate     < 45

Fail policy: all public functions are FAIL_OPEN.
Runtime: O(n), n = max_positions (typically 3).
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_EPS = 1e-9
SCHEMA_VERSION = "v1"
_JST = timezone(timedelta(hours=9))

# ── Priority tiers ─────────────────────────────────────────────────────────────
TIER_CORE    = "core_continuation"
TIER_HIGH    = "high_quality"
TIER_NEUTRAL = "neutral"
TIER_EXIT    = "exit_candidate"

TIER_THRESHOLDS: Dict[str, float] = {
    TIER_CORE:    80.0,
    TIER_HIGH:    65.0,
    TIER_NEUTRAL: 45.0,
    TIER_EXIT:    0.0,
}

# ── Score weights (must sum to 1.0) ───────────────────────────────────────────
W_COMPRESSION = 0.25
W_BQ          = 0.25
W_RSR         = 0.20
W_MFE         = 0.15
W_HOLD        = 0.15

# ── RSR normalization range ───────────────────────────────────────────────────
RSR_SCORE_FLOOR = 60.0    # RSR below this scores 0
RSR_SCORE_CEIL  = 100.0   # RSR at or above this scores 100

# ── MFE normalization breakpoints ─────────────────────────────────────────────
MFE_ZERO_AT   = 0.0
MFE_HALF_AT   = 0.05     # 5% unrealized → 50 pts
MFE_FULL_AT   = 0.15     # 15% unrealized → 100 pts

# ── Hold quality breakpoints ──────────────────────────────────────────────────
HOLD_MIN_DAYS     = 3     # < 3d: early, 30 pts
HOLD_RAMP_END     = 5     # 3-5d: ramp 30-60
HOLD_PRIME_START  = 5
HOLD_PRIME_END    = 40    # 5-40d: prime zone 60-90
HOLD_DECLINE_RATE = 2.5   # pts lost per day after 40d
HOLD_OVERSTAY_AT  = 50    # overstay penalty trigger

# ── Bonus constants ────────────────────────────────────────────────────────────
BONUS_HEALTHY_BREAKOUT   = 10.0
BONUS_SUPPRESSION        = 5.0
BONUS_MFE_RETENTION      = 8.0   # high MFE + low giveback
BONUS_RSR_STRONG         = 5.0   # RSR > 80
BONUS_ACTIVE_BREAKOUT    = 5.0   # compression_phase == "breakout"

# ── Penalty constants ──────────────────────────────────────────────────────────
PENALTY_DISTRIBUTION     = 15.0
PENALTY_FAILED_BREAKOUT  = 10.0
PENALTY_HIGH_GIVEBACK    = 8.0   # giveback_ratio > 0.40
PENALTY_RSR_NEGATIVE_MOM = 5.0
PENALTY_OVERSTAY         = 5.0   # hold_days > HOLD_OVERSTAY_AT

# ── Thresholds for bonus/penalty triggers ─────────────────────────────────────
MFE_RETENTION_MIN_MFE    = 0.12   # mfe_pct > 12% for MFE retention bonus
MFE_RETENTION_MAX_GB     = 0.20   # giveback_ratio < 20% for MFE retention bonus
RSR_STRONG_THRESHOLD     = 80.0
GIVEBACK_HIGH_THRESHOLD  = 0.40

# ── Materialization ───────────────────────────────────────────────────────────
_MAT_BUFFER_5D = 7    # calendar days before materializing 5d return


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContinuationPriorityInput:
    symbol:                str
    hold_days:             int
    unrealized_pnl_pct:    float
    compression_score:     float    # [0-100] from compression_continuation_hook
    breakout_quality_score: float   # [0-100] from breakout_quality module (or 50)
    rsr:                   float    # RSR percentile
    rsr_momentum:          float    # WinnerConfirmation.trend_quality
    mfe_pct:               float    # proxy: unrealized_pnl_pct when peak unknown
    giveback_ratio:        float    # proxy: 0.0 when peak unknown
    current_phase:         str      # BQ phase ("healthy_breakout" etc.) or compression phase
    suppression_eligible:  bool     # from compression_continuation_hook


@dataclass
class ContinuationPriorityResult:
    symbol:               str
    priority_score:       float     # 0-100 final score
    priority_tier:        str       # TIER_* constant
    base_score:           float     # weighted component sum before adjustments
    bonus:                float     # total bonus applied
    penalty:              float     # total penalty applied
    # Component scores (for explainability)
    compression_component: float
    bq_component:         float
    rsr_component:        float
    mfe_component:        float
    hold_component:       float


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers — pure functions
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def rsr_strength_score(rsr: float) -> float:
    """Map RSR percentile → [0-100] score. 60=0, 100=100. Linear interpolation."""
    v = _safe_float(rsr)
    span = RSR_SCORE_CEIL - RSR_SCORE_FLOOR
    if span <= 0:
        return 0.0
    return round(max(0.0, min(100.0, (v - RSR_SCORE_FLOOR) / span * 100.0)), 2)


def mfe_quality_score(mfe_pct: float) -> float:
    """
    Map MFE (proxy: unrealized_pnl_pct) → [0-100] score.
    Negative: 0. 0→5%: linear 0-50. 5%→15%: linear 50-100. >15%: 100.
    """
    v = _safe_float(mfe_pct)
    if v <= MFE_ZERO_AT:
        return 0.0
    if v <= MFE_HALF_AT:
        return round(v / MFE_HALF_AT * 50.0, 2)
    if v <= MFE_FULL_AT:
        return round(50.0 + (v - MFE_HALF_AT) / (MFE_FULL_AT - MFE_HALF_AT) * 50.0, 2)
    return 100.0


def hold_quality_score(hold_days: int) -> float:
    """
    Map hold_days → [0-100] score.
    <3d: very early (30 pts). 3-5d: ramp 30-60. 5-40d: prime zone 60-90.
    >40d: gradually declining (overstay risk). >50d: hard penalty.
    """
    d = max(0, int(_safe_float(hold_days, 0)))
    if d < HOLD_MIN_DAYS:
        return 30.0
    if d < HOLD_RAMP_END:
        return round(30.0 + (d - HOLD_MIN_DAYS) / (HOLD_RAMP_END - HOLD_MIN_DAYS) * 30.0, 2)
    if d <= HOLD_PRIME_END:
        # 5d=60, 40d=90: linear
        return round(60.0 + (d - HOLD_PRIME_START) / (HOLD_PRIME_END - HOLD_PRIME_START) * 30.0, 2)
    return round(max(30.0, 90.0 - (d - HOLD_PRIME_END) * HOLD_DECLINE_RATE), 2)


def classify_tier(score: float) -> str:
    """Map priority_score → priority tier."""
    s = _safe_float(score)
    if s >= TIER_THRESHOLDS[TIER_CORE]:
        return TIER_CORE
    if s >= TIER_THRESHOLDS[TIER_HIGH]:
        return TIER_HIGH
    if s >= TIER_THRESHOLDS[TIER_NEUTRAL]:
        return TIER_NEUTRAL
    return TIER_EXIT


# ─────────────────────────────────────────────────────────────────────────────
# Main pure API
# ─────────────────────────────────────────────────────────────────────────────

def compute_continuation_priority(inp: ContinuationPriorityInput) -> ContinuationPriorityResult:
    """
    Score a held position's continuation priority.

    Pure function. Never raises.
    Returns TIER_NEUTRAL with score=50 on any error.
    """
    try:
        comp_s = min(100.0, max(0.0, _safe_float(inp.compression_score)))
        bq_s   = min(100.0, max(0.0, _safe_float(inp.breakout_quality_score, 50.0)))
        rsr_s  = rsr_strength_score(inp.rsr)
        mfe_s  = mfe_quality_score(inp.mfe_pct)
        hold_s = hold_quality_score(inp.hold_days)

        comp_w = round(comp_s * W_COMPRESSION, 2)
        bq_w   = round(bq_s   * W_BQ,          2)
        rsr_w  = round(rsr_s  * W_RSR,          2)
        mfe_w  = round(mfe_s  * W_MFE,          2)
        hold_w = round(hold_s * W_HOLD,          2)
        base   = comp_w + bq_w + rsr_w + mfe_w + hold_w

        # ── Bonuses ───────────────────────────────────────────────────────────
        bonus = 0.0
        phase = str(inp.current_phase or "neutral")

        if phase == "healthy_breakout":
            bonus += BONUS_HEALTHY_BREAKOUT
        if inp.suppression_eligible:
            bonus += BONUS_SUPPRESSION
        if (
            _safe_float(inp.mfe_pct) > MFE_RETENTION_MIN_MFE
            and _safe_float(inp.giveback_ratio) < MFE_RETENTION_MAX_GB
        ):
            bonus += BONUS_MFE_RETENTION
        if _safe_float(inp.rsr) > RSR_STRONG_THRESHOLD:
            bonus += BONUS_RSR_STRONG
        if phase == "breakout":   # compression_phase = active breakout expansion
            bonus += BONUS_ACTIVE_BREAKOUT

        # ── Penalties ─────────────────────────────────────────────────────────
        penalty = 0.0
        if phase == "distribution":
            penalty += PENALTY_DISTRIBUTION
        if phase == "failed_breakout":
            penalty += PENALTY_FAILED_BREAKOUT
        if _safe_float(inp.giveback_ratio) > GIVEBACK_HIGH_THRESHOLD:
            penalty += PENALTY_HIGH_GIVEBACK
        if _safe_float(inp.rsr_momentum) < 0.0:
            penalty += PENALTY_RSR_NEGATIVE_MOM
        if inp.hold_days > HOLD_OVERSTAY_AT:
            penalty += PENALTY_OVERSTAY

        score = round(max(0.0, min(100.0, base + bonus - penalty)), 2)
        tier  = classify_tier(score)

        return ContinuationPriorityResult(
            symbol=inp.symbol,
            priority_score=score,
            priority_tier=tier,
            base_score=round(base, 2),
            bonus=round(bonus, 2),
            penalty=round(penalty, 2),
            compression_component=comp_w,
            bq_component=bq_w,
            rsr_component=rsr_w,
            mfe_component=mfe_w,
            hold_component=hold_w,
        )
    except Exception as exc:
        logger.warning("[CP] compute failed %s: %s", getattr(inp, "symbol", "?"), exc)
        return ContinuationPriorityResult(
            symbol=getattr(inp, "symbol", ""),
            priority_score=50.0,
            priority_tier=TIER_NEUTRAL,
            base_score=50.0, bonus=0.0, penalty=0.0,
            compression_component=12.5, bq_component=12.5,
            rsr_component=10.0, mfe_component=7.5, hold_component=7.5,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Report formatting
# ─────────────────────────────────────────────────────────────────────────────

_TIER_EMOJI = {
    TIER_CORE:    "★",
    TIER_HIGH:    "◎",
    TIER_NEUTRAL: "○",
    TIER_EXIT:    "△",
}


def format_priority_ranking_table(
    results: List[Tuple["Any", ContinuationPriorityResult]],
) -> str:
    """
    Format inline ranking table for DRY/LIVE print output.

    results: list of (signal_dict, ContinuationPriorityResult), sorted desc by score.
    FAIL_OPEN: returns empty string on error.
    """
    try:
        if not results:
            return ""
        lines = [
            "",
            "── TOP CONTINUATION PRIORITY POSITIONS ──────────────────────────",
            f"  {'銘柄':<10} {'score':>6} {'tier':<20} {'phase':<22} {'pnl':>7} {'RSR':>6} {'保有':>4}",
            "  " + "-" * 76,
        ]
        for sig, res in results:
            emoji = _TIER_EMOJI.get(res.priority_tier, " ")
            pnl   = float(sig.get("unrealized_pnl_pct", 0.0)) if sig else 0.0
            rsr   = float(sig.get("rsr", 0.0)) if sig else 0.0
            hold  = int(sig.get("hold_days", 0)) if sig else 0
            phase = str(sig.get("compression_phase", "?") if sig else "?")[:20]
            lines.append(
                f"  {res.symbol:<10}"
                f" {res.priority_score:>6.1f}"
                f" {emoji}{res.priority_tier:<19}"
                f" {phase:<22}"
                f" {pnl:>+7.1%}"
                f" {rsr:>6.1f}"
                f" {hold:>3}d"
            )
        lines.append("──────────────────────────────────────────────────────────────────────────")
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("[CP] ranking table failed: %s", exc)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry I/O
# ─────────────────────────────────────────────────────────────────────────────

def _append_jsonl(record: dict, path: Path) -> None:
    """Atomic JSONL append. FAIL_OPEN."""
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
        logger.warning("[CP] append failed: %s", exc)


def append_cp_telemetry(
    result:          ContinuationPriorityResult,
    inp:             ContinuationPriorityInput,
    telemetry_path:  Path,
    date_str:        Optional[str] = None,
) -> None:
    """Append one telemetry record. Forward return null; materialized later. FAIL_OPEN."""
    if date_str is None:
        date_str = datetime.now(_JST).strftime("%Y-%m-%d")
    record = {
        "date":                    date_str,
        "symbol":                  result.symbol,
        "priority_score":          result.priority_score,
        "priority_tier":           result.priority_tier,
        "base_score":              result.base_score,
        "bonus":                   result.bonus,
        "penalty":                 result.penalty,
        "compression_score":       round(inp.compression_score, 2),
        "breakout_quality_score":  round(inp.breakout_quality_score, 2),
        "rsr":                     round(inp.rsr, 2),
        "rsr_momentum":            round(inp.rsr_momentum, 4),
        "mfe_pct":                 round(inp.mfe_pct, 4),
        "giveback_ratio":          round(inp.giveback_ratio, 4),
        "hold_days":               inp.hold_days,
        "current_phase":           inp.current_phase,
        "suppression_eligible":    inp.suppression_eligible,
        "subsequent_5d_return":    None,
        "materialized_5d":         False,
        "schema_version":          SCHEMA_VERSION,
    }
    _append_jsonl(record, telemetry_path)


# ─────────────────────────────────────────────────────────────────────────────
# Forward return materialization
# ─────────────────────────────────────────────────────────────────────────────

def _fwd_return(df: Any, record_date_str: str, n_trading_days: int) -> Optional[float]:
    """n-trading-day forward return. Handles DatetimeIndex or 'date' column."""
    try:
        import pandas as pd
        import numpy as np
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            date_col = next((c for c in ("date", "Date") if c in df.columns), None)
            if date_col is None:
                return None
            df.index = pd.to_datetime(df[date_col])
        df = df.sort_index()
        rec_dt = pd.Timestamp(record_date_str)
        mask = df.index >= rec_dt
        if not mask.any():
            return None
        entry_pos  = int(np.argmax(mask))
        target_pos = entry_pos + n_trading_days
        if target_pos >= len(df):
            return None
        close_col = "Close" if "Close" in df.columns else "close"
        entry_close  = float(df[close_col].iloc[entry_pos])
        target_close = float(df[close_col].iloc[target_pos])
        if entry_close <= _EPS:
            return None
        return (target_close - entry_close) / entry_close
    except Exception as exc:
        logger.warning("[CP] _fwd_return failed for %s: %s", record_date_str, exc)
        return None


def materialize_cp_returns(
    telemetry_path: Path,
    ohlcv_loader:   Callable[[str], Optional[Any]],
) -> Dict[str, int]:
    """Fill null 5d forward returns for old records. Atomic rewrite. FAIL_OPEN."""
    try:
        if not telemetry_path.exists():
            return {}
        lines = [l for l in telemetry_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            return {}

        today = datetime.now(_JST).date()
        records: List[dict] = []
        for line in lines:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

        n5 = 0
        cache: Dict[str, Any] = {}
        for rec in records:
            sym      = rec.get("symbol", "")
            date_str = rec.get("date", "")
            if not sym or not date_str:
                continue
            try:
                rec_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            if rec.get("materialized_5d") or (today - rec_date).days < _MAT_BUFFER_5D:
                continue
            if sym not in cache:
                try:
                    cache[sym] = ohlcv_loader(sym)
                except Exception as exc:
                    logger.warning("[CP] ohlcv_loader failed for %s: %s", sym, exc)
                    cache[sym] = None
            if cache[sym] is not None:
                ret = _fwd_return(cache[sym], date_str, 5)
                if ret is not None:
                    rec["subsequent_5d_return"] = round(ret, 6)
                    rec["materialized_5d"] = True
                    n5 += 1

        new_content = (
            "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in records) + "\n"
        )
        fd, tmp = tempfile.mkstemp(dir=str(telemetry_path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tf:
                tf.write(new_content)
                tf.flush()
                os.fsync(tf.fileno())
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise
        Path(tmp).replace(telemetry_path)
        return {"n_materialized_5d": n5}
    except Exception as exc:
        logger.warning("[CP] materialize failed: %s", exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# KPI aggregation
# ─────────────────────────────────────────────────────────────────────────────

def compute_cp_kpis(telemetry_path: Path) -> Dict[str, Any]:
    """
    Aggregate priority KPIs from telemetry JSONL.
    Key insight: avg_return_delta_core_vs_exit validates capital efficiency.
    FAIL_OPEN.
    """
    try:
        if not telemetry_path.exists():
            return {}
        records: List[dict] = []
        for line in telemetry_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
        if not records:
            return {}

        by_tier: Dict[str, List[dict]] = {t: [] for t in (TIER_CORE, TIER_HIGH, TIER_NEUTRAL, TIER_EXIT)}
        for r in records:
            tier = r.get("priority_tier", TIER_NEUTRAL)
            if tier in by_tier:
                by_tier[tier].append(r)

        def _avg5(recs: List[dict]) -> Optional[float]:
            vals = [r["subsequent_5d_return"] for r in recs if r.get("subsequent_5d_return") is not None]
            return round(sum(vals) / len(vals), 6) if vals else None

        def _win_rate(recs: List[dict]) -> Optional[float]:
            with_ret = [r for r in recs if r.get("subsequent_5d_return") is not None]
            pos = [r for r in with_ret if r["subsequent_5d_return"] > 0]
            return round(len(pos) / len(with_ret), 4) if with_ret else None

        avg_5d = {t: _avg5(recs) for t, recs in by_tier.items()}
        c5 = avg_5d.get(TIER_CORE)
        e5 = avg_5d.get(TIER_EXIT)
        delta = round(c5 - e5, 6) if (c5 is not None and e5 is not None) else None

        exit_recs = by_tier[TIER_EXIT]
        with_ret_exit = [r for r in exit_recs if r.get("subsequent_5d_return") is not None]
        underperf_exit = [r for r in with_ret_exit if r["subsequent_5d_return"] < 0]
        exit_underperf_rate = round(len(underperf_exit) / len(with_ret_exit), 4) if with_ret_exit else None

        return {
            "total_records":                       len(records),
            "core_continuation_count":             len(by_tier[TIER_CORE]),
            "high_quality_count":                  len(by_tier[TIER_HIGH]),
            "neutral_count":                       len(by_tier[TIER_NEUTRAL]),
            "exit_candidate_count":                len(by_tier[TIER_EXIT]),
            "avg_5d_return_by_priority_tier":      avg_5d,
            "core_continuation_win_rate":          _win_rate(by_tier[TIER_CORE]),
            "exit_candidate_underperformance_rate": exit_underperf_rate,
            "avg_return_delta_core_vs_exit_candidate": delta,
        }
    except Exception as exc:
        logger.warning("[CP] kpi aggregation failed: %s", exc)
        return {}


def format_cp_kpi_section(telemetry_path: Path) -> str:
    """Morning report KPI section. FAIL_OPEN."""
    try:
        kpis = compute_cp_kpis(telemetry_path)
        if not kpis:
            return ""
        lines = [
            "",
            "── [CONTINUATION PRIORITY KPI] ─────────────────────────────",
            f"  Core cont.  : {kpis.get('core_continuation_count', 0)}件",
            f"  High quality: {kpis.get('high_quality_count', 0)}件",
            f"  Neutral     : {kpis.get('neutral_count', 0)}件",
            f"  Exit cand.  : {kpis.get('exit_candidate_count', 0)}件",
        ]
        c5 = kpis.get("avg_5d_return_by_priority_tier", {}).get(TIER_CORE)
        e5 = kpis.get("avg_5d_return_by_priority_tier", {}).get(TIER_EXIT)
        if c5 is not None:
            lines.append(f"  Core 5d avg : {c5:+.4f}")
        if e5 is not None:
            lines.append(f"  Exit 5d avg : {e5:+.4f}")
        delta = kpis.get("avg_return_delta_core_vs_exit_candidate")
        if delta is not None:
            arrow = "↑ core優勢" if delta > 0 else "↓ 要確認"
            lines.append(f"  Core vs Exit Δ5d: {delta:+.4f}  {arrow}")
        hq = kpis.get("core_continuation_win_rate")
        if hq is not None:
            lines.append(f"  Core 勝率   : {hq:.1%}")
        ex = kpis.get("exit_candidate_underperformance_rate")
        if ex is not None:
            lines.append(f"  Exit 損失率 : {ex:.1%}")
        lines.append("──────────────────────────────────────────────────────────")
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("[CP] format_kpi failed: %s", exc)
        return ""
