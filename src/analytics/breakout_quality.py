"""
breakout_quality.py — Breakout Quality Intelligence

Lightweight daily-candle quality scorer for continuation breakout add-on modulation.
Suppresses false breakout boosts without touching position sizing.

Four metrics (日足のみ・no ML・no intraday):
  1. Close Position in Range (CPR)  : (close - low) / (high - low)
  2. Upper Shadow Ratio (USR)       : (high - close) / (high - low)
  3. Gap Extension Risk (GER)       : |close - prev_close| / atr_20  [ATR units]
  4. Range Expansion (RE)           : (high - low) / atr_20          [ATR units]

Phase priority:  failed_breakout > extended_breakout > healthy_breakout > weak_breakout
Boost mapping:   healthy=1.15 (unchanged), extended/weak=1.00, failed=0.90

Design notes:
  - CPR and USR are mathematically complementary (CPR = 1 - USR for the same candle),
    so combined close-position weight = 0.60. Both are kept for report explainability.
  - GER scoring uses a trapezoid: 0.5-1.8 ATR is ideal (score=100); below gets partial
    credit (weak signal); above decays to 0 by 3.5 ATR (exhaustion).
  - RE scoring: <1.0 ATR = flat expansion (50pts), 1.0-2.5 = healthy, >4.5 = blow-off.

Fail policy: all public functions are FAIL_OPEN.
Runtime: O(1) per symbol, local parquet cache only, no extra API calls.
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

# ── Phase labels ───────────────────────────────────────────────────────────────
PHASE_HEALTHY  = "healthy_breakout"
PHASE_EXTENDED = "extended_breakout"
PHASE_WEAK     = "weak_breakout"
PHASE_FAILED   = "failed_breakout"

# ── Boost multipliers — confidence_score modulation only, not position sizing ──
BOOST_BY_PHASE: Dict[str, float] = {
    PHASE_HEALTHY:  1.15,   # unchanged from current default
    PHASE_EXTENDED: 1.00,   # suppress: potential blow-off / distribution trap
    PHASE_WEAK:     1.00,   # suppress: low-quality close / weak continuation
    PHASE_FAILED:   0.90,   # mild demotion: rejection evidence detected
}

# ── Classification gate thresholds ────────────────────────────────────────────
# healthy: ALL must pass
HEALTHY_CPR_MIN     = 0.70   # close in top 30% of daily range
HEALTHY_USR_MAX     = 0.25   # upper shadow < 25% of range
HEALTHY_GER_MAX     = 1.80   # gap < 1.8 ATR (not overextended)
HEALTHY_RSR_MOM_MIN = 0.0    # RSR momentum must be positive

# failed: breakout_phase=True AND these hold
FAILED_CPR_MAX = 0.50   # close at or below midpoint
FAILED_USR_MIN = 0.45   # heavy upper shadow confirming rejection

# extended: ANY triggers classification
EXTENDED_GER_MIN = 2.50   # gap > 2.5 ATR (exhaustion zone)
EXTENDED_RE_MIN  = 3.50   # range expansion > 3.5 ATR (blow-off candle)

# ── Score weights (sum to 1.0) ─────────────────────────────────────────────────
W_CPR = 0.35
W_USR = 0.25
W_GER = 0.25
W_RE  = 0.15

# ── GER trapezoid curve ───────────────────────────────────────────────────────
GER_IDEAL_LO = 0.50    # below: partial credit max=80
GER_IDEAL_HI = 1.80    # ideal zone ceiling (score=100)
GER_ZERO_AT  = 3.50    # score reaches 0 here

# ── RE trapezoid curve ─────────────────────────────────────────────────────────
RE_WEAK_THRESHOLD = 1.00   # below: flat expansion = 50pts
RE_IDEAL_HI       = 2.50   # ideal zone ceiling (score=100)
RE_ZERO_AT        = 4.50   # score reaches 0 here

# ── Materialization calendar buffers ─────────────────────────────────────────
_MAT_BUFFER_3D = 5    # calendar days after record date before materializing 3d
_MAT_BUFFER_5D = 7


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BreakoutQualityInput:
    symbol:            str
    breakout_phase:    bool      # True when continuation_breakout detected upstream
    close:             float
    high:              float
    low:               float
    prev_close:        float
    atr_20:            float
    volume_ratio_5d:   float
    compression_score: float
    rsr:               float
    rsr_momentum:      float     # WinnerConfirmation.trend_quality


@dataclass
class BreakoutQualityResult:
    symbol:                  str
    phase_type:              str      # PHASE_* constant
    quality_score:           float    # 0-100 composite
    # Raw ratios (for telemetry and reporting)
    close_position_in_range: float    # CPR ∈ [0,1]
    upper_shadow_ratio:      float    # USR ∈ [0,1]
    gap_extension_risk:      float    # GER in ATR units
    range_expansion_atr:     float    # RE in ATR units
    # Component scores 0-100 (higher = better quality signal)
    cpr_score:               float
    usr_score:               float
    ger_score:               float
    re_score:                float
    # Input echoes (for replay / diagnostics)
    rsr:                     float
    rsr_momentum:            float
    volume_ratio_5d:         float
    compression_score:       float


# ─────────────────────────────────────────────────────────────────────────────
# Pure scoring components
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _cpr(close: float, high: float, low: float) -> Tuple[float, float]:
    """Close Position in Range. Zero-range guard returns neutral (0.5, 50)."""
    rng = high - low
    if rng < _EPS:
        return 0.5, 50.0
    v = max(0.0, min(1.0, (close - low) / rng))
    return v, v * 100.0


def _usr(close: float, high: float, low: float) -> Tuple[float, float]:
    """Upper Shadow Ratio. Inverted to score: lower shadow = higher score."""
    rng = high - low
    if rng < _EPS:
        return 0.5, 50.0
    v = max(0.0, min(1.0, (high - close) / rng))
    return v, (1.0 - v) * 100.0


def _ger(close: float, prev_close: float, atr_20: float) -> Tuple[float, float]:
    """Gap Extension Risk. Ideal zone 0.5-1.8 ATR scores 100; outside decays."""
    if atr_20 < _EPS:
        return 1.0, 100.0
    v = abs(close - prev_close) / atr_20
    if v < GER_IDEAL_LO:
        score = (v / GER_IDEAL_LO) * 80.0
    elif v <= GER_IDEAL_HI:
        score = 100.0
    else:
        score = max(0.0, 100.0 * (1.0 - (v - GER_IDEAL_HI) / (GER_ZERO_AT - GER_IDEAL_HI)))
    return v, score


def _re(high: float, low: float, atr_20: float) -> Tuple[float, float]:
    """Range Expansion in ATR units. Healthy expansion 1.0-2.5 ATR scores 80-100."""
    if atr_20 < _EPS:
        return 1.5, 80.0
    v = (high - low) / atr_20
    if v < RE_WEAK_THRESHOLD:
        score = 50.0
    elif v <= RE_IDEAL_HI:
        score = 80.0 + (v - RE_WEAK_THRESHOLD) / (RE_IDEAL_HI - RE_WEAK_THRESHOLD) * 20.0
    else:
        score = max(0.0, 100.0 * (1.0 - (v - RE_IDEAL_HI) / (RE_ZERO_AT - RE_IDEAL_HI)))
    return v, score


def _classify_phase(
    cpr_val:       float,
    usr_val:       float,
    ger_val:       float,
    re_val:        float,
    rsr_momentum:  float,
    breakout_phase: bool,
) -> str:
    """
    Priority: failed > extended > healthy > weak.

    Failed requires breakout_phase=True so we only penalise signals
    where continuation was actively flagged but the close rejected hard.
    """
    # 1. Failed: breakout claimed but close rejected into lower half with heavy shadow
    if breakout_phase and cpr_val < FAILED_CPR_MAX and usr_val >= FAILED_USR_MIN:
        return PHASE_FAILED

    # 2. Extended / blow-off: large gap or climax-range candle
    if ger_val > EXTENDED_GER_MIN or re_val > EXTENDED_RE_MIN:
        return PHASE_EXTENDED

    # 3. Healthy: all quality gates pass simultaneously
    if (
        cpr_val > HEALTHY_CPR_MIN
        and usr_val < HEALTHY_USR_MAX
        and ger_val < HEALTHY_GER_MAX
        and rsr_momentum >= HEALTHY_RSR_MOM_MIN
    ):
        return PHASE_HEALTHY

    # 4. Weak (default)
    return PHASE_WEAK


# ─────────────────────────────────────────────────────────────────────────────
# Public pure API
# ─────────────────────────────────────────────────────────────────────────────

def compute_breakout_quality(inp: BreakoutQualityInput) -> BreakoutQualityResult:
    """
    Score breakout quality [0-100] from a BreakoutQualityInput.

    Pure function. Never raises.
    Returns PHASE_WEAK with score=50 on any error (neutral, no blast radius).
    """
    try:
        cpr_v, cpr_s = _cpr(inp.close, inp.high, inp.low)
        usr_v, usr_s = _usr(inp.close, inp.high, inp.low)
        ger_v, ger_s = _ger(inp.close, inp.prev_close, inp.atr_20)
        re_v,  re_s  = _re(inp.high, inp.low, inp.atr_20)

        quality = W_CPR * cpr_s + W_USR * usr_s + W_GER * ger_s + W_RE * re_s
        quality  = round(max(0.0, min(100.0, quality)), 2)

        phase = _classify_phase(
            cpr_v, usr_v, ger_v, re_v,
            _safe_float(inp.rsr_momentum),
            inp.breakout_phase,
        )

        return BreakoutQualityResult(
            symbol=inp.symbol,
            phase_type=phase,
            quality_score=quality,
            close_position_in_range=round(cpr_v, 4),
            upper_shadow_ratio=round(usr_v, 4),
            gap_extension_risk=round(ger_v, 4),
            range_expansion_atr=round(re_v, 4),
            cpr_score=round(cpr_s, 2),
            usr_score=round(usr_s, 2),
            ger_score=round(ger_s, 2),
            re_score=round(re_s, 2),
            rsr=round(_safe_float(inp.rsr), 2),
            rsr_momentum=round(_safe_float(inp.rsr_momentum), 4),
            volume_ratio_5d=round(_safe_float(inp.volume_ratio_5d), 4),
            compression_score=round(_safe_float(inp.compression_score), 2),
        )
    except Exception as exc:
        logger.warning("[BQ] compute failed %s: %s", getattr(inp, "symbol", "?"), exc)
        return BreakoutQualityResult(
            symbol=getattr(inp, "symbol", ""),
            phase_type=PHASE_WEAK,
            quality_score=50.0,
            close_position_in_range=0.5,
            upper_shadow_ratio=0.5,
            gap_extension_risk=1.0,
            range_expansion_atr=1.5,
            cpr_score=50.0, usr_score=50.0,
            ger_score=100.0, re_score=80.0,
            rsr=0.0, rsr_momentum=0.0,
            volume_ratio_5d=1.0, compression_score=0.0,
        )


def get_addon_boost_multiplier(phase_type: str) -> float:
    """Addon confidence multiplier for a given phase. Unknown phase → 1.00."""
    return BOOST_BY_PHASE.get(phase_type, 1.00)


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV helpers (pandas / numpy — lazy imported)
# ─────────────────────────────────────────────────────────────────────────────

def compute_atr20_from_df(df: Any) -> float:
    """
    20-period ATR from OHLCV DataFrame.
    Fallback: 2% of last close when data is insufficient.
    """
    try:
        import numpy as np
        hi = df["high"].values[-22:].astype(float)
        lo = df["low"].values[-22:].astype(float)
        cl = df["close"].values[-22:].astype(float)
        if len(hi) < 2:
            return float(df["close"].iloc[-1]) * 0.02
        tr_hl = hi[1:] - lo[1:]
        tr_hc = np.abs(hi[1:] - cl[:-1])
        tr_lc = np.abs(lo[1:] - cl[:-1])
        tr    = np.maximum(tr_hl, np.maximum(tr_hc, tr_lc))
        atr   = float(np.mean(tr[-20:]))
        return atr if atr > _EPS else float(df["close"].iloc[-1]) * 0.02
    except Exception:
        try:
            return float(df["close"].iloc[-1]) * 0.02
        except Exception:
            return 1.0


def extract_bq_inputs_from_df(
    symbol:       str,
    df:           Any,
    signal:       Dict[str, Any],
    rsr:          float = 75.0,
    rsr_momentum: float = 0.0,
) -> BreakoutQualityInput:
    """
    Build BreakoutQualityInput from parquet DataFrame + signal dict.
    Requires len(df) >= 2. Caller is responsible for the guard.
    rsr / rsr_momentum come from WinnerConfirmation.rsr / .trend_quality.
    """
    # Normalize OHLCV column case once (cache parquet uses "Close"/"High"/"Low";
    # this function and compute_atr20_from_df expect lowercase).
    if "Close" in df.columns and "close" not in df.columns:
        df = df.rename(columns={
            c: c.lower() for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns
        })
    atr20    = compute_atr20_from_df(df)
    row      = df.iloc[-1]
    prev_row = df.iloc[-2]
    return BreakoutQualityInput(
        symbol=symbol,
        breakout_phase=bool(signal.get("continuation_breakout", True)),
        close=_safe_float(row["close"]),
        high=_safe_float(row["high"]),
        low=_safe_float(row["low"]),
        prev_close=_safe_float(prev_row["close"]),
        atr_20=atr20,
        volume_ratio_5d=_safe_float(signal.get("compression_volume_ratio", 1.0)),
        compression_score=_safe_float(signal.get("compression_score", 0.0)),
        rsr=_safe_float(rsr),
        rsr_momentum=_safe_float(rsr_momentum),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry I/O
# ─────────────────────────────────────────────────────────────────────────────

def _append_jsonl(record: dict, path: Path) -> None:
    """Atomic JSONL append via read-append-rename. FAIL_OPEN."""
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
        logger.warning("[BQ] append failed: %s", exc)


def append_bq_telemetry(
    result:            BreakoutQualityResult,
    addon_boost_before: float,
    addon_boost_after:  float,
    telemetry_path:    Path,
    date_str:          Optional[str] = None,
) -> None:
    """
    Append one telemetry record.
    Forward returns are null at record time and materialized in subsequent runs.
    FAIL_OPEN.
    """
    if date_str is None:
        date_str = datetime.now(_JST).strftime("%Y-%m-%d")
    record = {
        "date":                      date_str,
        "symbol":                    result.symbol,
        "breakout_quality_score":    result.quality_score,
        "breakout_phase_type":       result.phase_type,
        "close_position_in_range":   result.close_position_in_range,
        "upper_shadow_ratio":        result.upper_shadow_ratio,
        "gap_extension_risk":        result.gap_extension_risk,
        "range_expansion_atr":       result.range_expansion_atr,
        "cpr_score":                 result.cpr_score,
        "usr_score":                 result.usr_score,
        "ger_score":                 result.ger_score,
        "re_score":                  result.re_score,
        "rsr":                       result.rsr,
        "rsr_momentum":              result.rsr_momentum,
        "volume_ratio_5d":           result.volume_ratio_5d,
        "compression_score":         result.compression_score,
        "addon_boost_before":        round(addon_boost_before, 4),
        "addon_boost_after":         round(addon_boost_after, 4),
        "subsequent_3d_return":      None,
        "subsequent_5d_return":      None,
        "materialized_3d":           False,
        "materialized_5d":           False,
        "schema_version":            SCHEMA_VERSION,
    }
    _append_jsonl(record, telemetry_path)


# ─────────────────────────────────────────────────────────────────────────────
# Forward return materialization
# ─────────────────────────────────────────────────────────────────────────────

def _fwd_return(df: Any, record_date_str: str, n_trading_days: int) -> Optional[float]:
    """
    Compute n-trading-day forward return from record_date.
    Handles both DatetimeIndex and 'date'/'Date' column layouts.
    Returns None when data is insufficient.
    """
    try:
        import pandas as pd
        import numpy as np
        df = df.copy()
        # Normalise index to DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            date_col = next((c for c in ("date", "Date") if c in df.columns), None)
            if date_col is None:
                return None
            df.index = pd.to_datetime(df[date_col])
        df = df.sort_index()
        rec_dt = pd.Timestamp(record_date_str)
        future_mask = df.index >= rec_dt
        if not future_mask.any():
            return None
        entry_pos  = int(np.argmax(future_mask))
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
        logger.warning("[BQ] _fwd_return failed for %s: %s", record_date_str, exc)
        return None


def materialize_bq_returns(
    telemetry_path: Path,
    ohlcv_loader:   Callable[[str], Optional[Any]],
) -> Dict[str, int]:
    """
    Fill null forward returns for records old enough to have realized.
    Atomically rewrites the telemetry file. FAIL_OPEN.
    """
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

        n3 = n5 = 0
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
            elapsed = (today - rec_date).days

            if not rec.get("materialized_3d") and elapsed >= _MAT_BUFFER_3D:
                if sym not in cache:
                    try:
                        cache[sym] = ohlcv_loader(sym)
                    except Exception as exc:
                        logger.warning("[BQ] ohlcv_loader failed for %s: %s", sym, exc)
                        cache[sym] = None
                if cache[sym] is not None:
                    ret = _fwd_return(cache[sym], date_str, 3)
                    if ret is not None:
                        rec["subsequent_3d_return"] = round(ret, 6)
                        rec["materialized_3d"] = True
                        n3 += 1

            if not rec.get("materialized_5d") and elapsed >= _MAT_BUFFER_5D:
                if sym not in cache:
                    try:
                        cache[sym] = ohlcv_loader(sym)
                    except Exception as exc:
                        logger.warning("[BQ] ohlcv_loader failed for %s: %s", sym, exc)
                        cache[sym] = None
                if cache[sym] is not None:
                    ret = _fwd_return(cache[sym], date_str, 5)
                    if ret is not None:
                        rec["subsequent_5d_return"] = round(ret, 6)
                        rec["materialized_5d"] = True
                        n5 += 1

        # Atomic rewrite
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

        return {"n_materialized_3d": n3, "n_materialized_5d": n5}
    except Exception as exc:
        logger.warning("[BQ] materialize failed: %s", exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# KPI aggregation
# ─────────────────────────────────────────────────────────────────────────────

def compute_bq_kpis(telemetry_path: Path) -> Dict[str, Any]:
    """
    Aggregate breakout quality KPIs from telemetry JSONL.
    Key KPI: healthy vs weak 3d return delta — validates false-breakout suppression.
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

        by_phase: Dict[str, List[dict]] = {
            p: [] for p in (PHASE_HEALTHY, PHASE_EXTENDED, PHASE_WEAK, PHASE_FAILED)
        }
        for r in records:
            phase = r.get("breakout_phase_type", PHASE_WEAK)
            if phase in by_phase:
                by_phase[phase].append(r)

        def _avg(recs: List[dict], key: str) -> Optional[float]:
            vals = [r[key] for r in recs if r.get(key) is not None]
            return round(sum(vals) / len(vals), 6) if vals else None

        total    = len(records)
        n_failed = len(by_phase[PHASE_FAILED])

        avg_3d = {p: _avg(recs, "subsequent_3d_return") for p, recs in by_phase.items()}
        avg_5d = {p: _avg(recs, "subsequent_5d_return") for p, recs in by_phase.items()}

        h3 = avg_3d.get(PHASE_HEALTHY)
        w3 = avg_3d.get(PHASE_WEAK)
        delta_hw = round(h3 - w3, 6) if (h3 is not None and w3 is not None) else None

        h_recs  = by_phase[PHASE_HEALTHY]
        h_ret   = [r for r in h_recs if r.get("subsequent_3d_return") is not None]
        h_pos   = [r for r in h_ret  if r["subsequent_3d_return"] > 0]
        hq_rate = round(len(h_pos) / len(h_ret), 4) if h_ret else None

        return {
            "total_records":                       total,
            "healthy_breakout_count":              len(by_phase[PHASE_HEALTHY]),
            "extended_breakout_count":             len(by_phase[PHASE_EXTENDED]),
            "weak_breakout_count":                 len(by_phase[PHASE_WEAK]),
            "failed_breakout_count":               n_failed,
            "failed_breakout_rate":                round(n_failed / total, 4) if total else 0.0,
            "avg_3d_return_by_phase":              avg_3d,
            "avg_5d_return_by_phase":              avg_5d,
            "avg_return_delta_healthy_vs_weak_3d": delta_hw,
            "high_quality_breakout_success_rate":  hq_rate,
        }
    except Exception as exc:
        logger.warning("[BQ] kpi aggregation failed: %s", exc)
        return {}


def format_bq_report_section(telemetry_path: Path) -> str:
    """Morning report section [BREAKOUT QUALITY INTELLIGENCE]. FAIL_OPEN."""
    try:
        kpis = compute_bq_kpis(telemetry_path)
        if not kpis:
            return ""
        lines = [
            "",
            "── [BREAKOUT QUALITY INTELLIGENCE] ──────────────────────────",
            f"  健全    : {kpis.get('healthy_breakout_count', 0)}件",
            (
                f"  失敗    : {kpis.get('failed_breakout_count', 0)}件"
                f"  (失敗率={kpis.get('failed_breakout_rate', 0):.1%})"
            ),
            f"  延長    : {kpis.get('extended_breakout_count', 0)}件",
            f"  弱い    : {kpis.get('weak_breakout_count', 0)}件",
        ]
        h3 = kpis.get("avg_3d_return_by_phase", {}).get(PHASE_HEALTHY)
        w3 = kpis.get("avg_3d_return_by_phase", {}).get(PHASE_WEAK)
        if h3 is not None:
            lines.append(f"  健全 3d avg   : {h3:+.4f}")
        if w3 is not None:
            lines.append(f"  弱い 3d avg   : {w3:+.4f}")
        delta = kpis.get("avg_return_delta_healthy_vs_weak_3d")
        if delta is not None:
            arrow = "↑ 健全優勢" if delta > 0 else "↓ 要確認"
            lines.append(f"  健全 vs 弱い Δ: {delta:+.4f}  {arrow}")
        hq = kpis.get("high_quality_breakout_success_rate")
        if hq is not None:
            lines.append(f"  高質成功率    : {hq:.1%}")
        lines.append("──────────────────────────────────────────────────────────")
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("[BQ] format_report failed: %s", exc)
        return ""
