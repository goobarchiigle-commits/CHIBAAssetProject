"""
src/live/future_leader_screener.py — Future Leader Detection: Shadow Observer Layer

Purpose:
  RSR 完成前の「強者化開始」を観察する research-only infrastructure。
  forward expectancy / institutional accumulation / holding survivability 研究用。

INVARIANTS (変更禁止):
  - FutureLeaderScreenResult.observation_only は常に True
  - order generation 禁止
  - execution mutation 禁止
  - sizing mutation 禁止
  - predictive_entry_enabled は変更しない
  - fail-closed: write_future_leader_log() が失敗した場合は IOError を raise
  - lookahead contamination 禁止: 前日終値確定データのみ使用 (parquet から読む)

Feature design:
  future_leader_score =
    0.30 * rsr_velocity         ← EMA-smoothed 5d/10d acceleration (spike suppressed)
    0.25 * accumulation_pressure ← up-day × close_strength × log1p(vol_ratio)
    0.20 * persistent_drift     ← cum_return / cum_true_range (smooth drift ratio)
    0.10 * close_strength_3d    ← (close-low)/(high-low) 3d avg (upper shadow filter)
    0.10 * rs_topix_delta       ← 10d change in RS vs TOPIX (or cross-sectional proxy)
    0.05 * affordability        ← lot_cost / (effective_capital × max_lot_cost_ratio)

RSR zone gate:
  weak:     RSR < 45   → 0.80× penalty (insufficient institutional interest)
  emerging: 45 ≤ RSR ≤ 72 ← target zone
  mature:   RSR > 72   → 0.70× penalty (late-entry risk)

Noise penalties (applied multiplicatively to the minimum):
  single_day_jump:  ret > 4% AND vol_ratio > 2.5x → ×0.50
  gap_exhaustion:   gap_up > 2% AND close_strength < 0.35 → ×0.40
  long_upper_shadow: cs3d_avg < 0.40 → ×0.70

Absolute threshold penalties (Phase 1 — applied multiplicatively, product):
  low_drift:              persistent_drift < 0.05 → ×0.70
  negative_accumulation:  accumulation_pressure ≤ 0 → ×0.40
  weak_close_strength:    close_strength_3d < 0.45 → ×0.75
  negative_rs_delta:      rs_vs_topix_10d_delta < 0 → ×0.80

Governance promotion gate (observation candidate — no execution connection):
  future_leader_score ≥ 65 AND consecutive ≥ 2 AND zone == "emerging" AND affordable
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

# ── Shadow-only machine-readable constant ─────────────────────────────────────
_OBSERVATION_ONLY: bool = True  # must never be set False

# ── RSR zone thresholds ────────────────────────────────────────────────────────
RSR_EMERGING_MIN: float = 45.0
RSR_EMERGING_MAX: float = 72.0

# ── Composite weights ─────────────────────────────────────────────────────────
WEIGHTS: Dict[str, float] = {
    "rsr_velocity":     0.30,
    "accumulation":     0.25,
    "persistent_drift": 0.20,
    "close_strength":   0.10,
    "rs_topix_delta":   0.10,
    "affordability":    0.05,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1.0"

# ── Zone multipliers ───────────────────────────────────────────────────────────
MATURE_ZONE_PENALTY: float = 0.70
WEAK_ZONE_PENALTY:   float = 0.80

# ── Noise / exhaustion penalties ──────────────────────────────────────────────
CLOSE_STRENGTH_WARN_THRESHOLD: float = 0.40
CLOSE_STRENGTH_PENALTY_MULT:   float = 0.70
SINGLE_DAY_JUMP_RET:           float = 0.040   # 4% single-day return
SINGLE_DAY_JUMP_VOL:           float = 2.5     # 2.5× avg volume
SINGLE_DAY_JUMP_MULT:          float = 0.50
GAP_UP_THRESHOLD:              float = 0.020   # 2% overnight gap
GAP_EXHAUSTION_CS_MAX:         float = 0.35    # weak close after gap
GAP_EXHAUSTION_MULT:           float = 0.40

# ── Absolute threshold penalties (Phase 1) ────────────────────────────────────
ABS_DRIFT_THRESHOLD:        float = 0.05   # persistent_drift < 0.05
ABS_DRIFT_MULT:             float = 0.70
ABS_ACCUM_THRESHOLD:        float = 0.0    # accumulation_pressure <= 0
ABS_ACCUM_MULT:             float = 0.40
ABS_CS_THRESHOLD:           float = 0.45   # close_strength_3d < 0.45
ABS_CS_MULT:                float = 0.75
ABS_RS_DELTA_THRESHOLD:     float = 0.0    # rs_vs_topix_10d_delta < 0
ABS_RS_DELTA_MULT:          float = 0.80

# ── Consecutive / governance gate ─────────────────────────────────────────────
CONSECUTIVE_SCORE_THRESHOLD: float = 60.0
GOVERNANCE_SCORE_MIN:        float = 65.0
GOVERNANCE_CONSECUTIVE_MIN:  int   = 2


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FutureLeaderScreenResult:
    """
    Shadow observation result for one symbol.
    observation_only is always True — never connected to order routing.
    """
    symbol:                   str
    future_leader_score:      float   # 0-100 (zone + noise + absolute penalized composite)
    rsr_zone:                 str     # "weak" | "emerging" | "mature"
    affordability_ok:         bool    # lot_cost ≤ effective_capital × max_lot_cost_ratio
    accumulation_pressure:    float   # raw up-day weighted accumulation value
    persistent_drift:         float   # raw cum_return / cum_range ratio
    close_strength_3d:        float   # raw (close-low)/(high-low) 3d average
    consecutive_high_score_days: int  # consecutive days score ≥ threshold (incl. today)
    penalties:                Dict[str, float]  # noise penalties {name: multiplier}
    failure_flags:            List[str]         # risk / attribution flags

    # Factor breakdown (0-100 cross-sectional scores before penalties)
    rsr_velocity_score:      float = 0.0
    accumulation_score:      float = 0.0
    persistent_drift_score:  float = 0.0
    close_strength_score:    float = 0.0
    rs_topix_delta_score:    float = 0.0
    affordability_score:     float = 0.0

    # Diagnostics
    current_rsr:           float = 0.0
    rs_vs_topix_10d_delta: float = 0.0
    lot_cost_jpy:          float = 0.0

    # Phase 1 — absolute threshold penalties {name: multiplier}
    absolute_penalties: Dict[str, float] = field(default_factory=dict)

    @property
    def observation_only(self) -> bool:
        """Invariant: always True. Never connected to execution path."""
        return True

    @property
    def is_governance_observation_candidate(self) -> bool:
        """
        Observation watchlist promotion gate — future leader specific.
        Isolated from deployable universe / execution governance.
        No connection to allocator, signal bridge, or order routing.
        """
        return (
            self.future_leader_score >= GOVERNANCE_SCORE_MIN
            and self.consecutive_high_score_days >= GOVERNANCE_CONSECUTIVE_MIN
            and self.rsr_zone == "emerging"
            and self.affordability_ok
        )

    @property
    def is_governance_candidate(self):
        raise RuntimeError(
            "FutureLeader observation layer does not support governance_candidate semantics."
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["observation_only"] = True   # explicit guarantee in serialized output
        d["is_governance_observation_candidate"] = self.is_governance_observation_candidate
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Internal cross-sectional normalizers
# ─────────────────────────────────────────────────────────────────────────────

def _cs_rank_high(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank → 0-100. High value = high score."""
    if s.isna().all() or len(s) < 2:
        return pd.Series(50.0, index=s.index)
    return (s.rank(pct=True, na_option="bottom") * 100.0).fillna(0.0)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Feature 1 — RSR velocity (EMA-smoothed, spike-suppressed)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rsr_velocity(
    rsr_df: pd.DataFrame,
    symbols: List[str],
) -> pd.Series:
    """
    EMA-smoothed RSR acceleration over 5d and 10d windows.
    Suppresses single-day spikes via span=3 EMA on raw velocity.
    """
    if rsr_df.empty or len(rsr_df) < 15:
        return pd.Series(0.0, index=symbols)

    rsr = rsr_df.reindex(columns=symbols)

    vel5  = rsr - rsr.shift(5)
    vel10 = rsr - rsr.shift(10)

    vel5_ema  = vel5.ewm(span=3, adjust=False).mean()
    vel10_ema = vel10.ewm(span=3, adjust=False).mean()

    latest5  = vel5_ema.iloc[-1].reindex(symbols).fillna(0.0)
    latest10 = vel10_ema.iloc[-1].reindex(symbols).fillna(0.0)

    composite = 0.60 * latest5 + 0.40 * latest10
    return composite.fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 2 — Accumulation pressure (institutional quiet accumulation proxy)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_accumulation_pressure(
    ohlcv: Dict[str, pd.DataFrame],
    symbols: List[str],
    window: int = 10,
    failures: Optional[Set[str]] = None,
) -> pd.Series:
    """
    Up-day only accumulation proxy.
    Each up-day contributes: return × close_strength × log1p(vol_ratio).
    failures: optional set populated with symbol names on per-symbol compute exception.
    """
    result: Dict[str, float] = {}
    for sym in symbols:
        df = ohlcv.get(sym)
        if df is None or len(df) < window + 2:
            result[sym] = 0.0
            continue
        try:
            tail = df.tail(window + 1).copy()
            close = tail["Close"].values.astype(float)
            high  = tail["High"].values.astype(float)
            low   = tail["Low"].values.astype(float)
            vol   = tail["Volume"].values.astype(float)
            vol   = np.where(vol <= 0, np.nan, vol)

            avg_vol_full = np.nanmean(vol[:-1]) if not np.all(np.isnan(vol[:-1])) else 1.0
            if avg_vol_full <= 0 or not math.isfinite(avg_vol_full):
                avg_vol_full = 1.0

            pressure = 0.0
            for i in range(1, len(tail)):
                prev_c = _safe_float(close[i - 1])
                curr_c = _safe_float(close[i])
                if prev_c <= 0 or not math.isfinite(curr_c):
                    continue
                ret = (curr_c - prev_c) / prev_c
                if ret <= 0.0:
                    continue

                h  = _safe_float(high[i])
                lo = _safe_float(low[i])
                rng = h - lo
                cs = (curr_c - lo) / rng if rng > 1e-8 else 0.5
                cs = max(0.0, min(1.0, cs))

                v = _safe_float(vol[i])
                vol_ratio = v / avg_vol_full if v > 0 and avg_vol_full > 0 else 1.0
                vol_ratio = min(vol_ratio, 5.0)

                pressure += ret * cs * math.log1p(vol_ratio)

            result[sym] = pressure
        except Exception:
            if failures is not None:
                failures.add(sym)
            result[sym] = 0.0

    return pd.Series(result, dtype=float).fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 3 — Persistent drift (low-noise directional strength)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_persistent_drift(
    ohlcv: Dict[str, pd.DataFrame],
    symbols: List[str],
    window: int = 12,
    failures: Optional[Set[str]] = None,
) -> pd.Series:
    """
    cumulative_return / cumulative_true_range_pct over [window] days.
    failures: optional set populated with symbol names on per-symbol compute exception.
    """
    result: Dict[str, float] = {}
    for sym in symbols:
        df = ohlcv.get(sym)
        if df is None or len(df) < window + 2:
            result[sym] = 0.0
            continue
        try:
            tail = df.tail(window + 1).copy()
            close = tail["Close"].values.astype(float)
            high  = tail["High"].values.astype(float)
            low   = tail["Low"].values.astype(float)
            prev_c = np.roll(close, 1)
            prev_c[0] = np.nan

            start_c = _safe_float(close[0])
            end_c   = _safe_float(close[-1])
            if start_c <= 0 or not math.isfinite(end_c):
                result[sym] = 0.0
                continue

            cum_return = (end_c - start_c) / start_c

            tr = np.maximum.reduce([
                high[1:] - low[1:],
                np.abs(high[1:] - prev_c[1:]),
                np.abs(low[1:]  - prev_c[1:]),
            ])
            tr = np.where(np.isfinite(tr), tr, 0.0)
            cum_tr = float(np.sum(tr))

            avg_price = float(np.nanmean(close))
            if avg_price <= 0 or cum_tr <= 0:
                result[sym] = 0.0
                continue

            cum_tr_pct = cum_tr / avg_price
            drift = cum_return / cum_tr_pct if cum_tr_pct > 1e-8 else 0.0
            result[sym] = float(np.clip(drift, -5.0, 5.0))
        except Exception:
            if failures is not None:
                failures.add(sym)
            result[sym] = 0.0

    return pd.Series(result, dtype=float).fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 4 — Close strength 3d (upper shadow / exhaustion filter)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_close_strength_3d(
    ohlcv: Dict[str, pd.DataFrame],
    symbols: List[str],
    failures: Optional[Set[str]] = None,
) -> pd.Series:
    """
    (close - low) / (high - low), 3-day average.
    < 0.40 triggers long_upper_shadow noise penalty.
    < 0.45 triggers weak_close_strength absolute threshold penalty.
    failures: optional set populated with symbol names on per-symbol compute exception.
    """
    result: Dict[str, float] = {}
    for sym in symbols:
        df = ohlcv.get(sym)
        if df is None or len(df) < 3:
            result[sym] = 0.5
            continue
        try:
            tail = df.tail(3)
            cs_vals = []
            for i in range(len(tail)):
                h  = _safe_float(tail["High"].iloc[i])
                lo = _safe_float(tail["Low"].iloc[i])
                c  = _safe_float(tail["Close"].iloc[i])
                rng = h - lo
                cs_vals.append((c - lo) / rng if rng > 1e-8 else 0.5)
            result[sym] = float(np.mean(cs_vals))
        except Exception:
            if failures is not None:
                failures.add(sym)
            result[sym] = 0.5

    return pd.Series(result, dtype=float).fillna(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 5 — RS vs TOPIX delta (relative strength trend)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rs_topix_delta(
    ohlcv: Dict[str, pd.DataFrame],
    symbols: List[str],
    window: int = 10,
    topix_symbol: str = "998407.T",
    failures: Optional[Set[str]] = None,
) -> pd.Series:
    """
    10d change in RS (symbol return - TOPIX return).
    Falls back to raw 10d return if TOPIX not in ohlcv.
    failures: optional set populated with symbol names on per-symbol compute exception.
    """
    topix_ret_10d: Optional[float] = None
    if topix_symbol in ohlcv:
        tdf = ohlcv[topix_symbol]
        if tdf is not None and len(tdf) >= window + 1:
            try:
                tc = tdf["Close"].values
                t_start = _safe_float(tc[-(window + 1)])
                t_end   = _safe_float(tc[-1])
                if t_start > 0:
                    topix_ret_10d = (t_end - t_start) / t_start
            except Exception:
                pass

    result: Dict[str, float] = {}
    for sym in symbols:
        df = ohlcv.get(sym)
        if df is None or len(df) < window + 1:
            result[sym] = 0.0
            continue
        try:
            close = df["Close"].values
            c_start = _safe_float(close[-(window + 1)])
            c_end   = _safe_float(close[-1])
            if c_start <= 0 or not math.isfinite(c_end):
                result[sym] = 0.0
                continue
            sym_ret = (c_end - c_start) / c_start
            if topix_ret_10d is not None:
                result[sym] = sym_ret - topix_ret_10d
            else:
                result[sym] = sym_ret
        except Exception:
            if failures is not None:
                failures.add(sym)
            result[sym] = 0.0

    return pd.Series(result, dtype=float).fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 6 — Affordability (capital constraint gate)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_affordability(
    ohlcv: Dict[str, pd.DataFrame],
    symbols: List[str],
    effective_capital: float,
    lot_size: int,
    max_lot_cost_ratio: float,
    failures: Optional[Set[str]] = None,
) -> Tuple[pd.Series, Dict[str, bool], Dict[str, float]]:
    """
    1.0 if lot_cost << threshold, 0.0 if at/over threshold.
    failures: optional set populated with symbol names on per-symbol compute exception.
    """
    threshold = effective_capital * max_lot_cost_ratio
    scores:    Dict[str, float] = {}
    ok_flags:  Dict[str, bool]  = {}
    lot_costs: Dict[str, float] = {}

    for sym in symbols:
        df = ohlcv.get(sym)
        if df is None or len(df) < 1:
            scores[sym]    = 0.5
            ok_flags[sym]  = True
            lot_costs[sym] = 0.0
            continue
        try:
            price = _safe_float(df["Close"].iloc[-1])
            lot_cost = price * lot_size
            lot_costs[sym] = lot_cost
            ok_flags[sym]  = (lot_cost <= threshold) if threshold > 0 else False
            if threshold <= 0:
                scores[sym] = 0.0
            else:
                ratio = lot_cost / threshold
                scores[sym] = float(max(0.0, 1.0 - ratio))
        except Exception:
            if failures is not None:
                failures.add(sym)
            scores[sym]    = 0.5
            ok_flags[sym]  = True
            lot_costs[sym] = 0.0

    return pd.Series(scores, dtype=float).fillna(0.5), ok_flags, lot_costs


# ─────────────────────────────────────────────────────────────────────────────
# Penalty detection (noise / exhaustion / jump)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_penalties(
    ohlcv: Dict[str, pd.DataFrame],
    symbols: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Detect jump / gap-exhaustion noise patterns per symbol.
    Returns {symbol: {penalty_name: multiplier_applied}}.
    """
    result: Dict[str, Dict[str, float]] = {}
    for sym in symbols:
        df = ohlcv.get(sym)
        pens: Dict[str, float] = {}
        if df is None or len(df) < 6:
            result[sym] = pens
            continue
        try:
            close = df["Close"]
            high  = df["High"]
            low   = df["Low"]
            vol   = df["Volume"].replace(0, np.nan).astype(float)
            # Use prior-day average (exclude today) so today's spike doesn't inflate denominator
            avg_v5 = float(vol.shift(1).rolling(5, min_periods=3).mean().iloc[-1])

            curr_c = _safe_float(close.iloc[-1])
            prev_c = _safe_float(close.iloc[-2])
            h      = _safe_float(high.iloc[-1])
            lo     = _safe_float(low.iloc[-1])
            v_today = _safe_float(vol.iloc[-1])

            open_today = _safe_float(
                df["Open"].iloc[-1]
                if "Open" in df.columns else h
            )

            if prev_c <= 0 or not math.isfinite(prev_c):
                result[sym] = pens
                continue

            ret_today = (curr_c - prev_c) / prev_c
            vol_ratio = v_today / avg_v5 if (avg_v5 > 1e-8 and math.isfinite(avg_v5)) else 1.0
            rng       = h - lo
            cs_today  = (curr_c - lo) / rng if rng > 1e-8 else 0.5
            gap_up    = (open_today - prev_c) / prev_c

            if ret_today > SINGLE_DAY_JUMP_RET and vol_ratio > SINGLE_DAY_JUMP_VOL:
                pens["single_day_jump"] = SINGLE_DAY_JUMP_MULT

            if gap_up > GAP_UP_THRESHOLD and cs_today < GAP_EXHAUSTION_CS_MAX:
                pens["gap_exhaustion"] = GAP_EXHAUSTION_MULT

        except Exception:
            pass

        result[sym] = pens

    return result


# ─────────────────────────────────────────────────────────────────────────────
# RSR zone classification
# ─────────────────────────────────────────────────────────────────────────────

def _classify_rsr_zone(rsr_value: float) -> str:
    if rsr_value < RSR_EMERGING_MIN:
        return "weak"
    if rsr_value <= RSR_EMERGING_MAX:
        return "emerging"
    return "mature"


# ─────────────────────────────────────────────────────────────────────────────
# Consecutive day tracking (reads prior JSONL, fail-safe)
# ─────────────────────────────────────────────────────────────────────────────

def _load_prior_day_state(
    log_path: Path,
    today_str: str,
) -> Dict[str, Tuple[float, int]]:
    """
    Load {symbol: (score, consecutive_days)} from the most recent
    JSONL records whose observation_date != today_str.
    Fail-safe: returns {} on any read / parse error.
    """
    if not log_path.exists():
        return {}
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
        seen: Dict[str, Tuple[float, int]] = {}
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("observation_date", "") == today_str:
                    continue
                sym = rec.get("symbol", "")
                if sym and sym not in seen:
                    score = _safe_float(rec.get("future_leader_score", 0.0))
                    consec = int(rec.get("consecutive_high_score_days", 0))
                    seen[sym] = (score, consec)
            except Exception:
                continue
        return seen
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Absolute threshold penalties (Phase 1 — cross-sectional illusion suppression)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_absolute_penalties(
    drift_raw_val:  float,
    accum_raw_val:  float,
    cs3d_val:       float,
    rs_delta_val:   float,
) -> Dict[str, float]:
    """
    Apply absolute threshold penalties to suppress cross-sectional illusions.
    Returns {name: multiplier} for all active penalties.
    Applied as product (each penalty fires independently).
    Purpose: suppression, NOT elimination.
    """
    abs_pens: Dict[str, float] = {}
    if drift_raw_val < ABS_DRIFT_THRESHOLD:
        abs_pens["low_drift"] = ABS_DRIFT_MULT
    if accum_raw_val <= ABS_ACCUM_THRESHOLD:
        abs_pens["negative_accumulation"] = ABS_ACCUM_MULT
    if cs3d_val < ABS_CS_THRESHOLD:
        abs_pens["weak_close_strength"] = ABS_CS_MULT
    if rs_delta_val < ABS_RS_DELTA_THRESHOLD:
        abs_pens["negative_rs_delta"] = ABS_RS_DELTA_MULT
    return abs_pens


# ─────────────────────────────────────────────────────────────────────────────
# Main screening function
# ─────────────────────────────────────────────────────────────────────────────

def screen_future_leaders(
    rsr_df: pd.DataFrame,
    ohlcv: Dict[str, pd.DataFrame],
    sector_map: Dict[str, str],
    rs_vs_topix_ohlcv: Optional[Dict[str, pd.DataFrame]] = None,
    effective_capital: float = 3_000_000.0,
    *,
    lot_size: int = 100,
    max_lot_cost_ratio: float = 0.30,
    prior_day_state: Optional[Dict[str, Tuple[float, int]]] = None,
    score_threshold_consecutive: float = CONSECUTIVE_SCORE_THRESHOLD,
    _failure_tracker: Optional[dict] = None,
) -> List[FutureLeaderScreenResult]:
    """
    Screen for future leader candidates. Returns list sorted by score desc.
    observation_only = True on ALL results (invariant).

    Args:
        rsr_df:              RSR history DataFrame (dates × symbols, 0-100).
        ohlcv:               {symbol: OHLCV DataFrame}. Must include High/Low/Close/Volume.
        sector_map:          {symbol: sector_name}. Defines the universe.
        rs_vs_topix_ohlcv:   Optional combined ohlcv including TOPIX ("998407.T").
        effective_capital:   Current effective capital (JPY).
        lot_size:            Standard lot size (100 shares for TSE).
        max_lot_cost_ratio:  Max single-lot cost as fraction of effective_capital.
        prior_day_state:     {symbol: (prior_score, prior_consecutive)} from yesterday.
        score_threshold_consecutive: Threshold for consecutive day counting.
        _failure_tracker:    Optional mutable dict populated with partial failure stats.
    """
    symbols = sorted(sector_map.keys())
    if not symbols:
        return []

    ohlcv_for_rs = rs_vs_topix_ohlcv if rs_vs_topix_ohlcv is not None else ohlcv

    # ── Collect per-symbol compute failures ───────────────────────────────────
    partial_failure_symbols: Set[str] = set()

    # ── Compute raw feature vectors ───────────────────────────────────────────
    rsr_vel_raw   = _compute_rsr_velocity(rsr_df, symbols)
    accum_raw     = _compute_accumulation_pressure(
        ohlcv, symbols, failures=partial_failure_symbols)
    drift_raw     = _compute_persistent_drift(
        ohlcv, symbols, failures=partial_failure_symbols)
    cs3d_raw      = _compute_close_strength_3d(
        ohlcv, symbols, failures=partial_failure_symbols)
    rs_delta_raw  = _compute_rs_topix_delta(
        ohlcv_for_rs, symbols, failures=partial_failure_symbols)
    afford_raw, ok_map, lot_cost_map = _compute_affordability(
        ohlcv, symbols, effective_capital, lot_size, max_lot_cost_ratio,
        failures=partial_failure_symbols,
    )
    penalties_map = _detect_penalties(ohlcv, symbols)

    # ── Cross-sectional rank → 0-100 ──────────────────────────────────────────
    rsr_vel_score   = _cs_rank_high(rsr_vel_raw)
    accum_score     = _cs_rank_high(accum_raw)
    drift_score     = _cs_rank_high(drift_raw)
    cs3d_score      = _cs_rank_high(cs3d_raw)
    rs_delta_score  = _cs_rank_high(rs_delta_raw)
    afford_score    = _cs_rank_high(afford_raw)

    # ── Latest RSR for zone classification ────────────────────────────────────
    latest_rsr: Dict[str, float] = {}
    if not rsr_df.empty:
        lr = rsr_df.iloc[-1]
        for sym in symbols:
            latest_rsr[sym] = _safe_float(lr.get(sym, 0.0) if sym in lr.index else 0.0)
    else:
        for sym in symbols:
            latest_rsr[sym] = 0.0

    prior = prior_day_state or {}
    today_str = datetime.now(JST).strftime("%Y-%m-%d")

    # ── Build results (symbol-level isolation) ────────────────────────────────
    results: List[FutureLeaderScreenResult] = []
    successfully_scored = 0

    for sym in symbols:
        if sym in partial_failure_symbols:
            logger.warning(
                "[FUTURE_LEADER] %s excluded — compute failure (research purity, not weak alpha)",
                sym,
            )
            continue
        try:
            # Weighted composite (pre-penalty, 0-100)
            raw_score = (
                WEIGHTS["rsr_velocity"]       * _safe_float(rsr_vel_score.get(sym, 0.0))
                + WEIGHTS["accumulation"]     * _safe_float(accum_score.get(sym, 0.0))
                + WEIGHTS["persistent_drift"] * _safe_float(drift_score.get(sym, 0.0))
                + WEIGHTS["close_strength"]   * _safe_float(cs3d_score.get(sym, 0.0))
                + WEIGHTS["rs_topix_delta"]   * _safe_float(rs_delta_score.get(sym, 0.0))
                + WEIGHTS["affordability"]    * _safe_float(afford_score.get(sym, 0.0))
            )

            # RSR zone
            current_rsr = _safe_float(latest_rsr.get(sym, 0.0))
            rsr_zone    = _classify_rsr_zone(current_rsr)
            zone_mult   = {"weak": WEAK_ZONE_PENALTY, "emerging": 1.0,
                           "mature": MATURE_ZONE_PENALTY}[rsr_zone]

            # Noise penalties (use minimum multiplier)
            pens: Dict[str, float] = dict(penalties_map.get(sym, {}))

            cs_val = _safe_float(cs3d_raw.get(sym, 0.5))
            if cs_val < CLOSE_STRENGTH_WARN_THRESHOLD:
                pens["long_upper_shadow"] = CLOSE_STRENGTH_PENALTY_MULT

            pen_mult = min(pens.values()) if pens else 1.0

            # Phase 1: absolute threshold penalties (product of all active)
            drift_raw_val = _safe_float(drift_raw.get(sym, 0.0))
            accum_raw_val = _safe_float(accum_raw.get(sym, 0.0))
            rs_delta_val  = _safe_float(rs_delta_raw.get(sym, 0.0))

            abs_pens = _compute_absolute_penalties(
                drift_raw_val=drift_raw_val,
                accum_raw_val=accum_raw_val,
                cs3d_val=cs_val,
                rs_delta_val=rs_delta_val,
            )
            abs_pen_mult = 1.0
            for m in abs_pens.values():
                abs_pen_mult *= m

            final_score = float(np.clip(
                raw_score * zone_mult * pen_mult * abs_pen_mult, 0.0, 100.0
            ))

            # Consecutive high score tracking
            prior_score, prior_consec = prior.get(sym, (0.0, 0))
            if (
                final_score >= score_threshold_consecutive
                and prior_score >= score_threshold_consecutive
            ):
                consecutive = prior_consec + 1
            elif final_score >= score_threshold_consecutive:
                consecutive = 1
            else:
                consecutive = 0

            # Failure / risk flags
            failure_flags: List[str] = []
            if rsr_zone == "mature":
                failure_flags.append("late_entry_risk")
            if rsr_zone == "weak":
                failure_flags.append("rsr_too_weak")
            if not ok_map.get(sym, True):
                failure_flags.append("affordability_fail")
            failure_flags.extend(pens.keys())
            failure_flags.extend(abs_pens.keys())

            results.append(FutureLeaderScreenResult(
                symbol=sym,
                future_leader_score=round(final_score, 2),
                rsr_zone=rsr_zone,
                affordability_ok=ok_map.get(sym, True),
                accumulation_pressure=round(_safe_float(accum_raw.get(sym, 0.0)), 6),
                persistent_drift=round(_safe_float(drift_raw.get(sym, 0.0)), 6),
                close_strength_3d=round(cs_val, 4),
                consecutive_high_score_days=consecutive,
                penalties=pens,
                failure_flags=failure_flags,
                rsr_velocity_score=round(_safe_float(rsr_vel_score.get(sym, 0.0)), 2),
                accumulation_score=round(_safe_float(accum_score.get(sym, 0.0)), 2),
                persistent_drift_score=round(_safe_float(drift_score.get(sym, 0.0)), 2),
                close_strength_score=round(_safe_float(cs3d_score.get(sym, 0.0)), 2),
                rs_topix_delta_score=round(_safe_float(rs_delta_score.get(sym, 0.0)), 2),
                affordability_score=round(_safe_float(afford_score.get(sym, 0.0)), 2),
                current_rsr=round(current_rsr, 2),
                rs_vs_topix_10d_delta=round(_safe_float(rs_delta_raw.get(sym, 0.0)), 4),
                lot_cost_jpy=round(_safe_float(lot_cost_map.get(sym, 0.0)), 0),
                absolute_penalties=abs_pens,
            ))
            successfully_scored += 1

        except Exception as e:
            partial_failure_symbols.add(sym)
            logger.warning("[FUTURE_LEADER] %s result build failed: %s — skipping", sym, e)
            continue

    # ── Propagate failure stats to caller ─────────────────────────────────────
    if _failure_tracker is not None:
        _failure_tracker["partial_failure_symbols"] = partial_failure_symbols
        _failure_tracker["partial_compute_failed"]  = len(partial_failure_symbols) > 0
        _failure_tracker["successfully_scored"]     = successfully_scored

    results.sort(key=lambda r: (-r.future_leader_score, r.symbol))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Logging (fail-closed)
# ─────────────────────────────────────────────────────────────────────────────

def write_future_leader_log(
    results: List[FutureLeaderScreenResult],
    log_path: Path,
    run_id: str = "",
) -> None:
    """
    Persist screening results to JSONL. FAIL-CLOSED: raises IOError on failure.
    Append-only: prior records are never modified.
    """
    if not results:
        return

    ts_str   = datetime.now(JST).isoformat()
    date_str = datetime.now(JST).strftime("%Y-%m-%d")

    lines: List[str] = []
    for r in results:
        record = {
            "observation_ts":   ts_str,
            "observation_date": date_str,
            "run_id":           run_id,
            **r.to_dict(),
        }
        lines.append(json.dumps(record, ensure_ascii=False, default=str))

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as exc:
        raise IOError(
            f"[FUTURE_LEADER] FAIL-CLOSED: log write failed → {log_path}: {exc}"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report writer
# ─────────────────────────────────────────────────────────────────────────────

def write_future_leader_report(
    results: List[FutureLeaderScreenResult],
    reports_dir: Path,
    top_k: int = 10,
    survivability_log_dir: Optional[Path] = None,
    candidates_log_path: Optional[Path] = None,
    failure_log_dir: Optional[Path] = None,
) -> Path:
    """Write daily markdown report. Fail-open."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(JST).strftime("%Y-%m-%d")
    ts_str   = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
    out_path = reports_dir / f"future_leader_{date_str}.md"

    emerging  = [r for r in results if r.rsr_zone == "emerging"]
    top       = emerging[:top_k] if len(emerging) >= top_k else (
        emerging + [r for r in results if r.rsr_zone != "emerging"]
    )[:top_k]
    obs_cands = [r for r in results if r.is_governance_observation_candidate]

    lines: List[str] = [
        f"# Future Leader Watch — {date_str}",
        "*Shadow observation only. observation_only=True. No execution connection.*",
        "",
        "## Scoring Legend",
        "| Factor | Weight | Signal |",
        "|--------|--------|--------|",
        "| RSR velocity (EMA-smoothed) | 30% | Acceleration before RSR completion |",
        "| Accumulation pressure | 25% | Up-day × close_strength × log1p(vol) |",
        "| Persistent drift | 20% | cum_return / cum_range (smooth trend) |",
        "| Close strength 3d | 10% | Upper shadow / exhaustion filter |",
        "| RS vs TOPIX delta | 10% | Relative strength improvement |",
        "| Affordability | 5% | Lot cost vs capital constraint |",
        "",
        f"## Top Emerging Candidates (zone=emerging, top {top_k})",
        "| Rank | Symbol | Score | RSR | Zone | Accum | Drift | CS3d | Consec | Observe | Penalties |",
        "|------|--------|-------|-----|------|-------|-------|------|--------|---------|-----------|",
    ]
    for i, r in enumerate(top, 1):
        pen_str = ",".join(r.penalties.keys()) if r.penalties else "—"
        obs_str = "★" if r.is_governance_observation_candidate else ""
        lines.append(
            f"| {i} | {r.symbol} | {r.future_leader_score:.1f}"
            f" | {r.current_rsr:.0f} | {r.rsr_zone}"
            f" | {r.accumulation_pressure:.4f}"
            f" | {r.persistent_drift:.3f}"
            f" | {r.close_strength_3d:.2f}"
            f" | {r.consecutive_high_score_days}"
            f" | {obs_str}"
            f" | {pen_str} |"
        )

    if obs_cands:
        lines += [
            "",
            f"## Observation Candidates ({len(obs_cands)})",
            f"*score ≥ {GOVERNANCE_SCORE_MIN:.0f} AND consecutive ≥ {GOVERNANCE_CONSECUTIVE_MIN}"
            f" AND zone=emerging AND affordable*",
            "",
        ]
        for r in obs_cands:
            lines.append(
                f"- **{r.symbol}** score={r.future_leader_score:.1f}"
                f" RSR={r.current_rsr:.0f} consecutive={r.consecutive_high_score_days}"
                f" lot_cost=¥{r.lot_cost_jpy:,.0f}"
            )
    else:
        lines += ["", "## Observation Candidates", "*(条件未達: なし)*"]

    # Expectancy analytics section (fail-open)
    if survivability_log_dir is not None and candidates_log_path is not None:
        try:
            from src.analytics.future_leader_expectancy import (
                format_future_leader_expectancy_section as _fmt_exp,
            )
            _exp_section = _fmt_exp(survivability_log_dir, candidates_log_path)
            if _exp_section:
                lines += ["", "---", "", _exp_section]
        except Exception as _exp_err:
            logger.debug("[FUTURE_LEADER] expectancy section failed (fail-open): %s", _exp_err)

    # Failure clustering section (fail-open)
    if failure_log_dir is not None:
        try:
            from src.analytics.future_leader_failure_clustering import (
                format_failure_clustering_section as _fmt_fail_cluster,
                format_failure_timing_section as _fmt_fail_timing,
                format_failure_overlap_section as _fmt_fail_overlap,
            )
            _fail_cluster = _fmt_fail_cluster(failure_log_dir)
            _fail_timing  = _fmt_fail_timing(failure_log_dir)
            _fail_overlap = _fmt_fail_overlap(failure_log_dir)
            if _fail_cluster:
                lines += ["", "---", "", _fail_cluster]
            if _fail_timing:
                lines += ["", _fail_timing]
            if _fail_overlap:
                lines += ["", _fail_overlap]
        except Exception as _fail_err:
            logger.debug("[FUTURE_LEADER] failure section failed (fail-open): %s", _fail_err)

    lines += ["", f"*Generated: {ts_str} JST*"]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("[FUTURE_LEADER] report → %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Stdout summary (for dry-run output)
# ─────────────────────────────────────────────────────────────────────────────

def print_future_leader_summary(
    results: List[FutureLeaderScreenResult],
    top_k: int = 5,
) -> None:
    """Print [FUTURE_LEADER] block to stdout. Shadow observation only."""
    emerging  = [r for r in results if r.rsr_zone == "emerging"]
    top       = emerging[:top_k] if len(emerging) >= top_k else results[:top_k]
    obs_cands = [r for r in results if r.is_governance_observation_candidate]

    print("\n  ── [FUTURE_LEADER] Strongman Initiation Watch (shadow only) ─────────")
    if not top:
        print("  (データ不足 or 候補なし)")
        return

    print(f"  {'Symbol':<10} {'Score':>6} {'RSR':>5} {'Zone':<9} {'Accum':>7} {'Drift':>7} {'CS3d':>6} {'Con':>3} {'Flags'}")
    print(f"  {'-'*10} {'-'*6} {'-'*5} {'-'*9} {'-'*7} {'-'*7} {'-'*6} {'-'*3} {'-'*20}")
    for r in top:
        flags = ",".join(r.failure_flags[:2]) if r.failure_flags else "—"
        obs_marker = "★" if r.is_governance_observation_candidate else " "
        print(
            f"  {r.symbol:<10}{obs_marker}{r.future_leader_score:>6.1f}"
            f" {r.current_rsr:>5.0f} {r.rsr_zone:<9}"
            f" {r.accumulation_pressure:>7.4f}"
            f" {r.persistent_drift:>7.3f}"
            f" {r.close_strength_3d:>6.2f}"
            f" {r.consecutive_high_score_days:>3}"
            f" {flags}"
        )
    if obs_cands:
        print(f"\n  Observation candidates: {', '.join(r.symbol for r in obs_cands)}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Mail section formatter
# ─────────────────────────────────────────────────────────────────────────────

def _level_str(v: float, low: float = 0.002, high: float = 0.010) -> str:
    if v >= high:
        return "strong"
    if v >= low:
        return "moderate"
    return "weak"


def format_future_leader_mail_section(
    results: List[FutureLeaderScreenResult],
    top_k: int = 5,
) -> str:
    """
    Format FUTURE LEADER WATCH section for daily mail/report.
    observation_only=True guaranteed.
    """
    emerging  = [r for r in results if r.rsr_zone == "emerging"]
    top       = emerging[:top_k] if len(emerging) >= top_k else results[:top_k]
    obs_cands = [r for r in results if r.is_governance_observation_candidate]

    lines: List[str] = [
        "=" * 56,
        "  FUTURE LEADER WATCH  (shadow observation only)",
        "=" * 56,
        "",
    ]

    if not top:
        lines.append("  候補なし (データ不足 or 全銘柄 zone=weak/mature)")
        lines.append("")
    else:
        for r in top:
            pen_str  = " [PENALTY: " + ",".join(r.penalties.keys()) + "]" if r.penalties else ""
            abs_str  = (" [ABS: " + ",".join(r.absolute_penalties.keys()) + "]"
                        if r.absolute_penalties else "")
            obs_str  = " ★OBSERVATION" if r.is_governance_observation_candidate else ""
            aff_warn = " [CAUTION:not-affordable]" if not r.affordability_ok else ""
            lines.append(f"  {r.symbol}{obs_str}")
            lines.append(f"    score:    {r.future_leader_score:.1f}  RSR: {r.current_rsr:.0f} ({r.rsr_zone})")
            lines.append(f"    accum:    {_level_str(r.accumulation_pressure)}"
                         f"  drift: {r.persistent_drift:+.3f}"
                         f"  cs3d: {r.close_strength_3d:.2f}")
            if r.consecutive_high_score_days > 0:
                lines.append(f"    consecutive: {r.consecutive_high_score_days}d")
            if pen_str or abs_str or aff_warn:
                lines.append(f"    {pen_str}{abs_str}{aff_warn}".strip())
            lines.append("")

    if obs_cands:
        lines.append(f"  Observation candidates ({len(obs_cands)}): "
                     + ", ".join(r.symbol for r in obs_cands))
        lines.append("  → watchlist promotion via universe_governance (shadow only)")
        lines.append("")

    lines.append("  observation_only=True | execution path unchanged")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_ohlcv_for_screener(
    symbols: List[str],
    backtest_dataset_dir: Path,
    default_data_version: str,
    cache_dir: Path,
) -> Dict[str, Any]:
    """Load OHLCV parquet files for given symbols. Fail-safe per symbol."""
    result: Dict[str, Any] = {}
    for sym in symbols:
        candidates = []
        if default_data_version:
            candidates.append(backtest_dataset_dir / default_data_version / f"{sym}.parquet")
        candidates.append(cache_dir / "ohlcv" / f"{sym}.parquet")

        for p in candidates:
            if not p.exists():
                continue
            try:
                df = pd.read_parquet(p)
                if "Close" not in df.columns and "Adj Close" in df.columns:
                    df = df.copy()
                    df["Close"] = df["Adj Close"]
                required = {"High", "Low", "Close"}
                if required.issubset(df.columns):
                    if "Volume" not in df.columns:
                        df = df.copy()
                        df["Volume"] = 0
                    result[sym] = df
                    break
            except Exception as e:
                logger.debug("[FUTURE_LEADER] OHLCV load %s@%s: %s", sym, p, e)

    return result


def _build_rsr_df_for_screener(ohlcv: Dict[str, Any]) -> pd.DataFrame:
    """Build RSR DataFrame from OHLCV dict. Returns empty DataFrame on failure."""
    try:
        from src.backtest.rsr import calc_universe_rsr
        prices: Dict[str, Any] = {}
        for sym, df in ohlcv.items():
            close = df.get("Close") if hasattr(df, "get") else df["Close"]
            if close is not None:
                prices[sym] = df["Close"].dropna()
        if len(prices) < 2:
            return pd.DataFrame()
        return calc_universe_rsr(prices)
    except Exception as e:
        logger.warning("[FUTURE_LEADER] RSR build failed: %s", e)
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Integrity state helper
# ─────────────────────────────────────────────────────────────────────────────

def _emit_integrity_state(
    integrity_log_dir: Optional[Path],
    today_str: str,
    total_symbols: int,
    successfully_scored: int,
    missing_count: int,
    rsr_build_failed: bool,
    ohlcv_load_failed: bool,
    partial_compute_failed: bool,
    warning_count: int,
    partial_failure_symbols_count: int = 0,
) -> None:
    """Build and write integrity state. FAIL-CLOSED: raises IOError on write failure."""
    if integrity_log_dir is None:
        return
    from src.live.future_leader_integrity import (
        ResearchIntegrityState, write_integrity_log,
    )
    state = ResearchIntegrityState.classify(
        observation_date=today_str,
        total_input_symbols=total_symbols,
        successfully_scored_symbols=successfully_scored,
        missing_symbols_count=missing_count,
        rsr_build_failed=rsr_build_failed,
        ohlcv_load_failed=ohlcv_load_failed,
        partial_compute_failed=partial_compute_failed,
        warning_count=warning_count,
        partial_failure_symbols_count=partial_failure_symbols_count,
    )
    date_str_ymd = datetime.now(JST).strftime("%Y%m%d")
    log_path = integrity_log_dir / f"future_leader_integrity_{date_str_ymd}.jsonl"
    write_integrity_log(state, log_path)  # FAIL-CLOSED — raises IOError


# ─────────────────────────────────────────────────────────────────────────────
# Main integration hook (called from run_live_signal.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_future_leader_hook(
    universe: Dict[str, str],
    *,
    backtest_dataset_dir: Path,
    default_data_version: str,
    cache_dir: Path,
    candidates_log_path: Path,
    reports_dir: Path,
    integrity_log_dir: Optional[Path] = None,
    survivability_log_dir: Optional[Path] = None,
    effective_capital: float = 3_000_000.0,
    lot_size: int = 100,
    max_lot_cost_ratio: float = 0.30,
    run_id: str = "",
    print_summary: bool = True,
    top_k: int = 5,
) -> List[FutureLeaderScreenResult]:
    """
    Main integration hook for run_live_signal.py. Shadow observer only.

    FAIL-CLOSED on: candidates log write, integrity log write (both raise IOError).
    Fail-open on computation errors (returns [] and logs warning).

    observation_only=True guaranteed on all returned results.
    No order routing. No execution mutation. No sizing mutation.

    Args:
        universe:             {symbol: sector} — RSR context universe.
        backtest_dataset_dir: Path to parquet dataset files.
        default_data_version: Version subdirectory.
        cache_dir:            Fallback OHLCV cache directory.
        candidates_log_path:  JSONL path for candidates (fail-closed).
        reports_dir:          Directory for markdown daily reports.
        integrity_log_dir:    Directory for integrity state JSONL (fail-closed). None = skip.
        effective_capital:    Current effective capital (JPY).
        lot_size:             Standard lot size (100 shares).
        max_lot_cost_ratio:   Max lot cost as fraction of effective capital.
        run_id:               Live run identifier for tracing.
        print_summary:        Print [FUTURE_LEADER] block to stdout.
        top_k:                Number of top candidates to display.

    Returns:
        List[FutureLeaderScreenResult] — sorted by score desc.
        Returns [] on computation error (does NOT propagate compute exceptions).

    Raises:
        IOError: if candidates log or integrity log write fails (fail-closed).
    """
    today_str = datetime.now(JST).strftime("%Y-%m-%d")
    symbols   = list(universe.keys())

    # Integrity tracking state
    ohlcv_load_failed    = False
    rsr_build_failed     = False
    partial_compute_failed = False
    warning_count        = 0

    logger.info("[FUTURE_LEADER] loading OHLCV for %d symbols", len(symbols))

    try:
        ohlcv = _load_ohlcv_for_screener(
            symbols=symbols,
            backtest_dataset_dir=backtest_dataset_dir,
            default_data_version=default_data_version,
            cache_dir=cache_dir,
        )

        missing_count = max(0, len(symbols) - len(ohlcv))

        if len(ohlcv) < 3:
            ohlcv_load_failed = True
            logger.warning(
                "[FUTURE_LEADER] insufficient OHLCV (%d loaded) — MAJOR_FAILURE", len(ohlcv)
            )
            _emit_integrity_state(
                integrity_log_dir=integrity_log_dir,
                today_str=today_str,
                total_symbols=len(symbols),
                successfully_scored=0,
                missing_count=missing_count,
                rsr_build_failed=rsr_build_failed,
                ohlcv_load_failed=ohlcv_load_failed,
                partial_compute_failed=False,
                warning_count=warning_count,
            )
            return []

        rsr_df = _build_rsr_df_for_screener(ohlcv)
        if rsr_df.empty:
            rsr_build_failed = True
            warning_count += 1
            logger.warning("[FUTURE_LEADER] RSR build returned empty — proceeding with zero RSR velocity")

        prior_day_state = _load_prior_day_state(candidates_log_path, today_str)

        failure_tracker: dict = {}
        results = screen_future_leaders(
            rsr_df=rsr_df,
            ohlcv=ohlcv,
            sector_map=universe,
            effective_capital=effective_capital,
            lot_size=lot_size,
            max_lot_cost_ratio=max_lot_cost_ratio,
            prior_day_state=prior_day_state,
            _failure_tracker=failure_tracker,
        )

        partial_compute_failed = failure_tracker.get("partial_compute_failed", False)
        successfully_scored    = failure_tracker.get("successfully_scored", len(results))

    except Exception as exc:
        logger.warning("[FUTURE_LEADER] computation failed (%s) — skipping", exc)
        # Write MAJOR_FAILURE integrity state before returning
        _emit_integrity_state(
            integrity_log_dir=integrity_log_dir,
            today_str=today_str,
            total_symbols=len(symbols),
            successfully_scored=0,
            missing_count=len(symbols),
            rsr_build_failed=True,
            ohlcv_load_failed=True,
            partial_compute_failed=False,
            warning_count=warning_count,
        )
        return []

    # Write integrity state (FAIL-CLOSED)
    _emit_integrity_state(
        integrity_log_dir=integrity_log_dir,
        today_str=today_str,
        total_symbols=len(symbols),
        successfully_scored=successfully_scored,
        missing_count=missing_count,
        rsr_build_failed=rsr_build_failed,
        ohlcv_load_failed=ohlcv_load_failed,
        partial_compute_failed=partial_compute_failed,
        warning_count=warning_count,
        partial_failure_symbols_count=len(
            failure_tracker.get("partial_failure_symbols", set())
        ),
    )

    if not results:
        logger.warning("[FUTURE_LEADER] no results — skipping candidates log write")
        return []

    # FAIL-CLOSED: raises IOError if write fails
    write_future_leader_log(results, candidates_log_path, run_id=run_id)

    # Write markdown report (fail-open)
    try:
        from src.paths import FUTURE_LEADER_FAILURE_DIR as _FL_FAIL_DIR
        write_future_leader_report(
            results, reports_dir, top_k=top_k,
            survivability_log_dir=survivability_log_dir,
            candidates_log_path=candidates_log_path,
            failure_log_dir=_FL_FAIL_DIR,
        )
    except Exception as exc:
        logger.warning("[FUTURE_LEADER] report write failed (%s) — continuing", exc)

    if print_summary:
        print_future_leader_summary(results, top_k=top_k)

    obs_cands = [r for r in results if r.is_governance_observation_candidate]
    logger.info(
        "[FUTURE_LEADER] screened=%d emerging=%d observation=%d top3=%s",
        len(results),
        sum(1 for r in results if r.rsr_zone == "emerging"),
        len(obs_cands),
        [r.symbol for r in results[:3]],
    )

    return results
