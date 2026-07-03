"""
src/analytics/future_leader_failure_clustering.py — Future Leader Failure Clustering

Purpose:
  Failure transition telemetry for Future Leader observations.
  Classifies how and when trend persistence collapses after initial breakout detection.
  Research answer: "what dies?" — not "what survives?".

INVARIANTS (変更禁止):
  - observation_only=True: no execution path connection
  - single-label failure: primary_failure_reason is the first-priority match
  - secondary_failure_flags: all additional triggered conditions (telemetry only)
  - append-only JSONL: (symbol, observation_date) once sealed, never re-written
  - lookahead prevention: forward bars strictly before today
  - research purity: CLEAN-only (data_quality_weight >= 1.0) for summary aggregation
  - PARTIAL_FAILURE: persisted to JSONL, excluded from aggregation

Priority order (fixed):
  1. market_selloff    — TOPIX 3d cumret <= -4%
  2. gap_fill          — initial breakout gain 70%+ lost
  3. rsr_reversal      — RSR delta negative AND RSR < observation_rsr
  4. volume_faded      — 5d avg vol ratio < 0.7 AND price stagnant (+ regime filter)
  5. weak_followthrough — 5d return <= 0 AND MFE_5d < 3%
  6. breakout_failed_3d — close < obs_close within 3 days AND MFE_3d < 2%
  7. none              — failure_detected=False

禁止:
  - order generation / signal mutation / allocator linkage / governance linkage
  - deployable universe mutation / auto activation / predictive_entry_enabled mutation
  - ML clustering / survival curves / Bayesian classification / regime trees
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

FORWARD_WINDOW: int = 10
FORWARD_SHORT:  int = 5

TOPIX_SYMBOL: str = "1306.T"

_QUALITY_CLEAN_MIN: float = 1.0
_MIN_SAMPLE:        int   = 3

_FAILURE_PRIORITY: List[str] = [
    "market_selloff",
    "gap_fill",
    "rsr_reversal",
    "volume_faded",
    "weak_followthrough",
    "breakout_failed_3d",
]


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FutureLeaderFailureRecord:
    """
    Failure transition telemetry for one Future Leader observation.
    observation_only=True: never connected to execution path.
    primary_failure_reason: first-priority match (single-label for aggregation).
    secondary_failure_flags: all additional triggered conditions (telemetry only).
    """
    symbol:                  str
    observation_date:        str
    observation_ts:          str
    future_leader_score:     float
    rsr_zone:                str
    integrity_state:         str
    data_quality_weight:     float
    failure_detected:        bool
    primary_failure_reason:  Optional[str]   # None when failure_detected=False
    failure_day_offset:      Optional[int]   # 1-based business day; None when no failure
    forward_return_5d:       Optional[float]
    forward_return_10d:      Optional[float]
    mfe_10d:                 Optional[float]
    mae_10d:                 Optional[float]
    secondary_failure_flags: List[str] = field(default_factory=list)
    mfe_3d:                  Optional[float] = None
    topix_volume_ratio:      Optional[float] = None
    observation_only:        bool = True     # invariant: always True

    def to_dict(self) -> dict:
        return {
            "symbol":                  self.symbol,
            "observation_date":        self.observation_date,
            "observation_ts":          self.observation_ts,
            "future_leader_score":     self.future_leader_score,
            "rsr_zone":                self.rsr_zone,
            "integrity_state":         self.integrity_state,
            "data_quality_weight":     self.data_quality_weight,
            "failure_detected":        self.failure_detected,
            "primary_failure_reason":  self.primary_failure_reason,
            "secondary_failure_flags": self.secondary_failure_flags,
            "failure_day_offset":      self.failure_day_offset,
            "forward_return_5d":       self.forward_return_5d,
            "forward_return_10d":      self.forward_return_10d,
            "mfe_10d":                 self.mfe_10d,
            "mae_10d":                 self.mae_10d,
            "mfe_3d":                  self.mfe_3d,
            "topix_volume_ratio":      self.topix_volume_ratio,
            "observation_only":        self.observation_only,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _normalize_df_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index).normalize()
    return df


def _find_obs_idx(df: pd.DataFrame, obs_date_str: str) -> Optional[int]:
    obs_ts = pd.Timestamp(obs_date_str).normalize()
    positions = np.where(df.index == obs_ts)[0]
    return int(positions[0]) if len(positions) > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# Failure detection — each returns 1-based day offset or None
# ─────────────────────────────────────────────────────────────────────────────

def _detect_market_selloff(
    topix_df: pd.DataFrame,
    fwd_dates: pd.DatetimeIndex,
) -> Optional[int]:
    """
    TOPIX 3d cumulative return <= -4%.
    Uses TOPIX positional calendar: looks back 3 positions in topix index.
    Returns 1-based offset within fwd_dates, or None.
    """
    if topix_df is None or topix_df.empty or "Close" not in topix_df.columns:
        return None

    topix_close = topix_df["Close"]
    topix_dates = topix_close.index
    topix_pos_map: Dict[pd.Timestamp, int] = {ts: i for i, ts in enumerate(topix_dates)}
    topix_vals = topix_close.values.astype(float)

    for d_offset, fwd_date in enumerate(fwd_dates, start=1):
        pos_d = topix_pos_map.get(fwd_date)
        if pos_d is None:
            continue
        pos_3ago = pos_d - 3
        if pos_3ago < 0:
            continue

        c_d    = topix_vals[pos_d]
        c_3ago = topix_vals[pos_3ago]

        if not (math.isfinite(c_d) and math.isfinite(c_3ago) and c_3ago > 0):
            continue

        if (c_d / c_3ago - 1.0) <= -0.04:
            return d_offset

    return None


def _detect_gap_fill(
    fwd_closes: np.ndarray,
    close_obs: float,
) -> Optional[int]:
    """
    Initial breakout gain 70%+ lost.
    initial_gain = (fwd_close[0] - close_obs) / close_obs.
    If initial_gain <= 0: no breakout, skip.
    gap_fill at day d >= 2 when: (close[d] - close_obs) / close_obs < 0.30 * initial_gain.
    Returns 1-based offset, or None.
    """
    if len(fwd_closes) < 2 or close_obs <= 0:
        return None

    initial_gain = (fwd_closes[0] - close_obs) / close_obs
    if initial_gain <= 0:
        return None

    threshold = 0.30 * initial_gain  # retaining only 30% of gain = losing 70%

    for d_idx in range(1, len(fwd_closes)):
        d_return = (fwd_closes[d_idx] - close_obs) / close_obs
        if d_return < threshold:
            return d_idx + 1  # 1-based

    return None


def _detect_rsr_reversal(
    rsr_df: pd.DataFrame,
    sym: str,
    observation_rsr: float,
    fwd_dates: pd.DatetimeIndex,
) -> Optional[int]:
    """
    RSR delta turns negative AND current_rsr < observation_rsr.
    Checks consecutive forward bars (prev → current).
    Returns 1-based offset within fwd_dates, or None.
    """
    if rsr_df is None or rsr_df.empty or sym not in rsr_df.columns:
        return None

    rsr_col = rsr_df[sym]
    rsr_dates = rsr_col.index
    rsr_pos_map: Dict[pd.Timestamp, int] = {ts: i for i, ts in enumerate(rsr_dates)}
    rsr_vals = rsr_col.values.astype(float)

    for d_offset, fwd_date in enumerate(fwd_dates, start=1):
        pos_curr = rsr_pos_map.get(fwd_date)
        if pos_curr is None or pos_curr == 0:
            continue

        curr_rsr = rsr_vals[pos_curr]
        prev_rsr = rsr_vals[pos_curr - 1]

        if not (math.isfinite(curr_rsr) and math.isfinite(prev_rsr)):
            continue

        delta = curr_rsr - prev_rsr
        if delta < 0 and curr_rsr < observation_rsr:
            return d_offset

    return None


def _detect_volume_faded(
    fwd_closes:      np.ndarray,
    fwd_volumes:     np.ndarray,
    close_obs:       float,
    baseline_vol:    float,
    topix_vol_ratio: Optional[float] = None,
) -> Optional[int]:
    """
    5d avg volume ratio < 0.7 AND price stagnation (|return| < 2%).
    Regime contamination filter: suppressed when topix_vol_ratio <= 0.8
    (market-wide inactivity — not individual failure signal).
    Checks from day 5 onward (0-indexed day 4+).
    Returns 1-based offset, or None.
    """
    if baseline_vol <= 0 or len(fwd_closes) < 5 or len(fwd_volumes) < 5 or close_obs <= 0:
        return None

    # Regime suppression: TOPIX itself inactive → market-wide, not individual failure
    if topix_vol_ratio is not None and topix_vol_ratio <= 0.8:
        return None

    for d_idx in range(4, len(fwd_closes)):  # day 5..N (0-indexed 4..N-1)
        vol_5d = float(np.nanmean(fwd_volumes[d_idx - 4: d_idx + 1]))
        vol_ratio = vol_5d / baseline_vol

        price_return = abs((fwd_closes[d_idx] - close_obs) / close_obs)
        stagnant = price_return < 0.02

        if vol_ratio < 0.70 and stagnant:
            return d_idx + 1  # 1-based

    return None


def _detect_weak_followthrough(
    fwd_closes: np.ndarray,
    fwd_highs:  np.ndarray,
    close_obs:  float,
) -> Optional[int]:
    """
    5d return <= 0 AND MFE_5d < 3%.
    Measured at day 5. Returns 5 if both conditions hold, else None.
    """
    if len(fwd_closes) < 5 or len(fwd_highs) < 5 or close_obs <= 0:
        return None

    ret_5d = (fwd_closes[4] - close_obs) / close_obs
    mfe_5d = float(np.max(fwd_highs[:5])) / close_obs - 1.0

    if ret_5d <= 0 and mfe_5d < 0.03:
        return 5

    return None


def _detect_breakout_failed_3d(
    fwd_closes: np.ndarray,
    fwd_highs:  np.ndarray,
    close_obs:  float,
) -> Optional[int]:
    """
    Close < observation close within first 3 business days AND MFE_3d < 2%.
    MFE_3d filter suppresses large-cap noise: if stock reached 2%+ above obs_close
    in the first 3 days, the later dip is not a genuine breakout failure.
    Returns 1-based offset (1, 2, or 3), or None.
    """
    if close_obs <= 0:
        return None

    n = min(3, len(fwd_closes), len(fwd_highs))
    if n == 0:
        return None

    # MFE_3d filter: suppress if stock reached 2%+ above obs_close in first 3 days
    mfe_3d = float(np.max(fwd_highs[:n])) / close_obs - 1.0
    if mfe_3d >= 0.02:
        return None

    for d_idx in range(n):
        if fwd_closes[d_idx] < close_obs:
            return d_idx + 1  # 1-based

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Primary/secondary failure orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def _classify_failure(
    surv_rec:        dict,
    sym_df:          pd.DataFrame,
    obs_idx:         int,
    topix_df:        Optional[pd.DataFrame],
    rsr_df:          Optional[pd.DataFrame],
    observation_rsr: float,
    today:           date,
) -> Tuple[bool, Optional[str], Optional[int], List[str], Optional[float], Optional[float]]:
    """
    Classify failure for one observation. Runs all priority conditions, collects all matches.
    Returns:
        (failure_detected, primary_failure_reason, failure_day_offset,
         secondary_failure_flags, mfe_3d, topix_volume_ratio)

    primary = first-priority match (single-label for aggregation).
    secondary = remaining triggered conditions (telemetry only).
    Lookahead prevention: forward bars strictly < today.
    """
    if obs_idx is None:
        return False, None, None, [], None, None

    today_ts  = pd.Timestamp(today).normalize()
    close_obs = _safe_float(surv_rec.get("close_price_at_observation", 0.0))
    if close_obs is None or close_obs <= 0:
        return False, None, None, [], None, None

    # Forward bars: confirmed, strictly < today, up to FORWARD_WINDOW
    fwd_df = sym_df.iloc[obs_idx + 1:]
    fwd_df = fwd_df[fwd_df.index < today_ts].iloc[:FORWARD_WINDOW]

    if len(fwd_df) < FORWARD_WINDOW:
        return False, None, None, [], None, None  # insufficient confirmed forward bars

    fwd_dates   = fwd_df.index
    fwd_closes  = fwd_df["Close"].values.astype(float)
    fwd_highs   = fwd_df["High"].values.astype(float)
    fwd_volumes = fwd_df["Volume"].values.astype(float)

    # Pre-observation baseline volume (20d)
    pre_df       = sym_df.iloc[:obs_idx]
    baseline_vol = 0.0
    if len(pre_df) >= 20:
        pre_vols = pre_df["Volume"].values[-20:].astype(float)
        valid    = pre_vols[np.isfinite(pre_vols) & (pre_vols > 0)]
        if len(valid) > 0:
            baseline_vol = float(np.mean(valid))

    # Compute MFE_3d: max forward high over first 3 days (for breakout_failed_3d filter + JSONL)
    mfe_3d: Optional[float] = None
    if len(fwd_highs) >= 3 and close_obs > 0:
        mfe_3d = float(np.max(fwd_highs[:3])) / close_obs - 1.0

    # Compute TOPIX volume ratio for regime contamination filter (volume_faded suppression)
    topix_vol_ratio: Optional[float] = None
    if topix_df is not None and not topix_df.empty and "Volume" in topix_df.columns:
        obs_date_str = surv_rec.get("observation_date", "")
        if obs_date_str:
            obs_ts_for_topix = pd.Timestamp(obs_date_str).normalize()
            topix_pos_map: Dict[pd.Timestamp, int] = {ts: i for i, ts in enumerate(topix_df.index)}
            topix_obs_pos = topix_pos_map.get(obs_ts_for_topix)
            if topix_obs_pos is not None and topix_obs_pos > 0:
                topix_vols = topix_df["Volume"].values.astype(float)
                pre_topix  = topix_vols[max(0, topix_obs_pos - 20):topix_obs_pos]
                valid_pre  = pre_topix[np.isfinite(pre_topix) & (pre_topix > 0)]
                if len(valid_pre) > 0:
                    topix_baseline = float(np.mean(valid_pre))
                    topix_fwd_v: List[float] = []
                    for fd in fwd_dates[:5]:
                        pos = topix_pos_map.get(fd)
                        if pos is not None:
                            v = float(topix_vols[pos])
                            if math.isfinite(v) and v > 0:
                                topix_fwd_v.append(v)
                    if topix_fwd_v and topix_baseline > 0:
                        topix_vol_ratio = float(np.mean(topix_fwd_v)) / topix_baseline

    # ── Collect all triggered conditions in priority order ────────────────────
    matches: List[Tuple[str, int]] = []

    # Priority 1: market_selloff
    if topix_df is not None:
        offset = _detect_market_selloff(topix_df, fwd_dates)
        if offset is not None:
            matches.append(("market_selloff", offset))

    # Priority 2: gap_fill
    offset = _detect_gap_fill(fwd_closes, close_obs)
    if offset is not None:
        matches.append(("gap_fill", offset))

    # Priority 3: rsr_reversal
    sym = surv_rec.get("symbol", "")
    if rsr_df is not None and sym:
        offset = _detect_rsr_reversal(rsr_df, sym, observation_rsr, fwd_dates)
        if offset is not None:
            matches.append(("rsr_reversal", offset))

    # Priority 4: volume_faded
    offset = _detect_volume_faded(fwd_closes, fwd_volumes, close_obs, baseline_vol, topix_vol_ratio)
    if offset is not None:
        matches.append(("volume_faded", offset))

    # Priority 5: weak_followthrough
    offset = _detect_weak_followthrough(fwd_closes, fwd_highs, close_obs)
    if offset is not None:
        matches.append(("weak_followthrough", offset))

    # Priority 6: breakout_failed_3d
    offset = _detect_breakout_failed_3d(fwd_closes, fwd_highs, close_obs)
    if offset is not None:
        matches.append(("breakout_failed_3d", offset))

    if not matches:
        return False, None, None, [], mfe_3d, topix_vol_ratio

    primary_reason, primary_offset = matches[0]
    secondary_flags = [m[0] for m in matches[1:]]

    return True, primary_reason, primary_offset, secondary_flags, mfe_3d, topix_vol_ratio


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_failure_keys(log_dir: Path) -> Set[Tuple[str, str]]:
    """Load already-classified (symbol, observation_date) pairs."""
    keys: Set[Tuple[str, str]] = set()
    if not log_dir.exists():
        return keys
    for p in sorted(log_dir.glob("future_leader_failure_*.jsonl")):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sym = rec.get("symbol", "")
                dt  = rec.get("observation_date", "")
                if sym and dt:
                    keys.add((sym, dt))
        except Exception:
            pass
    return keys


def _load_survivability_records(log_dir: Path) -> List[dict]:
    """Load all survivability JSONL records."""
    records: List[dict] = []
    if not log_dir.exists():
        return records
    for p in sorted(log_dir.glob("future_leader_survivability_*.jsonl")):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        except Exception:
            pass
    return records


def _load_candidates_lookup(candidates_log_path: Path) -> Dict[Tuple[str, str], dict]:
    """
    Load {(symbol, observation_date): candidate_record} from candidates JSONL.
    First occurrence per key wins (stable dedup).
    Provides observation_rsr (current_rsr field) for rsr_reversal detection.
    """
    lookup: Dict[Tuple[str, str], dict] = {}
    if not candidates_log_path.exists():
        return lookup
    try:
        for line in candidates_log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sym = rec.get("symbol", "")
                dt  = rec.get("observation_date", "")
                if sym and dt:
                    key = (sym, dt)
                    if key not in lookup:
                        lookup[key] = rec
            except Exception:
                pass
    except Exception:
        pass
    return lookup


def _load_failure_records(log_dir: Path) -> List[dict]:
    """Load all failure JSONL records for report aggregation."""
    records: List[dict] = []
    if not log_dir.exists():
        return records
    for p in sorted(log_dir.glob("future_leader_failure_*.jsonl")):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        except Exception:
            pass
    return records


def _write_failure_log(
    records: List[FutureLeaderFailureRecord],
    log_dir: Path,
    today:   Optional[date] = None,
) -> None:
    """Append new failure records to today's JSONL. Fail-open."""
    if not records:
        return
    if today is None:
        today = datetime.now(JST).date()

    today_str = today.strftime("%Y%m%d")
    log_path  = log_dir / f"future_leader_failure_{today_str}.jsonl"

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        ts    = datetime.now(JST).isoformat()
        lines = []
        for r in records:
            d = {"materialization_ts": ts, **r.to_dict()}
            lines.append(json.dumps(d, ensure_ascii=False, default=str))
        with log_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("[FL_FAIL] wrote %d records → %s", len(records), log_path)
    except Exception as e:
        logger.warning("[FL_FAIL] log write failed (fail-open): %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Main materialization pipeline
# ─────────────────────────────────────────────────────────────────────────────

def materialize_future_leader_failures(
    candidates_log_path:   Path,
    survivability_log_dir: Path,
    failure_log_dir:       Path,
    ohlcv_loader:          Callable[[str], Optional[pd.DataFrame]],
    topix_df:              Optional[pd.DataFrame],
    rsr_df:                Optional[pd.DataFrame],
    today:                 Optional[date] = None,
) -> int:
    """
    Materialize failure classifications for unmaterialized Future Leader observations.

    observation_only=True: research telemetry only, no execution connection.
    primary_failure_reason: first-priority failure per observation (single-label aggregation).
    secondary_failure_flags: all additional triggered conditions (telemetry only).
    append-only: (symbol, observation_date) pairs never re-processed.
    Lookahead prevention: forward bars strictly before today.
    Research purity: PARTIAL_FAILURE records persisted, excluded from aggregation.

    Returns:
        Count of newly classified records (including PARTIAL_FAILURE).
    """
    if today is None:
        today = datetime.now(JST).date()

    # Load survivability records (primary input)
    surv_records = _load_survivability_records(survivability_log_dir)
    if not surv_records:
        logger.info("[FL_FAIL] no survivability records — nothing to classify")
        return 0

    # Deduplicate by (symbol, observation_date) — first occurrence wins
    unique_surv: Dict[Tuple[str, str], dict] = {}
    for rec in surv_records:
        sym = rec.get("symbol", "")
        dt  = rec.get("observation_date", "")
        if sym and dt:
            key = (sym, dt)
            if key not in unique_surv:
                unique_surv[key] = rec

    # Filter out already-classified
    classified_keys = _load_failure_keys(failure_log_dir)
    pending = {k: v for k, v in unique_surv.items() if k not in classified_keys}

    if not pending:
        logger.debug("[FL_FAIL] %d observations already classified", len(classified_keys))
        return 0

    logger.info("[FL_FAIL] %d pending observations to classify", len(pending))

    # Load observation_rsr for rsr_reversal (from candidates JSONL)
    cand_lookup = _load_candidates_lookup(candidates_log_path)

    # Normalize TOPIX
    topix_norm: Optional[pd.DataFrame] = None
    if topix_df is not None and not topix_df.empty and "Close" in topix_df.columns:
        topix_norm = _normalize_df_index(topix_df)

    # Normalize RSR
    rsr_norm: Optional[pd.DataFrame] = None
    if rsr_df is not None and not rsr_df.empty:
        rsr_norm_idx = pd.DatetimeIndex(rsr_df.index).normalize()
        rsr_norm = rsr_df.copy()
        rsr_norm.index = rsr_norm_idx

    ohlcv_cache: Dict[str, Optional[pd.DataFrame]] = {}
    new_records: List[FutureLeaderFailureRecord] = []
    cnt_ohlcv = 0
    cnt_skip  = 0

    for (sym, obs_date_str), surv in sorted(pending.items()):
        # Skip MAJOR_FAILURE (data_quality_weight = 0.0)
        dq_weight = float(surv.get("data_quality_weight", 1.0))
        if dq_weight <= 0.0:
            cnt_skip += 1
            continue

        # Load OHLCV (cached)
        if sym not in ohlcv_cache:
            try:
                df_raw = ohlcv_loader(sym)
                if df_raw is not None and len(df_raw) >= FORWARD_WINDOW + 2:
                    ohlcv_cache[sym] = _normalize_df_index(df_raw)
                else:
                    ohlcv_cache[sym] = None
            except Exception as e:
                logger.debug("[FL_FAIL] OHLCV load failed %s: %s", sym, e)
                ohlcv_cache[sym] = None

        sym_df = ohlcv_cache.get(sym)
        if sym_df is None:
            cnt_ohlcv += 1
            continue

        obs_idx = _find_obs_idx(sym_df, obs_date_str)
        if obs_idx is None:
            cnt_skip += 1
            continue

        # Pre-check: sufficient confirmed forward bars (strictly < today)
        today_ts  = pd.Timestamp(today).normalize()
        fwd_avail = sym_df.iloc[obs_idx + 1:]
        fwd_avail = fwd_avail[fwd_avail.index < today_ts]
        if len(fwd_avail) < FORWARD_WINDOW:
            cnt_skip += 1
            continue  # insufficient confirmed bars — re-check on next run

        # observation_rsr from candidates JSONL, fall back to 0
        cand_rec = cand_lookup.get((sym, obs_date_str), {})
        observation_rsr = float(cand_rec.get("current_rsr", 0.0))

        # Classify failure
        try:
            (fail_detected, fail_reason, fail_offset,
             secondary_flags, mfe_3d, topix_vol_ratio) = _classify_failure(
                surv_rec=surv,
                sym_df=sym_df,
                obs_idx=obs_idx,
                topix_df=topix_norm,
                rsr_df=rsr_norm,
                observation_rsr=observation_rsr,
                today=today,
            )
        except Exception as e:
            logger.debug("[FL_FAIL] classify failed %s@%s: %s", sym, obs_date_str, e)
            cnt_skip += 1
            continue

        def _opt(key: str) -> Optional[float]:
            v = surv.get(key)
            return float(v) if v is not None else None

        new_records.append(FutureLeaderFailureRecord(
            symbol=sym,
            observation_date=obs_date_str,
            observation_ts=str(surv.get("observation_ts", "")),
            future_leader_score=float(surv.get("future_leader_score", 0.0)),
            rsr_zone=str(surv.get("rsr_zone", "")),
            integrity_state=str(surv.get("integrity_state", "CLEAN")),
            data_quality_weight=dq_weight,
            failure_detected=fail_detected,
            primary_failure_reason=fail_reason,
            failure_day_offset=fail_offset,
            forward_return_5d=_opt("forward_return_5d"),
            forward_return_10d=_opt("forward_return_10d"),
            mfe_10d=_opt("mfe_10d"),
            mae_10d=_opt("mae_10d"),
            secondary_failure_flags=secondary_flags,
            mfe_3d=mfe_3d,
            topix_volume_ratio=topix_vol_ratio,
            observation_only=True,
        ))

    if new_records:
        _write_failure_log(new_records, failure_log_dir, today=today)

    logger.info(
        "[FL_FAIL] classified=%d skipped_ohlcv=%d skipped_other=%d",
        len(new_records), cnt_ohlcv, cnt_skip,
    )
    return len(new_records)


# ─────────────────────────────────────────────────────────────────────────────
# Report formatters — pure functions
# ─────────────────────────────────────────────────────────────────────────────

def _filter_clean_complete(records: List[dict]) -> List[dict]:
    """CLEAN-only filter: data_quality_weight >= 1.0. PARTIAL_FAILURE excluded."""
    return [r for r in records if float(r.get("data_quality_weight", 0.0)) >= _QUALITY_CLEAN_MIN]


def _pct(v: Optional[float]) -> str:
    return f"{v * 100:+.1f}%" if v is not None else "N/A"


def _get_primary_reason(r: dict) -> str:
    """Backward-compatible accessor: prefers primary_failure_reason, falls back to failure_reason."""
    return r.get("primary_failure_reason") or r.get("failure_reason") or "none"


def format_failure_clustering_section(failure_log_dir: Path) -> str:
    """
    Format FAILURE CLUSTERING section for daily report.
    CLEAN observations only (data_quality_weight >= 1.0).
    PARTIAL_FAILURE excluded from aggregation.
    Pure function: no I/O beyond loading the failure JSONL.
    """
    all_records   = _load_failure_records(failure_log_dir)
    clean_records = _filter_clean_complete(all_records)

    failed  = [r for r in clean_records if r.get("failure_detected", False)]
    no_fail = [r for r in clean_records if not r.get("failure_detected", False)]

    lines = [
        "=== FAILURE CLUSTERING ===",
        f"clean_classified: {len(clean_records)}",
        f"failure_detected: {len(failed)}",
        f"no_failure:       {len(no_fail)}",
    ]

    if len(all_records) > len(clean_records):
        lines.append(f"partial_failure_excluded: {len(all_records) - len(clean_records)}")

    if not failed:
        lines.append("(insufficient failure records)")
        return "\n".join(lines)

    # Group by primary_failure_reason
    reason_groups: Dict[str, List[dict]] = {}
    for r in failed:
        reason = _get_primary_reason(r)
        if reason not in reason_groups:
            reason_groups[reason] = []
        reason_groups[reason].append(r)

    lines.append("")

    # Output in priority order
    for reason in _FAILURE_PRIORITY:
        recs = reason_groups.get(reason, [])
        if not recs:
            continue

        fwd10 = [float(r["forward_return_10d"]) for r in recs if r.get("forward_return_10d") is not None]
        mae   = [float(r["mae_10d"])  for r in recs if r.get("mae_10d")  is not None]

        lines.append(f"{reason}:")
        lines.append(f"  count: {len(recs)}")

        if reason == "market_selloff":
            lines.append("  (exogenous regime failure)")
        else:
            if len(fwd10) >= _MIN_SAMPLE:
                lines.append(f"  median 10d return: {_pct(float(np.median(fwd10)))}")
            if len(mae) >= _MIN_SAMPLE:
                lines.append(f"  median MAE:        {_pct(float(np.median(mae)))}")

    return "\n".join(lines)


def format_failure_timing_section(failure_log_dir: Path) -> str:
    """
    Format FAILURE TIMING section.
    CLEAN observations only. Groups by primary_failure_reason → median failure_day_offset.
    """
    all_records   = _load_failure_records(failure_log_dir)
    clean_records = _filter_clean_complete(all_records)
    failed        = [r for r in clean_records if r.get("failure_detected", False)]

    lines = ["=== FAILURE TIMING ==="]

    if not failed:
        lines.append("(insufficient failure records)")
        return "\n".join(lines)

    reason_offsets: Dict[str, List[int]] = {}
    for r in failed:
        reason = _get_primary_reason(r)
        offset = r.get("failure_day_offset")
        if offset is not None:
            if reason not in reason_offsets:
                reason_offsets[reason] = []
            reason_offsets[reason].append(int(offset))

    for reason in _FAILURE_PRIORITY:
        offsets = reason_offsets.get(reason, [])
        if not offsets:
            continue
        med = float(np.median(offsets))
        lines.append(f"{reason}:")
        lines.append(f"  median day: {med:.1f}")

    return "\n".join(lines)


def format_failure_overlap_section(failure_log_dir: Path) -> str:
    """
    Format FAILURE OVERLAP section: secondary_failure_flags co-occurrence per primary_failure_reason.
    CLEAN observations only.
    Telemetry only — secondary flags are not used for ranking, weighting, or activation.
    Pure function: no I/O beyond loading the failure JSONL.
    """
    all_records   = _load_failure_records(failure_log_dir)
    clean_records = _filter_clean_complete(all_records)
    failed        = [r for r in clean_records if r.get("failure_detected", False)]

    lines = ["=== FAILURE OVERLAP ==="]

    if not failed:
        lines.append("(insufficient failure records)")
        return "\n".join(lines)

    # Group secondary co-occurrence counts by primary reason
    primary_secondary: Dict[str, Dict[str, int]] = {}
    any_secondary = False
    for r in failed:
        primary  = _get_primary_reason(r)
        secondary = r.get("secondary_failure_flags", [])
        if not isinstance(secondary, list):
            continue
        if secondary:
            any_secondary = True
        if primary not in primary_secondary:
            primary_secondary[primary] = {}
        for flag in secondary:
            primary_secondary[primary][flag] = primary_secondary[primary].get(flag, 0) + 1

    if not any_secondary:
        lines.append("(no secondary flags recorded)")
        return "\n".join(lines)

    for reason in _FAILURE_PRIORITY:
        sec_counts = primary_secondary.get(reason, {})
        if not sec_counts:
            continue
        lines.append(f"{reason}:")
        for flag, cnt in sorted(sec_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  co-occurs: {flag} ({cnt}x)")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# run_live_signal.py integration hook
# ─────────────────────────────────────────────────────────────────────────────

def run_future_leader_failure_materialization(
    candidates_log_path:   Path,
    survivability_log_dir: Path,
    failure_log_dir:       Path,
    backtest_dataset_dir:  Path,
    default_data_version:  str,
    cache_dir:             Path,
    today:                 Optional[date] = None,
) -> int:
    """
    Integration hook for run_live_signal.py. Fail-open.

    observation_only=True: telemetry only, no execution mutation.
    Loads OHLCV via screener's bulk loader, TOPIX parquet, and RSR from universe.

    Returns count of newly classified records (0 on any error).
    """
    from src.live.future_leader_screener import (
        _load_ohlcv_for_screener,
        _build_rsr_df_for_screener,
    )

    # Collect all candidate symbols
    symbols: Set[str] = set()
    if candidates_log_path.exists():
        try:
            for line in candidates_log_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    sym = rec.get("symbol", "")
                    if sym:
                        symbols.add(sym)
                except Exception:
                    pass
        except Exception:
            pass

    if not symbols:
        logger.info("[FL_FAIL] no candidate symbols — skipping")
        return 0

    # Load OHLCV (include TOPIX)
    all_syms  = sorted(symbols) + [TOPIX_SYMBOL]
    ohlcv_map = _load_ohlcv_for_screener(
        symbols=all_syms,
        backtest_dataset_dir=backtest_dataset_dir,
        default_data_version=default_data_version,
        cache_dir=cache_dir,
    )

    # Extract and normalize TOPIX
    topix_raw = ohlcv_map.pop(TOPIX_SYMBOL, None)
    topix_df: Optional[pd.DataFrame] = None
    if topix_raw is not None and "Close" in topix_raw.columns:
        topix_df = _normalize_df_index(topix_raw)

    # Build RSR from candidate symbols
    rsr_df: Optional[pd.DataFrame] = None
    if len(ohlcv_map) >= 2:
        try:
            rsr_raw = _build_rsr_df_for_screener(ohlcv_map)
            if not rsr_raw.empty:
                rsr_df = rsr_raw.copy()
                rsr_df.index = pd.DatetimeIndex(rsr_df.index).normalize()
        except Exception as e:
            logger.debug("[FL_FAIL] RSR build failed (fail-open): %s", e)

    def _loader(sym: str) -> Optional[pd.DataFrame]:
        df = ohlcv_map.get(sym)
        return _normalize_df_index(df) if df is not None else None

    try:
        return materialize_future_leader_failures(
            candidates_log_path=candidates_log_path,
            survivability_log_dir=survivability_log_dir,
            failure_log_dir=failure_log_dir,
            ohlcv_loader=_loader,
            topix_df=topix_df,
            rsr_df=rsr_df,
            today=today,
        )
    except Exception as e:
        logger.warning("[FL_FAIL] materialization failed (fail-open): %s", e)
        return 0
