"""
src/analytics/future_leader_survivability.py — Future Leader Survivability Telemetry

Purpose:
  Forward expectancy materialization for Future Leader observations.
  Minimal telemetry: 5d/10d return, MFE, MAE.

INVARIANTS (変更禁止):
  - observation_only=True: no execution path connection
  - append-only JSONL: (symbol, observation_date) once sealed, never re-written
  - lookahead prevention: forward bars strictly before today's date
  - research purity: data_quality_weight=0.0 (MAJOR_FAILURE) excluded entirely
  - PARTIAL_FAILURE: persisted in JSONL, excluded from summary aggregation
  - Only fully materialized records (days_materialized >= FORWARD_WINDOW) are persisted

禁止:
  - order generation
  - signal mutation
  - allocator linkage
  - governance promotion
  - auto activation
  - predictive_entry_enabled mutation
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

# Forward windows (business days)
FORWARD_WINDOW: int = 10  # days for full materialization
FORWARD_SHORT: int  = 5   # days for 5d return

# Research purity thresholds
_QUALITY_WEIGHT_EXCLUDE:     float = 0.0  # MAJOR_FAILURE → skip materialization
_QUALITY_WEIGHT_SUMMARY_MIN: float = 1.0  # CLEAN only → include in summary aggregation


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FutureLeaderSurvivabilityRecord:
    """
    Forward return telemetry for one Future Leader observation.
    observation_only=True: not connected to execution path.
    """
    symbol:                     str
    observation_date:           str
    observation_ts:             str
    future_leader_score:        float
    rsr_zone:                   str
    data_quality_weight:        float
    integrity_state:            str
    close_price_at_observation: float
    forward_return_5d:          Optional[float]
    forward_return_10d:         Optional[float]
    mfe_10d:                    Optional[float]
    mae_10d:                    Optional[float]
    days_materialized:          int
    insufficient_forward_data:  bool

    def to_dict(self) -> dict:
        return {
            "symbol":                     self.symbol,
            "observation_date":           self.observation_date,
            "observation_ts":             self.observation_ts,
            "future_leader_score":        self.future_leader_score,
            "rsr_zone":                   self.rsr_zone,
            "data_quality_weight":        self.data_quality_weight,
            "integrity_state":            self.integrity_state,
            "close_price_at_observation": self.close_price_at_observation,
            "forward_return_5d":          self.forward_return_5d,
            "forward_return_10d":         self.forward_return_10d,
            "mfe_10d":                    self.mfe_10d,
            "mae_10d":                    self.mae_10d,
            "days_materialized":          self.days_materialized,
            "insufficient_forward_data":  self.insufficient_forward_data,
        }


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _normalize_df_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame index to date-only timestamps (remove time/tz).
    Ensures consistent date comparison regardless of parquet index precision.
    """
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index).normalize()
    return df


def _find_obs_idx(df: pd.DataFrame, obs_date_str: str) -> Optional[int]:
    """
    Find positional index of observation date in df (index already normalized).
    Returns None if not found.
    """
    obs_ts = pd.Timestamp(obs_date_str).normalize()
    positions = np.where(df.index == obs_ts)[0]
    return int(positions[0]) if len(positions) > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# Core metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_forward_metrics(
    df: pd.DataFrame,
    obs_idx: int,
    close_obs: float,
    today: date,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], int, bool]:
    """
    Compute forward returns and MFE/MAE from confirmed OHLCV bars.

    Lookahead prevention:
        Forward bars are strictly AFTER obs_idx AND with date < today.
        today's bar is excluded — may be incomplete during live session.

    Returns:
        (fwd_5d, fwd_10d, mfe_10d, mae_10d, days_materialized, insufficient)

    insufficient=True when days_materialized < FORWARD_WINDOW.
    """
    today_ts = pd.Timestamp(today).normalize()

    fwd_df = df.iloc[obs_idx + 1:]
    fwd_df = fwd_df[fwd_df.index < today_ts]

    days_avail = len(fwd_df)
    days_used  = min(days_avail, FORWARD_WINDOW)
    insufficient = days_used < FORWARD_WINDOW

    fwd_5d:  Optional[float] = None
    fwd_10d: Optional[float] = None
    mfe:     Optional[float] = None
    mae:     Optional[float] = None

    if days_used >= FORWARD_SHORT:
        c5 = _safe_float(fwd_df["Close"].iloc[FORWARD_SHORT - 1])
        if c5 is not None and close_obs > 0:
            fwd_5d = round((c5 / close_obs) - 1.0, 6)

    if days_used >= FORWARD_WINDOW:
        c10 = _safe_float(fwd_df["Close"].iloc[FORWARD_WINDOW - 1])
        if c10 is not None and close_obs > 0:
            fwd_10d = round((c10 / close_obs) - 1.0, 6)

        highs = fwd_df["High"].iloc[:FORWARD_WINDOW].values.astype(float)
        lows  = fwd_df["Low"].iloc[:FORWARD_WINDOW].values.astype(float)
        valid_h = highs[np.isfinite(highs)]
        valid_l = lows[np.isfinite(lows)]
        if len(valid_h) > 0 and close_obs > 0:
            mfe = round(float(np.max(valid_h)) / close_obs - 1.0, 6)
        if len(valid_l) > 0 and close_obs > 0:
            mae = round(float(np.min(valid_l)) / close_obs - 1.0, 6)

    return fwd_5d, fwd_10d, mfe, mae, days_used, insufficient


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_materialized_keys(log_dir: Path) -> Set[Tuple[str, str]]:
    """
    Load already-materialized (symbol, observation_date) pairs.
    Prevents duplicate materialization on re-run.
    """
    keys: Set[Tuple[str, str]] = set()
    if not log_dir.exists():
        return keys
    for p in sorted(log_dir.glob("future_leader_survivability_*.jsonl")):
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


def _load_integrity_map(integrity_log_dir: Path) -> Dict[str, Tuple[str, float]]:
    """
    Load {observation_date: (integrity_state, data_quality_weight)}.
    Takes the last record per date (append-only logs may accumulate multiple).
    Defaults to ("CLEAN", 1.0) for dates predating integrity logging.
    """
    result: Dict[str, Tuple[str, float]] = {}
    if not integrity_log_dir.exists():
        return result
    for p in sorted(integrity_log_dir.glob("future_leader_integrity_*.jsonl")):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                dt     = rec.get("observation_date", "")
                state  = rec.get("integrity_state", "CLEAN")
                weight = float(rec.get("data_quality_weight", 1.0))
                if dt:
                    result[dt] = (state, weight)  # last occurrence wins
        except Exception:
            pass
    return result


def _load_all_survivability_records(log_dir: Path) -> List[dict]:
    """Load all survivability JSONL records for summary aggregation."""
    records: List[dict] = []
    if not log_dir.exists():
        return records
    for p in sorted(log_dir.glob("future_leader_survivability_*.jsonl")):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        except Exception:
            pass
    return records


def _write_survivability_log(
    records: List[FutureLeaderSurvivabilityRecord],
    log_dir: Path,
    today: Optional[date] = None,
) -> None:
    """Append new records to today's survivability JSONL. Fail-open."""
    if not records:
        return
    if today is None:
        today = datetime.now(JST).date()

    today_str = today.strftime("%Y%m%d")
    log_path  = log_dir / f"future_leader_survivability_{today_str}.jsonl"

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        ts    = datetime.now(JST).isoformat()
        lines = []
        for r in records:
            d = {"materialization_ts": ts, **r.to_dict()}
            lines.append(json.dumps(d, ensure_ascii=False, default=str))
        with log_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("[FL_SURV] wrote %d records → %s", len(records), log_path)
    except Exception as e:
        logger.warning("[FL_SURV] log write failed (fail-open): %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Main materialization pipeline
# ─────────────────────────────────────────────────────────────────────────────

def materialize_future_leader_survivability(
    candidates_log_path: Path,
    integrity_log_dir:   Path,
    survivability_log_dir: Path,
    ohlcv_loader:        Callable[[str], Optional[pd.DataFrame]],
    today:               Optional[date] = None,
) -> int:
    """
    Materialize forward returns for unmaterialized Future Leader observations.

    observation_only=True: no execution path connection.
    append-only: (symbol, observation_date) pairs are never re-processed.
    lookahead prevention: forward bars strictly before today.
    research purity: data_quality_weight=0.0 (MAJOR_FAILURE) excluded.
    Only fully materialized records (days >= FORWARD_WINDOW) are persisted;
    observations with insufficient bars are silently skipped and re-checked next run.

    Returns:
        Count of newly materialized records.
    """
    if today is None:
        today = datetime.now(JST).date()

    if not candidates_log_path.exists():
        logger.info("[FL_SURV] candidates log not found — nothing to materialize")
        return 0

    materialized_keys = _load_materialized_keys(survivability_log_dir)
    integrity_map     = _load_integrity_map(integrity_log_dir)

    # Load raw candidate observations
    raw_obs: List[dict] = []
    try:
        for line in candidates_log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw_obs.append(json.loads(line))
            except Exception:
                pass
    except Exception as e:
        logger.warning("[FL_SURV] candidates log load failed: %s", e)
        return 0

    if not raw_obs:
        return 0

    # Deduplicate — first occurrence per (symbol, observation_date)
    unique_obs: Dict[Tuple[str, str], dict] = {}
    for rec in raw_obs:
        sym = rec.get("symbol", "")
        dt  = rec.get("observation_date", "")
        if sym and dt:
            key = (sym, dt)
            if key not in unique_obs:
                unique_obs[key] = rec

    pending = {k: v for k, v in unique_obs.items() if k not in materialized_keys}
    if not pending:
        logger.debug("[FL_SURV] %d observations already materialized", len(materialized_keys))
        return 0

    logger.info("[FL_SURV] %d pending observations", len(pending))

    ohlcv_cache: Dict[str, Optional[pd.DataFrame]] = {}
    new_records: List[FutureLeaderSurvivabilityRecord] = []
    cnt_purity   = 0
    cnt_ohlcv    = 0
    cnt_skip     = 0

    for (sym, obs_date_str), obs in sorted(pending.items()):
        # Research purity: MAJOR_FAILURE excluded entirely
        int_state, dq_weight = integrity_map.get(obs_date_str, ("CLEAN", 1.0))
        if dq_weight <= _QUALITY_WEIGHT_EXCLUDE:
            cnt_purity += 1
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
                logger.debug("[FL_SURV] OHLCV load failed %s: %s", sym, e)
                ohlcv_cache[sym] = None

        df = ohlcv_cache.get(sym)
        if df is None:
            cnt_ohlcv += 1
            continue

        # Locate observation row
        obs_idx = _find_obs_idx(df, obs_date_str)
        if obs_idx is None:
            cnt_skip += 1
            continue

        close_obs = _safe_float(df["Close"].iloc[obs_idx])
        if close_obs is None or close_obs <= 0:
            cnt_skip += 1
            continue

        # Compute forward metrics (lookahead prevention built in)
        try:
            fwd_5d, fwd_10d, mfe, mae, days_mat, insufficient = _compute_forward_metrics(
                df, obs_idx, close_obs, today
            )
        except Exception as e:
            logger.debug("[FL_SURV] metrics failed %s@%s: %s", sym, obs_date_str, e)
            cnt_skip += 1
            continue

        # Skip if not fully materialized — will be re-checked on next run
        if insufficient:
            cnt_skip += 1
            continue

        new_records.append(FutureLeaderSurvivabilityRecord(
            symbol=sym,
            observation_date=obs_date_str,
            observation_ts=str(obs.get("observation_ts", "")),
            future_leader_score=float(obs.get("future_leader_score", 0.0)),
            rsr_zone=str(obs.get("rsr_zone", "")),
            data_quality_weight=dq_weight,
            integrity_state=int_state,
            close_price_at_observation=round(close_obs, 2),
            forward_return_5d=fwd_5d,
            forward_return_10d=fwd_10d,
            mfe_10d=mfe,
            mae_10d=mae,
            days_materialized=days_mat,
            insufficient_forward_data=insufficient,
        ))

    if new_records:
        _write_survivability_log(new_records, survivability_log_dir, today=today)

    logger.info(
        "[FL_SURV] materialized=%d skipped_purity=%d skipped_ohlcv=%d skipped_other=%d",
        len(new_records), cnt_purity, cnt_ohlcv, cnt_skip,
    )
    return len(new_records)


# ─────────────────────────────────────────────────────────────────────────────
# run_live_signal.py integration hook
# ─────────────────────────────────────────────────────────────────────────────

def run_future_leader_survivability_materialization(
    candidates_log_path:   Path,
    integrity_log_dir:     Path,
    survivability_log_dir: Path,
    backtest_dataset_dir:  Path,
    default_data_version:  str,
    cache_dir:             Path,
    today:                 Optional[date] = None,
) -> int:
    """
    Integration hook for run_live_signal.py. Fail-open.

    observation_only=True: telemetry only, no execution mutation.
    Loads OHLCV via the screener's bulk loader and materializes forward returns.

    Returns count of newly materialized records (0 on any error).
    """
    from src.live.future_leader_screener import _load_ohlcv_for_screener

    symbols: Set[str] = set()
    if candidates_log_path.exists():
        try:
            for line in candidates_log_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sym = rec.get("symbol", "")
                if sym:
                    symbols.add(sym)
        except Exception:
            pass

    if not symbols:
        logger.info("[FL_SURV] no candidate symbols — skipping materialization")
        return 0

    ohlcv_map = _load_ohlcv_for_screener(
        symbols=sorted(symbols),
        backtest_dataset_dir=backtest_dataset_dir,
        default_data_version=default_data_version,
        cache_dir=cache_dir,
    )
    logger.info("[FL_SURV] OHLCV loaded %d/%d symbols", len(ohlcv_map), len(symbols))

    def _loader(sym: str) -> Optional[pd.DataFrame]:
        return ohlcv_map.get(sym)

    try:
        return materialize_future_leader_survivability(
            candidates_log_path=candidates_log_path,
            integrity_log_dir=integrity_log_dir,
            survivability_log_dir=survivability_log_dir,
            ohlcv_loader=_loader,
            today=today,
        )
    except Exception as e:
        logger.warning("[FL_SURV] materialization failed (fail-open): %s", e)
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Survivability summary report
# ─────────────────────────────────────────────────────────────────────────────

def format_survivability_summary(survivability_log_dir: Path) -> str:
    """
    Format FUTURE LEADER SURVIVABILITY section.
    CLEAN observations only (data_quality_weight >= 1.0) for summary aggregation.
    """
    all_records = _load_all_survivability_records(survivability_log_dir)

    clean_complete = [
        r for r in all_records
        if float(r.get("data_quality_weight", 0.0)) >= _QUALITY_WEIGHT_SUMMARY_MIN
        and not r.get("insufficient_forward_data", True)
    ]

    lines = [
        "=== FUTURE LEADER SURVIVABILITY ===",
        f"materialized_records: {len(all_records)}",
    ]
    if len(all_records) > len(clean_complete):
        lines.append(f"clean_complete_records: {len(clean_complete)}")

    if not clean_complete:
        lines.append("(insufficient clean records for aggregation)")
        return "\n".join(lines)

    def _vals(key: str) -> List[float]:
        return [float(r[key]) for r in clean_complete if r.get(key) is not None]

    def _pct(v: float) -> str:
        return f"{v * 100:+.1f}%"

    fwd5  = _vals("forward_return_5d")
    fwd10 = _vals("forward_return_10d")
    mfe_v = _vals("mfe_10d")
    mae_v = _vals("mae_10d")

    if fwd5:
        lines += [
            "",
            "5d expectancy:",
            f"  mean:   {_pct(float(np.mean(fwd5)))}",
            f"  median: {_pct(float(np.median(fwd5)))}",
        ]
    if fwd10:
        lines += [
            "",
            "10d expectancy:",
            f"  mean:   {_pct(float(np.mean(fwd10)))}",
            f"  median: {_pct(float(np.median(fwd10)))}",
        ]
    if mfe_v:
        lines += [
            "",
            "10d MFE:",
            f"  median: {_pct(float(np.median(mfe_v)))}",
        ]
    if mae_v:
        lines += [
            "",
            "10d MAE:",
            f"  median: {_pct(float(np.median(mae_v)))}",
        ]

    return "\n".join(lines)
