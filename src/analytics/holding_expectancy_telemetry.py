"""
holding_expectancy_telemetry.py — Daily holding state snapshot + 3d forward return materialization.

Purpose:
  Accumulate daily state snapshots for held positions.
  Materialize actual 3-business-day forward returns post-hoc.
  Feeds future probabilistic holding expectancy model (heuristic first, ML later).

Design:
  observation_only: never modifies signals, never affects execution.
  append-only snapshot JSONL: one record per (symbol, date), never rewritten.
  materialized records: separate append-only JSONL keyed by (symbol, date).
  fail-open: any error logs and continues.

Snapshot schema:
  {
    "date": "2026-05-27",
    "symbol": "6920.T",
    "hold_day": 8,
    "unrealized_pnl_pct": 0.052,
    "compression_score": 71.3,
    "phase": "compression",
    "rsr": 83.2,
    "rsr_momentum": 0.8,
    "near_high_pct": 0.82,
    "higher_low_streak": 3,
    "suppression_eligible": true,
    "regime": "bull",
    "actual_next_3d_return": null,
    "materialized": false,
    "schema_version": "v1"
  }

Materialized schema: same + "actual_next_3d_return" filled, "materialized": true,
  "materialized_date", "forward_date".
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

SCHEMA_VERSION = "v1"
FORWARD_DAYS = 3       # business days for materialization
_EPS = 1e-9
# Conservative calendar-day buffer: require > FORWARD_DAYS + 2 calendar days before materializing
_CALENDAR_BUFFER = FORWARD_DAYS + 2


def _today_str() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d")


def _safe(v: Any, fallback: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


# ── Atomic JSONL append ───────────────────────────────────────────────────────

def _append_jsonl(record: dict, path: Path) -> None:
    """Atomic append to JSONL. Fail-open."""
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
        os.replace(tmp, str(path))
    except Exception as exc:
        logger.debug("[HET] append failed: %s", exc)


def _load_jsonl(path: Path) -> List[dict]:
    """Load JSONL. Corrupted lines silently skipped."""
    if not path.exists():
        return []
    records: List[dict] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception as exc:
            logger.debug("[HET] parse error line %d: %s", i + 1, exc)
    return records


def _business_day_offset(date_str: str, offset: int) -> str:
    """Return YYYY-MM-DD of date_str + offset business days."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    bdr = pd.bdate_range(start=dt, periods=offset + 1)
    return bdr[-1].strftime("%Y-%m-%d")


def _get_close_at_date(
    df: Optional[pd.DataFrame],
    date_str: str,
    tolerance: int = 2,
) -> Optional[float]:
    """
    Look up Close price on date_str (±tolerance calendar days for holidays).
    Returns None if not available.
    """
    if df is None or df.empty:
        return None
    close_col = "Close" if "Close" in df.columns else "Adj Close"
    if close_col not in df.columns:
        return None
    try:
        idx = pd.DatetimeIndex(df.index)
        target = pd.Timestamp(date_str)
        for delta in range(tolerance + 1):
            candidate = target + pd.Timedelta(days=delta)
            mask = idx == candidate
            if mask.any():
                v = float(df.loc[mask, close_col].iloc[0])
                return v if math.isfinite(v) and v > 0 else None
    except Exception as exc:
        logger.debug("[HET] price lookup failed %s: %s", date_str, exc)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot creation
# ─────────────────────────────────────────────────────────────────────────────

def build_snapshot(
    signal: Dict[str, Any],
    regime: str = "unknown",
    today: Optional[str] = None,
) -> dict:
    """Build a daily holding snapshot from a signal dict (observation only)."""
    return {
        "date":                 today or _today_str(),
        "symbol":               str(signal.get("symbol", "")),
        "hold_day":             int(signal.get("hold_days", 0)),
        "unrealized_pnl_pct":   round(_safe(signal.get("current_return", 0.0)), 4),
        "compression_score":    round(_safe(signal.get("compression_score", 0.0)), 2),
        "phase":                str(signal.get("compression_phase", "neutral")),
        "rsr":                  round(_safe(signal.get("rsr", 50.0), 50.0), 2),
        "rsr_momentum":         round(_safe(signal.get("rsr_momentum", 0.0)), 4),
        "near_high_pct":        round(_safe(signal.get("near_high_pct", 0.5), 0.5), 4),
        "higher_low_streak":    int(signal.get("higher_low_streak", 0)),
        "suppression_eligible": bool(signal.get("suppression_eligible", False)),
        "regime":               regime,
        "actual_next_3d_return": None,
        "materialized":         False,
        "schema_version":       SCHEMA_VERSION,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Materialization
# ─────────────────────────────────────────────────────────────────────────────

def materialize_pending(
    snapshot_path: Path,
    materialized_path: Path,
    ohlcv_loader: Callable[[str], Optional[pd.DataFrame]],
    today: Optional[str] = None,
) -> int:
    """
    Scan snapshot_path for un-materialized records >= FORWARD_DAYS business days old.
    For each, look up 3-business-day forward price and append to materialized_path.
    Returns count of newly materialized records. Fail-open per record.
    """
    today = today or _today_str()
    today_dt = datetime.strptime(today, "%Y-%m-%d")

    snapshots = _load_jsonl(snapshot_path)
    if not snapshots:
        return 0

    already = {
        (r.get("symbol", ""), r.get("date", ""))
        for r in _load_jsonl(materialized_path)
    }

    n_new = 0
    for snap in snapshots:
        sym  = snap.get("symbol", "")
        date = snap.get("date", "")
        if not sym or not date:
            continue
        if snap.get("materialized"):
            continue
        if (sym, date) in already:
            continue

        try:
            snap_dt = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            continue

        # Conservative: require > _CALENDAR_BUFFER calendar days
        if (today_dt - snap_dt).days < _CALENDAR_BUFFER:
            continue

        try:
            forward_date = _business_day_offset(date, FORWARD_DAYS)
            df = ohlcv_loader(sym)
            price_snap    = _get_close_at_date(df, date)
            price_forward = _get_close_at_date(df, forward_date)

            if price_snap is None or price_forward is None or price_snap < _EPS:
                continue

            fwd_return = (price_forward - price_snap) / price_snap
            if not math.isfinite(fwd_return):
                continue

            mat_record = dict(snap)
            mat_record["actual_next_3d_return"] = round(fwd_return, 6)
            mat_record["materialized"]          = True
            mat_record["materialized_date"]     = today
            mat_record["forward_date"]          = forward_date

            _append_jsonl(mat_record, materialized_path)
            already.add((sym, date))
            n_new += 1
            logger.debug(
                "[HET] materialized %s@%s fwd3d=%.2f%%",
                sym, date, fwd_return * 100,
            )
        except Exception as exc:
            logger.debug("[HET] materialization error %s@%s: %s", sym, date, exc)

    return n_new


# ─────────────────────────────────────────────────────────────────────────────
# Integration hook
# ─────────────────────────────────────────────────────────────────────────────

def run_holding_expectancy_hook(
    signals: List[Dict[str, Any]],
    *,
    snapshot_path: Path,
    materialized_path: Path,
    ohlcv_loader: Callable[[str], Optional[pd.DataFrame]],
    regime_info: Optional[Dict[str, Any]] = None,
    today: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Append daily snapshots for held positions and attempt forward return materialization.
    observation_only: never modifies signals.
    Fail-open.

    Returns summary: {"n_written": int, "n_materialized": int}.
    """
    try:
        return _run_internal(
            signals=signals,
            snapshot_path=snapshot_path,
            materialized_path=materialized_path,
            ohlcv_loader=ohlcv_loader,
            regime_info=regime_info or {},
            today=today or _today_str(),
        )
    except Exception as exc:
        logger.warning("[HET] hook failed (%s) — continuing", exc)
        return {}


def _run_internal(
    signals: List[Dict[str, Any]],
    snapshot_path: Path,
    materialized_path: Path,
    ohlcv_loader: Callable[[str], Optional[pd.DataFrame]],
    regime_info: Dict[str, Any],
    today: str,
) -> Dict[str, Any]:
    regime = str(regime_info.get("regime", "unknown"))

    existing = {
        (r.get("symbol", ""), r.get("date", ""))
        for r in _load_jsonl(snapshot_path)
    }

    n_written = 0
    for sig in signals:
        if not sig.get("currently_holding"):
            continue
        sym = str(sig.get("symbol", ""))
        if not sym:
            continue
        if (sym, today) in existing:
            continue

        snap = build_snapshot(sig, regime=regime, today=today)
        _append_jsonl(snap, snapshot_path)
        existing.add((sym, today))
        n_written += 1

    n_materialized = materialize_pending(
        snapshot_path=snapshot_path,
        materialized_path=materialized_path,
        ohlcv_loader=ohlcv_loader,
        today=today,
    )

    logger.info(
        "[HET] written=%d materialized=%d date=%s",
        n_written, n_materialized, today,
    )
    return {"n_written": n_written, "n_materialized": n_materialized}
