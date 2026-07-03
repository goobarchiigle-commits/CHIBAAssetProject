"""
addon_continuation_telemetry.py — Attribution telemetry for continuation breakout addon timing.

Purpose:
  Compares performance of phase_transition_addon (compression→breakout) vs normal_addon.
  Key question: does VCP-style continuation breakout produce superior addon timing?

Design:
  append-only JSONL per addon evaluation (confirmed winners only).
  Materialization: 3d/5d forward returns filled post-hoc from OHLCV.
  KPI aggregation: phase_transition vs normal comparison.

Schema (JSONL record):
  date, symbol, addon_type, continuation_breakout,
  previous_phase, current_phase, compression_score,
  volume_ratio_5d, rsr_momentum,
  addon_confidence_before, addon_confidence_after, addon_boost_applied,
  subsequent_3d_return (null → materialized later),
  subsequent_5d_return (null → materialized later),
  materialized_3d, materialized_5d, schema_version

observation_only: never modifies signals or orders.
fail-open: any error logs and continues.
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

SCHEMA_VERSION = "v1"
_EPS = 1e-9
FORWARD_DAYS_3 = 3
FORWARD_DAYS_5 = 5
# Conservative calendar-day buffer before materializing
_CALENDAR_BUFFER_3D = FORWARD_DAYS_3 + 2
_CALENDAR_BUFFER_5D = FORWARD_DAYS_5 + 2


def _today_str() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d")


def _safe(v: Any, fallback: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


# ── Atomic I/O helpers ────────────────────────────────────────────────────────

def _append_jsonl(record: dict, path: Path) -> None:
    """Atomic JSONL append. Fail-open."""
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
        logger.debug("[ACT] append failed: %s", exc)


def _load_jsonl(path: Path) -> List[dict]:
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
            logger.debug("[ACT] parse error line %d: %s", i + 1, exc)
    return records


# ── Price helpers (reuses holding_expectancy_telemetry pattern) ───────────────

def _get_close_at_date(
    df: Optional[pd.DataFrame],
    date_str: str,
    tolerance: int = 2,
) -> Optional[float]:
    """Close price on date_str ±tolerance calendar days (holiday-tolerant)."""
    if df is None or df.empty:
        return None
    close_col = "Close" if "Close" in df.columns else "Adj Close"
    if close_col not in df.columns:
        return None
    try:
        idx = pd.DatetimeIndex(df.index)
        target = pd.Timestamp(date_str)
        for d in range(tolerance + 1):
            candidate = target + pd.Timedelta(days=d)
            mask = idx == candidate
            if mask.any():
                v = float(df.loc[mask, close_col].iloc[0])
                return v if math.isfinite(v) and v > 0 else None
    except Exception as exc:
        logger.debug("[ACT] price lookup failed %s: %s", date_str, exc)
    return None


def _business_day_offset(date_str: str, offset: int) -> str:
    """Return YYYY-MM-DD of date_str + offset business days."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    bdr = pd.bdate_range(start=dt, periods=offset + 1)
    return bdr[-1].strftime("%Y-%m-%d")


# ── Record construction ───────────────────────────────────────────────────────

def build_addon_record(
    confirmation: Any,  # WinnerConfirmation (avoid circular import)
    signal: dict,
    today: str,
) -> dict:
    """
    Build an addon evaluation record from a WinnerConfirmation and signal dict.

    addon_type: "continuation_breakout" when continuation_breakout=True, else "normal"
    confidence_before: score before boost (inferred from boost_applied)
    confidence_after:  score after boost (== confidence_before when no boost)
    """
    boost_applied = float(getattr(confirmation, "continuation_boost_applied", 1.0))
    confidence_after = float(confirmation.confidence_score)
    confidence_before = round(
        confidence_after / max(boost_applied, 1.0 + _EPS)
        if boost_applied > 1.0 + _EPS
        else confidence_after,
        4,
    )

    is_cb = bool(getattr(confirmation, "continuation_breakout", False))
    addon_type = "continuation_breakout" if is_cb else "normal"

    return {
        "date": today,
        "symbol": str(confirmation.symbol),
        "addon_triggered": bool(confirmation.confirmed),
        "addon_type": addon_type,
        "continuation_breakout": is_cb,
        "previous_phase": signal.get("prev_phase_at_compression", None),
        "current_phase": signal.get("compression_phase", "neutral"),
        "compression_score": round(_safe(signal.get("compression_score", 0.0)), 2),
        "volume_ratio_5d": round(_safe(signal.get("compression_volume_ratio", 1.0)), 4),
        "rsr_momentum": round(_safe(signal.get("rsr_momentum", 0.0)), 4),
        "addon_confidence_before": round(confidence_before, 4),
        "addon_confidence_after": round(confidence_after, 4),
        "addon_boost_applied": round(boost_applied, 4),
        "subsequent_3d_return": None,
        "subsequent_5d_return": None,
        "materialized_3d": False,
        "materialized_5d": False,
        "schema_version": SCHEMA_VERSION,
    }


# ── Materialization ───────────────────────────────────────────────────────────

def materialize_returns(
    log_path: Path,
    ohlcv_loader: Callable[[str], Optional[pd.DataFrame]],
    today: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Scan log_path for un-materialized records old enough to have forward prices.
    Fill in subsequent_3d_return and subsequent_5d_return from OHLCV.
    Rewrites the file with materialized records in-place (atomic).

    Returns (n_new_3d, n_new_5d) — count of newly materialized entries.
    Fail-open: any error returns (0, 0).
    """
    today = today or _today_str()
    try:
        return _materialize_internal(log_path, ohlcv_loader, today)
    except Exception as exc:
        logger.debug("[ACT] materialize failed: %s", exc)
        return 0, 0


def _materialize_internal(
    log_path: Path,
    ohlcv_loader: Callable[[str], Optional[pd.DataFrame]],
    today: str,
) -> Tuple[int, int]:
    records = _load_jsonl(log_path)
    if not records:
        return 0, 0

    today_dt = datetime.strptime(today, "%Y-%m-%d")
    n_3d = n_5d = 0
    changed = False
    ohlcv_cache: Dict[str, Optional[pd.DataFrame]] = {}

    for r in records:
        sym  = r.get("symbol", "")
        date = r.get("date", "")
        if not sym or not date:
            continue

        try:
            snap_dt = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            continue

        days_ago = (today_dt - snap_dt).days

        # Materialize 3d return
        if not r.get("materialized_3d") and days_ago >= _CALENDAR_BUFFER_3D:
            if sym not in ohlcv_cache:
                ohlcv_cache[sym] = ohlcv_loader(sym)
            df = ohlcv_cache[sym]
            price_base = _get_close_at_date(df, date)
            fwd_date   = _business_day_offset(date, FORWARD_DAYS_3)
            price_fwd  = _get_close_at_date(df, fwd_date)
            if price_base and price_fwd and price_base > _EPS:
                ret = (price_fwd - price_base) / price_base
                if math.isfinite(ret):
                    r["subsequent_3d_return"] = round(ret, 6)
                    r["materialized_3d"] = True
                    n_3d += 1
                    changed = True

        # Materialize 5d return
        if not r.get("materialized_5d") and days_ago >= _CALENDAR_BUFFER_5D:
            if sym not in ohlcv_cache:
                ohlcv_cache[sym] = ohlcv_loader(sym)
            df = ohlcv_cache[sym]
            price_base = _get_close_at_date(df, date)
            fwd_date   = _business_day_offset(date, FORWARD_DAYS_5)
            price_fwd  = _get_close_at_date(df, fwd_date)
            if price_base and price_fwd and price_base > _EPS:
                ret = (price_fwd - price_base) / price_base
                if math.isfinite(ret):
                    r["subsequent_5d_return"] = round(ret, 6)
                    r["materialized_5d"] = True
                    n_5d += 1
                    changed = True

    if changed:
        _rewrite_jsonl(records, log_path)

    return n_3d, n_5d


def _rewrite_jsonl(records: List[dict], path: Path) -> None:
    """Atomically rewrite entire JSONL from records list."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(
            json.dumps(r, ensure_ascii=False, default=str) for r in records
        ) + ("\n" if records else "")
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tf:
                tf.write(content)
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
        logger.debug("[ACT] rewrite failed: %s", exc)


# ── KPI aggregation ───────────────────────────────────────────────────────────

def compute_kpis(log_path: Path) -> dict:
    """
    Aggregate addon continuation KPIs from the JSONL log.
    Returns {} if no data.

    KPIs per addon_type ("continuation_breakout" | "normal"):
      count, avg_3d_return, avg_5d_return
    Cross-type:
      avg_return_delta_vs_normal (continuation_breakout - normal), 3d and 5d
      compression_to_breakout_success_rate (3d return > 0)
      false_breakout_rate (3d return <= 0)
    """
    records = _load_jsonl(log_path)
    if not records:
        return {}

    confirmed = [r for r in records if r.get("addon_triggered")]
    if not confirmed:
        return {}

    def _group_stats(group: List[dict], return_key: str) -> dict:
        vals = [
            _safe(r[return_key])
            for r in group
            if r.get(return_key) is not None
        ]
        if not vals:
            return {"count": len(group), "avg_return": None, "n_materialized": 0}
        return {
            "count": len(group),
            "n_materialized": len(vals),
            "avg_return": round(sum(vals) / len(vals), 4),
            "positive_rate": round(sum(1 for v in vals if v > 0) / len(vals), 4),
        }

    cb_records   = [r for r in confirmed if r.get("addon_type") == "continuation_breakout"]
    norm_records = [r for r in confirmed if r.get("addon_type") == "normal"]

    result: dict = {
        "n_total_addon_evaluations": len(confirmed),
        "phase_transition_addon_count": len(cb_records),
        "normal_addon_count":          len(norm_records),
    }

    for horizon, key in [("3d", "subsequent_3d_return"), ("5d", "subsequent_5d_return")]:
        cb_stats   = _group_stats(cb_records, key)
        norm_stats = _group_stats(norm_records, key)
        result[f"continuation_breakout_{horizon}"] = cb_stats
        result[f"normal_{horizon}"] = norm_stats

        # Delta: continuation_breakout avg - normal avg (only when both have data)
        cb_avg   = cb_stats.get("avg_return")
        norm_avg = norm_stats.get("avg_return")
        if cb_avg is not None and norm_avg is not None:
            result[f"avg_return_delta_vs_normal_{horizon}"] = round(cb_avg - norm_avg, 4)
        else:
            result[f"avg_return_delta_vs_normal_{horizon}"] = None

    # compression→breakout success rate and false breakout rate (3d)
    cb_3d_vals = [
        _safe(r["subsequent_3d_return"])
        for r in cb_records
        if r.get("subsequent_3d_return") is not None
    ]
    if cb_3d_vals:
        result["compression_to_breakout_success_rate"] = round(
            sum(1 for v in cb_3d_vals if v > 0) / len(cb_3d_vals), 4
        )
        result["false_breakout_rate"] = round(
            sum(1 for v in cb_3d_vals if v <= 0) / len(cb_3d_vals), 4
        )
    else:
        result["compression_to_breakout_success_rate"] = None
        result["false_breakout_rate"] = None

    return result


# ── Integration hook ──────────────────────────────────────────────────────────

def run_addon_continuation_hook(
    confirmations: List[Any],  # List[WinnerConfirmation]
    signals_by_sym: Dict[str, dict],
    *,
    log_path: Path,
    ohlcv_loader: Callable[[str], Optional[pd.DataFrame]],
    today: Optional[str] = None,
) -> dict:
    """
    Main hook for run_live_signal.py.
    observation_only: never modifies confirmations or signals.
    Fail-open.

    Call AFTER apply_continuation_breakout_boost() so boost fields are set.

    Returns summary: {n_recorded, n_materialized_3d, n_materialized_5d}.
    """
    try:
        return _run_internal(
            confirmations=confirmations,
            signals_by_sym=signals_by_sym,
            log_path=log_path,
            ohlcv_loader=ohlcv_loader,
            today=today or _today_str(),
        )
    except Exception as exc:
        logger.warning("[ACT] hook failed (%s) — continuing", exc)
        return {}


def _run_internal(
    confirmations: List[Any],
    signals_by_sym: Dict[str, dict],
    log_path: Path,
    ohlcv_loader: Callable[[str], Optional[pd.DataFrame]],
    today: str,
) -> dict:
    # Load existing records to dedup (same symbol+date)
    existing = _load_jsonl(log_path)
    already = {
        (r.get("symbol", ""), r.get("date", ""))
        for r in existing
    }

    n_recorded = 0
    for conf in confirmations:
        if not getattr(conf, "confirmed", False):
            continue
        sym = str(getattr(conf, "symbol", ""))
        if not sym:
            continue
        if (sym, today) in already:
            continue

        sig = signals_by_sym.get(sym, {})
        record = build_addon_record(conf, sig, today)
        _append_jsonl(record, log_path)
        already.add((sym, today))
        n_recorded += 1
        logger.info(
            "[ACT] recorded: %s type=%s boost=%.2f score_after=%.3f",
            sym, record["addon_type"],
            record["addon_boost_applied"], record["addon_confidence_after"],
        )

    n_3d, n_5d = materialize_returns(log_path, ohlcv_loader, today)

    logger.info(
        "[ACT] recorded=%d mat_3d=%d mat_5d=%d date=%s",
        n_recorded, n_3d, n_5d, today,
    )
    return {
        "n_recorded":        n_recorded,
        "n_materialized_3d": n_3d,
        "n_materialized_5d": n_5d,
    }
