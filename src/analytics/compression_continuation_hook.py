"""
compression_continuation_hook.py — Live integration hook for run_live_signal.py

Loads OHLCV for held positions, computes compression continuation scores,
injects results into signal dicts for exit orchestrator consumption,
and appends JSONL log.

Call BEFORE run_exit_orchestration_hook so the orchestrator sees
compression_score / compression_phase / suppression_eligible in signals.

Fail-open: any error returns signals unchanged.
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.analytics.compression_continuation import (
    CompressionContinuationInput,
    CompressionContinuationResult,
    compute_compression_continuation,
)

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))
_EPS = 1e-9


def _today_str() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d")


def _safe(v: Any, fallback: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


# ── OHLCV feature computation from raw arrays ─────────────────────────────────

def _compute_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> float:
    """Average true range over last `period` bars. Fallback: 2% of last close."""
    n = min(period, len(high) - 1)
    if n < 1:
        return float(close[-1]) * 0.02 if len(close) > 0 else 1.0
    trs: list[float] = []
    for i in range(-n, 0):
        h = float(high[i]) if np.isfinite(high[i]) else 0.0
        l = float(low[i])  if np.isfinite(low[i])  else 0.0
        c_prev_idx = i - 1
        if c_prev_idx < -len(close):
            c_prev_idx = 0
        c = float(close[c_prev_idx]) if np.isfinite(close[c_prev_idx]) else 0.0
        if h > 0:
            tr = max(h - l, abs(h - c), abs(l - c))
            trs.append(tr)
    return float(np.mean(trs)) if trs else float(abs(close[-1])) * 0.02


def compute_ohlcv_features(df: pd.DataFrame, window: int = 20) -> Optional[Dict[str, float]]:
    """
    Compute compression continuation features from OHLCV DataFrame.
    Returns None if insufficient data (< 65 bars needed for ATR60).
    All intermediate arrays are numpy float64. NaN values replaced where possible.
    """
    MIN_BARS = 65  # need 60 for ATR60 + 5 buffer
    if len(df) < MIN_BARS:
        return None

    df = df.sort_index()
    close_col = "Close" if "Close" in df.columns else "Adj Close"
    high_col  = "High"  if "High"  in df.columns else close_col
    low_col   = "Low"   if "Low"   in df.columns else close_col
    vol_col   = "Volume" if "Volume" in df.columns else None

    close  = np.array(df[close_col].values, dtype=float)
    high   = np.array(df[high_col].values,  dtype=float)
    low    = np.array(df[low_col].values,   dtype=float)
    volume = (
        np.array(df[vol_col].values, dtype=float)
        if vol_col is not None
        else np.ones(len(close), dtype=float)
    )

    # Guard: if last 20 closes are all NaN, cannot compute
    recent_close = close[-20:]
    if np.all(~np.isfinite(recent_close)):
        return None

    # Forward-fill NaN for price arrays (carry last known value)
    for arr in (close, high, low):
        mask = ~np.isfinite(arr)
        for i in range(len(arr)):
            if mask[i] and i > 0:
                arr[i] = arr[i - 1]
    # Volume: replace NaN/zero with median
    vol_valid = volume[np.isfinite(volume) & (volume > 0)]
    vol_median = float(np.median(vol_valid)) if len(vol_valid) > 0 else 1.0
    volume[~np.isfinite(volume)] = vol_median
    volume[volume <= 0] = vol_median

    cur = float(close[-1])

    # ── near_high_pct ─────────────────────────────────────────────────────────
    h20 = float(np.nanmax(high[-window:]))
    l20 = float(np.nanmin(low[-window:]))
    near_high_pct = (cur - l20) / (h20 - l20 + _EPS)
    near_high_pct = float(np.clip(near_high_pct, 0.0, 1.0))

    # ── higher_low_streak ─────────────────────────────────────────────────────
    streak = 0
    lows_w = low[-window:]
    for i in range(len(lows_w) - 1, 0, -1):
        if np.isfinite(lows_w[i]) and np.isfinite(lows_w[i - 1]):
            if lows_w[i] > lows_w[i - 1]:
                streak += 1
            else:
                break

    # ── dip_recovery_pct: max-close-drop / ATR20 over last 10 bars ───────────
    atr20 = _compute_atr(high, low, close, 20)
    close10 = close[-10:][np.isfinite(close[-10:])]
    if len(close10) >= 2 and atr20 > _EPS:
        drawdown_abs = float(np.nanmax(close10) - np.nanmin(close10))
        dip_recovery_pct = float(drawdown_abs / atr20)
    else:
        dip_recovery_pct = 1.0
    dip_recovery_pct = min(dip_recovery_pct, 5.0)

    # ── compression_ratio: ATR5 / ATR60 ─────────────────────────────────────
    atr5  = _compute_atr(high, low, close, 5)
    atr60 = _compute_atr(high, low, close, 60)
    compression_ratio = atr5 / (atr60 + _EPS)

    # ── price_tightness: std5 / std20 ────────────────────────────────────────
    close_valid = close[np.isfinite(close)]
    std5  = float(np.std(close_valid[-5:]))  if len(close_valid) >= 5  else 0.0
    std20 = float(np.std(close_valid[-20:])) if len(close_valid) >= 20 else (std5 + _EPS)
    price_tightness = std5 / (std20 + _EPS)

    # ── volume_ratio_5d: mean5 / mean20 ──────────────────────────────────────
    vol5  = float(np.mean(volume[-5:]))
    vol20 = float(np.mean(volume[-20:]))
    volume_ratio_5d = vol5 / (vol20 + _EPS) if vol20 > _EPS else 1.0

    return {
        "compression_ratio": float(np.clip(compression_ratio, 0.05, 5.0)),
        "price_tightness":   float(np.clip(price_tightness,   0.0,  5.0)),
        "volume_ratio_5d":   float(np.clip(volume_ratio_5d,   0.0,  10.0)),
        "higher_low_streak": int(streak),
        "dip_recovery_pct":  float(dip_recovery_pct),
        "near_high_pct":     float(near_high_pct),
    }


# ── OHLCV loader ──────────────────────────────────────────────────────────────

def _load_ohlcv(
    symbol: str,
    backtest_dataset_dir: Path,
    default_data_version: str,
    cache_dir: Path,
) -> Optional[pd.DataFrame]:
    """Load OHLCV parquet for one symbol. Returns None if unavailable."""
    candidates: list[Path] = []
    if default_data_version:
        candidates.append(backtest_dataset_dir / default_data_version / f"{symbol}.parquet")
    candidates.append(cache_dir / "ohlcv" / f"{symbol}.parquet")
    for p in candidates:
        if not p.exists():
            continue
        try:
            return pd.read_parquet(p)
        except Exception as exc:
            logger.debug("[COMPRESS_HOOK] load failed %s @ %s: %s", symbol, p, exc)
    return None


# ── Previous-phase lookup for continuation breakout detection ────────────────

_MAX_PREV_PHASE_LOOKBACK_DAYS: int = 3  # only look back this many calendar days


def _get_previous_phase(
    symbol: str,
    today: str,
    log_path: Path,
) -> Optional[str]:
    """
    Return the most recent phase for `symbol` from the snapshot log,
    excluding today's record. Only considers records within
    _MAX_PREV_PHASE_LOOKBACK_DAYS calendar days.

    Returns None if not found or data is too old.
    Fail-open: any error returns None.
    """
    if not log_path.exists():
        return None
    try:
        today_dt = datetime.strptime(today, "%Y-%m-%d")
        cutoff = (today_dt - timedelta(days=_MAX_PREV_PHASE_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        best: Optional[dict] = None
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("symbol") != symbol:
                    continue
                r_date = r.get("date", "")
                if r_date >= today or r_date < cutoff:
                    continue
                if best is None or r_date > best.get("date", ""):
                    best = r
            except Exception:
                pass
        return best.get("phase") if best else None
    except Exception as exc:
        logger.debug("[COMPRESS_HOOK] prev_phase lookup failed %s: %s", symbol, exc)
        return None


# ── Atomic JSONL append ───────────────────────────────────────────────────────

def _append_jsonl(record: dict, path: Path) -> None:
    """Atomic append. Fail-open: logging errors never block execution."""
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
        logger.debug("[COMPRESS_HOOK] append failed: %s", exc)


# ── Main hook ─────────────────────────────────────────────────────────────────

def run_compression_continuation_hook(
    signals: List[Dict[str, Any]],
    *,
    backtest_dataset_dir: Path,
    default_data_version: str,
    cache_dir: Path,
    log_path: Path,
    run_id: str = "",
    as_of_date: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Score held positions for compression continuation.

    Injects into each held-position signal dict:
      compression_score       float [0, 100]
      compression_phase       str "compression" | "distribution" | "breakout" | "neutral"
      suppression_eligible    bool
      compression_invalidations  list[str]

    Appends snapshot to log_path JSONL.

    Returns (modified_signals, summary_dict).
    Fail-open: any error returns (signals_unchanged, {}).
    """
    try:
        return _run_internal(
            signals=signals,
            backtest_dataset_dir=backtest_dataset_dir,
            default_data_version=default_data_version,
            cache_dir=cache_dir,
            log_path=log_path,
            run_id=run_id,
            today=as_of_date or _today_str(),
        )
    except Exception as exc:
        logger.warning("[COMPRESS_HOOK] hook failed (%s) — signals unchanged", exc)
        return signals, {}


def _run_internal(
    signals: List[Dict[str, Any]],
    backtest_dataset_dir: Path,
    default_data_version: str,
    cache_dir: Path,
    log_path: Path,
    run_id: str,
    today: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    results: Dict[str, CompressionContinuationResult] = {}
    continuation_breakouts: Dict[str, bool] = {}
    volume_ratios: Dict[str, float] = {}

    for sig in signals:
        if not sig.get("currently_holding"):
            continue
        symbol = str(sig.get("symbol", ""))
        if not symbol:
            continue

        df = _load_ohlcv(symbol, backtest_dataset_dir, default_data_version, cache_dir)
        if df is None:
            logger.debug("[COMPRESS_HOOK] no OHLCV for %s — skip", symbol)
            continue

        features = compute_ohlcv_features(df)
        if features is None:
            logger.debug("[COMPRESS_HOOK] insufficient history for %s — skip", symbol)
            continue

        inp = CompressionContinuationInput(
            symbol=symbol,
            hold_days=int(sig.get("hold_days", 0)),
            unrealized_pnl_pct=_safe(sig.get("current_return", 0.0)),
            compression_ratio=features["compression_ratio"],
            price_tightness=features["price_tightness"],
            volume_ratio_5d=features["volume_ratio_5d"],
            higher_low_streak=features["higher_low_streak"],
            dip_recovery_pct=features["dip_recovery_pct"],
            near_high_pct=features["near_high_pct"],
            rsr=_safe(sig.get("rsr", 50.0), 50.0),
            rsr_momentum=_safe(sig.get("rsr_momentum", 0.0)),
        )

        result = compute_compression_continuation(inp)
        results[symbol] = result
        volume_ratios[symbol] = features["volume_ratio_5d"]

        # ── Continuation breakout detection ───────────────────────────────────
        # Checks compression→breakout transition with volume expansion.
        # Read previous phase BEFORE appending today's record.
        prev_phase = _get_previous_phase(symbol, today, log_path)
        rsr_mom = _safe(sig.get("rsr_momentum", 0.0))
        is_cb = (
            prev_phase == "compression"
            and result.phase == "breakout"
            and features["volume_ratio_5d"] >= 1.30  # volume expansion required
            and rsr_mom > 0.0
            and result.score >= 70.0
        )
        continuation_breakouts[symbol] = is_cb
        if is_cb:
            logger.info(
                "[COMPRESS_HOOK] continuation_breakout: %s vol=%.2f rsr_mom=%.2f score=%.1f",
                symbol, features["volume_ratio_5d"], rsr_mom, result.score,
            )

        _append_jsonl(
            {
                "date":               today,
                "run_id":             run_id,
                "symbol":             symbol,
                "hold_days":          inp.hold_days,
                "unrealized_pnl_pct": round(inp.unrealized_pnl_pct, 4),
                "continuation_breakout": is_cb,
                "prev_phase":         prev_phase,
                **result.to_dict(),
                "raw_features": {k: round(v, 4) if isinstance(v, float) else v
                                 for k, v in features.items()},
            },
            log_path,
        )

        logger.info(
            "[COMPRESS_HOOK] %s phase=%s score=%.1f suppress=%s inv=%s cb=%s",
            symbol, result.phase, result.score,
            result.suppression_eligible, result.invalidation_flags, is_cb,
        )

    # Inject scores into signal dicts
    modified: List[Dict[str, Any]] = []
    for sig in signals:
        sym = str(sig.get("symbol", ""))
        if sym in results:
            r = results[sym]
            new_sig = dict(sig)
            new_sig["compression_score"]          = round(r.score, 2)
            new_sig["compression_phase"]          = r.phase
            new_sig["suppression_eligible"]       = r.suppression_eligible
            new_sig["compression_invalidations"]  = r.invalidation_flags
            new_sig["continuation_breakout"]      = continuation_breakouts.get(sym, False)
            new_sig["compression_volume_ratio"]   = round(volume_ratios.get(sym, 1.0), 4)
            modified.append(new_sig)
        else:
            modified.append(sig)

    n_eligible = sum(1 for r in results.values() if r.suppression_eligible)
    n_distrib  = sum(1 for r in results.values() if r.is_distribution)
    n_cb       = sum(1 for v in continuation_breakouts.values() if v)

    return modified, {
        "n_scored":              len(results),
        "n_eligible":            n_eligible,
        "n_distrib":             n_distrib,
        "n_continuation_breakout": n_cb,
        "results":               {s: r.to_dict() for s, r in results.items()},
    }
