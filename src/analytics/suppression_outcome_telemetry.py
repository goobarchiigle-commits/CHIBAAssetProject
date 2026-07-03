"""
suppression_outcome_telemetry.py — Attribution telemetry for exit suppression decisions.

Purpose:
  Measures whether compression-gate exit suppressions produce alpha extension
  or loss delay. Feeds P(alpha_extension | suppression) estimation.

Design:
  active_state_path:  suppression_active_state.json   — mutable JSON, max 3 entries
  outcome_path:       suppression_outcomes.jsonl       — append-only outcome records
  transition_path:    suppression_phase_transitions.jsonl — compression→breakout tracking

  observation_only: never modifies signals, never affects execution.
  fail-open: any error logs and continues.

Outcome classification (fixed thresholds, no adaptive logic):
  delta = return_after_suppression - return_if_exited_original
  delta >= +0.03  → "alpha_extension"
  delta <= -0.03  → "loss_delay"
  else            → "neutral"

Suppression detection:
  Signal has exit_orchestrator_applied=True AND action="HOLD" AND currently_holding=True.
  This fires when runtime_exit_orchestrator.apply_to_signals() overwrites signal=0→1.

Exit detection:
  Symbol was in active_state but is no longer in currently_holding signals
  (next run after actual exit, due to portfolio sync order).

Sufficiency criteria for probabilistic model readiness:
  MIN_SAMPLES = 20          total suppression outcomes
  MIN_ALPHA_EXT = 5         alpha_extension outcomes observed
  MIN_PHASE_TRANSITIONS = 10  compression→breakout transitions
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

# Outcome thresholds — fixed, never adaptive
ALPHA_EXT_THRESHOLD: float = 0.03
LOSS_DELAY_THRESHOLD: float = -0.03

# Sufficiency thresholds for model readiness gate
SUFFICIENCY_MIN_SAMPLES: int = 20
SUFFICIENCY_MIN_ALPHA_EXT: int = 5
SUFFICIENCY_MIN_PHASE_TRANSITIONS: int = 10

# Safety: warn if suppression active longer than this many calendar days
MAX_ACTIVE_SUPPRESSION_DAYS: int = 15


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
    """Atomic JSONL append via tempfile + os.replace. Fail-open."""
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
        logger.debug("[SOT] append_jsonl failed: %s", exc)


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
            logger.debug("[SOT] parse error line %d: %s", i + 1, exc)
    return records


def _load_active_state(path: Path) -> Dict[str, dict]:
    """Load active suppression state (JSON dict keyed by symbol). Returns {} on error."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("[SOT] active state load failed: %s", exc)
        return {}


def _save_active_state(state: Dict[str, dict], path: Path) -> None:
    """Atomic JSON write of active state. Fail-open."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(state, ensure_ascii=False, indent=2, default=str)
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
        logger.debug("[SOT] active state save failed: %s", exc)


# ── Price helpers ─────────────────────────────────────────────────────────────

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
        logger.debug("[SOT] price lookup failed %s: %s", date_str, exc)
    return None


def _compute_window_excursions(
    df: Optional[pd.DataFrame],
    start_date: str,
    end_date: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    (MFE, MAE) of incremental close changes from start_date close
    over the window [start_date, end_date], as fractions of start_date close.

    Returns (None, None) if insufficient data.
    """
    if df is None or df.empty:
        return None, None
    try:
        close_col = "Close" if "Close" in df.columns else "Adj Close"
        if close_col not in df.columns:
            return None, None

        idx = pd.DatetimeIndex(df.index)
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        mask = (idx >= start) & (idx <= end)
        window = df.loc[mask, close_col].dropna()
        if len(window) < 2:
            return None, None

        start_price = float(window.iloc[0])
        if start_price < _EPS:
            return None, None

        changes = (window - start_price) / start_price
        mfe = float(changes.max())
        mae = float(changes.min())

        if not (math.isfinite(mfe) and math.isfinite(mae)):
            return None, None
        return round(mfe, 6), round(mae, 6)
    except Exception as exc:
        logger.debug("[SOT] excursion compute failed: %s", exc)
        return None, None


# ── Core logic ────────────────────────────────────────────────────────────────

def classify_outcome(delta: float) -> str:
    """Classify suppression outcome from return delta."""
    if delta >= ALPHA_EXT_THRESHOLD:
        return "alpha_extension"
    if delta <= LOSS_DELAY_THRESHOLD:
        return "loss_delay"
    return "neutral"


def detect_new_suppressions(signals: List[dict]) -> List[dict]:
    """
    Return signals where exit was just suppressed by the exit orchestrator.
    Detection: exit_orchestrator_applied=True AND action="HOLD" AND currently_holding=True.
    """
    return [
        s for s in signals
        if s.get("exit_orchestrator_applied") is True
        and s.get("action") == "HOLD"
        and s.get("currently_holding") is True
        and s.get("symbol")
    ]


def _write_suppression_outcome(
    event: dict,
    final_return: float,
    phase_at_exit: str,
    today: str,
    ohlcv_loader: Callable[[str], Optional[pd.DataFrame]],
    outcome_path: Path,
) -> None:
    """Compute and append suppression outcome record. Fail-open per record."""
    try:
        sym = event["symbol"]
        start_return = _safe(event.get("suppression_start_return", 0.0))
        delta = final_return - start_return

        # MFE/MAE from OHLCV (optional — nullable on missing data)
        mfe: Optional[float] = None
        mae: Optional[float] = None
        try:
            df = ohlcv_loader(sym)
            mfe, mae = _compute_window_excursions(
                df, event["original_exit_date"], today
            )
        except Exception as exc:
            logger.debug("[SOT] excursion load failed %s: %s", sym, exc)

        start_dt = datetime.strptime(event["original_exit_date"], "%Y-%m-%d")
        end_dt = datetime.strptime(today, "%Y-%m-%d")
        suppression_days = (end_dt - start_dt).days

        record = {
            "date": today,
            "symbol": sym,
            "suppressed_exit": True,
            "original_exit_date": event["original_exit_date"],
            "final_exit_date": today,
            "suppression_days": suppression_days,
            "phase_at_suppression": event.get("phase_at_suppression", "neutral"),
            "phase_at_final_exit": phase_at_exit,
            "compression_score": round(_safe(event.get("compression_score", 0.0)), 2),
            "return_if_exited_original": round(start_return, 6),
            "return_after_suppression": round(final_return, 6),
            "max_favorable_excursion_after_suppression": mfe,
            "max_adverse_excursion_after_suppression": mae,
            "suppression_outcome": classify_outcome(delta),
            "state_at_resolution": event.get("state", "active"),
            "schema_version": SCHEMA_VERSION,
        }
        _append_jsonl(record, outcome_path)
        logger.info(
            "[SOT] outcome: %s delta=%+.2f%% outcome=%s days=%d",
            sym, delta * 100, record["suppression_outcome"], suppression_days,
        )
    except Exception as exc:
        logger.debug("[SOT] write_outcome failed for %s: %s", event.get("symbol", "?"), exc)


def detect_phase_transitions(
    snapshots_path: Path,
    transition_path: Path,
    ohlcv_loader: Callable[[str], Optional[pd.DataFrame]],
    today: str,
) -> int:
    """
    Scan compression_snapshots.jsonl for phase changes (e.g. compression→breakout).
    Append new transitions to transition_path.
    Attempt 5d subsequent return materialization for transitions >7 calendar days old.

    Returns count of newly written transition records. Fail-open.
    """
    snapshots = _load_jsonl(snapshots_path)
    if not snapshots:
        return 0

    already = {
        (r.get("symbol", ""), r.get("transition_date", ""))
        for r in _load_jsonl(transition_path)
    }

    by_sym: Dict[str, List[dict]] = defaultdict(list)
    for s in snapshots:
        sym = s.get("symbol", "")
        if sym:
            by_sym[sym].append(s)

    today_dt = datetime.strptime(today, "%Y-%m-%d")
    n_new = 0

    for sym, records in by_sym.items():
        records.sort(key=lambda r: r.get("date", ""))

        for i in range(1, len(records)):
            from_phase = records[i - 1].get("phase", "neutral")
            to_phase = records[i].get("phase", "neutral")
            curr_date = records[i].get("date", "")

            if not curr_date or from_phase == to_phase:
                continue
            if (sym, curr_date) in already:
                continue

            # Count consecutive days in from_phase ending at i-1
            days_in_phase = 0
            j = i - 1
            while j >= 0 and records[j].get("phase", "neutral") == from_phase:
                days_in_phase += 1
                j -= 1

            # Materialize 5d subsequent return if old enough
            subsequent_5d_return: Optional[float] = None
            try:
                trans_dt = datetime.strptime(curr_date, "%Y-%m-%d")
                if (today_dt - trans_dt).days >= 7:
                    df = ohlcv_loader(sym)
                    price_start = _get_close_at_date(df, curr_date)
                    bdr = pd.bdate_range(start=pd.Timestamp(curr_date), periods=6)
                    fwd_date = bdr[-1].strftime("%Y-%m-%d")
                    price_fwd = _get_close_at_date(df, fwd_date)
                    if price_start and price_fwd and price_start > _EPS:
                        subsequent_5d_return = round(
                            (price_fwd - price_start) / price_start, 6
                        )
            except Exception:
                pass

            transition_rec = {
                "symbol": sym,
                "transition_date": curr_date,
                "from_phase": from_phase,
                "to_phase": to_phase,
                "days_in_from_phase": days_in_phase,
                "subsequent_5d_return": subsequent_5d_return,
                "schema_version": SCHEMA_VERSION,
            }
            _append_jsonl(transition_rec, transition_path)
            already.add((sym, curr_date))
            n_new += 1
            logger.debug(
                "[SOT] transition: %s %s→%s days_in=%d",
                sym, from_phase, to_phase, days_in_phase,
            )

    return n_new


# ── Attribution analytics ─────────────────────────────────────────────────────

def compute_suppression_attribution(outcome_path: Path) -> dict:
    """
    Aggregate KPIs from suppression_outcomes.jsonl.
    Returns {} if no outcomes yet.
    """
    outcomes = _load_jsonl(outcome_path)
    if not outcomes:
        return {}

    total = len(outcomes)
    alpha_ext  = sum(1 for o in outcomes if o.get("suppression_outcome") == "alpha_extension")
    neutral    = sum(1 for o in outcomes if o.get("suppression_outcome") == "neutral")
    loss_delay = sum(1 for o in outcomes if o.get("suppression_outcome") == "loss_delay")

    deltas = [
        _safe(o.get("return_after_suppression", 0.0))
        - _safe(o.get("return_if_exited_original", 0.0))
        for o in outcomes
    ]

    maes = [
        _safe(o["max_adverse_excursion_after_suppression"])
        for o in outcomes
        if o.get("max_adverse_excursion_after_suppression") is not None
    ]
    mfes = [
        _safe(o["max_favorable_excursion_after_suppression"])
        for o in outcomes
        if o.get("max_favorable_excursion_after_suppression") is not None
    ]

    phase_breakdown: Dict[str, int] = defaultdict(int)
    for o in outcomes:
        phase_breakdown[o.get("phase_at_suppression", "unknown")] += 1

    days_list = [int(o.get("suppression_days", 0)) for o in outcomes]

    return {
        "n_suppressions":             total,
        "alpha_extension_rate":       round(alpha_ext / total, 4)  if total else None,
        "neutral_rate":               round(neutral / total, 4)    if total else None,
        "loss_delay_rate":            round(loss_delay / total, 4) if total else None,
        "avg_return_delta":           round(sum(deltas) / len(deltas), 4) if deltas else None,
        "avg_MAE_during_suppression": round(sum(maes) / len(maes), 4)    if maes else None,
        "avg_MFE_during_suppression": round(sum(mfes) / len(mfes), 4)    if mfes else None,
        "phase_breakdown":            dict(phase_breakdown),
        "avg_suppression_days":       round(sum(days_list) / len(days_list), 1) if days_list else None,
    }


def compute_sufficiency_check(outcome_path: Path, transition_path: Path) -> dict:
    """
    Check whether suppression telemetry has accumulated enough for a
    probabilistic model to be calibrated.

    Returns {"sufficient": bool, "reason": str, "details": dict}.
    """
    outcomes    = _load_jsonl(outcome_path)
    transitions = _load_jsonl(transition_path)

    n_total  = len(outcomes)
    n_alpha  = sum(1 for o in outcomes if o.get("suppression_outcome") == "alpha_extension")
    n_trans  = len(transitions)

    reasons: List[str] = []
    if n_total < SUFFICIENCY_MIN_SAMPLES:
        reasons.append(f"n_suppressions={n_total} < min={SUFFICIENCY_MIN_SAMPLES}")
    if n_alpha < SUFFICIENCY_MIN_ALPHA_EXT:
        reasons.append(f"n_alpha_extension={n_alpha} < min={SUFFICIENCY_MIN_ALPHA_EXT}")
    if n_trans < SUFFICIENCY_MIN_PHASE_TRANSITIONS:
        reasons.append(f"n_transitions={n_trans} < min={SUFFICIENCY_MIN_PHASE_TRANSITIONS}")

    return {
        "sufficient": len(reasons) == 0,
        "reason":     "; ".join(reasons) if reasons else "all criteria met",
        "details": {
            "n_suppressions":       n_total,
            "n_alpha_extension":    n_alpha,
            "n_phase_transitions":  n_trans,
            "thresholds": {
                "min_samples":          SUFFICIENCY_MIN_SAMPLES,
                "min_alpha_ext":        SUFFICIENCY_MIN_ALPHA_EXT,
                "min_phase_transitions": SUFFICIENCY_MIN_PHASE_TRANSITIONS,
            },
        },
    }


# ── Integration hook ──────────────────────────────────────────────────────────

def run_suppression_outcome_hook(
    signals: List[dict],
    *,
    active_state_path: Path,
    outcome_path: Path,
    transition_path: Path,
    compression_snapshots_path: Path,
    ohlcv_loader: Callable[[str], Optional[pd.DataFrame]],
    today: Optional[str] = None,
) -> dict:
    """
    Main hook for run_live_signal.py.
    observation_only: never modifies signals.
    Fail-open.

    Call AFTER run_exit_orchestration_hook so suppressed signals are visible.

    Returns summary: {n_suppression_starts, n_exits_resolved, n_transitions, n_active}.
    """
    try:
        return _run_internal(
            signals=signals,
            active_state_path=active_state_path,
            outcome_path=outcome_path,
            transition_path=transition_path,
            compression_snapshots_path=compression_snapshots_path,
            ohlcv_loader=ohlcv_loader,
            today=today or _today_str(),
        )
    except Exception as exc:
        logger.warning("[SOT] hook failed (%s) — continuing", exc)
        return {}


def _run_internal(
    signals: List[dict],
    active_state_path: Path,
    outcome_path: Path,
    transition_path: Path,
    compression_snapshots_path: Path,
    ohlcv_loader: Callable[[str], Optional[pd.DataFrame]],
    today: str,
) -> dict:
    today_dt = datetime.strptime(today, "%Y-%m-%d")

    # ── Load active state ─────────────────────────────────────────────────────
    active_state = _load_active_state(active_state_path)

    # ── Build currently-holding lookup ────────────────────────────────────────
    currently_holding: Dict[str, dict] = {
        str(s.get("symbol", "")): s
        for s in signals
        if s.get("currently_holding") and s.get("symbol")
    }

    # ── Record newly suppressed positions ─────────────────────────────────────
    n_starts = 0
    for sig in detect_new_suppressions(signals):
        sym = str(sig.get("symbol", ""))
        if sym in active_state:
            logger.debug("[SOT] re-suppression for %s — keeping original start", sym)
            continue
        active_state[sym] = {
            "symbol": sym,
            "state": "active",
            "original_exit_date": today,
            "suppression_start_return": round(_safe(sig.get("current_return", 0.0)), 6),
            "phase_at_suppression": sig.get("compression_phase", "neutral"),
            "compression_score": round(_safe(sig.get("compression_score", 0.0)), 2),
            "hold_extension_days": int(sig.get("hold_extension_days", 0)),
            "original_exit_reason": sig.get("exit_reason", ""),
            "last_seen_return": round(_safe(sig.get("current_return", 0.0)), 6),
            "last_seen_phase": sig.get("compression_phase", "neutral"),
            "last_updated": today,
        }
        n_starts += 1
        logger.info(
            "[SOT] suppression start: %s score=%.1f phase=%s ext=%dd return=%+.2f%%",
            sym,
            active_state[sym]["compression_score"],
            active_state[sym]["phase_at_suppression"],
            active_state[sym]["hold_extension_days"],
            active_state[sym]["suppression_start_return"] * 100,
        )

    # ── Update tracking state + resolve exits ─────────────────────────────────
    n_exits = 0
    resolved: List[str] = []

    for sym, event in list(active_state.items()):
        if sym in currently_holding:
            # Still holding — update live tracking fields
            sig = currently_holding[sym]
            event["last_seen_return"] = round(
                _safe(sig.get("current_return", event.get("last_seen_return", 0.0))), 6
            )
            event["last_seen_phase"] = sig.get(
                "compression_phase", event.get("last_seen_phase", "neutral")
            )
            event["last_updated"] = today

            # Warn if held longer than safety limit
            start_dt = datetime.strptime(event["original_exit_date"], "%Y-%m-%d")
            days_active = (today_dt - start_dt).days
            if days_active > MAX_ACTIVE_SUPPRESSION_DAYS:
                logger.warning(
                    "[SOT] suppression for %s active %dd (> %dd limit) — still holding",
                    sym, days_active, MAX_ACTIVE_SUPPRESSION_DAYS,
                )
        else:
            # Position exited — resolve outcome
            final_return = _safe(
                event.get("last_seen_return", event.get("suppression_start_return", 0.0))
            )
            phase_at_exit = event.get("last_seen_phase", "neutral")

            start_dt = datetime.strptime(event["original_exit_date"], "%Y-%m-%d")
            days_held = (today_dt - start_dt).days
            ext_days = int(event.get("hold_extension_days", 0))
            event["state"] = "timeout_exit" if days_held > ext_days else "success_exit"

            _write_suppression_outcome(
                event=event,
                final_return=final_return,
                phase_at_exit=phase_at_exit,
                today=today,
                ohlcv_loader=ohlcv_loader,
                outcome_path=outcome_path,
            )
            resolved.append(sym)
            n_exits += 1

    for sym in resolved:
        del active_state[sym]

    # ── Persist updated active state ──────────────────────────────────────────
    _save_active_state(active_state, active_state_path)

    # ── Detect phase transitions from compression snapshots ───────────────────
    n_transitions = 0
    try:
        n_transitions = detect_phase_transitions(
            snapshots_path=compression_snapshots_path,
            transition_path=transition_path,
            ohlcv_loader=ohlcv_loader,
            today=today,
        )
    except Exception as exc:
        logger.debug("[SOT] phase transition detection failed: %s", exc)

    logger.info(
        "[SOT] starts=%d exits=%d transitions=%d active=%d date=%s",
        n_starts, n_exits, n_transitions, len(active_state), today,
    )
    return {
        "n_suppression_starts": n_starts,
        "n_exits_resolved":     n_exits,
        "n_transitions":        n_transitions,
        "n_active":             len(active_state),
    }
