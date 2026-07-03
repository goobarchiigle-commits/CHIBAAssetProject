"""
src/execution/dd_engine.py
Canonical single-implementation drawdown and risk engine.

RULE:
  - All DD / CB / SAFE_WARN logic must call compute_drawdown().
  - Forbidden: (current - peak) / peak inline anywhere else.
  - All risk logic consumes canonical BrokerSnapshot equity values.

Replaces any inline DD formulas in startup_check.py, signal_bridge.py, etc.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Constants (mirrors CLAUDE.md PARAMS_LOCKED / CIRCUIT) ────────────────────
MAX_DD_LIMIT:               float = -0.15   # BUY_STOP threshold
SAFE_WARN_ANOMALY_RATIO:    float =  1.25   # peak/equity anomaly (state corruption)
SAFE_WARN_CONFIRM_REQUIRED: int   =  3      # consecutive SAFE_WARN before CB escalation
RECOVERY_THRESHOLD_RATIO:   float =  0.90   # equity must reach this % of peak to recover


def compute_drawdown(current_equity: float, peak_equity: float) -> float:
    """
    Canonical drawdown: (current - peak) / peak.

    Returns 0.0 for invalid inputs (never raises).
    Negative value = drawdown, positive = new-high territory.

    [DD] equity=... peak=... dd=... is logged by assess_risk().
    """
    if peak_equity <= 0 or current_equity <= 0:
        return 0.0
    return (current_equity - peak_equity) / peak_equity


def is_cb_trigger(dd: float) -> bool:
    """True when dd ≤ MAX_DD_LIMIT (BUY_STOP condition)."""
    return dd <= MAX_DD_LIMIT


def compute_recovery_threshold(peak_equity: float) -> float:
    """Equity must exceed this value for CB to lift."""
    return peak_equity * RECOVERY_THRESHOLD_RATIO


def assess_risk(
    current_equity:  float,
    peak_equity:     float,
    cb_state:        str  = "NORMAL",
    safe_warn_count: int  = 0,
) -> dict:
    """
    Single canonical risk assessment producing a recommendation.

    Consumes equity values only — no external state reads.

    Returns:
        {
          "dd":              float,
          "cb_trigger":      bool,
          "peak_anomaly":    bool,
          "safe_warn_count": int,
          "recommendation":  str,   # "NORMAL" | "SAFE_WARN" | "CB_ACTIVE" | "RECOVERY"
          "message":         str,
        }
    """
    dd = compute_drawdown(current_equity, peak_equity)

    logger.info(
        "[DD] equity=¥%.0f peak=¥%.0f dd=%.4f cb_state=%s",
        current_equity, peak_equity, dd, cb_state,
    )

    peak_anomaly = (
        peak_equity > 0
        and current_equity > 0
        and peak_equity / current_equity > SAFE_WARN_ANOMALY_RATIO
    )

    # ── CB trigger ───────────────────────────────────────────────────
    if is_cb_trigger(dd):
        logger.warning(
            "[DD] CIRCUIT BREAKER TRIGGER: dd=%.2f%% <= %.2f%% equity=%.0f peak=%.0f",
            dd * 100, MAX_DD_LIMIT * 100, current_equity, peak_equity,
        )
        return {
            "dd":              dd,
            "cb_trigger":      True,
            "peak_anomaly":    peak_anomaly,
            "safe_warn_count": 0,
            "recommendation":  "CB_ACTIVE",
            "message":         f"DD={dd:.2%} ≤ {MAX_DD_LIMIT:.2%} — circuit breaker",
        }

    # ── CB recovery ──────────────────────────────────────────────────
    if cb_state in ("CB_ACTIVE", "RECOVERY"):
        threshold = compute_recovery_threshold(peak_equity)
        if current_equity >= threshold:
            return {
                "dd":              dd,
                "cb_trigger":      False,
                "peak_anomaly":    False,
                "safe_warn_count": 0,
                "recommendation":  "NORMAL",
                "message":         (
                    f"CB recovery: equity=¥{current_equity:,.0f} "
                    f"≥ threshold=¥{threshold:,.0f}"
                ),
            }
        return {
            "dd":              dd,
            "cb_trigger":      False,
            "peak_anomaly":    False,
            "safe_warn_count": safe_warn_count,
            "recommendation":  "RECOVERY",
            "message":         (
                f"CB recovery pending: equity=¥{current_equity:,.0f} "
                f"< threshold=¥{threshold:,.0f}"
            ),
        }

    # ── Peak anomaly → SAFE_WARN ──────────────────────────────────────
    if peak_anomaly:
        new_count = safe_warn_count + 1
        if new_count >= SAFE_WARN_CONFIRM_REQUIRED:
            return {
                "dd":              dd,
                "cb_trigger":      False,
                "peak_anomaly":    True,
                "safe_warn_count": new_count,
                "recommendation":  "CB_ACTIVE",
                "message":         (
                    f"SAFE_WARN escalated: confirm={new_count}/{SAFE_WARN_CONFIRM_REQUIRED}"
                ),
            }
        return {
            "dd":              dd,
            "cb_trigger":      False,
            "peak_anomaly":    True,
            "safe_warn_count": new_count,
            "recommendation":  "SAFE_WARN",
            "message":         (
                f"PEAK_ANOMALY: peak/equity={peak_equity/current_equity:.3f} "
                f"> {SAFE_WARN_ANOMALY_RATIO} confirm={new_count}/{SAFE_WARN_CONFIRM_REQUIRED}"
            ),
        }

    # ── Normal ────────────────────────────────────────────────────────
    return {
        "dd":              dd,
        "cb_trigger":      False,
        "peak_anomaly":    False,
        "safe_warn_count": 0,
        "recommendation":  "NORMAL",
        "message":         f"DD={dd:.2%} — normal",
    }
