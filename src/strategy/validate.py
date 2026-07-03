"""
validate.py — statistical safety guard for live strategy decisions

Environment variables (all overridable at runtime):
    MIN_OBS          int   minimum observation count before full evaluation  (default 20)
    MIN_SHARPE       float Sharpe threshold for ACCEPT                       (default 0.2)
    WARMUP_DAYS      int   observation count treated as warmup phase         (default 30)
    FORCE_MIN_TRADES int   trade count below which bootstrap exploration runs (default 5)
    EPS              float zero-variance guard threshold                     (default 1e-8)
"""

from __future__ import annotations

import datetime
import math
import os
from typing import Any

import pandas as pd

# ── Environment variables ─────────────────────────────────────────────────────
MIN_OBS          = int(os.environ.get("MIN_OBS",          20))
MIN_SHARPE       = float(os.environ.get("MIN_SHARPE",      0.2))
WARMUP_DAYS      = int(os.environ.get("WARMUP_DAYS",      30))
FORCE_MIN_TRADES = int(os.environ.get("FORCE_MIN_TRADES", 5))
EPS              = float(os.environ.get("EPS",             1e-8))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _safe(value: Any, fallback: float = 0.0) -> float:
    """NaN / inf / None → fallback.  Division-by-zero cannot reach here."""
    try:
        v = float(value)
        return v if math.isfinite(v) else fallback
    except (TypeError, ValueError):
        return fallback


def _build(
    decision:   str,
    multiplier: float,
    reason:     str,
    mean_net:   float = 0.0,
    sharpe:     float = 0.0,
    stability:  float = 0.0,
    n_obs:      int   = 0,
    timestamp:  str   = "",
) -> dict[str, Any]:
    return {
        "decision":   decision,
        "mean_net":   _safe(mean_net),
        "sharpe":     _safe(sharpe),
        "stability":  _safe(stability),
        "n_obs":      int(n_obs),
        "multiplier": _safe(max(0.0, multiplier)),
        "reason":     reason,
        "timestamp":  timestamp,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def validate_strategy(
    daily_decomp: pd.DataFrame,
    total_trades: int,
) -> dict[str, Any]:
    """
    Statistical safety guard.  Pure function — no side effects.

    Parameters
    ----------
    daily_decomp : DataFrame with column "net" (daily net returns).
                   Extra columns are ignored.
    total_trades : cumulative executed trades for the strategy so far.

    Returns
    -------
    {
      "decision":   "FORCE" | "WARMUP" | "SKIP" | "DISCARD" | "ACCEPT",
      "mean_net":   float,
      "sharpe":     float,
      "stability":  float,   # == std_net
      "n_obs":      int,
      "multiplier": float,   # position-size scalar ∈ [0, 1]
      "reason":     str,
      "timestamp":  str,     # ISO-8601 UTC
    }

    Decision ladder (evaluated in order — first match wins):
      total_trades < FORCE_MIN_TRADES  → FORCE      multiplier=0.1
      n_obs        < WARMUP_DAYS       → WARMUP     multiplier=0.0
      n_obs        < MIN_OBS           → SKIP       multiplier=0.0
      std_net      < EPS               → SKIP       multiplier=0.0
      sharpe       < MIN_SHARPE        → DISCARD    multiplier=0.0
      else                             → ACCEPT     multiplier=tanh(sharpe)

    Guarantees
    ----------
    - Never divides by zero.
    - All NaN / inf in inputs are coerced to 0.0 in output.
    - Single-sample decisions are prohibited (n_obs < MIN_OBS path catches them).
    - mean_net alone never drives a decision (Sharpe required for ACCEPT).
    """
    ts = _now_utc()

    # ── 1. Basic statistics ───────────────────────────────────────────────────
    if "net" not in daily_decomp.columns:
        return _build("SKIP", 0.0, "missing_net_column", timestamp=ts)

    net = daily_decomp["net"].dropna()
    n_obs = len(net)

    mean_net = _safe(net.mean()) if n_obs > 0 else 0.0
    # ddof=1 requires n_obs >= 2; fallback 0.0 for n_obs < 2
    std_net  = _safe(net.std(ddof=1)) if n_obs >= 2 else 0.0

    # ── 2. Bootstrap exploration ──────────────────────────────────────────────
    if total_trades < FORCE_MIN_TRADES:
        return _build(
            "FORCE", 0.1, "bootstrap_exploration",
            mean_net=mean_net, stability=std_net, n_obs=n_obs, timestamp=ts,
        )

    # ── 3. Warmup phase ───────────────────────────────────────────────────────
    if n_obs < WARMUP_DAYS:
        return _build(
            "WARMUP", 0.0, "warmup_phase",
            mean_net=mean_net, stability=std_net, n_obs=n_obs, timestamp=ts,
        )

    # ── 4. Minimum sample constraint ─────────────────────────────────────────
    if n_obs < MIN_OBS:
        return _build(
            "SKIP", 0.0, "insufficient_history",
            mean_net=mean_net, stability=std_net, n_obs=n_obs, timestamp=ts,
        )

    # ── 5. Zero-variance guard ────────────────────────────────────────────────
    if std_net < EPS:
        return _build(
            "SKIP", 0.0, "zero_variance",
            mean_net=mean_net, stability=std_net, n_obs=n_obs, timestamp=ts,
        )

    # ── 6. Sharpe (division-by-zero impossible: std_net >= EPS) ──────────────
    sharpe = _safe(mean_net / std_net)

    # ── 7. Decision ───────────────────────────────────────────────────────────
    if sharpe < MIN_SHARPE:
        return _build(
            "DISCARD", 0.0, "low_sharpe",
            mean_net=mean_net, sharpe=sharpe, stability=std_net,
            n_obs=n_obs, timestamp=ts,
        )

    multiplier = max(0.0, math.tanh(sharpe))
    return _build(
        "ACCEPT", multiplier, "valid_signal",
        mean_net=mean_net, sharpe=sharpe, stability=std_net,
        n_obs=n_obs, timestamp=ts,
    )
