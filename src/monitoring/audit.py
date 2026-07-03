"""
src/monitoring/audit.py
メトリクス計算とログ完全性チェック。純粋関数ベース。副作用なし。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

_REQUIRED: frozenset[str] = frozenset({"pnl", "expected_risk", "slippage_real", "slippage_est"})
_METRICS_REQUIRED: list[str] = ["risk_ratio", "max_dd", "slip_diff"]
_MIN_SAMPLE_DEGRADED: int = 5


def _safe(v: Any, fallback: float = 0.0) -> float:
    """NaN / inf / None → fallback。"""
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


def detect_strategy_degradation(df: pd.DataFrame, cfg: dict) -> bool:
    """
    戦略劣化検出（純粋関数）。小サンプル安全版。
    有限値のみで mean_pnl / win_rate を計算する。
    """
    if len(df) == 0:
        return False
    pnl_arr = df["pnl"].values.astype(float)
    finite_mask = np.isfinite(pnl_arr)
    if not finite_mask.any():
        return False
    finite_pnl = pnl_arr[finite_mask]
    mean_pnl = float(np.mean(finite_pnl))
    win_rate = float(np.mean(finite_pnl > 0))
    pnl_floor = _safe(cfg.get("PNL_FLOOR", math.inf), fallback=math.inf)
    min_win = _safe(cfg.get("MIN_WIN_RATE", 0.0), fallback=0.0)
    if not math.isfinite(pnl_floor):
        return False
    return mean_pnl < -pnl_floor and win_rate < min_win


def sanitize_metrics(metrics: dict) -> dict | None:
    """
    メトリクス dict の完全性チェック（純粋関数）。
    必須キー欠損 / None / NaN / inf を含む場合は None を返す。
    """
    if metrics is None:
        return None
    for k in _METRICS_REQUIRED:
        if k not in metrics:
            return None
    cleaned: dict = {}
    for k, v in metrics.items():
        if v is None:
            return None
        if isinstance(v, (float, np.floating)):
            if not math.isfinite(float(v)):
                return None
        cleaned[k] = v
    return cleaned


def compute_metrics(df: pd.DataFrame, cfg: dict | None = None) -> dict[str, Any]:
    """
    取引ログ DataFrame からメトリクスを計算する。
    全出力が有限値であることを保証する。

    必須列:
        pnl            損益
        expected_risk  想定リスク
        slippage_real  実際スリッページ
        slippage_est   推定スリッページ

    Returns:
        max_dd       float  最大ドローダウン（0以下）
        slip_diff    float  slippage_real - slippage_est の平均
        risk_ratio   float  |pnl| / expected_risk の平均
        n_trades     int
        degraded     bool
        status       "OK" | "HALT_CORRUPT"
        corrupt_reason  str | None
    """
    result: dict[str, Any] = {
        "max_dd":         0.0,
        "slip_diff":      0.0,
        "risk_ratio":     0.0,
        "n_trades":       0,
        "degraded":       False,
        "status":         "OK",
        "corrupt_reason": None,
    }

    # 欠損列チェック
    missing = _REQUIRED - set(df.columns)
    if missing:
        result["status"] = "HALT_CORRUPT"
        result["corrupt_reason"] = f"欠損列: {sorted(missing)}"
        return result

    # NaN チェック
    nan_cols = [c for c in _REQUIRED if df[c].isna().any()]
    if nan_cols:
        result["status"] = "HALT_CORRUPT"
        result["corrupt_reason"] = f"NaN検出列: {sorted(nan_cols)}"
        return result

    # inf チェック（追加）
    inf_cols = [
        c for c in _REQUIRED
        if np.isinf(pd.to_numeric(df[c], errors="coerce").values).any()
    ]
    if inf_cols:
        result["status"] = "HALT_CORRUPT"
        result["corrupt_reason"] = f"inf検出列: {sorted(inf_cols)}"
        return result

    n = len(df)
    result["n_trades"] = n
    if n == 0:
        return result

    pnl_arr = df["pnl"].values.astype(float)

    # max_dd: equity ≤ 0 崩壊保護
    cum = np.cumsum(pnl_arr)
    if not np.all(np.isfinite(cum)):
        result["max_dd"] = 0.0
    else:
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        max_dd_abs = float(np.min(dd))
        peak_max = float(np.max(peak))
        if peak_max > 1e-12:
            raw = max_dd_abs / peak_max
            result["max_dd"] = float(raw) if math.isfinite(raw) else 0.0
        else:
            result["max_dd"] = 0.0

    # slip_diff: NaN 平均保護
    slip_arr = (df["slippage_real"] - df["slippage_est"]).values.astype(float)
    finite_slip = slip_arr[np.isfinite(slip_arr)]
    result["slip_diff"] = float(np.mean(finite_slip)) if len(finite_slip) > 0 else 0.0

    # risk_ratio: ゼロ除算 + NaN 保護
    risk_arr = df["expected_risk"].values.astype(float)
    pnl_abs = np.abs(pnl_arr)
    valid_risk = np.abs(risk_arr) > 1e-12
    safe_risk = np.where(valid_risk, risk_arr, np.nan)
    ratios = pnl_abs / safe_risk
    finite_ratios = ratios[np.isfinite(ratios)]
    result["risk_ratio"] = float(np.mean(finite_ratios)) if len(finite_ratios) > 0 else 0.0

    # 戦略劣化フラグ: 小サンプルでは判定しない
    if cfg is not None:
        min_sample = int(cfg.get("MIN_TRADE_FOR_DEGRADED", _MIN_SAMPLE_DEGRADED))
        if n >= min_sample:
            result["degraded"] = detect_strategy_degradation(df, cfg)

    return result
