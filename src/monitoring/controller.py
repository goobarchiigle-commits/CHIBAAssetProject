"""
src/monitoring/controller.py
段階制御: OK → ADJUST → DEGRADED → HALT（ラッチ）
純粋関数ベース。state dict を入力として受け取り、更新済み state を返す。
全タイムスタンプは int（秒）に正規化する。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

# ── 閾値定数 ────────────────────────────────────────────────────────────────────
RISK_SOFT        = 1.5
RISK_SOFT_RECOV  = 1.3
RISK_HARD        = 3.0
MAX_DD_HARD      = -0.15
SLIP_SOFT        = 0.002
COOLDOWN_SEC     = 3_600
DRIFT            = 0.30
MIN_ACTIVITY     = 5
MIN_MULTIPLIER   = 0.5

_STATE_FILE = Path(__file__).resolve().parents[2] / "runtime" / "monitoring_state.json"

# 全必須フィールド（これが揃わない state は破損扱い）
_STATE_REQUIRED: frozenset[str] = frozenset({
    "mode", "halted", "halt_reason", "halt_ts",
    "global_multiplier", "risk_scale", "impact_coeff",
    "last_adjust_ts", "base_risk_scale", "base_impact_coeff",
    "last_metrics", "n_trades",
})


# ── 初期状態 ─────────────────────────────────────────────────────────────────────
def init_controller_state(ts: int = 0) -> dict[str, Any]:
    """
    完全仕様の初期コントローラー状態を返す。
    全フィールドが明示的に存在し、タイムスタンプは int。
    暗黙のデフォルト不使用。
    """
    return {
        "mode":               "OK",
        "halted":             False,
        "halt_reason":        None,
        "halt_ts":            int(ts),
        "global_multiplier":  1.0,
        "risk_scale":         1.0,
        "impact_coeff":       1.0,
        "last_adjust_ts":     0,
        "base_risk_scale":    1.0,
        "base_impact_coeff":  1.0,
        "last_metrics":       {},
        "n_trades":           0,
    }


# 後方互換エイリアス
def make_default_state() -> dict[str, Any]:
    return init_controller_state(ts=0)


# ── 永続化 ──────────────────────────────────────────────────────────────────────
def load_state(path: Path = _STATE_FILE) -> dict[str, Any]:
    if not path.exists():
        return init_controller_state()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return init_controller_state()


def save_state(state: dict[str, Any], path: Path = _STATE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── フィールド検証 ───────────────────────────────────────────────────────────────
def validate_state_fields(state: dict[str, Any]) -> dict[str, Any] | None:
    """
    state dict の完全性チェック（純粋関数）。
    必須フィールド欠損 / float NaN / float inf → None を返す。
    正常時は shallow copy を返す（入力を変更しない）。
    """
    if state is None:
        return None
    missing = _STATE_REQUIRED - set(state.keys())
    if missing:
        return None
    cleaned: dict = {}
    for k, v in state.items():
        if isinstance(v, float) and not math.isfinite(v):
            return None
        cleaned[k] = v
    return cleaned


# ── 内部ユーティリティ ──────────────────────────────────────────────────────────
def _clamp_drift(val: float, base: float, drift: float = DRIFT) -> float:
    lo = base * (1.0 - drift)
    hi = base * (1.0 + drift)
    return max(lo, min(hi, val))


def _mean_revert(val: float, base: float, alpha: float = 0.05) -> float:
    return val + alpha * (base - val)


# ── メイン制御関数 ──────────────────────────────────────────────────────────────
def apply_controller(
    state: dict[str, Any],
    metrics: dict[str, Any],
    ts: int | float | None = None,
    cfg: dict | None = None,
) -> "dict[str, Any] | tuple[dict[str, Any], str]":
    """
    メトリクスを元に state を更新して返す（副作用なし・新 dict を返す）。
    全タイムスタンプは int に正規化する。

    Args:
        state:   現在の monitoring state（変更しない）
        metrics: compute_metrics() の出力
        ts:      現在タイムスタンプ（int 秒）。None なら int(time.time())
        cfg:     設定 dict（指定時は config-driven パス使用、(state, mode) を返す）

    Returns:
        cfg=None → 更新済み state dict
        cfg 指定 → (更新済み state dict, mode 文字列) のタプル
    """
    if cfg is not None:
        now_int = int(ts) if ts is not None else int(time.time())
        return _apply_controller_cfg(state, metrics, now_int, cfg)

    now = int(ts) if ts is not None else int(time.time())

    # state フィールド検証 → 破損時は強制 HALT
    validated = validate_state_fields(state)
    if validated is None:
        return {
            **init_controller_state(ts=now),
            "halted":            True,
            "mode":              "HALT_STATE_CORRUPT",
            "halt_reason":       "state_corruption_detected",
            "halt_ts":           now,
            "global_multiplier": 0.0,
        }

    s = dict(validated)

    # HALT ラッチ: 一度 halted なら全処理停止
    if s.get("halted"):
        s["mode"]              = "HALT"
        s["global_multiplier"] = 0.0
        return s

    # HALT_CORRUPT: ログ欠損・NaN
    if metrics.get("status") == "HALT_CORRUPT":
        s["halted"]            = True
        s["mode"]              = "HALT_CORRUPT"
        s["halt_reason"]       = f"HALT_CORRUPT: {metrics.get('corrupt_reason')}"
        s["halt_ts"]           = now
        s["global_multiplier"] = 0.0
        s["last_metrics"]      = dict(metrics)
        return s

    max_dd     = float(metrics.get("max_dd",     0.0))
    risk_ratio = float(metrics.get("risk_ratio", 0.0))
    slip_diff  = float(metrics.get("slip_diff",  0.0))
    n_trades   = int(  metrics.get("n_trades",   0  ))

    # HARD 条件 → HALT（不可逆ラッチ）
    hard_reason: str | None = None
    if max_dd < MAX_DD_HARD:
        hard_reason = f"max_dd={max_dd:.4f} < MAX_DD_HARD={MAX_DD_HARD}"
    elif risk_ratio > RISK_HARD:
        hard_reason = f"risk_ratio={risk_ratio:.3f} > RISK_HARD={RISK_HARD}"

    if hard_reason:
        s["halted"]            = True
        s["mode"]              = "HALT"
        s["halt_reason"]       = hard_reason
        s["halt_ts"]           = now
        s["global_multiplier"] = 0.0
        s["last_metrics"]      = dict(metrics)
        return s

    # クールダウン判定（0 は「未調整」sentinel → 非クールダウン）
    last_adj    = int(s.get("last_adjust_ts", 0))
    in_cooldown = last_adj > 0 and (now - last_adj) < COOLDOWN_SEC

    current_mode = s.get("mode", "OK")

    if not in_cooldown:
        base_rs = float(s.get("base_risk_scale",   1.0))
        base_ic = float(s.get("base_impact_coeff", 1.0))
        rs      = float(s.get("risk_scale",         1.0))
        ic      = float(s.get("impact_coeff",       1.0))
        gm      = float(s.get("global_multiplier",  1.0))

        if risk_ratio > RISK_SOFT and current_mode != "DEGRADED":
            rs = _clamp_drift(rs * 0.7, base_rs)
            gm = _clamp_drift(gm * 0.7, base_rs)
            s["risk_scale"]        = rs
            s["global_multiplier"] = gm
            s["mode"]              = "DEGRADED"
            s["last_adjust_ts"]    = now

        elif slip_diff > SLIP_SOFT and current_mode not in ("DEGRADED", "HALT", "HALT_CORRUPT"):
            ic = _clamp_drift(ic * 1.1, base_ic)
            s["impact_coeff"]   = ic
            s["mode"]           = "ADJUST"
            s["last_adjust_ts"] = now

        elif risk_ratio < RISK_SOFT_RECOV and current_mode in ("DEGRADED", "ADJUST"):
            s["mode"] = "OK"

    else:
        if current_mode not in ("HALT", "HALT_CORRUPT"):
            s["mode"] = "COOLDOWN"

    # 弱い平均回帰（ドリフト抑制）
    base_rs = float(s.get("base_risk_scale",   1.0))
    base_ic = float(s.get("base_impact_coeff", 1.0))
    s["risk_scale"]   = _clamp_drift(_mean_revert(float(s["risk_scale"]),   base_rs), base_rs)
    s["impact_coeff"] = _clamp_drift(_mean_revert(float(s["impact_coeff"]), base_ic), base_ic)

    # 最小稼働保証
    if n_trades < MIN_ACTIVITY and not s.get("halted"):
        s["global_multiplier"] = max(float(s.get("global_multiplier", 1.0)), MIN_MULTIPLIER)

    s["last_metrics"] = dict(metrics)
    s["n_trades"]     = n_trades

    return s


def get_system_status(state: dict[str, Any]) -> str:
    if state.get("halted"):
        mode = state.get("mode", "HALT")
        return mode if mode in ("HALT", "HALT_CORRUPT", "HALT_STATE_CORRUPT") else "HALT"
    return state.get("mode", "OK")


# ── Config-driven helpers ────────────────────────────────────────────────────────

def sanitize_state(state: dict[str, Any]) -> dict[str, Any] | None:
    """
    state dict の完全性チェック（純粋関数）。
    NaN / inf を含む float 値が存在すれば None を返す。
    """
    if state is None:
        return None
    cleaned: dict = {}
    for k, v in state.items():
        if isinstance(v, float) and not math.isfinite(v):
            return None
        cleaned[k] = v
    return cleaned


def can_recover(state: dict, metrics: dict, ts: int, cfg: dict) -> bool:
    """HALT 状態からの回復条件を判定（純粋関数）。"""
    if not state.get("halted"):
        return False
    cond_metrics = (
        metrics["risk_ratio"] < cfg["RECOVERY_RISK"]
        and metrics["max_dd"] > -cfg["RECOVERY_DD"]
        and abs(metrics["slip_diff"]) < cfg["RECOVERY_SLIP"]
    )
    halt_ts = int(state.get("halt_ts", ts))
    cond_time = (ts - halt_ts) > int(cfg.get("MIN_HALTED_TIME", 0))
    return cond_metrics and cond_time


def _apply_controller_cfg(
    state: dict[str, Any],
    metrics: dict[str, Any],
    ts: int,
    cfg: dict,
) -> tuple[dict[str, Any], str]:
    """
    Config-driven controller（純粋関数）。
    state を shallow copy → 変換済み (new_state, mode) を返す。
    副作用なし・入力を変更しない。
    """
    from .audit import sanitize_metrics

    # state 完全性チェック
    clean_state = sanitize_state(state)
    if clean_state is None:
        return {
            **init_controller_state(ts=ts),
            "halted":            True,
            "global_multiplier": 0.0,
            "halt_ts":           ts,
            "halt_reason":       "HALT_STATE_CORRUPT",
        }, "HALT_STATE_CORRUPT"

    s = dict(clean_state)  # 入力を変更しない

    # metrics 完全性チェック
    clean_metrics = sanitize_metrics(metrics)
    if clean_metrics is None:
        s["halted"]            = True
        s["global_multiplier"] = 0.0
        s["halt_reason"]       = "HALT_CORRUPT"
        s["halt_ts"]           = ts
        return s, "HALT_CORRUPT"

    prev_mode = s.get("mode")

    # HALT ラッチ
    if s.get("halted"):
        if can_recover(s, clean_metrics, ts, cfg):
            s["halted"]            = False
            s["global_multiplier"] = min(float(cfg.get("RECOVERY_MULTIPLIER", 0.5)), 1.0)
            s["mode"]              = "RECOVER"
            s["last_adjust_ts"]    = ts
            return s, "RECOVER"
        return s, "HALT"

    # リカバリー猶予判定
    in_recovery = s.get("mode") == "RECOVER"
    in_grace = (
        in_recovery
        and (ts - int(s.get("last_adjust_ts", ts)))
        < int(cfg.get("RECOVERY_GRACE_PERIOD", 0))
    )

    effective_risk_hard = float(cfg.get("RISK_HARD", float("inf")))
    if in_grace:
        effective_risk_hard *= float(cfg.get("RECOVERY_HARD_RELAX", 1.0))

    # HARD 条件 → HALT
    if not in_grace and (
        clean_metrics["max_dd"] < -float(cfg.get("MAX_DD_HARD", float("inf")))
        or clean_metrics["risk_ratio"] > effective_risk_hard
    ):
        s["halted"]            = True
        s["halt_ts"]           = ts
        s["global_multiplier"] = 0.0
        s["halt_reason"]       = "HARD_LIMIT"
        return s, "HALT"

    # 戦略劣化
    if clean_metrics.get("degraded", False):
        s["mode"] = "DEGRADED_STRATEGY"
        gm = float(s.get("global_multiplier", 1.0))
        s["global_multiplier"] = gm * (0.5 if prev_mode != "DEGRADED_STRATEGY" else 0.9)
        s["global_multiplier"] = max(
            float(cfg.get("MIN_MULTIPLIER", 0.0)),
            min(1.0, s["global_multiplier"]),
        )
        return s, "DEGRADED_STRATEGY"

    # SOFT 調整
    if clean_metrics["risk_ratio"] > float(cfg.get("RISK_SOFT", float("inf"))):
        s["mode"]              = "DEGRADED"
        s["global_multiplier"] = float(s.get("global_multiplier", 1.0)) * 0.7
    elif clean_metrics["slip_diff"] > float(cfg.get("SLIP_SOFT", float("inf"))):
        s["mode"]        = "ADJUST"
        s["impact_coeff"] = float(s.get("impact_coeff", 1.0)) * 1.1
    else:
        s["mode"] = "OK"

    # 不変条件: halted なら multiplier=0、上下限クランプ
    if s.get("halted"):
        s["global_multiplier"] = 0.0
    s["global_multiplier"] = max(0.0, min(1.0, float(s.get("global_multiplier", 1.0))))

    return s, s["mode"]
