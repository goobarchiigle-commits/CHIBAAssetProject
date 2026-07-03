"""
src/execution/validate_pipeline.py
フルリプレイバリデーター。純粋関数ベース。副作用なし。

validate_pipeline(df, cfg) → JSON-serializable report:
{
    "deterministic":        bool,
    "nan_detected":         bool,
    "invalid_transitions":  int,
    "max_multiplier":       float,
    "violations":           list[str],
}

実行方法:
    python -m src.execution.validate_pipeline
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from src.monitoring.audit import compute_metrics
from src.monitoring.controller import apply_controller, init_controller_state
from src.execution.bucket import BucketAccumulator
from src.execution.sizing import compute_opportunity, compute_ema_fill, compute_final_multiplier


def _safe(v: Any, fallback: float = 0.0) -> float:
    """NaN / inf / None → fallback。"""
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


# ── 有効な状態遷移表 ────────────────────────────────────────────────────────────
# HALT からは HALT / RECOVER のみ許可（直接 OK / ADJUST 等は禁止）
_VALID_TRANSITIONS: dict[str, frozenset[str]] = {
    "OK":                 frozenset({"OK", "ADJUST", "DEGRADED", "DEGRADED_STRATEGY",
                                     "HALT", "COOLDOWN", "HALT_CORRUPT"}),
    "ADJUST":             frozenset({"OK", "ADJUST", "DEGRADED",
                                     "HALT", "COOLDOWN", "HALT_CORRUPT"}),
    "DEGRADED":           frozenset({"OK", "ADJUST", "DEGRADED",
                                     "HALT", "COOLDOWN", "HALT_CORRUPT"}),
    "DEGRADED_STRATEGY":  frozenset({"OK", "DEGRADED_STRATEGY", "HALT", "HALT_CORRUPT"}),
    "COOLDOWN":           frozenset({"OK", "ADJUST", "DEGRADED",
                                     "HALT", "COOLDOWN", "HALT_CORRUPT"}),
    "HALT":               frozenset({"HALT", "RECOVER",
                                     "HALT_CORRUPT", "HALT_STATE_CORRUPT"}),
    "RECOVER":            frozenset({"OK", "HALT", "RECOVER"}),
    "HALT_CORRUPT":       frozenset({"HALT_CORRUPT"}),
    "HALT_STATE_CORRUPT": frozenset({"HALT_STATE_CORRUPT"}),
}

_REQUIRED_COLS: frozenset[str] = frozenset(
    {"pnl", "expected_risk", "slippage_real", "slippage_est"}
)
_MAX_VIOLATIONS: int = 100


# ── JSON-safe 変換 ──────────────────────────────────────────────────────────────

def _to_json_safe(obj: Any) -> Any:
    """numpy / pandas 型を JSON ネイティブ型に変換する（再帰的）。"""
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else 0.0
    if obj is None:
        return None
    if isinstance(obj, (np.ndarray,)):
        return _to_json_safe(obj.tolist())
    return str(obj)


# ── 有限値違反検査 ──────────────────────────────────────────────────────────────

def _finite_violations(obj: Any, path: str = "", limit: int = 10) -> list[str]:
    """オブジェクト内の NaN / inf を再帰的に検出する（パス文字列付き）。"""
    found: list[str] = []

    def _walk(o: Any, p: str) -> None:
        if len(found) >= limit:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                _walk(v, f"{p}.{k}" if p else str(k))
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                _walk(v, f"{p}[{i}]")
        elif isinstance(o, (float, np.floating)):
            if not math.isfinite(float(o)):
                found.append(f"{p}={o}")
        elif isinstance(o, (int, np.integer, bool, str, type(None))):
            pass
        else:
            pass  # 未対応型は無視

    _walk(obj, path)
    return found


# ── シングルリプレイパス ─────────────────────────────────────────────────────────

def _replay(
    df: pd.DataFrame,
    cfg: dict,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    df を先頭から 1 行ずつリプレイし (final_state, per_row_results) を返す。

    タイムスタンプ: 行インデックス整数（0, 1, 2, ...）→ 完全決定論的。
    ランダム性: ゼロ。
    先読み: ゼロ（累積スライス df.iloc[:i+1] のみ参照）。

    per_row_results[i] のキー:
        ts            : int
        metrics       : dict
        state         : dict
        bucket_stats  : dict
        opportunity   : float
        fill          : float
        final_multiplier : float
    """
    state        = init_controller_state(ts=0)
    acc          = BucketAccumulator()
    opp_hist:    list[float] = []   # opportunity 中央値計算用（PNLではない）
    fill_hist:   list[float] = []   # fill ratio 履歴（0-1）
    results:     list[dict[str, Any]] = []

    max_slip     = _safe(cfg.get("max_slippage",    0.01), fallback=0.01)
    cost_mult    = _safe(cfg.get("cost_mult",        1.0),  fallback=1.0)
    base_alpha   = _safe(cfg.get("base_alpha",       0.1),  fallback=0.1)
    default_fill = _safe(cfg.get("default_fill_rate", 1.0), fallback=1.0)

    for i in range(len(df)):
        ts = i  # 決定論的な整数タイムスタンプ

        # 先読みなし: 行 0..i の累積スライスのみ使用
        slice_df = df.iloc[: i + 1]

        # [2] メトリクス計算
        metrics = compute_metrics(slice_df, cfg)

        # [1] コントローラー更新（副作用なし）
        new_state = apply_controller(state, metrics, ts=ts)

        # [5] バケット統計 O(N) 更新（PNL を蓄積）
        mode_key = str(new_state.get("mode", "OK"))
        pnl_val  = float(df.iloc[i]["pnl"]) if "pnl" in df.columns else 0.0
        acc.update(mode_key, pnl_val, track_values=True)
        bucket_stats = {mode_key: acc.stats(mode_key)}

        # [7] median_opportunity: opportunity 履歴の中央値（PNL の中央値ではない）
        if opp_hist:
            arr = np.array([_safe(v) for v in opp_hist], dtype=float)
            finite = arr[np.isfinite(arr)]
            median_opp: float | None = float(np.median(finite)) if len(finite) > 0 else None
        else:
            median_opp = None

        # [3] opportunity（slippage_est から計算）
        row = df.iloc[i]
        row_slip = _safe(
            row.get("slippage_est", 0.001) if hasattr(row, "get") else row["slippage_est"],
            fallback=_safe(cfg.get("default_slippage_est", 0.001), fallback=0.001),
        )
        impact_coef = float(new_state.get("impact_coeff", 1.0))
        opp = compute_opportunity(
            slippage_est=row_slip,
            max_slippage=max_slip,
            impact_coef=impact_coef,
            cfg=cfg,
            confidence=1.0,
            median_opp=median_opp,
        )
        opp_hist.append(opp)  # opportunity 履歴に追加（PNL 履歴と分離）

        # [4] 適応 fill（fill ratio ∈ [0,1]、trades_log に fill_rate 列があれば使用）
        if "fill_rate" in df.columns:
            raw_fill = _safe(
                row.get("fill_rate") if hasattr(row, "get") else row["fill_rate"],
                fallback=default_fill,
            )
        else:
            raw_fill = default_fill
        fill = compute_ema_fill(fill_hist, raw_fill, base_alpha)
        fill_hist.append(raw_fill)

        # [8] final multiplier
        base_mult = float(new_state.get("global_multiplier", 1.0))
        final_mult = compute_final_multiplier(
            base=base_mult,
            cost_mult=cost_mult,
            opportunity=opp,
            fill=fill,
            cfg=cfg,
        )

        results.append({
            "ts":               ts,
            "metrics":          _to_json_safe(metrics),
            "state":            _to_json_safe(new_state),
            "bucket_stats":     _to_json_safe(bucket_stats),
            "opportunity":      float(opp),
            "fill":             float(fill),
            "final_multiplier": float(final_mult),
        })

        state = new_state

    return state, results


# ── [9] 重みバリデーター ────────────────────────────────────────────────────────

def _validate_weights(weight_df: pd.DataFrame, cfg: dict) -> dict[str, Any]:
    """
    compute_dynamic_weights() の出力を検証する（純粋関数）。

    Computes:
        avg_weight       : 重みの平均
        weight_std       : 重みの標準偏差
        weight_turnover  : per-bucket |diff(weight)| の平均

    Asserts:
        - NaN / inf なし
        - max(weight) <= cfg["MAX_WEIGHT"]
        - weight_turnover < cfg["TURNOVER_LIMIT"]（設定がある場合）

    Returns:
        avg_weight, weight_std, weight_turnover, weight_valid, weight_violations
    """
    w_violations: list[str] = []

    if "weight" not in weight_df.columns:
        return {
            "avg_weight":        0.0,
            "weight_std":        0.0,
            "weight_turnover":   0.0,
            "weight_valid":      False,
            "weight_violations": ["missing 'weight' column"],
        }

    w_arr = weight_df["weight"].values.astype(float)

    # NaN / inf チェック
    nan_mask = ~np.isfinite(w_arr)
    if nan_mask.any():
        w_violations.append(f"weight_nan_inf: {int(nan_mask.sum())} values")

    finite_w = w_arr[np.isfinite(w_arr)]
    avg_weight = float(np.mean(finite_w))   if len(finite_w) > 0 else 0.0
    weight_std = float(np.std(finite_w))    if len(finite_w) > 0 else 0.0
    max_w      = float(np.max(finite_w))    if len(finite_w) > 0 else 0.0

    # max weight チェック
    cfg_max_w = _safe(cfg.get("MAX_WEIGHT", 5.0), fallback=5.0)
    if max_w > cfg_max_w + 1e-9:
        w_violations.append(
            f"weight_exceeds_max: {max_w:.4f} > {cfg_max_w:.4f}"
        )

    # weight_turnover = mean(|diff(weight)|) per bucket
    has_keys = (
        "bucket_key" in weight_df.columns
        and "timestamp" in weight_df.columns
    )
    if has_keys and len(weight_df) > 1:
        w_sorted = weight_df.sort_values(["bucket_key", "timestamp"])
        diffs    = w_sorted.groupby("bucket_key")["weight"].diff().abs()
        raw_turn = float(diffs.mean()) if diffs.notna().any() else 0.0
    else:
        raw_turn = float(pd.Series(w_arr).diff().abs().mean()) if len(w_arr) > 1 else 0.0
    turnover = raw_turn if math.isfinite(raw_turn) else 0.0

    cfg_limit = _safe(cfg.get("TURNOVER_LIMIT", float("inf")), fallback=float("inf"))
    if math.isfinite(cfg_limit) and turnover > cfg_limit:
        w_violations.append(
            f"weight_turnover_exceeded: {turnover:.6f} > {cfg_limit:.6f}"
        )

    return {
        "avg_weight":        avg_weight,
        "weight_std":        weight_std,
        "weight_turnover":   turnover,
        "weight_valid":      len(w_violations) == 0,
        "weight_violations": w_violations,
    }


# ── メインバリデーター ───────────────────────────────────────────────────────────

def validate_pipeline(
    df: pd.DataFrame,
    cfg: dict,
    weight_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    パイプライン全体をリプレイして健全性を検証する。

    2 回同一実行し結果を比較することで決定論を確認する。
    weight_df が指定された場合は重み検証 [9] も実施する。

    Args:
        df        : 取引ログ（必須列: pnl, expected_risk, slippage_real, slippage_est）
        cfg       : 設定 dict
        weight_df : compute_dynamic_weights() の出力（省略可）

    Returns (JSON-serializable):
        {
            "deterministic":       bool,
            "nan_detected":        bool,
            "invalid_transitions": int,
            "max_multiplier":      float,
            "violations":          list[str],
            # weight_df 指定時に追加:
            "avg_weight":          float,
            "weight_std":          float,
            "weight_turnover":     float,
            "weight_valid":        bool,
            "weight_violations":   list[str],
        }
    """
    violations: list[str] = []

    # 欠損列チェック
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        return {
            "deterministic":       False,
            "nan_detected":        False,
            "invalid_transitions": 0,
            "max_multiplier":      0.0,
            "violations":          [f"missing_columns: {sorted(missing)}"],
        }

    if df.empty:
        return {
            "deterministic":       True,
            "nan_detected":        False,
            "invalid_transitions": 0,
            "max_multiplier":      0.0,
            "violations":          [],
        }

    # df のコピーを作成（外部を変更しない）
    work_df = df.reset_index(drop=True)

    # ── 2 回リプレイ（決定論チェック用）──────────────────────────────────────────
    _, results1 = _replay(work_df, dict(cfg))
    _, results2 = _replay(work_df, dict(cfg))

    # JSON シリアライズして文字列比較（型差異に強い）
    key1 = json.dumps(results1, sort_keys=True, ensure_ascii=False)
    key2 = json.dumps(results2, sort_keys=True, ensure_ascii=False)
    deterministic = key1 == key2
    if not deterministic:
        violations.append("non_deterministic_output")

    # ── NaN / inf 検査 + multiplier 範囲検査 ────────────────────────────────────
    nan_detected = False
    max_multiplier = 0.0

    for i, row in enumerate(results1):
        if len(violations) >= _MAX_VIOLATIONS:
            break

        # NaN / inf チェック
        nf = _finite_violations(row, path=f"row_{i}", limit=5)
        if nf:
            nan_detected = True
            violations.extend(nf)

        # final_multiplier 範囲チェック
        mult = row.get("final_multiplier", 0.0)
        if isinstance(mult, (int, float)) and math.isfinite(mult):
            max_multiplier = max(max_multiplier, mult)
            if mult < 0.0:
                violations.append(f"row_{i}: negative_multiplier={mult:.6g}")
            elif mult > 1e6:
                violations.append(f"row_{i}: explosive_multiplier={mult:.6g}")
        else:
            violations.append(f"row_{i}: non_finite_multiplier={mult}")

        # メトリクス出力の有限性
        metrics = row.get("metrics", {})
        for mkey in ("max_dd", "slip_diff", "risk_ratio"):
            mv = metrics.get(mkey)
            if isinstance(mv, (int, float)) and not math.isfinite(mv):
                nan_detected = True
                violations.append(f"row_{i}: metrics.{mkey}={mv}")

    # ── 状態遷移チェック ──────────────────────────────────────────────────────────
    invalid_transitions = 0
    prev_mode = "OK"

    for i, row in enumerate(results1):
        new_mode = row.get("state", {}).get("mode", "OK")
        allowed  = _VALID_TRANSITIONS.get(prev_mode, frozenset())
        if new_mode not in allowed:
            invalid_transitions += 1
            if len(violations) < _MAX_VIOLATIONS:
                violations.append(
                    f"row_{i}: invalid_transition {prev_mode!r}→{new_mode!r}"
                )
        prev_mode = new_mode

    report: dict[str, Any] = {
        "deterministic":       deterministic,
        "nan_detected":        nan_detected,
        "invalid_transitions": invalid_transitions,
        "max_multiplier":      float(max_multiplier) if math.isfinite(max_multiplier) else 0.0,
        "violations":          violations,
    }

    # ── [9] 重み検証（weight_df 指定時のみ）──────────────────────────────────────
    if weight_df is not None:
        w_report = _validate_weights(weight_df, cfg)
        report["avg_weight"]        = w_report["avg_weight"]
        report["weight_std"]        = w_report["weight_std"]
        report["weight_turnover"]   = w_report["weight_turnover"]
        report["weight_valid"]      = w_report["weight_valid"]
        report["weight_violations"] = w_report["weight_violations"]
        # 重み違反を共通 violations に追記
        report["violations"] = violations + w_report["weight_violations"]

    return report


# ── 最小使用例（外部依存なし）────────────────────────────────────────────────────

def _example() -> None:
    """
    外部依存なしの最小使用例。
    python -m src.execution.validate_pipeline で実行可能。
    """
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    # サンプル取引ログ（NaN / inf なし、正常系）
    data = {
        "pnl":           [0.01, -0.005, 0.02, -0.008, 0.015,
                          0.012, -0.003, 0.018, -0.006, 0.009],
        "expected_risk": [0.02,  0.02,  0.02,  0.02,  0.02,
                          0.02,   0.02,  0.02,  0.02,  0.02],
        "slippage_real": [0.0012, 0.0008, 0.0015, 0.0010, 0.0011,
                          0.0013, 0.0009, 0.0014, 0.0010, 0.0012],
        "slippage_est":  [0.001,  0.001,  0.001,  0.001,  0.001,
                          0.001,  0.001,  0.001,  0.001,  0.001],
    }
    df = pd.DataFrame(data)

    cfg = {
        "PNL_FLOOR":           0.005,
        "MIN_WIN_RATE":        0.4,
        "MIN_TRADE_FOR_DEGRADED": 5,
        "max_slippage":        0.01,
        "cost_mult":           1.0,
        "base_alpha":          0.1,
        "max_opportunity":     5.0,
        "default_slippage_est": 0.001,
        "multiplier_min":      0.0,
        "multiplier_max":      1e6,
    }

    print("=== validate_pipeline 使用例 ===")
    report = validate_pipeline(df, cfg)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # エラー注入テスト（NaN pnl）
    print("\n=== NaN 注入テスト ===")
    bad_df = df.copy()
    bad_df.loc[3, "pnl"] = float("nan")
    bad_report = validate_pipeline(bad_df, cfg)
    print(json.dumps(bad_report, ensure_ascii=False, indent=2))

    # 空 DataFrame テスト
    print("\n=== 空 DataFrame テスト ===")
    empty_report = validate_pipeline(pd.DataFrame(columns=list(data.keys())), cfg)
    print(json.dumps(empty_report, ensure_ascii=False, indent=2))

    # [9] 重み検証テスト
    print("\n=== [9] weight_df 検証テスト ===")
    from src.execution.weight_update import compute_dynamic_weights

    w_data = {
        "timestamp":  [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "bucket_key": ["A", "B"] * 5,
        "pnl":        [0.01, -0.005, 0.02, 0.003, -0.008,
                       0.015, 0.012, -0.003, 0.018, -0.006],
    }
    w_cfg = {
        **cfg,
        "EMA_ALPHA":           0.1,
        "CONF_SCALE":          5.0,
        "MIN_CONF":            0.0,
        "MAX_WEIGHT":          5.0,
        "MIN_COUNT":           2,
        "COLD_START_WEIGHT":   1.0,
        "VAR_WINDOW":          3,
        "VAR_SPIKE_K":         3.0,
        "WEIGHT_SMOOTH_ALPHA": 0.3,
        "TURNOVER_LIMIT":      2.0,
    }
    weight_df = compute_dynamic_weights(pd.DataFrame(w_data), w_cfg)
    print("weight_df:")
    print(weight_df.to_string(index=False))

    # validate_pipeline + 重み検証
    w_report = validate_pipeline(df, w_cfg, weight_df=weight_df)
    print("\nvalidate_pipeline (with weight_df):")
    print(json.dumps(w_report, ensure_ascii=False, indent=2))

    # MAX_WEIGHT 違反テスト
    print("\n=== MAX_WEIGHT 違反テスト ===")
    bad_weight_df = weight_df.copy()
    bad_weight_df.loc[0, "weight"] = 999.0  # 上限超え
    bad_w_report = validate_pipeline(df, w_cfg, weight_df=bad_weight_df)
    print(json.dumps({
        "weight_valid":      bad_w_report.get("weight_valid"),
        "weight_violations": bad_w_report.get("weight_violations"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _example()
