"""
src/execution/weight_update.py
決定論的・先読みなし動的バケット重み計算。純粋関数ベース。副作用なし。

契約:
  - NO LOOKAHEAD : 時刻 t の全特徴量は t-1 以前のデータのみ使用（shift(1) 実施済み）
  - PURE FUNCTION: 入力を変更しない。隠れ状態なし。
  - DETERMINISTIC: 同一入力 → 同一出力。ランダム性ゼロ。
  - EMA METHOD   : index-based cumulative array（approach B）
                   mutable dict/state 更新は不使用

pipeline:
  [1] per-bucket cumulative EMA (shift 1)
  [2] edge signal (finite-safe)
  [3] confidence
  [4] raw weight = edge × confidence
  [5] cross-sectional normalization per timestamp
  [6] stability constraints + variance spike
  [6.5] temporal smoothness per bucket
  [7] return DataFrame
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# ── 定数 ────────────────────────────────────────────────────────────────────────
_EPS: float = 1e-12
_REQUIRED_COLS: frozenset[str] = frozenset({"timestamp", "bucket_key", "pnl"})

_CFG_DEFAULTS: dict[str, Any] = {
    "EMA_ALPHA":           0.05,  # EMA alpha 下限
    "CONF_SCALE":          10.0,  # confidence スケーリング係数
    "MIN_CONF":            0.0,   # confidence 下限
    "MAX_WEIGHT":          5.0,   # 重み上限
    "MIN_COUNT":           5,     # cold start 閾値
    "COLD_START_WEIGHT":   1.0,   # count < MIN_COUNT 時の重み
    "VAR_WINDOW":          10,    # 分散スパイク検出ウィンドウ
    "VAR_SPIKE_K":         3.0,   # 分散スパイク倍率閾値
    "WEIGHT_SMOOTH_ALPHA": 0.3,   # 時系列平滑化 beta
}


# ── ユーティリティ ───────────────────────────────────────────────────────────────

def _safe(v: Any, fallback: float = 0.0) -> float:
    """NaN / inf / None → fallback。"""
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


def _safe_arr(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """配列内の NaN / inf を fill 値で置換する（新配列を返す）。"""
    return np.where(np.isfinite(arr), arr, fill).astype(float)


def _shift1(arr: np.ndarray) -> np.ndarray:
    """
    配列を 1 ステップ右シフトし、位置 0 を 0.0 で埋める。
    先読み回避用（result[t] = arr[t-1]）。
    """
    out = np.empty_like(arr, dtype=float)
    out[0] = 0.0
    if len(arr) > 1:
        out[1:] = arr[:-1]
    return out


# ── [1] per-bucket EMA（approach B: index-based cumulative array）──────────────

def _ema_array(x: np.ndarray, ema_alpha: float) -> np.ndarray:
    """
    適応 alpha EMA の計算（index-based cumulative array、approach B）。

    alpha_t = max(ema_alpha, 1 / (n + 20))  where n = row index within group

    - result[t] は x[0..t] のみを使用（shift(1) 前）
    - mutable dict/state 不使用: numpy 配列インデックスのみ
    - 純粋関数: 入力 x を変更しない

    Args:
        x         : 1D float array（NaN/inf 除去済み）
        ema_alpha : EMA alpha 下限

    Returns:
        EMA 配列（同じ shape）
    """
    n = len(x)
    if n == 0:
        return np.empty(0, dtype=float)

    alphas = np.maximum(ema_alpha, 1.0 / (np.arange(n, dtype=float) + 20.0))
    result = np.empty(n, dtype=float)
    result[0] = alphas[0] * x[0]
    for i in range(1, n):
        result[i] = alphas[i] * x[i] + (1.0 - alphas[i]) * result[i - 1]
    return result


def _compute_group_emas(group: pd.DataFrame, ema_alpha: float) -> pd.DataFrame:
    """
    1 バケット分の EMA 統計を計算し shift(1) を適用する。

    GROUP は timestamp 昇順ソート済みであること（compute_dynamic_weights が保証）。
    shift(1) により、出力位置 t は x[0..t-1] のデータのみ反映する。
    位置 0 は全て 0.0（事前データなし）。

    Returns:
        DataFrame with columns: count, ema_pnl, ema_abs_pnl, ema_hit, ema_downside
        index は group と同一（positional alignment を保証）
    """
    idx = group.index
    n   = len(idx)
    x   = _safe_arr(group["pnl"].values.astype(float))  # NaN/inf → 0.0

    # 1-indexed count: shift(1) 後に count[t] = "位置 t で利用可能な過去観測数 t"
    counts_raw = np.arange(1, n + 1, dtype=float)

    ema_pnl  = _ema_array(x,                           ema_alpha)
    ema_abs  = _ema_array(np.abs(x),                   ema_alpha)
    ema_hit  = _ema_array((x > 0.0).astype(float),     ema_alpha)
    ema_down = _ema_array(np.maximum(-x, 0.0),          ema_alpha)

    return pd.DataFrame({
        "count":        _shift1(counts_raw),
        "ema_pnl":      _shift1(ema_pnl),
        "ema_abs_pnl":  _shift1(ema_abs),
        "ema_hit":      _shift1(ema_hit),
        "ema_downside": _shift1(ema_down),
    }, index=idx)


# ── [2] エッジシグナル ─────────────────────────────────────────────────────────

def _compute_edge(
    ema_pnl:  np.ndarray,
    ema_abs:  np.ndarray,
    ema_hit:  np.ndarray,
    ema_down: np.ndarray,
) -> np.ndarray:
    """
    edge = (ema_pnl / (ema_abs + ε))
         × sqrt(ema_hit)
         × (1 - clip(ema_downside / (ema_abs + ε), 0, 1))

    |ema_pnl| ≤ ema_abs（三角不等式）より directional ∈ [-1, 1]。
    全出力が有限値であることを保証する。
    """
    p   = _safe_arr(ema_pnl)
    a   = _safe_arr(ema_abs)
    h   = _safe_arr(ema_hit)
    d   = _safe_arr(ema_down)

    directional    = np.clip(p / (a + _EPS), -1.0, 1.0)
    sqrt_hit       = np.sqrt(np.clip(h, 0.0, 1.0))
    downside_ratio = np.clip(d / (a + _EPS), 0.0, 1.0)
    raw            = directional * sqrt_hit * (1.0 - downside_ratio)
    return np.where(np.isfinite(raw), raw, 0.0)


# ── [3] Confidence ─────────────────────────────────────────────────────────────

def _compute_confidence(
    count: np.ndarray, conf_scale: float, min_conf: float
) -> np.ndarray:
    """
    conf = log1p(count) / CONF_SCALE
    conf = clip(conf, MIN_CONF, 1.0)
    """
    raw = np.log1p(np.maximum(_safe_arr(count), 0.0)) / (conf_scale + _EPS)
    return np.clip(raw, min_conf, 1.0)


# ── [6] 分散スパイク制御 ────────────────────────────────────────────────────────

def _variance_spike_mult(
    df: pd.DataFrame, var_window: int, var_spike_k: float
) -> np.ndarray:
    """
    per-bucket ローリング分散スパイク検出（先読みなし）。

    1. ローリング分散を計算し shift(1) を適用（先読みなし）
    2. ベースライン = 同ウィンドウのローリング中央値
    3. var_shifted > spike_k × baseline → 重み ×0.8

    Returns:
        multiplier 配列（1.0 または 0.8）、len(df) 要素
    """
    mult = np.ones(len(df), dtype=float)

    for _, grp in df.groupby("bucket_key"):
        pos = grp.index.values  # df は reset_index 済み → 位置インデックス
        x   = _safe_arr(grp["pnl"].values.astype(float))
        n   = len(x)

        # ローリング分散（min_periods=1）→ shift(1) で先読み除去
        roll_var = pd.Series(x).rolling(var_window, min_periods=1).var().values
        var_s    = _shift1(roll_var)

        # ベースライン: var_s のローリング中央値
        baseline = pd.Series(var_s).rolling(var_window, min_periods=1).median().values

        spike     = var_s > var_spike_k * np.maximum(baseline, _EPS)
        mult[pos] = np.where(spike, 0.8, 1.0)

    return mult


# ── [6.5] 時系列平滑化 ─────────────────────────────────────────────────────────

def _smooth_per_bucket(
    df: pd.DataFrame, weight: np.ndarray, beta: float
) -> np.ndarray:
    """
    per-bucket 時系列重み平滑化（index-based cumulative array、approach B）。

    smooth_t = prev_weight × (1 - beta) + current_weight × beta
    - t-1 の重みのみ使用（smooth[i-1] を参照）
    - prev がない場合（最初の行）は current_weight をそのまま使用
    - スケール保存: 凸結合により mean ≈ 1.0 を維持

    Args:
        df     : work DataFrame（reset_index 済み、timestamp 昇順ソート済み）
        weight : 正規化・安定化済み重み配列
        beta   : ∈ [0.05, 0.5]

    Returns:
        平滑化済み重み配列（同じ shape、入力を変更しない）
    """
    result = weight.copy()  # 入力を変更しない

    for _, grp in df.groupby("bucket_key"):
        pos    = grp.index.values  # timestamp 昇順（work のソート順を反映）
        w      = weight[pos]
        n      = len(w)
        smooth = np.empty(n, dtype=float)

        smooth[0] = w[0]  # 事前重みなし → current をそのまま
        for i in range(1, n):
            smooth[i] = (1.0 - beta) * smooth[i - 1] + beta * w[i]

        result[pos] = smooth

    return result


# ── [7] メイン関数 ─────────────────────────────────────────────────────────────

def compute_dynamic_weights(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    動的バケット重みを計算する（純粋関数）。

    Contract:
        - 全特徴量は t-1 以前のデータのみ使用（先読みゼロ）
        - 入力 df を変更しない
        - ランダム性ゼロ
        - 同一入力 → 同一出力（決定論的）

    Args:
        df  : DataFrame with columns {timestamp, bucket_key, pnl}
              timestamp は比較可能な任意の型（int, float, datetime）
        cfg : 設定 dict（_CFG_DEFAULTS 参照。未指定キーはデフォルト値を使用）

    Returns:
        DataFrame with columns:
            timestamp, bucket_key, weight, edge, confidence, ema_pnl
        (timestamp, bucket_key) 昇順ソート済み。全値が有限。

    Raises:
        ValueError : 必須列欠損
    """
    # ── 入力検証 ──────────────────────────────────────────────────────────────
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        return pd.DataFrame(columns=[
            "timestamp", "bucket_key", "weight", "edge", "confidence", "ema_pnl"
        ])

    # cfg のデフォルト値マージ
    c = {**_CFG_DEFAULTS, **cfg}
    ema_alpha   = _safe(c["EMA_ALPHA"],           0.05)
    conf_scale  = max(_EPS, _safe(c["CONF_SCALE"], 10.0))
    min_conf    = _safe(c["MIN_CONF"],             0.0)
    max_weight  = _safe(c["MAX_WEIGHT"],           5.0)
    min_count   = max(1, int(_safe(c["MIN_COUNT"], 5.0)))
    cold_start  = _safe(c["COLD_START_WEIGHT"],    1.0)
    var_window  = max(2, int(_safe(c["VAR_WINDOW"], 10.0)))
    var_spike_k = _safe(c["VAR_SPIKE_K"],          3.0)
    beta        = float(np.clip(_safe(c["WEIGHT_SMOOTH_ALPHA"], 0.3), 0.05, 0.5))

    # ── timestamp 昇順・bucket_key 昇順 厳密ソート（決定論保証）────────────────
    work = (
        df[list(_REQUIRED_COLS)]
        .copy()
        .sort_values(["timestamp", "bucket_key"])
        .reset_index(drop=True)
    )

    # ── [1] per-bucket cumulative EMA（shift(1) 適用済み = 先読みなし）──────────
    # groupby 後の concat を sort_index で元の (timestamp, bucket_key) 順に戻す
    ema_stats = (
        work
        .groupby("bucket_key", group_keys=False)
        .apply(lambda g: _compute_group_emas(g, ema_alpha))
        .sort_index()
    )

    count    = ema_stats["count"].values
    ema_pnl  = ema_stats["ema_pnl"].values
    ema_abs  = ema_stats["ema_abs_pnl"].values
    ema_hit  = ema_stats["ema_hit"].values
    ema_down = ema_stats["ema_downside"].values

    # ── [2] エッジシグナル ─────────────────────────────────────────────────────
    edge = _compute_edge(ema_pnl, ema_abs, ema_hit, ema_down)

    # ── [3] Confidence ─────────────────────────────────────────────────────────
    confidence = _compute_confidence(count, conf_scale, min_conf)

    # ── [4] raw weight = edge × confidence ───────────────────────────────────
    raw_w = _safe_arr(edge * confidence)

    # ── [5] cross-sectional 正規化（per timestamp）────────────────────────────
    # バケット数が timestamp ごとに異なる場合も安全に処理
    work["_raw_w"] = raw_w
    grp_mean = work.groupby("timestamp")["_raw_w"].transform("mean").values
    norm_w   = raw_w / (_safe_arr(grp_mean) + _EPS)
    weight   = np.clip(norm_w, 0.0, max_weight)

    # ── [6] 安定性制約 ─────────────────────────────────────────────────────────
    # count < MIN_COUNT → COLD_START_WEIGHT
    weight = np.where(count < min_count, cold_start, weight)
    # edge が有限でない → 1.0（中立）
    weight = np.where(~np.isfinite(edge), 1.0, weight)

    # 分散スパイク制御: var > spike_k × baseline → ×0.8
    var_mult = _variance_spike_mult(work, var_window, var_spike_k)
    weight   = weight * var_mult

    # ── [6.5] 時系列平滑化（per-bucket、beta = clip(WEIGHT_SMOOTH_ALPHA, 0.05, 0.5)）
    # 正規化・安定化の後に適用。スケール保存。
    weight = _smooth_per_bucket(work, weight, beta)

    # ── 最終 finite ガード ─────────────────────────────────────────────────────
    weight = _safe_arr(weight, fill=cold_start)

    # ── 出力 DataFrame 組み立て ────────────────────────────────────────────────
    result = work[["timestamp", "bucket_key"]].copy()
    result["weight"]     = weight
    result["edge"]       = _safe_arr(edge)
    result["confidence"] = _safe_arr(confidence)
    result["ema_pnl"]    = _safe_arr(ema_pnl)

    return result.reset_index(drop=True)
