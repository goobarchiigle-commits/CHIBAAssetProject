"""
src/execution/sizing.py
実行サイジング計算。純粋関数ベース。副作用なし。ランダム性なし。

[3] opportunity: slippage_est 依存（定数禁止）
    opportunity ∝ max_slippage / (slippage_est * impact_coef)

[4] fill モデル: 適応 alpha
    alpha = max(base_alpha, 1 / (n_obs + 10))

[7] opportunity 正規化:
    - None 安全処理
    - confidence 重み付けブレンド
    - 10x median 超え禁止
    - NaN / inf 禁止

[8] final multiplier:
    final = min(base * cost_mult, opportunity * fill)
    hard clip [0, 1e6]
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# ── デフォルト定数 ──────────────────────────────────────────────────────────────
_DEFAULT_SLIPPAGE_EST: float = 0.001
_DEFAULT_IMPACT_COEF:  float = 1.0
_BLEND_NEUTRAL:        float = 1.0   # confidence=0 時の中立値
_MAX_MEDIAN_AMPLIFY:   float = 10.0  # median の何倍まで許可するか
_HARD_CLIP_LO:         float = 0.0
_HARD_CLIP_HI:         float = 1e6


def _safe(v: Any, fallback: float = 0.0) -> float:
    """NaN / inf / None → fallback。"""
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


# ── [4] 適応 alpha fill モデル ──────────────────────────────────────────────────

def adaptive_alpha(n_obs: int, base_alpha: float = 0.1) -> float:
    """
    適応 EMA alpha を返す（純粋関数）。

    alpha = max(base_alpha, 1 / (n_obs + 10))

    サンプルが少ない初期ほど大きな alpha（直近値を重視）、
    サンプルが増えるにつれて base_alpha に収束する。

    Returns: alpha ∈ (0.0, 1.0]
    """
    n = max(0, int(n_obs))
    base = _safe(base_alpha, fallback=0.1)
    base = max(1e-6, min(1.0, base))
    return min(1.0, max(base, 1.0 / (n + 10)))


def compute_ema_fill(
    fill_history: list[float],
    new_value: float,
    base_alpha: float = 0.1,
) -> float:
    """
    適応 alpha EMA で fill 率を推定する（純粋関数）。

    fill_history が空の場合は new_value をそのまま返す。
    全処理が finite を保証する。

    Args:
        fill_history : 過去の fill 値リスト（最古→最新順）
        new_value    : 追加する最新 fill 値
        base_alpha   : EMA alpha の下限

    Returns:
        更新済み EMA 推定値 ∈ [0.0, 1.0]（通常）または任意 float（fill が 0-1 外の場合）
    """
    n_obs = len([x for x in fill_history if math.isfinite(_safe(x, float("nan")))])
    alpha = adaptive_alpha(n_obs, base_alpha)

    if not fill_history:
        return _safe(new_value, fallback=_BLEND_NEUTRAL)

    # fill_history から EMA を再構築（決定的）
    ema = _safe(fill_history[0], fallback=_BLEND_NEUTRAL)
    for v in fill_history[1:]:
        safe_v = _safe(v, fallback=ema)
        ema = alpha * safe_v + (1.0 - alpha) * ema

    # 最新値を適用
    ema = alpha * _safe(new_value, fallback=ema) + (1.0 - alpha) * ema
    return _safe(ema, fallback=_BLEND_NEUTRAL)


# ── [3][7] opportunity 計算 ────────────────────────────────────────────────────

def compute_opportunity(
    slippage_est: float | None,
    max_slippage: float,
    impact_coef: float,
    cfg: dict,
    confidence: float = 1.0,
    median_opp: float | None = None,
) -> float:
    """
    実行機会スコアを計算する（純粋関数）。

    opportunity ∝ max_slippage / (slippage_est * impact_coef)
    - slippage_est=None → cfg の default_slippage_est にフォールバック
    - confidence 重み付けブレンド（0=中立、1=raw opportunity）
    - 10x median を超える増幅を禁止
    - NaN / inf を返さない

    Args:
        slippage_est : 推定スリッページ（None 可）
        max_slippage : 許容最大スリッページ（cfg から）
        impact_coef  : 市場インパクト係数（state.impact_coeff から）
        cfg          : 設定 dict
                         default_slippage_est: float（default 0.001）
                         max_opportunity:      float（default 10.0）
        confidence   : シグナル信頼度 ∈ [0.0, 1.0]
        median_opp   : 過去 opportunity の中央値（None 可）

    Returns:
        opportunity スコア ∈ [0.0, max_opportunity]、有限値保証
    """
    # slippage_est: None 安全処理
    default_slip = _safe(cfg.get("default_slippage_est", _DEFAULT_SLIPPAGE_EST),
                         fallback=_DEFAULT_SLIPPAGE_EST)
    if slippage_est is None:
        slip = default_slip
    else:
        slip = _safe(slippage_est, fallback=default_slip)
    if slip < 1e-12:
        slip = default_slip

    ic  = max(1e-12, _safe(impact_coef, fallback=_DEFAULT_IMPACT_COEF))
    ms  = max(0.0,   _safe(max_slippage, fallback=0.0))
    conf = max(0.0, min(1.0, _safe(confidence, fallback=1.0)))

    # raw opportunity
    denom = slip * ic
    raw_opp = (ms / denom) if denom > 1e-12 else 0.0
    raw_opp = _safe(raw_opp, fallback=0.0)

    # config clip
    max_opp = max(0.0, _safe(cfg.get("max_opportunity", 10.0), fallback=10.0))
    raw_opp = min(raw_opp, max_opp)

    # confidence-weight ブレンド（confidence=0 → 中立値 1.0）
    blended = conf * raw_opp + (1.0 - conf) * _BLEND_NEUTRAL
    blended = _safe(blended, fallback=0.0)

    # 10x median 増幅禁止
    if median_opp is not None:
        med = _safe(median_opp, fallback=0.0)
        if med > 1e-12:
            blended = min(blended, _MAX_MEDIAN_AMPLIFY * med)

    return max(0.0, _safe(blended, fallback=0.0))


# ── [8] final multiplier ────────────────────────────────────────────────────────

def compute_final_multiplier(
    base: float,
    cost_mult: float,
    opportunity: float,
    fill: float,
    cfg: dict,
) -> float:
    """
    final = min(base * cost_mult, opportunity * fill)
    全入力を _safe で正規化。hard clip [0, 1e6]。

    Args:
        base        : 基本乗数（state.global_multiplier 等）
        cost_mult   : コスト調整係数（cfg.cost_mult 等）
        opportunity : compute_opportunity() の出力
        fill        : compute_ema_fill() の出力
        cfg         : 設定 dict
                        multiplier_min: float（default 0.0）
                        multiplier_max: float（default 1e6）

    Returns:
        final multiplier ∈ [multiplier_min, multiplier_max]、有限値保証
    """
    b   = _safe(base,        fallback=0.0)
    cm  = _safe(cost_mult,   fallback=1.0)
    opp = _safe(opportunity, fallback=0.0)
    f   = _safe(fill,        fallback=1.0)

    lo = _safe(cfg.get("multiplier_min", _HARD_CLIP_LO), fallback=_HARD_CLIP_LO)
    hi = _safe(cfg.get("multiplier_max", _HARD_CLIP_HI), fallback=_HARD_CLIP_HI)
    lo = max(_HARD_CLIP_LO, lo)
    hi = min(_HARD_CLIP_HI, hi)

    path_cost = _safe(b  * cm,  fallback=0.0)
    path_opp  = _safe(opp * f,  fallback=0.0)
    raw       = min(path_cost, path_opp)

    return _safe(max(lo, min(hi, raw)), fallback=lo)
